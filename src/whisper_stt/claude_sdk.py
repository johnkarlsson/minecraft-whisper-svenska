"""Claude Agent SDK wrapper for Minecraft Q&A pipeline.

Provides Haiku-based filtering/summarization and Opus-based answer generation
with web search, using the Claude Agent SDK.
"""

from __future__ import annotations

import asyncio
import logging
import re

import claude_agent_sdk._internal.client as _sdk_client
import claude_agent_sdk._internal.message_parser as _sdk_parser
from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk._errors import MessageParseError
from claude_agent_sdk.types import ResultMessage, SystemMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monkey-patch: make the SDK skip unknown message types instead of raising
# ---------------------------------------------------------------------------

_original_parse_message = _sdk_parser.parse_message


def _tolerant_parse_message(data: dict) -> SystemMessage | None:  # type: ignore[type-arg,override]
    """Wrapper around parse_message that returns None for unknown types."""
    try:
        return _original_parse_message(data)  # type: ignore[return-value]
    except MessageParseError as e:
        if "Unknown message type" in str(e):
            logger.debug("Skipping unknown SDK message type: %s", data.get("type"))
            return None
        raise


_sdk_client.parse_message = _tolerant_parse_message  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Filter: is this a Minecraft question?
# ---------------------------------------------------------------------------

_FILTER_PROMPT = """\
Du är ett filter. Kontexten är ett barn som spelar Minecraft. \
Svara JA BARA om texten är en fråga (frågar om något, ber om hjälp, undrar \
hur man gör något) OCH refererar till saker som finns i Minecraft. \
Det inkluderar både Minecraft-specifika termer (creeper, crafting, enchanting, \
nether, pickaxe, redstone, spawna, ender dragon, villager, biome) OCH \
vardagliga ord som representerar Minecraft-element (hus, häst, gris, ko, vatten, \
skog, grotta, berg, säng, dörr, bord, ugn, järn, guld, diamant, ull, sten, \
träd, bro, båt, karta, svärd, pilbåge, monster). \
Om texten slutar med "?" är det troligen en Minecraft-fråga — var generös. \
Frågor om huruvida något finns i Minecraft ("finns det bilar i Minecraft?", \
"kan man flyga?") ska ge JA oavsett om svaret är nej. \
Om texten innehåller många separata meningar avslutade med punkt så är det mindre sannolikt att
det handlar om Minecraft. \
Svara NEJ om texten är bakgrundsprat ("ska vi äta?", "kom hit"), \
uttryckligen handlar om något utanför Minecraft ("hur var det i skolan?"), \
eller inte är en fråga (påståenden, utrop, kommentarer). \
Svara BARA med "JA" eller "NEJ" på första raden.

Text: {text}
"""


async def _filter_minecraft_question(text: str) -> bool:
    options = ClaudeAgentOptions(
        model="claude-haiku-4-5",
        max_turns=1,
        allowed_tools=[],
        effort="low",
    )
    result_text = ""
    async with asyncio.timeout(30):
        async for message in query(
            prompt=_FILTER_PROMPT.format(text=text), options=options
        ):
            if message is None:
                continue
            if isinstance(message, ResultMessage) and message.result:
                result_text = message.result
    return result_text.strip().upper().startswith("JA")


_SENTENCE_END = re.compile(r"[.!?…]+")


def _count_sentences(text: str) -> int:
    """Count sentences by splitting on terminal punctuation."""
    parts = _SENTENCE_END.split(text.strip())
    return sum(1 for p in parts if p.strip())


def filter_minecraft_question(text: str) -> bool:
    """Check if text is a Minecraft question. Sync wrapper (call from threads)."""
    if _count_sentences(text) >= 5:
        return False
    try:
        return asyncio.run(_filter_minecraft_question(text))
    except Exception:
        logger.exception("Filter call failed")
        return False


# ---------------------------------------------------------------------------
# Summarize: short Swedish headline
# ---------------------------------------------------------------------------

_HEADLINE_PROMPT = """\
Skriv en kort svensk rubrik (max 6 ord) som sammanfattar denna Minecraft-fråga. \
Svara BARA med rubriken, inget annat.

Fråga: {text}
"""


async def _summarize_question(text: str) -> str:
    options = ClaudeAgentOptions(
        model="claude-haiku-4-5",
        max_turns=1,
        allowed_tools=[],
        effort="low",
    )
    result_text = ""
    async with asyncio.timeout(30):
        async for message in query(
            prompt=_HEADLINE_PROMPT.format(text=text), options=options
        ):
            if message is None:
                continue
            if isinstance(message, ResultMessage) and message.result:
                result_text = message.result
    return result_text.strip() or text[:40]


def summarize_question(text: str) -> str:
    """Generate a short Swedish headline for a Minecraft question. Sync wrapper."""
    try:
        return asyncio.run(_summarize_question(text))
    except Exception:
        logger.exception("Summarize call failed")
        return text[:40]


# ---------------------------------------------------------------------------
# Answer generation: Opus with web search + resumable session
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Du är en vänlig svensk Minecraft-expert som hjälper barn. \
Du svarar BARA på frågor om Minecraft — ignorera allt annat. \
Alla frågor du får ska tolkas i en Minecraft-kontext, även om de är vagt formulerade. \
Till exempel: "hur gör man en säng?" betyder "hur craftar man en säng i Minecraft?", \
"var hittar jag diamanter?" betyder "på vilken y-nivå finns diamanter i Minecraft?". \
Användaren spelar på PS5 (Bedrock Edition). Anpassa dina svar efter det — \
till exempel kontroller, knappar och plattformsspecifika detaljer ska gälla PS5. \
Om ett svar, en funktion, ett kommando eller en mekanism INTE fungerar eller \
inte finns i Bedrock Edition på PS5, ta INTE med det. Nämn aldrig Java Edition-exklusiva \
funktioner (t.ex. spectator mode, hardcore mode, dual wielding, specifika Java-kommandon) \
utan att tydligt säga att de inte finns på PS5. Om frågan handlar om något som inte \
finns på PS5, svara att det tyvärr inte är tillgängligt på deras version. \
Svara på svenska, kort och tydligt. Använd webbsökning vid behov för att ge korrekta \
svar. Håll svaren under 3-4 meningar om inte frågan kräver mer detalj.\
"""


async def _generate_answer(
    new_questions: list[str], session_id: str | None
) -> tuple[str, str | None]:
    if session_id is not None:
        options = ClaudeAgentOptions(resume=session_id)
    else:
        options = ClaudeAgentOptions(
            model="claude-opus-4-6",
            system_prompt=_SYSTEM_PROMPT,
            max_turns=10,
            allowed_tools=["WebSearch", "WebFetch"],
            effort="high",
        )

    prompt = (
        "Kontext: Användaren är ett barn som spelar Minecraft på PS5 "
        "(Bedrock Edition). Svara på svenska. Ge bara svar som gäller "
        "Bedrock Edition på PS5.\n\n"
        + "\n\n".join(f"Fråga: {q}" for q in new_questions)
    )

    result_text = ""
    new_session_id: str | None = None
    async with asyncio.timeout(120):
        async for message in query(prompt=prompt, options=options):
            if message is None:
                continue
            if isinstance(message, ResultMessage):
                new_session_id = message.session_id
                if message.result:
                    result_text = message.result
    return result_text.strip(), new_session_id


def generate_answer(
    new_questions: list[str], session_id: str | None
) -> tuple[str, str | None]:
    """Generate a Swedish Minecraft answer using Opus with web search. Sync wrapper."""
    return asyncio.run(_generate_answer(new_questions, session_id))
