"""riki-memory as an MCP server, so Claude Code, Cursor and other MCP clients use it directly.

    pip install "riki-mem[mcp]"
    riki serve --user me              # stdio; settings from RIKI_* / .env

One server process serves one memory namespace (``--user`` or ``RIKI_USER_ID``):
the usual setup for a person's own assistant. Tools:

- ``recall``: what memory knows that is relevant to a message (the ``context()`` text)
- ``remember``: something worth keeping (a note, or a user message plus the reply)
- ``list_memories`` / ``forget``: see and remove what is stored
- ``get_core_memory`` / ``set_core_memory``: the always-relevant blocks
"""

from .service import RikiMemory


def build_server(memory: RikiMemory, user_id: str):
    """An MCP server bound to ``user_id``'s memory (mcp 2.x ``MCPServer`` or 1.x ``FastMCP``)."""
    try:
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as Server
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise SystemExit('The MCP server needs the extra: pip install "riki-mem[mcp]"') from exc

    server = Server("riki-memory")

    @server.tool()
    async def recall(message: str, recent_chat: str = "") -> str:
        """What long-term memory knows that is relevant to this message. Call it before
        answering anything personal; counts and dates in it are computed, use them as given."""
        text, _ = await memory.context(user_id, message, recent_chat)
        return text

    @server.tool()
    async def remember(user_message: str, reply: str = "") -> str:
        """Keep what is worth remembering from an exchange (or from a note, as user_message).
        Small talk and one-off questions are ignored."""
        saved = await memory.remember(user_id, user_message, reply)
        return "Saved:\n" + "\n".join(f"- {m.content} (id {m.id})" for m in saved) if saved else "Nothing to keep."

    @server.tool()
    async def list_memories(limit: int = 20, offset: int = 0, only_guesses: bool = False) -> str:
        """What is stored, newest first. only_guesses: what was inferred rather than said."""
        found = await memory.list_memories(
            user_id, limit=min(limit, 100), offset=offset, source="inferred" if only_guesses else None
        )
        return "\n".join(f"- {m.content} (id {m.id})" for m in found) or "(nothing stored)"

    @server.tool()
    async def forget(memory_id: str) -> str:
        """Stop using a memory (kept on record, never recalled again)."""
        return "Forgotten." if await memory.forget(user_id, memory_id) else "No such memory."

    @server.tool()
    async def get_core_memory() -> dict[str, list[str]]:
        """The always-relevant blocks: identity, preferences, constraints, people, projects."""
        return await memory.get_blocks(user_id)

    @server.tool()
    async def set_core_memory(block: str, lines: list[str]) -> str:
        """Replace one block (identity, preferences, constraints, people or projects) with these lines."""
        await memory.set_block(user_id, block, lines)
        return f"{block} updated."

    return server


def serve(user_id: str) -> None:  # pragma: no cover - runs until the client disconnects
    memory = RikiMemory.from_env()
    build_server(memory, user_id).run()  # stdio transport
