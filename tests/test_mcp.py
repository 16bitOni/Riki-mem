"""The MCP server exposes memory as tools bound to one namespace."""

import pytest

pytest.importorskip("mcp")

from riki_memory import MemoryConfig, MemoryCreate, RikiMemory
from riki_memory.mcp_server import build_server


def _text(result) -> str:
    return "\n".join(getattr(part, "text", "") for part in result.content)


async def test_the_tools_read_and_change_one_users_memory(db_url, provider):
    async with RikiMemory(database_url=db_url, llm=provider, config=MemoryConfig(planner_mode="heuristic")) as mem:
        dog = await mem.save(MemoryCreate(user_id="me", content="Sub adopted a dog named Bruno.", kind="fact"))
        await mem.save(MemoryCreate(user_id="someone-else", content="Their dog is Rex.", kind="fact"))
        server = build_server(mem, "me")

        names = {tool.name for tool in await server.list_tools()}
        assert names == {"recall", "remember", "list_memories", "forget", "get_core_memory", "set_core_memory"}

        recalled = _text(await server.call_tool("recall", {"message": "What is my dog called? Bruno?"}))
        assert "Bruno" in recalled and "Rex" not in recalled

        await server.call_tool("set_core_memory", {"block": "constraints", "lines": ["Allergic to peanuts."]})
        assert await mem.get_blocks("me") == {"constraints": ["Allergic to peanuts."]}

        assert "Forgotten." in _text(await server.call_tool("forget", {"memory_id": dog.id}))
        assert "(nothing stored)" in _text(await server.call_tool("list_memories", {}))
