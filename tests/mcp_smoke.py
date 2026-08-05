"""Smoke test: drive the bigbrain MCP server over stdio like a real client."""

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command="uv",
        args=["run", "bigbrain-mcp"],
        env={**os.environ, "BIGBRAIN_HOME": "/tmp/bb_mcp"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("tools:", sorted(t.name for t in tools.tools))

            r = await session.call_tool(
                "memory_store",
                {
                    "topic": "MCP server for bigbrain",
                    "content": "FastMCP over stdio; tools memory_store/recall/etc.",
                    "tags": ["mcp", "bigbrain"],
                    "importance": 0.8,
                },
            )
            print("store:", r.structuredContent)

            r = await session.call_tool(
                "memory_recall", {"query": "how does the agent talk to memory?", "limit": 3}
            )
            content = r.structuredContent
            print("recall count:", len(content.get("result", [])))
            for m in content.get("result", []):
                print("  ", round(m["similarity"], 3), m["topic"])

            r = await session.call_tool("memory_count", {})
            print("count:", r.structuredContent)


if __name__ == "__main__":
    asyncio.run(main())
