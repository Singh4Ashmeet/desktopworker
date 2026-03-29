from __future__ import annotations

from models import ToolResult
from tools.registry import ToolContext, tool

_CTX: ToolContext | None = None


def init_context(ctx: ToolContext) -> None:
    global _CTX
    _CTX = ctx


def _ctx() -> ToolContext:
    if _CTX is None:
        raise RuntimeError("Tool context not initialized")
    return _CTX


@tool(
    name="remember_fact",
    description="Store a user fact in fact store and vector memory.",
    parameters_schema={
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "string"},
            "category": {"type": "string"},
        },
        "required": ["key", "value"],
    },
)
async def remember_fact(key: str, value: str, category: str = "general") -> ToolResult:
    ctx = _ctx()
    if ctx.fact_store is None or ctx.vector_store is None:
        return {
            "success": False,
            "output": "Memory stores not configured",
            "data": None,
        }
    await ctx.fact_store.set_fact(key, value, category)
    await ctx.vector_store.add_interaction(
        f"remember_fact:{key}", value, ["remember_fact"]
    )
    return {"success": True, "output": f"Remembered {key}", "data": {"key": key}}


@tool(
    name="recall_fact",
    description="Recall a specific fact by key.",
    parameters_schema={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
)
async def recall_fact(key: str) -> ToolResult:
    ctx = _ctx()
    if ctx.fact_store is None:
        return {"success": False, "output": "Fact store not configured", "data": None}
    value = await ctx.fact_store.get_fact(key)
    if value is None:
        return {
            "success": False,
            "output": f"No fact found for key: {key}",
            "data": None,
        }
    return {"success": True, "output": value, "data": {"key": key, "value": value}}


@tool(
    name="search_memory",
    description="Semantic search over interaction memory.",
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "n_results": {"type": "integer"},
            "no_decay": {"type": "boolean"},
        },
        "required": ["query"],
    },
)
async def search_memory(
    query: str, n_results: int = 5, no_decay: bool = False
) -> ToolResult:
    ctx = _ctx()
    if ctx.vector_store is None:
        return {"success": False, "output": "Vector store not configured", "data": None}
    items = await ctx.vector_store.query_relevant(
        query, n=n_results, use_decay=not no_decay
    )
    return {"success": True, "output": "\n".join(items), "data": {"results": items}}


@tool(
    name="list_facts",
    description="List stored facts by optional category.",
    parameters_schema={
        "type": "object",
        "properties": {"category": {"type": "string"}},
    },
)
async def list_facts(category: str | None = None) -> ToolResult:
    ctx = _ctx()
    if ctx.fact_store is None:
        return {"success": False, "output": "Fact store not configured", "data": None}
    rows = await ctx.fact_store.list_facts(category)
    data = [{"key": r.key, "value": r.value, "category": r.category} for r in rows]
    preview = "\n".join(f"{r['key']}: {r['value']} ({r['category']})" for r in data)
    return {"success": True, "output": preview, "data": {"facts": data}}


@tool(
    name="forget_fact",
    description="Delete a fact by key.",
    parameters_schema={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
)
async def forget_fact(key: str) -> ToolResult:
    ctx = _ctx()
    if ctx.fact_store is None:
        return {"success": False, "output": "Fact store not configured", "data": None}
    ok = await ctx.fact_store.delete_fact(key)
    return {
        "success": ok,
        "output": "Fact deleted" if ok else "Fact not found",
        "data": None,
    }


@tool(
    name="audit_superseded_facts",
    description="List superseded facts and their replacements.",
    parameters_schema={"type": "object", "properties": {}},
)
async def audit_superseded_facts() -> ToolResult:
    ctx = _ctx()
    if ctx.fact_store is None:
        return {"success": False, "output": "Fact store not configured", "data": None}
    rows = await ctx.fact_store.list_superseded()
    preview = "\n".join(
        f"{row['old_key']}={row['old_value']} -> {row.get('new_key')}={row.get('new_value')}"
        for row in rows
    )
    return {
        "success": True,
        "output": preview or "No superseded facts",
        "data": {"rows": rows},
    }
