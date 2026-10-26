# Extending agent tools

Agent tools are server-side adapters and are independent of the public site tool
catalog. Every tool still has to be present in an agent's `allowed_tools` list.

Set `AGENT_TOOL_MODULES` to a comma-separated list of trusted Python modules.
Each module exports a synchronous registration hook:

```python
async def lookup_customer(context, payload):
    customer_id = payload["customer_id"]
    # Call the internal service here and return JSON-serializable data.
    return {"execution": "completed", "customer_id": customer_id}


def register_agent_tools(registry):
    registry.register(
        "lookup_customer",
        lookup_customer,
        description="Look up one customer in the internal CRM.",
        input_schema={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
            "additionalProperties": False,
        },
        category="crm",
    )
```

Then add `lookup_customer` to the relevant `ai_agents.allowed_tools` JSON value.
The registered catalog is available at `GET /tasks/agents/tools/catalog`.

Do not load request-controlled module names. `AGENT_TOOL_MODULES` is trusted
deployment configuration and imported inside the API process.
