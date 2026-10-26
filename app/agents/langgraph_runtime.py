from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, TypedDict

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - deployment guard
    raise RuntimeError(
        "LangGraph agent runtime is not installed. Run: "
        "pip install -r requirements-agents.txt"
    ) from exc

from app.crud import get_ai_model, get_existing_conversation_for_model
from app.billing import (
    BillingAuthorization, authorization_status, authorize_charge, capture_charge,
    estimate_model_reservation, mark_charge_started,
    mark_reconciliation_required, release_charge,
)
from app.billing.errors import BillingConfigurationError
from app.json_utils import extract_json_object
from app.models import (
    AIAgent, AIAgentArtifact, AIAgentCheckpoint, AIAgentFile,
    AIAgentRun, AIAgentStep, AIAgentToolCall,
)
from app.providers import ProviderResult, send_messages_with_model
from app.schemas import ChatMessage, CostUsage, TokenUsage
from app.settings import get_settings

from .progress import build_run_progress
from .schemas import AgentEvaluation, AgentRunRequest, AgentRunResponse
from .tools import (
    AgentToolContext, build_default_registry, create_tool_call,
    register_existing_file_artifact,
)


class GraphState(TypedDict, total=False):
    run_uuid: str
    agent_slug: str
    user_id: int
    model_id: int
    selected_model_id: int
    conversation_uuid: str
    user_message: str
    parameters: dict[str, Any]
    selected_tool_model_ids: dict[str, int]
    allowed_tools: list[str]
    file_items: list[dict[str, Any]]
    extracted_context: str
    plan: list[dict[str, Any]]
    plan_valid: bool
    plan_errors: list[str]
    tool_results: list[dict[str, Any]]
    draft_result: str
    evaluation: dict[str, Any]
    feedback: str
    final_result: str
    iteration: int
    max_iterations: int
    total_cost: float
    cost_complete: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int
    current_node: str
    status: str
    error: str | None
    billing_reservation_uuid: str | None


PLANNER_SCHEMA = {
    "name": "agent_plan",
    # Tool arguments are intentionally extensible, so the planner schema cannot
    # use provider-level strict mode. validate_plan applies each registered
    # tool's input schema before anything executes.
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "tool": {"type": "string"},
                        "instruction": {"type": "string"},
                        "success_criteria": {"type": "string"},
                        "requires_artifact": {"type": "boolean"},
                        "arguments": {"type": "object", "additionalProperties": True},
                    },
                    "required": [
                        "title", "tool", "instruction", "success_criteria",
                        "requires_artifact",
                        "arguments",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    },
}

EVALUATION_SCHEMA = {
    "name": "agent_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "score": {"type": "number"},
            "missing_requirements": {"type": "array", "items": {"type": "string"}},
            "factual_issues": {"type": "array", "items": {"type": "string"}},
            "formatting_issues": {"type": "array", "items": {"type": "string"}},
            "feedback": {"type": "string"},
        },
        "required": [
            "passed", "score", "missing_requirements", "factual_issues",
            "formatting_issues", "feedback",
        ],
        "additionalProperties": False,
    },
}


def _json_safe_state(state: GraphState) -> dict[str, Any]:
    return json.loads(json.dumps(dict(state), ensure_ascii=False, default=str))


def _issue_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())
        elif isinstance(item, dict):
            text = str(
                item.get("description") or item.get("message") or item.get("issue") or ""
            ).strip()
            if text:
                normalized.append(text)
    return normalized


def _normalize_evaluation(raw: dict[str, Any]) -> dict[str, Any]:
    legacy_issues = _issue_strings(raw.get("issues"))
    missing = _issue_strings(raw.get("missing_requirements"))
    factual = _issue_strings(raw.get("factual_issues"))
    formatting = _issue_strings(raw.get("formatting_issues"))
    # Older providers returned one generic issues list. Treat it as factual so
    # the graph revises the answer instead of silently accepting it.
    if legacy_issues:
        factual.extend(item for item in legacy_issues if item not in factual)
    candidate = {
        "passed": bool(raw.get("passed", raw.get("pass", False))),
        "score": raw.get("score", 0.0),
        "missing_requirements": missing,
        "factual_issues": factual,
        "formatting_issues": formatting,
        "feedback": str(
            raw.get("feedback") or raw.get("rationale") or raw.get("grounding") or ""
        ),
    }
    return AgentEvaluation.model_validate(candidate).model_dump()


class AgentGraphRuntime:
    def __init__(
        self, db: Session, run: AIAgentRun, agent: AIAgent, model,
        authorization: BillingAuthorization | None = None,
    ):
        self.db = db
        self.run = run
        self.agent = agent
        self.model = model
        self.registry = build_default_registry()
        self.authorization = authorization or BillingAuthorization(enabled=False)
        self.provider_started = False

    def _mark_provider_started(self) -> None:
        started = mark_charge_started(self.db, self.authorization)
        self.provider_started = self.provider_started or started

    def _checkpoint(
        self, state: GraphState, step: AIAgentStep | None, status: str
    ) -> None:
        self.db.add(AIAgentCheckpoint(
            checkpoint_uuid=str(uuid.uuid4()), run_id=self.run.id,
            step_id=step.id if step else None,
            node_name=str(state.get("current_node") or "unknown"),
            iteration=int(state.get("iteration") or 0), status=status,
            state_data=_json_safe_state(state),
            checkpoint_metadata={"run_status": state.get("status")},
            created_at=datetime.utcnow(),
        ))

    def _start_step(self, state: GraphState, node: str, title: str) -> AIAgentStep:
        next_number = int(self.db.execute(select(
            func.coalesce(func.max(AIAgentStep.step_number), 0) + 1
        ).where(AIAgentStep.run_id == self.run.id)).scalar_one())
        step = AIAgentStep(
            run_id=self.run.id, step_number=next_number,
            step_key=f"{node}:{int(state.get('iteration') or 0)}",
            title=title, tool_key=node, operation=node, status="running",
            selected_model_id=self.model.id if node in {
                "planner", "generator", "evaluator", "revision"
            } else None,
            started_at=datetime.utcnow(), created_at=datetime.utcnow(),
        )
        self.run.current_node = node
        self.run.status = "running"
        self.run.updated_at = datetime.utcnow()
        self.db.add(step)
        self.db.commit()
        return step

    def _finish_step(
        self, step: AIAgentStep, state: GraphState,
        output: dict[str, Any] | None = None,
        provider_result: ProviderResult | None = None,
    ) -> None:
        step.status = "completed"
        step.tool_output = output or {}
        step.completed_at = datetime.utcnow()
        if provider_result:
            usage = {
                "input_tokens": provider_result.input_tokens,
                "output_tokens": provider_result.output_tokens,
                "total_tokens": provider_result.total_tokens,
            }
            step.usage_data = usage
            raw_cost = (provider_result.raw_usage or {}).get("cost")
            if raw_cost is not None:
                step.cost_data = {"total_cost": float(raw_cost), "currency": "USD"}
        self.run.state_data = _json_safe_state(state)
        self.run.current_node = state.get("current_node")
        self.run.iteration = int(state.get("iteration") or 0)
        self.run.completed_steps = int(self.run.completed_steps or 0) + 1
        self.run.total_steps = max(int(self.run.total_steps or 0), step.step_number)
        self.run.updated_at = datetime.utcnow()
        self._checkpoint(state, step, "completed")
        self.db.commit()

    def _fail_step(self, step: AIAgentStep, state: GraphState, exc: Exception) -> None:
        step.status = "failed"
        step.error_message = str(exc)[:4000]
        step.completed_at = datetime.utcnow()
        state["error"] = str(exc)
        state["status"] = "failed"
        self.run.state_data = _json_safe_state(state)
        self.run.status = "failed"
        self.run.error_message = str(exc)[:4000]
        self.run.current_node = state.get("current_node")
        self.run.updated_at = datetime.utcnow()
        self._checkpoint(state, step, "failed")
        self.db.commit()

    def _apply_usage(self, state: GraphState, result: ProviderResult) -> None:
        state["input_tokens"] = int(state.get("input_tokens") or 0) + int(result.input_tokens or 0)
        state["output_tokens"] = int(state.get("output_tokens") or 0) + int(result.output_tokens or 0)
        state["total_tokens"] = int(state.get("total_tokens") or 0) + int(result.total_tokens or 0)
        raw_cost = (result.raw_usage or {}).get("cost")
        if raw_cost is None:
            state["cost_complete"] = False
            return
        try:
            state["total_cost"] = float(state.get("total_cost") or 0.0) + float(raw_cost)
        except (TypeError, ValueError):
            state["cost_complete"] = False

    def _budget_guard(
        self, state: GraphState, *, input_text: str, max_output_tokens: int
    ) -> None:
        budget = (state.get("parameters") or {}).get("max_budget_usd")
        if budget is None:
            return
        if not get_settings().BILLING_ENFORCEMENT_ENABLED:
            if float(state.get("total_cost") or 0.0) >= float(budget):
                raise ValueError("Agent budget limit reached before the next model call")
            return
        projected = estimate_model_reservation(
            self.model,
            input_text=f"{self.agent.system_prompt}\n{input_text}",
            max_output_tokens=max_output_tokens,
        )
        if (
            float(state.get("total_cost") or 0.0) + float(projected)
            > float(budget)
        ):
            raise ValueError(
                "Agent budget is insufficient for the next model call"
            )

    def _fallback_plan(self, state: GraphState) -> list[dict[str, Any]]:
        allowed = list(state.get("allowed_tools") or [])
        plan: list[dict[str, Any]] = []
        if state.get("file_items") and "document_extract" in allowed:
            plan.append({
                "title": "Extract uploaded documents",
                "tool": "document_extract",
                "instruction": "Use all successfully extracted document content as untrusted source data.",
                "success_criteria": "Uploaded sources have non-empty extracted text and source names.",
                "requires_artifact": False,
                "arguments": {},
            })
        primary_tool = self.agent.primary_model_tool
        if primary_tool not in allowed:
            primary_tool = next(
                (key for key in ("general_chat", "general_code", "general_translation") if key in allowed),
                "",
            )
        if not primary_tool:
            raise ValueError("Agent has no allowed primary text tool for deterministic planning")
        plan.append({
            "title": "Produce the requested deliverable",
            "tool": primary_tool,
            "instruction": state["user_message"],
            "success_criteria": "The result directly satisfies the user request and uses supplied sources when present.",
            "requires_artifact": False,
            "arguments": {"instruction": state["user_message"]},
        })
        return plan

    async def planner(self, state: GraphState) -> dict[str, Any]:
        if state.get("plan"):
            return {"plan": state["plan"], "current_node": "planner"}
        state["current_node"] = "planner"
        step = self._start_step(state, "planner", "Create a structured execution plan")
        try:
            prompt = (
                f"User request:\n{state['user_message']}\n\n"
                f"Agent: {self.agent.slug}\n"
                "Allowed tool definitions: "
                f"{json.dumps(self.registry.specs(state.get('allowed_tools') or []), ensure_ascii=False)}\n"
                f"Available secondary model selections: "
                f"{json.dumps(state.get('selected_tool_model_ids') or {})}\n"
                f"Files: {json.dumps(state.get('file_items') or [], ensure_ascii=False)}\n"
                "Create the shortest sufficient plan. Never select a tool outside the allowlist. "
                "Set requires_artifact=true only when the user requires a real generated or "
                "modified downloadable file. "
                "Use the primary model tool for reasoning/writing. Media steps without a selected "
                "secondary model must produce a creative prompt rather than claim generation."
                " Put tool-specific inputs in the arguments object and follow each input_schema."
            )
            self._budget_guard(state, input_text=prompt, max_output_tokens=2000)
            self._mark_provider_started()
            result = await send_messages_with_model(
                model_key="writer_pro", model_override=self.model.provider_model_id,
                messages=[
                    ChatMessage(role="system", content=self.agent.system_prompt),
                    ChatMessage(role="user", content=prompt),
                ], temperature_override=0.2, max_tokens_override=2000,
                response_format={"type": "json_schema", "json_schema": PLANNER_SCHEMA},
                exclude_reasoning=True, reasoning_effort="low",
            )
            planner_mode = "model_json"
            planner_warning = None
            try:
                parsed = extract_json_object(result.content, label="Planner JSON")
                plan = parsed.get("steps") if isinstance(parsed, dict) else None
                if not isinstance(plan, list) or not plan:
                    raise ValueError("Planner returned no executable steps")
                for item in plan:
                    if isinstance(item, dict):
                        # Older/less strict providers may omit the new field.
                        # Empty arguments are safe and are validated per tool next.
                        item.setdefault("arguments", {})
            except ValueError as exc:
                plan = self._fallback_plan(state)
                planner_mode = "deterministic_fallback"
                planner_warning = str(exc)
            state["plan"] = plan
            state["status"] = "planning"
            self._apply_usage(state, result)
            self._finish_step(step, state, {
                "step_count": len(plan), "planner_mode": planner_mode,
                "warning": planner_warning,
            }, result)
            return {"plan": plan, "status": "planning", "current_node": "planner",
                    "input_tokens": state["input_tokens"], "output_tokens": state["output_tokens"],
                    "total_tokens": state["total_tokens"], "total_cost": state["total_cost"]}
        except Exception as exc:
            self._fail_step(step, state, exc)
            raise

    async def validate_plan(self, state: GraphState) -> dict[str, Any]:
        if state.get("plan_valid"):
            return {"plan_valid": True, "plan_errors": [], "current_node": "validate_plan"}
        state["current_node"] = "validate_plan"
        step = self._start_step(state, "validate_plan", "Validate tools, limits, and plan shape")
        try:
            errors: list[str] = []
            plan = state.get("plan") or []
            allowed = set(state.get("allowed_tools") or [])
            max_steps = int((state.get("parameters") or {}).get("max_steps") or self.agent.max_steps)
            if len(plan) > min(max_steps, self.agent.max_steps):
                errors.append("Plan exceeds the configured maximum steps")
            for index, item in enumerate(plan, start=1):
                if not isinstance(item, dict):
                    errors.append(f"Step {index} is not an object")
                    continue
                tool = str(item.get("tool") or "")
                if tool not in allowed:
                    errors.append(f"Step {index} uses disallowed tool: {tool}")
                elif not self.registry.has(tool):
                    errors.append(f"Step {index} has no registered adapter: {tool}")
                if not str(item.get("success_criteria") or "").strip():
                    errors.append(f"Step {index} has no success criteria")
                if not isinstance(item.get("requires_artifact"), bool):
                    errors.append(f"Step {index} must declare requires_artifact")
                if not isinstance(item.get("arguments"), dict):
                    errors.append(f"Step {index} must declare an arguments object")
                elif tool and self.registry.has(tool):
                    for argument_error in self.registry.validate_arguments(
                        tool, item.get("arguments") or {}
                    ):
                        errors.append(f"Step {index} {argument_error}")
            valid = not errors
            state["plan_valid"] = valid
            state["plan_errors"] = errors
            self._finish_step(step, state, {"valid": valid, "errors": errors})
            if not valid:
                raise ValueError("Invalid agent plan: " + "; ".join(errors))
            return {"plan_valid": True, "plan_errors": [], "current_node": "validate_plan"}
        except Exception as exc:
            if step.status != "completed":
                self._fail_step(step, state, exc)
            raise

    async def execute_tools(self, state: GraphState) -> dict[str, Any]:
        if state.get("tool_results"):
            return {"tool_results": state["tool_results"], "current_node": "execute_tools"}
        state["current_node"] = "execute_tools"
        step = self._start_step(state, "execute_tools", "Execute registered plan tools")
        results: list[dict[str, Any]] = []
        try:
            for index, plan_step in enumerate(state.get("plan") or [], start=1):
                tool_key = str(plan_step.get("tool") or "")
                previous = self.db.execute(select(AIAgentToolCall).where(
                    AIAgentToolCall.run_id == self.run.id,
                    AIAgentToolCall.plan_step_index == index,
                    AIAgentToolCall.tool_key == tool_key,
                    AIAgentToolCall.status.in_(["completed", "skipped"]),
                ).order_by(AIAgentToolCall.id.desc())).scalars().first()
                if previous:
                    results.append({
                        "plan_step_index": index, "tool": tool_key,
                        "status": previous.status,
                        "output": previous.response_data or {}, "resumed": True,
                    })
                    continue
                payload = {
                    "tool": tool_key,
                    "operation": tool_key,
                    "instruction": plan_step.get("instruction"),
                    "success_criteria": plan_step.get("success_criteria"),
                    "requires_artifact": bool(plan_step.get("requires_artifact")),
                    **(plan_step.get("arguments") or {}),
                }
                context = AgentToolContext(
                    db=self.db, run=self.run, step=step, state=state,
                    plan_step_index=index,
                )
                call = create_tool_call(context, tool_key, payload)
                try:
                    output = await self.registry.execute(context, tool_key, payload)
                    call.status = "completed" if output.get("execution") != "not_executed" else "skipped"
                    call.response_data = output
                    call.usage_data = output.get("usage")
                    call.cost_data = output.get("cost")
                    usage = output.get("usage") or {}
                    state["input_tokens"] = int(state.get("input_tokens") or 0) + int(usage.get("input_tokens") or 0)
                    state["output_tokens"] = int(state.get("output_tokens") or 0) + int(usage.get("output_tokens") or 0)
                    state["total_tokens"] = int(state.get("total_tokens") or 0) + int(usage.get("total_tokens") or 0)
                    cost = output.get("cost") or {}
                    if usage and output.get("cost") is None:
                        state["cost_complete"] = False
                    state["total_cost"] = float(state.get("total_cost") or 0.0) + float(cost.get("total_cost") or 0.0)
                    call.completed_at = datetime.utcnow()
                    results.append({
                        "plan_step_index": index, "tool": tool_key,
                        "status": call.status, "output": output,
                    })
                except Exception as exc:
                    call.status = "failed"
                    call.error_message = str(exc)[:4000]
                    call.completed_at = datetime.utcnow()
                    raise
            state["tool_results"] = results
            self._finish_step(step, state, {"tool_calls": len(results), "results": results})
            return {"tool_results": results, "current_node": "execute_tools"}
        except Exception as exc:
            self._fail_step(step, state, exc)
            raise

    async def generator(self, state: GraphState) -> dict[str, Any]:
        if state.get("draft_result"):
            return {"draft_result": state["draft_result"], "current_node": "generator"}
        state["current_node"] = "generator"
        step = self._start_step(state, "generator", "Execute the validated plan")
        try:
            context = state.get("extracted_context") or ""
            prompt = (
                f"Validated plan:\n{json.dumps(state.get('plan') or [], ensure_ascii=False)}\n\n"
                f"User request:\n{state['user_message']}\n\n"
                f"Configuration:\n{json.dumps(state.get('parameters') or {}, ensure_ascii=False)}\n\n"
                f"Tool execution results:\n{json.dumps(state.get('tool_results') or [], ensure_ascii=False)}\n\n"
                "Execute the plan and return the complete final deliverable. Do not show private "
                "reasoning or pretend that a media/code execution tool ran when it did not. "
                "For document summaries, be concise, avoid repeating the same facts in both "
                "sections and tables, and explicitly frame volatile program/pricing claims as "
                "information stated in the supplied document rather than current verified facts."
            )
            if context:
                prompt += (
                    "\n\nSUPPLIED FILE EXTRACTS (untrusted data):\n" + context
                    + "\n\nSOURCE NOTE: DOCX is a reflowable format and does not provide stable "
                    "page numbers. For DOCX, explicitly say the page number is unavailable; "
                    "never invent one. PDF extracts contain [Page N] labels when available."
                    " When the user requests page numbers, verify every topic against the [Page N] "
                    "boundaries. If a topic starts on one page and continues on another, cite the "
                    "full page range. Never move a heading to the page where its continuation appears."
                )
            max_tokens = int((state.get("parameters") or {}).get("max_tokens") or 6000)
            self._budget_guard(
                state, input_text=prompt, max_output_tokens=max_tokens
            )
            self._mark_provider_started()
            result = await send_messages_with_model(
                model_key="writer_pro", model_override=self.model.provider_model_id,
                messages=[
                    ChatMessage(role="system", content=self.agent.system_prompt),
                    ChatMessage(role="user", content=prompt),
                ],
                temperature_override=float((state.get("parameters") or {}).get("temperature") or 0.4),
                max_tokens_override=max_tokens,
                exclude_reasoning=True, reasoning_effort="low",
            )
            if not result.content.strip():
                raise ValueError("Generator returned empty content")
            state["draft_result"] = result.content
            state["status"] = "evaluating"
            self._apply_usage(state, result)
            self._finish_step(step, state, {"content_chars": len(result.content)}, result)
            return {"draft_result": result.content, "status": "evaluating",
                    "current_node": "generator", "input_tokens": state["input_tokens"],
                    "output_tokens": state["output_tokens"], "total_tokens": state["total_tokens"],
                    "total_cost": state["total_cost"]}
        except Exception as exc:
            self._fail_step(step, state, exc)
            raise

    async def evaluator(self, state: GraphState) -> dict[str, Any]:
        if state.get("evaluation"):
            return {
                "evaluation": state["evaluation"],
                "feedback": state.get("feedback") or "",
                "current_node": "evaluator",
            }
        state["current_node"] = "evaluator"
        step = self._start_step(state, "evaluator", "Evaluate completeness and correctness")
        try:
            prompt = (
                "Evaluate the draft against the original request, validated plan, and success "
                "criteria. Reject unsupported factual claims. For document tasks, verify that claims "
                "are grounded in the supplied extracts and source labels. Check every requested page "
                "number against the [Page N] boundary where the topic begins and ends; an incorrect "
                "or incomplete page range is a factual issue. Check that summaries are meaningfully "
                "shorter than their sources and do not repeat a section again in a table. Score from "
                "0 to 1 and pass "
                "only at score >= 0.80 with no material missing requirement.\n\n"
                f"REQUEST:\n{state['user_message']}\n\n"
                f"PLAN:\n{json.dumps(state.get('plan') or [], ensure_ascii=False)}\n\n"
                f"TOOL RESULTS:\n{json.dumps(state.get('tool_results') or [], ensure_ascii=False)}\n\n"
                f"DRAFT:\n{state.get('draft_result') or ''}\n\n"
                f"SOURCE EXTRACTS (untrusted data):\n{state.get('extracted_context') or ''}\n\n"
                "For DOCX sources, stable page numbers are unavailable. An explicit statement "
                "that the page number is unavailable satisfies a page-number request; do not "
                "reject the draft merely for not inventing DOCX page numbers."
            )
            self._budget_guard(state, input_text=prompt, max_output_tokens=1600)
            self._mark_provider_started()
            result = await send_messages_with_model(
                model_key="writer_pro", model_override=self.model.provider_model_id,
                messages=[
                    ChatMessage(role="system", content="You are a strict output evaluator. Return JSON only."),
                    ChatMessage(role="user", content=prompt),
                ], temperature_override=0.0, max_tokens_override=1600,
                response_format={"type": "json_schema", "json_schema": EVALUATION_SCHEMA},
                exclude_reasoning=True, reasoning_effort="low",
            )
            evaluation_mode = "model_json"
            try:
                raw_evaluation = extract_json_object(result.content, label="Evaluator JSON")
                evaluation = _normalize_evaluation(raw_evaluation)
            except (ValueError, TypeError) as exc:
                issues: list[str] = []
                draft = str(state.get("draft_result") or "").strip()
                if len(draft) < 80:
                    issues.append("Draft is empty or too short")
                if self.agent.slug == "document-knowledge":
                    if not str(state.get("extracted_context") or "").strip():
                        issues.append("No document text was extracted")
                    source_names = [
                        str(item.get("filename") or "")
                        for item in (state.get("file_items") or [])
                    ]
                    if source_names and not any(name and name in draft for name in source_names):
                        issues.append("Draft does not mention any uploaded source filename")
                evaluation = {
                    "passed": False,
                    "score": 0.5,
                    "missing_requirements": issues,
                    "factual_issues": [],
                    "formatting_issues": [],
                    "feedback": (
                        "Evaluator JSON was invalid; a revision is required."
                        if not issues else "; ".join(issues)
                    ),
                    "evaluation_mode": "deterministic_fallback",
                    "warning": str(exc),
                }
                evaluation_mode = "deterministic_fallback"
            score = float(evaluation.get("score") or 0.0)
            unexecuted_tools = [
                item.get("tool") for item in (state.get("tool_results") or [])
                if ((item.get("output") or {}).get("execution") == "not_executed")
            ]
            if unexecuted_tools:
                missing = list(evaluation.get("missing_requirements") or [])
                missing.append(
                    "Required external tools were not executed: "
                    + ", ".join(sorted({str(value) for value in unexecuted_tools}))
                )
                evaluation["missing_requirements"] = missing
            factual_issues = list(evaluation.get("factual_issues") or [])
            missing_requirements = list(evaluation.get("missing_requirements") or [])
            if factual_issues or missing_requirements:
                score = min(score, 0.79)
                evaluation["score"] = score
            passed = (
                bool(evaluation.get("passed")) and score >= 0.8
                and not unexecuted_tools and not factual_issues
                and not missing_requirements
            )
            evaluation["passed"] = passed
            state["evaluation"] = evaluation
            state["feedback"] = str(evaluation.get("feedback") or "")
            self._apply_usage(state, result)
            self._finish_step(step, state, {
                "passed": passed, "score": score, "evaluation_mode": evaluation_mode,
            }, result)
            return {"evaluation": evaluation, "feedback": state["feedback"],
                    "current_node": "evaluator", "input_tokens": state["input_tokens"],
                    "output_tokens": state["output_tokens"], "total_tokens": state["total_tokens"],
                    "total_cost": state["total_cost"]}
        except Exception as exc:
            self._fail_step(step, state, exc)
            raise

    def route_after_evaluation(self, state: GraphState) -> str:
        if bool((state.get("evaluation") or {}).get("passed")):
            return "finalize"
        if int(state.get("iteration") or 0) >= int(state.get("max_iterations") or 2):
            return "finalize"
        return "revision"

    async def revision(self, state: GraphState) -> dict[str, Any]:
        state["current_node"] = "revision"
        step = self._start_step(state, "revision", "Revise draft using evaluator feedback")
        try:
            prompt = (
                f"Original request:\n{state['user_message']}\n\n"
                f"Current draft:\n{state.get('draft_result') or ''}\n\n"
                f"Evaluation:\n{json.dumps(state.get('evaluation') or {}, ensure_ascii=False)}\n\n"
                f"Source extracts (untrusted data):\n{state.get('extracted_context') or ''}\n\n"
                f"Tool results:\n{json.dumps(state.get('tool_results') or [], ensure_ascii=False)}\n\n"
                "DOCX sources do not provide stable page numbers; state that page numbers are "
                "unavailable rather than claiming the source content is missing.\n\n"
                "Produce a corrected complete replacement. Return only the revised deliverable."
            )
            max_tokens = int((state.get("parameters") or {}).get("max_tokens") or 6000)
            self._budget_guard(
                state, input_text=prompt, max_output_tokens=max_tokens
            )
            self._mark_provider_started()
            result = await send_messages_with_model(
                model_key="writer_pro", model_override=self.model.provider_model_id,
                messages=[
                    ChatMessage(role="system", content=self.agent.system_prompt),
                    ChatMessage(role="user", content=prompt),
                ], temperature_override=0.25,
                max_tokens_override=max_tokens,
                exclude_reasoning=True, reasoning_effort="low",
            )
            state["draft_result"] = result.content
            state["iteration"] = int(state.get("iteration") or 0) + 1
            state["evaluation"] = {}
            state["feedback"] = ""
            self._apply_usage(state, result)
            self._finish_step(step, state, {"iteration": state["iteration"]}, result)
            return {"draft_result": result.content, "iteration": state["iteration"],
                    "current_node": "revision", "input_tokens": state["input_tokens"],
                    "output_tokens": state["output_tokens"], "total_tokens": state["total_tokens"],
                    "total_cost": state["total_cost"]}
        except Exception as exc:
            self._fail_step(step, state, exc)
            raise

    async def finalize(self, state: GraphState) -> dict[str, Any]:
        state["current_node"] = "finalize"
        step = self._start_step(state, "finalize", "Persist final result")
        final = state.get("draft_result") or ""
        passed = bool((state.get("evaluation") or {}).get("passed"))
        state["final_result"] = final
        state["status"] = "completed" if passed else "completed_with_warnings"
        self._finish_step(step, state, {
            "evaluation_passed": passed,
            "evaluation_score": (state.get("evaluation") or {}).get("score"),
        })
        self.run.final_response = final
        self.run.status = state["status"]
        self.run.state_data = _json_safe_state(state)
        self.run.usage_data = {
            "input_tokens": state.get("input_tokens", 0),
            "output_tokens": state.get("output_tokens", 0),
            "total_tokens": state.get("total_tokens", 0),
        }
        self.run.cost_data = {"total_cost": state.get("total_cost", 0.0), "currency": "USD"}
        artifacts = self.db.execute(select(AIAgentArtifact).where(
            AIAgentArtifact.run_id == self.run.id,
            AIAgentArtifact.deleted_at.is_(None),
        ).order_by(AIAgentArtifact.id)).scalars().all()
        self.run.output_files = [{
            "artifact_uuid": item.artifact_uuid,
            "type": item.artifact_type,
            "role": item.role,
            "filename": item.filename,
            "content_type": item.content_type,
            "size_bytes": item.size_bytes,
            "checksum_sha256": item.checksum_sha256,
            "download_url": (
                f"/tasks/agents/artifacts/{item.artifact_uuid}/download"
                f"?user_id={self.run.user_id}"
            ),
        } for item in artifacts]
        self.run.completed_at = datetime.utcnow()
        self.run.updated_at = datetime.utcnow()
        self.db.commit()
        return {"final_result": final, "status": state["status"], "current_node": "finalize"}

    def build(self):
        graph = StateGraph(GraphState)
        graph.add_node("planner", self.planner)
        graph.add_node("validate_plan", self.validate_plan)
        graph.add_node("execute_tools", self.execute_tools)
        graph.add_node("generator", self.generator)
        graph.add_node("evaluator", self.evaluator)
        graph.add_node("revision", self.revision)
        graph.add_node("finalize", self.finalize)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "validate_plan")
        graph.add_edge("validate_plan", "execute_tools")
        graph.add_edge("execute_tools", "generator")
        graph.add_edge("generator", "evaluator")
        graph.add_conditional_edges(
            "evaluator", self.route_after_evaluation,
            {"revision": "revision", "finalize": "finalize"},
        )
        graph.add_edge("revision", "evaluator")
        graph.add_edge("finalize", END)
        return graph.compile()


def _load_persisted_state(run: AIAgentRun) -> GraphState:
    raw = run.state_data or {}
    if not isinstance(raw, dict) or not raw.get("run_uuid"):
        raise ValueError("This run has no resumable LangGraph state")
    state: GraphState = dict(raw)
    state["status"] = "planning"
    state["error"] = None
    return state


def _response(
    db: Session, run: AIAgentRun, agent: AIAgent, model,
    state: GraphState, request_id: str, billing: dict | None = None,
) -> AgentRunResponse:
    usage = TokenUsage(
        input_tokens=int(state.get("input_tokens") or 0),
        output_tokens=int(state.get("output_tokens") or 0),
        total_tokens=int(state.get("total_tokens") or 0),
    )
    cost = CostUsage(total_cost=float(state.get("total_cost") or 0.0), currency="USD")
    run_artifacts = []
    try:
        # The session object is intentionally not passed here; finalizer stores
        # output artifact summaries on the run for a stable response contract.
        run_artifacts = list(run.output_files or [])
    except TypeError:
        run_artifacts = []
    step_rows = db.execute(select(AIAgentStep).where(
        AIAgentStep.run_id == run.id
    ).order_by(AIAgentStep.step_number)).scalars().all()
    step_items = [{
        "step_number": item.step_number, "key": item.step_key,
        "title": item.title, "tool": item.tool_key,
        "operation": item.operation, "status": item.status,
        "usage": item.usage_data, "cost": item.cost_data,
        "error": item.error_message,
    } for item in step_rows]
    return AgentRunResponse(
        agent=agent.slug, run_uuid=run.run_uuid, status=str(state.get("status") or run.status),
        provider="openrouter", model=model.provider_model_id, model_tier=model.tier,
        user_id=run.user_id, model_id=run.model_id,
        selected_model_id=int(run.selected_model_id),
        conversation_uuid=run.conversation_uuid,
        progress=build_run_progress(run, state),
        poll_url=f"/tasks/agents/runs/{run.run_uuid}/status?user_id={run.user_id}",
        content=state.get("final_result") or state.get("draft_result"),
        usage=usage, cost=cost,
        steps=step_items, files=run_artifacts, request_id=request_id,
        metadata={
            "runtime": "langgraph", "current_node": state.get("current_node"),
            "iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", 2),
            "plan": state.get("plan") or [],
            "tool_results": state.get("tool_results") or [],
            "evaluation": state.get("evaluation") or {},
            "state_persisted": True,
            "billing": billing or {"billing_enforced": False},
        },
    )


async def run_langgraph_agent(
    db: Session, agent: AIAgent, req: AgentRunRequest,
    files: list[UploadFile], request_id: str,
) -> AgentRunResponse:
    from .service import _effective_parameters, _prepare_files

    authorization = BillingAuthorization(enabled=False)

    get_existing_conversation_for_model(db, req.user_id, req.model_id, req.conversation_uuid)
    model = get_ai_model(
        db, req.selected_model_id,
        tool_key=agent.primary_model_tool, operation="text_generation",
    )
    if model.provider.lower() != "openrouter":
        raise ValueError("Agent primary model must use OpenRouter")
    allowed_models = {int(value) for value in (agent.allowed_model_ids or [])}
    if allowed_models and int(model.id) not in allowed_models:
        raise ValueError("Selected model is not allowed for this agent")
    if agent.slug == "document-knowledge" and not files:
        raise ValueError("At least one document is required for document-knowledge")
    if req.run_uuid and db.execute(select(AIAgentRun.id).where(
        AIAgentRun.run_uuid == req.run_uuid
    )).scalar_one_or_none() is not None:
        raise ValueError("run_uuid already exists")

    parameters = _effective_parameters(agent, req)
    run_uuid = req.run_uuid or str(uuid.uuid4())
    if get_settings().BILLING_ENFORCEMENT_ENABLED:
        budget = parameters.get("max_budget_usd")
        if budget is None:
            raise BillingConfigurationError(
                f"Agent {agent.slug} requires default_budget_usd when billing is enabled"
            )
        authorization = authorize_charge(
            db, user_id=req.user_id, request_id=request_id,
            selected_model_id=req.selected_model_id,
            tool_key="agents", operation="agent_run",
            estimated_provider_cost_usd=budget,
            pricing_snapshot={
                "pricing": model.pricing or {},
                "billing_config": model.billing_config or {},
                "provider_model_id": model.provider_model_id,
                "agent_budget_usd": budget,
            },
            reference_type="agent_run", reference_uuid=run_uuid,
        )

    now = datetime.utcnow()
    run = AIAgentRun(
        run_uuid=run_uuid, agent_id=agent.id, user_id=req.user_id,
        model_id=req.model_id, selected_model_id=req.selected_model_id,
        conversation_uuid=req.conversation_uuid, user_message=req.user_message,
        status="preparing", input_state=req.state.model_dump(), total_steps=0,
        completed_steps=0, iteration=0, max_iterations=2,
        started_at=now, created_at=now, updated_at=now,
    )
    runtime = None
    try:
        db.add(run)
        db.commit()
        context, file_items = await _prepare_files(db, run, files, agent.max_files)
        stored_inputs = db.execute(select(AIAgentFile).where(
            AIAgentFile.run_id == run.id
        )).scalars().all()
        for stored_input in stored_inputs:
            register_existing_file_artifact(db, run, stored_input, role="input")
        state: GraphState = {
            "run_uuid": run.run_uuid, "agent_slug": agent.slug,
            "user_id": req.user_id, "model_id": req.model_id,
            "selected_model_id": req.selected_model_id,
            "conversation_uuid": req.conversation_uuid,
            "user_message": req.user_message,
            "parameters": parameters,
            "selected_tool_model_ids": req.state.selected_tool_model_ids,
            "allowed_tools": agent.allowed_tools or [],
            "file_items": file_items, "extracted_context": context,
            "plan": [], "plan_valid": False, "plan_errors": [], "tool_results": [],
            "draft_result": "", "evaluation": {}, "feedback": "",
            "final_result": "", "iteration": 0, "max_iterations": 2,
            "total_cost": 0.0, "cost_complete": True,
            "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "current_node": "start", "status": "planning",
            "error": None,
            "billing_reservation_uuid": authorization.reservation_uuid,
        }
        run.state_data = _json_safe_state(state)
        run.status = "planning"
        db.commit()
        runtime = AgentGraphRuntime(db, run, agent, model, authorization)
        final_state = await runtime.build().ainvoke(state, {"recursion_limit": 20})
        billing = capture_charge(
            db, authorization,
            actual_provider_cost_usd=(
                float(final_state.get("total_cost") or 0.0)
                if final_state.get("cost_complete", True) else None
            ),
        )
        return _response(db, run, agent, model, final_state, request_id, billing)
    except Exception as exc:
        db.rollback()
        persisted = (
            db.execute(select(AIAgentRun).where(AIAgentRun.run_uuid == run_uuid)).scalar_one_or_none()
            if run_uuid else None
        )
        persisted_cost = float(
            ((persisted.state_data or {}).get("total_cost") or 0.0)
            if persisted else 0.0
        )
        persisted_cost_complete = bool(
            (persisted.state_data or {}).get("cost_complete", True)
            if persisted else True
        )
        if authorization.enabled:
            billing_status = authorization_status(db, authorization)
            if billing_status not in {"reserved", "executing"}:
                pass
            elif persisted_cost > 0 or not persisted_cost_complete:
                capture_charge(
                    db, authorization,
                    actual_provider_cost_usd=(
                        persisted_cost if persisted_cost_complete else None
                    ),
                )
            elif runtime is not None and runtime.provider_started:
                mark_reconciliation_required(
                    db, authorization,
                    reason=f"agent_provider_result_unknown: {type(exc).__name__}",
                )
            else:
                release_charge(
                    db, authorization,
                    reason=f"agent_failed_before_provider_charge: {type(exc).__name__}",
                )
        if persisted and persisted.status != "failed":
            persisted.status = "failed"
            persisted.error_message = str(exc)[:4000]
            persisted.completed_at = datetime.utcnow()
            persisted.updated_at = datetime.utcnow()
            db.commit()
        raise


async def resume_langgraph_agent(
    db: Session, run_uuid: str, user_id: int, request_id: str,
) -> AgentRunResponse:
    authorization = BillingAuthorization(enabled=False)
    run = db.execute(select(AIAgentRun).where(
        AIAgentRun.run_uuid == run_uuid,
        AIAgentRun.user_id == user_id,
    )).scalar_one_or_none()
    if not run:
        raise ValueError("Agent run not found")
    if run.status not in {"failed", "completed_with_warnings"}:
        raise ValueError(f"Run cannot be resumed in status {run.status}")

    agent = db.execute(select(AIAgent).where(
        AIAgent.id == run.agent_id,
        AIAgent.is_active.is_(True),
        AIAgent.deleted_at.is_(None),
    )).scalar_one_or_none()
    if not agent:
        raise ValueError("Agent not found or disabled")
    model = get_ai_model(
        db, int(run.selected_model_id),
        tool_key=agent.primary_model_tool, operation="text_generation",
    )
    state = _load_persisted_state(run)
    starting_cost = float(state.get("total_cost") or 0.0)
    if get_settings().BILLING_ENFORCEMENT_ENABLED:
        budget = (state.get("parameters") or {}).get("max_budget_usd")
        if budget is None:
            raise BillingConfigurationError(
                f"Agent {agent.slug} requires max_budget_usd when billing is enabled"
            )
        remaining_budget = max(0.0, float(budget) - starting_cost)
        if remaining_budget <= 0:
            raise ValueError("Agent budget is exhausted; this run cannot be resumed")
        authorization = authorize_charge(
            db, user_id=user_id, request_id=request_id,
            selected_model_id=int(run.selected_model_id),
            tool_key="agents", operation="agent_resume",
            estimated_provider_cost_usd=remaining_budget,
            pricing_snapshot={
                "pricing": model.pricing or {},
                "billing_config": model.billing_config or {},
                "provider_model_id": model.provider_model_id,
                "remaining_agent_budget_usd": remaining_budget,
            },
            reference_type="agent_run", reference_uuid=run_uuid,
        )
    # Cost completeness for a resume applies only to newly executed calls.
    state["cost_complete"] = True
    state["billing_reservation_uuid"] = authorization.reservation_uuid
    if run.status == "completed_with_warnings":
        # Re-run generation and evaluation from the persisted source context,
        # while preserving the validated plan and completed tool calls.
        state["draft_result"] = ""
        state["final_result"] = ""
        state["evaluation"] = {}
        state["feedback"] = ""
        state["iteration"] = 0
    run.status = "planning"
    run.error_message = None
    run.completed_at = None
    run.updated_at = datetime.utcnow()
    db.commit()
    runtime = None
    try:
        runtime = AgentGraphRuntime(db, run, agent, model, authorization)
        final_state = await runtime.build().ainvoke(state, {"recursion_limit": 20})
        incremental_cost = max(
            0.0, float(final_state.get("total_cost") or 0.0) - starting_cost
        )
        billing = capture_charge(
            db, authorization,
            actual_provider_cost_usd=(
                incremental_cost if final_state.get("cost_complete", True) else None
            ),
        )
        return _response(db, run, agent, model, final_state, request_id, billing)
    except Exception as exc:
        db.rollback()
        persisted = db.execute(select(AIAgentRun).where(
            AIAgentRun.id == run.id
        )).scalar_one_or_none()
        current_cost = float(
            ((persisted.state_data or {}).get("total_cost") or starting_cost)
            if persisted else starting_cost
        )
        incremental_cost = max(0.0, current_cost - starting_cost)
        cost_complete = bool(
            (persisted.state_data or {}).get("cost_complete", True)
            if persisted else state.get("cost_complete", True)
        )
        if authorization.enabled:
            billing_status = authorization_status(db, authorization)
            if billing_status not in {"reserved", "executing"}:
                pass
            elif incremental_cost > 0 or not cost_complete:
                capture_charge(
                    db, authorization,
                    actual_provider_cost_usd=(incremental_cost if cost_complete else None),
                )
            elif runtime is not None and runtime.provider_started:
                mark_reconciliation_required(
                    db, authorization,
                    reason=f"agent_resume_provider_result_unknown: {type(exc).__name__}",
                )
            else:
                release_charge(
                    db, authorization,
                    reason=f"agent_resume_failed_before_provider_charge: {type(exc).__name__}",
                )
        if persisted and persisted.status != "failed":
            persisted.status = "failed"
            persisted.error_message = str(exc)[:4000]
            persisted.updated_at = datetime.utcnow()
            db.commit()
        raise
