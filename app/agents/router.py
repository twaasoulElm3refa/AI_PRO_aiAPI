from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.billing.errors import BillingError
from app.crud import list_ai_models
from app.database import get_db
from app.general_tools.model_catalog import serialize_ai_model
from app.models import (
    AIAgent, AIAgentArtifact, AIAgentCheckpoint, AIAgentRun,
    AIAgentStep, AIAgentToolCall,
)
from app.security import verify_internal_api_key
from app.settings import get_settings

from .schemas import AgentRunRequest, AgentRunResponse
from .progress import build_run_progress
from .service import get_agent, serialize_agent
from .tools import build_default_registry


router = APIRouter(prefix="/tasks/agents", tags=["agents"])


def _error(exc: Exception) -> HTTPException:
    if isinstance(exc, BillingError):
        return HTTPException(status_code=exc.status_code, detail=exc.as_detail())
    return HTTPException(status_code=400 if isinstance(exc, ValueError) else 502, detail=str(exc))


def _serialize_run(run: AIAgentRun) -> dict:
    return {
        "run_uuid": run.run_uuid, "agent_id": run.agent_id,
        "user_id": run.user_id, "model_id": run.model_id,
        "selected_model_id": run.selected_model_id,
        "conversation_uuid": run.conversation_uuid, "status": run.status,
        "progress": build_run_progress(run),
        "poll_url": f"/tasks/agents/runs/{run.run_uuid}/status?user_id={run.user_id}",
        "total_steps": run.total_steps, "completed_steps": run.completed_steps,
        "content": run.final_response, "usage": run.usage_data,
        "cost": run.cost_data, "files": run.output_files or [],
        "error": run.error_message,
        "current_node": run.current_node, "iteration": run.iteration,
        "max_iterations": run.max_iterations,
        "state_summary": {
            "plan_steps": len((run.state_data or {}).get("plan") or []),
            "tool_results": len((run.state_data or {}).get("tool_results") or []),
            "evaluation": (run.state_data or {}).get("evaluation") or {},
        },
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


@router.get("")
def list_agents(
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    rows = db.execute(select(AIAgent).where(
        AIAgent.is_active.is_(True), AIAgent.deleted_at.is_(None)
    ).order_by(AIAgent.sort_order, AIAgent.id)).scalars().all()
    return {"success": True, "items": [serialize_agent(row) for row in rows]}


@router.get("/tools/catalog")
def agent_tools_catalog(
    _: None = Depends(verify_internal_api_key),
):
    """List every server-registered agent tool, including external adapters."""
    return {"success": True, "items": build_default_registry().specs()}


@router.get("/runs/{run_uuid}")
def get_run(
    run_uuid: str,
    user_id: int = Query(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    run = db.execute(select(AIAgentRun).where(
        AIAgentRun.run_uuid == run_uuid, AIAgentRun.user_id == user_id
    )).scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return {"success": True, **_serialize_run(run)}


@router.get("/runs/{run_uuid}/status")
def get_run_status(
    run_uuid: str,
    user_id: int = Query(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    """Small polling response for a progress indicator in the client UI."""
    run = db.execute(select(AIAgentRun).where(
        AIAgentRun.run_uuid == run_uuid, AIAgentRun.user_id == user_id
    )).scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return {
        "success": True,
        "run_uuid": run.run_uuid,
        "progress": build_run_progress(run),
        "result_url": f"/tasks/agents/runs/{run.run_uuid}?user_id={user_id}",
    }


@router.get("/runs/{run_uuid}/steps")
def get_run_steps(
    run_uuid: str,
    user_id: int = Query(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    run = db.execute(select(AIAgentRun).where(
        AIAgentRun.run_uuid == run_uuid, AIAgentRun.user_id == user_id
    )).scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    steps = db.execute(select(AIAgentStep).where(
        AIAgentStep.run_id == run.id
    ).order_by(AIAgentStep.step_number)).scalars().all()
    return {
        "success": True, "run_uuid": run.run_uuid,
        "items": [{
            "step_number": step.step_number, "key": step.step_key,
            "title": step.title, "tool": step.tool_key,
            "operation": step.operation, "status": step.status,
            "usage": step.usage_data, "cost": step.cost_data,
            "error": step.error_message,
        } for step in steps],
    }


@router.get("/runs/{run_uuid}/tool-calls")
def get_run_tool_calls(
    run_uuid: str, user_id: int = Query(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    run = db.execute(select(AIAgentRun).where(
        AIAgentRun.run_uuid == run_uuid, AIAgentRun.user_id == user_id
    )).scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    rows = db.execute(select(AIAgentToolCall).where(
        AIAgentToolCall.run_id == run.id
    ).order_by(AIAgentToolCall.id)).scalars().all()
    return {"success": True, "run_uuid": run_uuid, "items": [{
        "call_uuid": row.call_uuid, "plan_step_index": row.plan_step_index,
        "tool": row.tool_key, "operation": row.operation,
        "selected_model_id": row.selected_model_id, "status": row.status,
        "request": row.request_data, "response": row.response_data,
        "usage": row.usage_data, "cost": row.cost_data, "error": row.error_message,
    } for row in rows]}


@router.get("/runs/{run_uuid}/checkpoints")
def get_run_checkpoints(
    run_uuid: str, user_id: int = Query(...),
    include_state: bool = Query(False),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    run = db.execute(select(AIAgentRun).where(
        AIAgentRun.run_uuid == run_uuid, AIAgentRun.user_id == user_id
    )).scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    rows = db.execute(select(AIAgentCheckpoint).where(
        AIAgentCheckpoint.run_id == run.id
    ).order_by(AIAgentCheckpoint.id)).scalars().all()
    return {"success": True, "run_uuid": run_uuid, "items": [{
        "checkpoint_uuid": row.checkpoint_uuid, "node": row.node_name,
        "iteration": row.iteration, "status": row.status,
        "metadata": row.checkpoint_metadata,
        "state": row.state_data if include_state else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    } for row in rows]}


@router.get("/runs/{run_uuid}/artifacts")
def get_run_artifacts(
    run_uuid: str, user_id: int = Query(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    run = db.execute(select(AIAgentRun).where(
        AIAgentRun.run_uuid == run_uuid, AIAgentRun.user_id == user_id
    )).scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    rows = db.execute(select(AIAgentArtifact).where(
        AIAgentArtifact.run_id == run.id, AIAgentArtifact.deleted_at.is_(None)
    ).order_by(AIAgentArtifact.id)).scalars().all()
    return {"success": True, "run_uuid": run_uuid, "items": [{
        "artifact_uuid": row.artifact_uuid, "type": row.artifact_type,
        "role": row.role, "filename": row.filename,
        "content_type": row.content_type, "size_bytes": row.size_bytes,
        "checksum_sha256": row.checksum_sha256,
        "download_url": f"/tasks/agents/artifacts/{row.artifact_uuid}/download?user_id={user_id}",
        "metadata": row.artifact_metadata,
    } for row in rows]}


@router.get("/artifacts/{artifact_uuid}/download")
def download_artifact(
    artifact_uuid: str, user_id: int = Query(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    row = db.execute(select(AIAgentArtifact).join(
        AIAgentRun, AIAgentRun.id == AIAgentArtifact.run_id
    ).where(
        AIAgentArtifact.artifact_uuid == artifact_uuid,
        AIAgentArtifact.deleted_at.is_(None), AIAgentRun.user_id == user_id,
    )).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Agent artifact not found")
    path = Path(row.storage_path).expanduser().resolve()
    settings = get_settings()
    roots = [
        Path(settings.GENERATED_FILES_DIR).expanduser().resolve(),
        Path(settings.GENERATED_MEDIA_FILES_DIR).expanduser().resolve(),
    ]
    if not any(_is_within(path, root) for root in roots):
        raise HTTPException(status_code=403, detail="Artifact path is outside allowed storage")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file was not found")
    return FileResponse(path, media_type=row.content_type or "application/octet-stream", filename=row.filename)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@router.post("/runs/{run_uuid}/cancel")
def cancel_run(
    run_uuid: str,
    user_id: int = Query(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    run = db.execute(select(AIAgentRun).where(
        AIAgentRun.run_uuid == run_uuid, AIAgentRun.user_id == user_id
    )).scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    if run.status not in {"queued", "planning"}:
        raise HTTPException(status_code=409, detail=f"Run cannot be cancelled in status {run.status}")
    run.status = "cancelled"
    db.commit()
    return {"success": True, "run_uuid": run_uuid, "status": "cancelled"}


@router.post("/runs/{run_uuid}/resume", response_model=AgentRunResponse)
async def resume_run(
    run_uuid: str,
    request: Request,
    user_id: int = Query(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> AgentRunResponse:
    try:
        from .langgraph_runtime import resume_langgraph_agent

        return await resume_langgraph_agent(
            db, run_uuid, user_id, request.state.request_id
        )
    except Exception as exc:
        raise _error(exc) from exc


@router.get("/{agent_slug}")
def agent_details(
    agent_slug: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    try:
        return {"success": True, **serialize_agent(get_agent(db, agent_slug))}
    except Exception as exc:
        raise _error(exc) from exc


@router.get("/{agent_slug}/models")
def agent_models(
    agent_slug: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    try:
        agent = get_agent(db, agent_slug)
        rows, _ = list_ai_models(
            db, tool_key=agent.primary_model_tool, operation="text_generation",
            provider="openrouter", page=1, per_page=100,
        )
        allowed = {int(value) for value in (agent.allowed_model_ids or [])}
        if allowed:
            rows = [row for row in rows if int(row.id) in allowed]
        return {
            "success": True, "agent": serialize_agent(agent),
            "items": [serialize_ai_model(row) for row in rows],
        }
    except Exception as exc:
        raise _error(exc) from exc


@router.post("/{agent_slug}/run", response_model=AgentRunResponse)
async def execute_agent(
    agent_slug: str,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> AgentRunResponse:
    try:
        from .langgraph_runtime import run_langgraph_agent

        content_type = (request.headers.get("content-type") or "").lower()
        files = []
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            raw_payload = form.get("payload")
            if not raw_payload:
                raise ValueError("payload is required for multipart agent requests")
            req = AgentRunRequest.model_validate_json(str(raw_payload))
            files = [item for item in form.getlist("files") if getattr(item, "filename", None)]
        elif content_type.startswith("application/json"):
            req = AgentRunRequest.model_validate(await request.json())
        else:
            raise ValueError("Agent request must use application/json or multipart/form-data")
        return await run_langgraph_agent(
            db, get_agent(db, agent_slug), req, files, request.state.request_id
        )
    except Exception as exc:
        raise _error(exc) from exc
