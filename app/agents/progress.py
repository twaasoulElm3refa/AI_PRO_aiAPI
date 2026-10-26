from __future__ import annotations

from typing import Any


PHASES = {
    "start": ("Preparing", "جاري التجهيز", 5),
    "planner": ("Planning", "جاري إعداد خطة التنفيذ", 15),
    "validate_plan": ("Validating plan", "جاري التحقق من الخطة", 25),
    "execute_tools": ("Using tools", "جاري تنفيذ الأدوات", 45),
    "generator": ("Generating answer", "جاري إعداد الإجابة", 65),
    "evaluator": ("Checking quality", "جاري مراجعة الجودة", 85),
    "revision": ("Improving answer", "جاري تحسين الإجابة", 75),
    "finalize": ("Saving result", "جاري حفظ النتيجة", 95),
}

TERMINAL_STATUSES = {"completed", "completed_with_warnings", "failed", "cancelled"}


def build_run_progress(run: Any, state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = state if isinstance(state, dict) else (run.state_data or {})
    status = str(run.status or state.get("status") or "queued")
    node = str(run.current_node or state.get("current_node") or "start")
    label_en, label_ar, percent = PHASES.get(
        node, ("Processing", "جاري التنفيذ", 10)
    )
    if status == "queued":
        label_en, label_ar, percent = "Queued", "في قائمة الانتظار", 0
    elif status == "preparing":
        label_en, label_ar, percent = "Preparing files", "جاري تجهيز الملفات", 5
    elif status == "completed":
        label_en, label_ar, percent = "Completed", "اكتمل التنفيذ", 100
    elif status == "completed_with_warnings":
        label_en, label_ar, percent = (
            "Completed with warnings", "اكتمل مع ملاحظات", 100
        )
    elif status == "failed":
        label_en, label_ar = "Failed", "فشل التنفيذ"
    elif status == "cancelled":
        label_en, label_ar = "Cancelled", "تم إلغاء التنفيذ"

    return {
        "status": status,
        "phase": node,
        "label": label_en,
        "label_ar": label_ar,
        "percent": percent,
        "iteration": int(run.iteration or state.get("iteration") or 0),
        "max_iterations": int(
            run.max_iterations or state.get("max_iterations") or 0
        ),
        "completed_steps": int(run.completed_steps or 0),
        "total_steps": int(run.total_steps or 0),
        "is_terminal": status in TERMINAL_STATUSES,
        "can_cancel": status in {"queued", "preparing", "planning"},
        "can_resume": status in {"failed", "completed_with_warnings"},
        "message": run.error_message if status == "failed" else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
    }
