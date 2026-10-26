from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models import AIModel, User, SubTool, Conversation, Message, ModelsConversation


def get_ai_model(
    db: Session,
    model_id: int,
    *,
    tool_key: str | None = None,
    operation: str | None = None,
    require_available: bool = True,
) -> AIModel:
    stmt = select(AIModel).where(
        AIModel.id == model_id,
        AIModel.deleted_at.is_(None),
        AIModel.is_active.is_(True),
    )
    if require_available:
        stmt = stmt.where(AIModel.is_available.is_(True))
    if tool_key:
        stmt = stmt.where(AIModel.tool_key == tool_key)
    if operation:
        stmt = stmt.where(AIModel.operation == operation)
    model = db.execute(stmt).scalar_one_or_none()
    if not model:
        raise ValueError("AI model not found, disabled, unavailable, or not allowed for this tool")
    return model


def list_ai_models(
    db: Session,
    *,
    tool_key: str | None = None,
    operation: str | None = None,
    provider: str | None = None,
    tier: str | None = None,
    search: str | None = None,
    page: int = 1,
    per_page: int = 20,
) -> tuple[list[AIModel], int]:
    filters = [AIModel.deleted_at.is_(None), AIModel.is_active.is_(True)]
    if tool_key:
        filters.append(AIModel.tool_key == tool_key)
    if operation:
        filters.append(AIModel.operation == operation)
    if provider:
        filters.append(AIModel.provider == provider)
    if tier:
        filters.append(AIModel.tier == tier)
    if search:
        pattern = f"%{search.strip()}%"
        filters.append(or_(AIModel.name.like(pattern), AIModel.provider_model_id.like(pattern)))
    total = int(db.execute(select(func.count(AIModel.id)).where(*filters)).scalar_one())
    rows = list(db.execute(
        select(AIModel).where(*filters)
        .order_by(AIModel.is_recommended.desc(), AIModel.sort_order.asc(), AIModel.id.asc())
        .offset((page - 1) * per_page).limit(per_page)
    ).scalars().all())
    return rows, total


def get_user(db: Session, user_id: int) -> User | None:
    stmt = select(User).where(
        User.id == user_id,
        User.deleted_at.is_(None),
    )
    return db.execute(stmt).scalar_one_or_none()


def get_sub_tool(db: Session, sub_tool_id: int) -> SubTool | None:
    stmt = select(SubTool).where(
        SubTool.id == sub_tool_id,
        SubTool.deleted_at.is_(None),
    )
    return db.execute(stmt).scalar_one_or_none()


def get_conversation_by_uuid_for_user(
    db: Session,
    conversation_uuid: str,
    user_id: int,
) -> Conversation | None:
    stmt = select(Conversation).where(
        Conversation.uuid == conversation_uuid,
        Conversation.user_id == user_id,
        Conversation.deleted_at.is_(None),
    )
    return db.execute(stmt).scalar_one_or_none()


def get_existing_conversation_for_task(
    db: Session,
    user_id: int,
    sub_tool_id: int,
    conversation_uuid: str,
) -> Conversation:
    """
    Read-only conversation resolver.

    FastAPI must NOT create conversations and must NOT insert/update messages.
    Laravel should create/find the conversation and save messages.
    This function only validates that the requested conversation exists and belongs
    to the same user and sub-tool.
    """
    user = get_user(db, user_id)
    if not user:
        raise ValueError("User not found")

    sub_tool = get_sub_tool(db, sub_tool_id)
    if not sub_tool:
        raise ValueError("Sub tool not found")

    conversation = get_conversation_by_uuid_for_user(
        db=db,
        conversation_uuid=conversation_uuid,
        user_id=user_id,
    )
    if not conversation:
        raise ValueError(
            "Conversation not found. Create it in Laravel before calling FastAPI."
        )

    if int(conversation.sub_tool_id) != int(sub_tool_id):
        raise ValueError("Conversation sub_tool_id does not match request")

    return conversation


def get_general_conversation_by_uuid_for_user(
    db: Session,
    conversation_uuid: str,
    user_id: int,
) -> ModelsConversation | None:
    stmt = select(ModelsConversation).where(
        ModelsConversation.uuid == str(conversation_uuid).strip(),
        ModelsConversation.user_id == user_id,
    )
    return db.execute(stmt).scalar_one_or_none()


def get_existing_conversation_for_model(
    db: Session,
    user_id: int,
    model_id: int,
    conversation_uuid: str,
) -> ModelsConversation:
    """Validate the tool/model conversation; AI selection is independent."""
    conversation = get_general_conversation_by_uuid_for_user(
        db=db,
        conversation_uuid=conversation_uuid,
        user_id=user_id,
    )
    if not conversation:
        raise ValueError(
            "Model conversation not found. Create it in models_conversations before calling FastAPI."
        )
    if int(conversation.model_id) != int(model_id):
        raise ValueError("Conversation model_id does not match request")
    return conversation


def list_conversations_for_user(db: Session, user_id: int) -> list[Conversation]:
    stmt = (
        select(Conversation)
        .where(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
        )
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
    )
    return list(db.execute(stmt).scalars().all())


def get_recent_messages(db: Session, conversation_id: int, limit: int) -> list[Message]:
    limit = max(0, int(limit or 0))
    if limit <= 0:
        return []

    stmt = (
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
        )
        .order_by(Message.id.desc())
        .limit(limit)
    )
    rows = list(db.execute(stmt).scalars().all())
    rows.reverse()
    return rows


def list_messages(db: Session, conversation_id: int, limit: int = 500) -> list[Message]:
    stmt = (
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
        )
        .order_by(Message.id.asc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())
