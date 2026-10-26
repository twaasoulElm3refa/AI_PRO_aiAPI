from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.settings import get_settings


settings = get_settings()

if not settings.DATABASE_URL.strip():
    raise RuntimeError("DATABASE_URL is required to start the API")

if (
    settings.APP_ENV.lower() != "test"
    and not settings.DATABASE_URL.lower().startswith(
        ("mysql://", "mysql+pymysql://", "mariadb://", "mariadb+pymysql://")
    )
):
    raise RuntimeError("DATABASE_URL must use the production MySQL/MariaDB database")

engine = create_engine(
    settings.DATABASE_URL,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
