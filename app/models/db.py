from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

engine = create_engine(f"sqlite:///{settings.sqlite_path}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    description = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=True)


class ChatMessage(Base):
    """Stores per-session conversation history, used to make follow-up
    questions ('what about the second one?') resolvable across turns."""
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    module = Column(String, nullable=False)  # "docs", "github", "gmail", "news"
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)
