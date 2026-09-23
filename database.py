import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("Не задан DATABASE_URL")

# pool_pre_ping=True — чтобы не отваливались старые соединения к Neon
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)  # ← внутренний ID (для FK)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)  # ← ID из Telegram
    xp = Column(Integer, default=0)
    level = Column(Integer, default=0)
    balance = Column(Float, default=0.0)
    # Настройки бота
    model = Column(String, default="gemini-3.5-flash-lite")
    thinking = Column(String, default="medium")
    history_len = Column(Integer, default=6)
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=4096)
    hf_model = Column(String, default="flux_schnell")
    format = Column(String, default="square")

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))  # ← теперь users.id
    description = Column(String)
    task_type = Column(String)  # "one_time", "daily", "weekly"
    time_str = Column(String)
    days = Column(String)
    streak = Column(Integer, default=0)
    last_completed = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))  # ← теперь users.id
    amount = Column(Float)
    category = Column(String)
    description = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

# Создаем таблицы при первом запуске
Base.metadata.create_all(bind=engine)
