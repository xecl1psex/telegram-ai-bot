import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("Не задан DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(Integer, unique=True, index=True)
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
    user_id = Column(Integer, ForeignKey("users.telegram_id"))
    description = Column(String)
    task_type = Column(String)  # "one_time", "daily", "weekly"
    time_str = Column(String)   # "15:00"
    days = Column(String)       # "Mon,Wed" для еженедельных
    streak = Column(Integer, default=0)
    last_completed = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.telegram_id"))
    amount = Column(Float)
    category = Column(String)
    description = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

# Создаем таблицы при первом запуске
Base.metadata.create_all(bind=engine)
