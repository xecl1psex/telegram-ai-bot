import os
import re
import html
import base64
import time
import threading
import json
from io import BytesIO
from PIL import Image
import requests
import telebot
from telebot.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from dotenv import load_dotenv
from pydub import AudioSegment
from huggingface_hub import InferenceClient
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import func

from database import SessionLocal, User, Task, Transaction, Achievement, XPLog

load_dotenv()

TOKEN = os.environ.get("BOT_TOKEN")
API_KEY = os.environ.get("AI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not TOKEN:
    raise ValueError("Не задан BOT_TOKEN")
if not API_KEY:
    raise ValueError("Не задан AI_API_KEY")

AI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None

HF_MODELS = {
    "sdxl": {"id": "stabilityai/stable-diffusion-xl-base-1.0", "name": "🎨 SDXL Base 1.0", "desc": "Мощная, бесплатная."},
    "sdxl_turbo": {"id": "stabilityai/sdxl-turbo", "name": "⚡ SDXL Turbo", "desc": "Быстрая, 1-4 шага."},
    "sd15": {"id": "runwayml/stable-diffusion-v1-5", "name": "🖼 Stable Diffusion 1.5", "desc": "Классика, быстрая."},
}

bot = telebot.TeleBot(TOKEN)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_THINKING = "medium"
DEFAULT_HISTORY_LEN = 6
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_FORMAT = "square"
DEFAULT_HF_MODEL = "sdxl"

MAX_SAVED_ANSWER_LEN = 800
TIMEOUT = 600

TIMEZONE = "Europe/Moscow"
TZ = ZoneInfo(TIMEZONE)

def now_local():
    return datetime.now(TZ).replace(tzinfo=None)

WEEKDAYS_RU = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
WEEKDAYS_MAP = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
    "пн": 0, "вт": 1, "ср": 2, "чт": 3, "пт": 4, "сб": 5, "вс": 6,
}

chat_history = {}

MODEL_INFO = {
    "gemini-3.8-flash": {"name": "🌟 Gemini 3.8 Flash", "desc": "Сложные задачи, рассуждения."},
    "gemini-3.7-flash": {"name": "⚡ Gemini 3.7 Flash", "desc": "Кодинг, видео, агентные задачи."},
    "gemini-3.6-flash": {"name": "🚀 Gemini 3.6 Flash", "desc": "Баланс скорости и качества."},
    "gemini-3.5-flash-lite": {"name": "🍃 Gemini 3.5 Flash-Lite", "desc": "Самая быстрая."},
}

THINKING_LEVELS = {
    "low": "⚡ Быстрый",
    "medium": "🧠 Сбалансированный",
    "high": "🔬 Глубокий",
}

HISTORY_OPTIONS = [3, 6, 10, 20, 50]
TEMPERATURE_OPTIONS = [0.0, 0.3, 0.7, 1.0, 1.5]
MAX_TOKENS_OPTIONS = [512, 1024, 2048, 4096, 8192]

IMAGE_FORMATS = {
    "square": {"name": "⬛ Квадрат 1:1", "desc": "1024×1024", "width": 1024, "height": 1024},
    "wide": {"name": "🖼 Широкий 16:9", "desc": "1344×768", "width": 1344, "height": 768},
    "portrait": {"name": "📱 Вертикальный 9:16", "desc": "768×1344", "width": 768, "height": 1344},
}

ACHIEVEMENTS = {
    "first_task": {"name": "🎯 Первый шаг", "desc": "Первая задача"},
    "tasks_10": {"name": "💪 Трудяга", "desc": "10 задач"},
    "tasks_50": {"name": "🔥 Машина", "desc": "50 задач"},
    "streak_3": {"name": "🥉 3 дня", "desc": "Streak 3"},
    "streak_7": {"name": "🥈 Неделя", "desc": "Streak 7"},
    "streak_30": {"name": "🥇 Месяц", "desc": "Streak 30"},
    "level_5": {"name": "⭐ Опытный", "desc": "Уровень 5"},
    "level_10": {"name": "🌟 Ветеран", "desc": "Уровень 10"},
    "first_income": {"name": "💵 Доход", "desc": "Первый доход"},
    "first_expense": {"name": "🛒 Трата", "desc": "Первая трата"},
    "positive_balance": {"name": "🏦 В плюсе", "desc": "Баланс > 1000"},
    "week_warrior": {"name": "⚔️ Воин", "desc": "10 задач за неделю"},
}

STREAK_BONUSES = {
    "streak_3": 20,
    "streak_7": 50,
    "streak_30": 300,
}

RESET_TITLES = {
    "money": ("💰 Сброс финансов", "Баланс = 0, транзакции удалятся."),
    "xp": ("✨ Сброс XP", "XP, уровень и история XP удалятся."),
    "ach": ("🏆 Сброс достижений", "Все достижения удалятся."),
    "tasks": ("📋 Удаление задач", "Все задачи будут удалены."),
    "all": ("💣 Полный сброс", "Всё удалится: баланс, XP, достижения, задачи."),
}

scheduler = BackgroundScheduler(timezone=TIMEZONE)

def hf_model_info(model_id):
    return HF_MODELS.get(model_id) or HF_MODELS[DEFAULT_HF_MODEL]

def model_info(model_id):
    return MODEL_INFO.get(model_id) or MODEL_INFO[DEFAULT_MODEL]

def format_info(fmt_id):
    return IMAGE_FORMATS.get(fmt_id) or IMAGE_FORMATS[DEFAULT_FORMAT]

def get_db_user(chat_id):
    db = SessionLocal()
    user = db.query(User).filter(User.telegram_id == chat_id).first()
    if not user:
        user = User(telegram_id=chat_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user, db

def add_xp(user, amount, source, db):
    if amount <= 0:
        return None
    user.xp += amount
    db.add(XPLog(user_id=user.id, amount=amount, source=source))
    level_up_msg = None
    xp_needed = int(100 * (1.5 ** user.level))
    while user.xp >= xp_needed:
        user.xp -= xp_needed
        user.level += 1
        xp_needed = int(100 * (1.5 ** user.level))
        level_up_msg = f"🎉 Level {user.level}!"
    return level_up_msg

def unlock_achievement(user, code, db):
    existing = db.query(Achievement).filter(
        Achievement.user_id == user.id,
        Achievement.code == code
    ).first()
    if existing:
        return False
    ach = Achievement(user_id=user.id, code=code, unlocked_at=now_local())
    db.add(ach)
    return True

def check_task_achievements(user, db):
    new_achievements = []
    completed_count = db.query(Task).filter(
        Task.user_id == user.id,
        Task.last_completed.isnot(None)
    ).count()
    if completed_count >= 1 and unlock_achievement(user, "first_task", db):
        new_achievements.append("first_task")
    if completed_count >= 10 and unlock_achievement(user, "tasks_10", db):
        new_achievements.append("tasks_10")
    if completed_count >= 50 and unlock_achievement(user, "tasks_50", db):
        new_achievements.append("tasks_50")
    if user.level >= 5 and unlock_achievement(user, "level_5", db):
        new_achievements.append("level_5")
    if user.level >= 10 and unlock_achievement(user, "level_10", db):
        new_achievements.append("level_10")
    max_streak = db.query(func.max(Task.streak)).filter(Task.user_id == user.id).scalar() or 0
    if max_streak >= 3 and unlock_achievement(user, "streak_3", db):
        new_achievements.append("streak_3")
    if max_streak >= 7 and unlock_achievement(user, "streak_7", db):
        new_achievements.append("streak_7")
    if max_streak >= 30 and unlock_achievement(user, "streak_30", db):
        new_achievements.append("streak_30")
    week_ago = now_local() - timedelta(days=7)
    week_count = db.query(Task).filter(
        Task.user_id == user.id,
        Task.last_completed >= week_ago
    ).count()
    if week_count >= 10 and unlock_achievement(user, "week_warrior", db):
        new_achievements.append("week_warrior")
    return new_achievements

def check_money_achievements(user, money_change, db):
    new_achievements = []
    if money_change > 0 and unlock_achievement(user, "first_income", db):
        new_achievements.append("first_income")
    if money_change < 0 and unlock_achievement(user, "first_expense", db):
        new_achievements.append("first_expense")
    if user.balance >= 1000 and unlock_achievement(user, "positive_balance", db):
        new_achievements.append("positive_balance")
    return new_achievements

def format_achievements_msg(codes, user):
    if not codes:
        return "", 0
    lines = ["\n🏆 " + ", ".join(ACHIEVEMENTS.get(c, {}).get("name", c) for c in codes)]
    bonus_xp = sum(STREAK_BONUSES.get(c, 0) for c in codes)
    if bonus_xp > 0:
        lines.append(f"🎁 Бонус: +{bonus_xp} XP")
    return "\n".join(lines), bonus_xp

def get_finance_summary(user, db):
    week_ago = now_local() - timedelta(days=7)
    expenses = db.query(
        Transaction.category,
        func.sum(Transaction.amount).label("total"),
        func.count(Transaction.id).label("count")
    ).filter(
        Transaction.user_id == user.id,
        Transaction.amount < 0,
        Transaction.date >= week_ago
    ).group_by(Transaction.category).order_by(func.sum(Transaction.amount)).all()

    income = db.query(func.sum(Transaction.amount)).filter(
        Transaction.user_id == user.id,
        Transaction.amount > 0,
        Transaction.date >= week_ago
    ).scalar() or 0

    total_expense = db.query(func.sum(Transaction.amount)).filter(
        Transaction.user_id == user.id,
        Transaction.amount < 0,
        Transaction.date >= week_ago
    ).scalar() or 0

    parts = [f"Баланс: {user.balance:.2f}"]
    if income:
        parts.append(f"Доход за 7 дней: +{income:.0f}")
    if total_expense:
        parts.append(f"Расход за 7 дней: {total_expense:.0f}")

    if expenses:
        cat_lines = []
        for cat, total, cnt in expenses[:5]:
            cat_lines.append(f"  {cat}: {total:.0f} ({cnt})")
        parts.append("Расходы по категориям (7 дней):\n" + "\n".join(cat_lines))
    else:
        parts.append("Транзакций за неделю нет.")

    active_tasks = db.query(Task).filter(Task.user_id == user.id, Task.is_active == True).count()
    parts.append(f"Активных задач: {active_tasks}")
    xp_needed = int(100 * (1.5 ** user.level))
    parts.append(f"Уровень: {user.level} (XP {user.xp}/{xp_needed})")

    return "\n".join(parts)

def execute_query(user, db, query):
    try:
        qtype = query.get("type")
        period = query.get("period", "week")
        limit = int(query.get("limit", 10))
        category = query.get("category")

        now = now_local()
        if period == "week":
            start = now - timedelta(days=7)
        elif period == "month":
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == "all":
            start = None
        else:
            start = now - timedelta(days=7)

        if qtype == "balance":
            return {"balance": round(user.balance, 2)}

        if qtype == "expenses_by_category":
            q = db.query(
                Transaction.category,
                func.sum(Transaction.amount).label("total"),
                func.count(Transaction.id).label("count")
            ).filter(Transaction.user_id == user.id, Transaction.amount < 0)
            if start:
                q = q.filter(Transaction.date >= start)
            if category:
                q = q.filter(Transaction.category == category)
            rows = q.group_by(Transaction.category).order_by(func.sum(Transaction.amount)).all()
            total = sum(float(t) for _, t, _ in rows)
            return {
                "period": period,
                "total": round(total, 2),
                "by_category": [{"category": c, "total": round(float(t), 2), "count": n} for c, t, n in rows]
            }

        if qtype == "income_by_category":
            q = db.query(
                Transaction.category,
                func.sum(Transaction.amount).label("total"),
                func.count(Transaction.id).label("count")
            ).filter(Transaction.user_id == user.id, Transaction.amount > 0)
            if start:
                q = q.filter(Transaction.date >= start)
            rows = q.group_by(Transaction.category).order_by(func.sum(Transaction.amount).desc()).all()
            total = sum(float(t) for _, t, _ in rows)
            return {
                "period": period,
                "total": round(total, 2),
                "by_category": [{"category": c, "total": round(float(t), 2), "count": n} for c, t, n in rows]
            }

        if qtype == "recent_transactions":
            q = db.query(Transaction).filter(Transaction.user_id == user.id)
            if category:
                q = q.filter(Transaction.category == category)
            rows = q.order_by(Transaction.date.desc()).limit(limit).all()
            return {
                "transactions": [
                    {
                        "amount": round(t.amount, 2),
                        "category": t.category,
                        "description": t.description,
                        "date": t.date.strftime("%Y-%m-%d %H:%M") if t.date else None
                    }
                    for t in rows
                ]
            }

        if qtype == "top_categories":
            q = db.query(
                Transaction.category,
                func.sum(Transaction.amount).label("total"),
                func.count(Transaction.id).label("count")
            ).filter(Transaction.user_id == user.id, Transaction.amount < 0)
            if start:
                q = q.filter(Transaction.date >= start)
            rows = q.group_by(Transaction.category).order_by(func.sum(Transaction.amount)).limit(limit).all()
            return {
                "period": period,
                "top": [{"category": c, "total": round(float(t), 2), "count": n} for c, t, n in rows]
            }

        return {"error": f"Неизвестный тип: {qtype}"}
    except Exception as e:
        print(f"Ошибка execute_query: {e}", flush=True)
        return {"error": str(e)}

def get_main_keyboard():
    markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    markup.add(
        KeyboardButton("👤 Профиль"), KeyboardButton("📋 Задачи"),
        KeyboardButton("🏆 Достижения"), KeyboardButton("📊 Неделя"),
        KeyboardButton("🧠 Модели"), KeyboardButton("⚙️ Настройки"),
        KeyboardButton("🎨 Нарисовать"), KeyboardButton("🎨 Модель картинок"),
        KeyboardButton("📐 Формат"), KeyboardButton("🔄 Сбросить историю"),
        KeyboardButton("🗑 Сброс статистики"), KeyboardButton("ℹ️ Помощь"),
        KeyboardButton("📊 Статус"),
    )
    return markup

def get_history(chat_id):
    if chat_id not in chat_history:
        chat_history[chat_id] = []
    return chat_history[chat_id]

def trim_history(chat_id):
    history = get_history(chat_id)
    user, db = get_db_user(chat_id)
    limit = user.history_len
    db.close()
    if len(history) > limit + 1:
        non_system = [m for m in history if m["role"] != "system"]
        if len(non_system) > limit:
            non_system = non_system[-limit:]
        chat_history[chat_id] = non_system

def update_history(chat_id, role, content):
    history = get_history(chat_id)
    if history and history[-1]["role"] == role and history[-1]["content"] == content:
        return
    if role == "assistant" and len(content) > MAX_SAVED_ANSWER_LEN:
        content = content[:MAX_SAVED_ANSWER_LEN] + "..."
    history.append({"role": role, "content": content})
    trim_history(chat_id)

def clear_history(chat_id):
    chat_history[chat_id] = []

def format_plain_text(text):
    escaped = html.escape(text, quote=False)
    return re.sub(r'`([^`\n]+)`', r'<code>\1</code>', escaped)

def markdown_to_html(text):
    parts = re.split(r'```(\w*)\n?(.*?)```', text, flags=re.DOTALL)
    result = []
    i = 0
    while i < len(parts):
        if i % 3 == 0:
            result.append(format_plain_text(parts[i]))
            i += 1
        elif i % 3 == 1:
            lang = parts[i]
            code = parts[i + 1] if i + 1 < len(parts) else ""
            escaped_code = html.escape(code, quote=False)
            if lang:
                result.append(f'<pre><code class="language-{lang}">{escaped_code}</code></pre>')
            else:
                result.append(f'<pre>{escaped_code}</pre>')
            i += 2
        else:
            result.append(html.escape(parts[i], quote=False))
            i += 1
    return ''.join(result)

def split_for_telegram(text, max_len=3500):
    if len(text) <= max_len:
        return [text]
    parts = re.split(r'(```\w*\n?.*?```)', text, flags=re.DOTALL)
    chunks, current = [], ""
    for part in parts:
        if not part:
            continue
        if len(current) + len(part) <= max_len:
            current += part
        else:
            if current:
                chunks.append(current)
            if len(part) > max_len:
                remaining = part
                while len(remaining) > max_len:
                    sub = remaining[:max_len]
                    cut = -1
                    for sep in ('\n\n', '\n', '. ', ' '):
                        p = sub.rfind(sep)
                        if p > max_len // 2:
                            cut = p + len(sep)
                            break
                    if cut == -1:
                        cut = max_len
                    chunks.append(remaining[:cut])
                    remaining = remaining[cut:]
                current = remaining
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks

def send_long_message(chat_id, text, reply_to_message_id=None):
    if not text:
        return
    chunks = split_for_telegram(text)
    for i, chunk in enumerate(chunks):
        html_chunk = markdown_to_html(chunk)
        try:
            if i == 0:
                bot.send_message(chat_id, html_chunk, parse_mode='HTML', reply_to_message_id=reply_to_message_id)
            else:
                bot.send_message(chat_id, html_chunk, parse_mode='HTML')
        except Exception as e:
            print(f"⚠️ HTML fallback: {e}", flush=True)
            if i == 0:
                bot.send_message(chat_id, chunk, reply_to_message_id=reply_to_message_id)
            else:
                bot.send_message(chat_id, chunk)
        if i < len(chunks) - 1:
            time.sleep(0.5)

def translate_to_english(text):
    try:
        payload = {
            "model": "gemini-3.5-flash-lite",
            "messages": [{
                "role": "user",
                "content": f"Translate to English. Output ONLY the translation: {text}"
            }],
            "max_tokens": 300,
            "temperature": 0.1
        }
        r = requests.post(AI_URL, json=payload, headers=HEADERS, timeout=30)
        data = r.json()
        if 'choices' in data and data['choices']:
            english = data['choices'][0]['message']['content'].strip().strip('"\'')
            if english and len(english) > 2:
                return english
    except Exception as e:
        print(f"⚠️ Ошибка перевода: {e}", flush=True)
    return text

def call_gemini(messages, model, temperature, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(AI_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
    data = r.json()
    if 'choices' not in data or not data['choices']:
        raise ValueError(f"Нет choices: {str(data)[:300]}")
    return data['choices'][0]['message']['content'].strip()

def parse_gemini_json(raw_reply):
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw_reply)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', raw_reply)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if not isinstance(parsed, dict):
        return None
    return parsed

def build_system_prompt(user, db):
    now = now_local()
    today_str = now.strftime("%Y-%m-%d")
    weekday_str = WEEKDAYS_RU[now.weekday()]
    time_str = now.strftime("%H:%M")
    finance_summary = get_finance_summary(user, db)

    return (
        "Ты — ассистент. Отвечай кратко.\n\n"
        f"Дата: {today_str} ({weekday_str}), {time_str} МСК.\n\n"
        f"Данные пользователя:\n{finance_summary}\n\n"
        "=== ФИНАНСЫ — ВАЖНО ===\n"
        "Используй РАЗНЫЕ поля в зависимости от того, что сказал пользователь:\n\n"
        "1) money — ИЗМЕНИТЬ баланс (прибавить или вычесть):\n"
        "   • «потратил 500 на еду» → money: -500, category: \"Еда\"\n"
        "   • «купил кофе 300» → money: -300, category: \"Кафе\"\n"
        "   • «зарплата 50000» → money: 50000, category: \"Зарплата\"\n"
        "   • «получил 1000» → money: 1000\n\n"
        "2) set_balance — УСТАНОВИТЬ баланс в точное значение (ЗАМЕНИТЬ, а не прибавить):\n"
        "   • «установи баланс 3327.62» → set_balance: 3327.62\n"
        "   • «мой баланс 5000» → set_balance: 5000\n"
        "   • «на карте 3327.62» → set_balance: 3327.62\n"
        "   • «пусть баланс будет 1000» → set_balance: 1000\n"
        "   • «запиши что баланс 3327.62» → set_balance: 3327.62\n"
        "   • «не вычитай, а просто установи 3327» → set_balance: 3327\n"
        "   • «измени баланс на 3000» → set_balance: 3000\n\n"
        "ВАЖНО: если пользователь называет точное число и хочет, чтобы баланс СТАЛ таким — используй set_balance, а money: 0.\n"
        "Если пользователь ЗАРАБОТАЛ или ПОТРАТИЛ — используй money, а set_balance: null.\n\n"
        "=== ЗАДАЧИ ===\n"
        "- Разовые (one_time): description + date (YYYY-MM-DD) + time (HH:MM). "
        "Если «в четверг» без времени — найди ближайший четверг, time=09:00.\n"
        "- Ежедневные (daily): description + time (HH:MM).\n"
        "- Еженедельные (weekly): description + time + days (Mon,Wed).\n"
        "- Заполняй task всегда, когда просят напомнить / поставить задачу / не забыть.\n\n"
        "=== ОСТАЛЬНОЕ ===\n"
        "- полезное действие → xp (5-100)\n"
        "- вопрос про финансы → query\n\n"
        "Ответ — один JSON:\n"
        "{\"reply\": \"...\", \"xp\": 0, \"money\": 0, \"set_balance\": null, \"category\": \"\", \"task\": null, \"query\": null}\n\n"
        "task: {\"description\": \"...\", \"type\": \"one_time|daily|weekly\", \"date\": \"YYYY-MM-DD\", \"time\": \"HH:MM\", \"days\": \"Mon,Wed\"}\n"
        "query: {\"type\": \"expenses_by_category|income_by_category|recent_transactions|balance|top_categories\", \"category\": \"...\", \"period\": \"week|month|all\", \"limit\": 10}"
    )

def handle_parsed_response(parsed, user, db, user_text, chat_id, message_id, now):
    reply_text_out = (
        parsed.get("reply") or parsed.get("response") or parsed.get("message")
        or parsed.get("text") or parsed.get("content")
    )
    if not reply_text_out:
        for v in parsed.values():
            if isinstance(v, str) and v.strip():
                reply_text_out = v
                break
        if not reply_text_out:
            reply_text_out = "..."

    try:
        xp_gain = int(parsed.get("xp", 0) or 0)
    except (ValueError, TypeError):
        xp_gain = 0

    try:
        money_change = float(parsed.get("money", 0) or 0)
    except (ValueError, TypeError):
        money_change = 0.0

    # НОВОЕ: set_balance — ЗАМЕНИТЬ баланс
    set_balance = parsed.get("set_balance")
    if set_balance is not None:
        try:
            new_balance = float(set_balance)
            user.balance = new_balance
            reply_text_out += f"\n💰 Баланс установлен: {user.balance:.2f}"
            # Установка баланса не считается доходом — не пишем транзакцию
        except (ValueError, TypeError):
            set_balance = None

    category = parsed.get("category") or "Разное"
    task_data = parsed.get("task")

    new_achievements = []

    if xp_gain > 0:
        level_up = add_xp(user, xp_gain, "chat", db)
        reply_text_out += f"\n\n+{xp_gain} XP"
        if level_up:
            reply_text_out += f"\n{level_up}"

    if money_change != 0:
        user.balance += money_change
        db.add(Transaction(user_id=user.id, amount=money_change, category=category, description=user_text[:50]))
        sign = "+" if money_change > 0 else ""
        reply_text_out += f"\n💰 {sign}{money_change} ({category}). Баланс: {user.balance:.2f}"
        new_achievements.extend(check_money_achievements(user, money_change, db))

    if task_data and isinstance(task_data, dict):
        try:
            t_type = task_data.get("type", "one_time")
            t_time = task_data.get("time")
            t_days = task_data.get("days")
            t_date = task_data.get("date")
            t_desc = task_data.get("description", "Задача")

            stored_time = t_time
            if t_type == "one_time":
                if t_date and t_time:
                    stored_time = f"{t_date} {t_time}"
                elif t_date and not t_time:
                    stored_time = f"{t_date} 09:00"
                elif t_time and not t_date:
                    stored_time = f"{now.strftime('%Y-%m-%d')} {t_time}"
            elif t_type in ("daily", "weekly"):
                stored_time = t_time

            if t_type == "weekly" and t_days:
                days_list = []
                for d in str(t_days).split(","):
                    d_clean = d.strip().lower()
                    if d_clean in WEEKDAYS_MAP:
                        days_list.append(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][WEEKDAYS_MAP[d_clean]])
                    else:
                        days_list.append(d.strip())
                t_days = ",".join(days_list)

            new_task = Task(
                user_id=user.id,
                description=t_desc,
                task_type=t_type,
                time_str=stored_time,
                days=t_days,
            )
            db.add(new_task)

            if t_type == "one_time":
                reply_text_out += f"\n📝 {t_desc} — {stored_time}"
            elif t_type == "daily":
                reply_text_out += f"\n🔁 {t_desc} в {t_time}"
            elif t_type == "weekly":
                reply_text_out += f"\n📅 {t_desc} — {t_days} {t_time}"
        except Exception as e:
            print(f"Ошибка задачи: {e}", flush=True)

    new_achievements.extend(check_task_achievements(user, db))

    if new_achievements:
        new_achievements = list(dict.fromkeys(new_achievements))
        ach_msg, bonus_xp = format_achievements_msg(new_achievements, user)
        reply_text_out += ach_msg
        if bonus_xp > 0:
            lvl = add_xp(user, bonus_xp, "streak_bonus", db)
            if lvl:
                reply_text_out += f"\n{lvl}"

    return reply_text_out

# === ФОРМАТ ===
def build_formats_keyboard(chat_id):
    user, db = get_db_user(chat_id)
    current = user.format
    db.close()
    markup = InlineKeyboardMarkup(row_width=1)
    for fmt_id, info in IMAGE_FORMATS.items():
        check = " ✅" if fmt_id == current else ""
        markup.add(InlineKeyboardButton(text=f"{info['name']}{check}", callback_data=f"set_format:{fmt_id}"))
    return markup

@bot.message_handler(commands=['format'])
def show_formats(message):
    bot.send_message(message.chat.id, "📐 Формат:", reply_markup=build_formats_keyboard(message.chat.id))

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_format:'))
def callback_set_format(call):
    chat_id = call.message.chat.id
    fmt_id = call.data.split(':', 1)[1]
    user, db = get_db_user(chat_id)
    user.format = fmt_id
    db.commit()
    db.close()
    info = format_info(fmt_id)
    bot.answer_callback_query(call.id, info['name'])
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"✅ {info['name']}")

# === HF MODELS ===
def build_hf_models_keyboard(chat_id):
    user, db = get_db_user(chat_id)
    current = user.hf_model
    db.close()
    markup = InlineKeyboardMarkup(row_width=1)
    for model_id, info in HF_MODELS.items():
        check = " ✅" if model_id == current else ""
        markup.add(InlineKeyboardButton(text=f"{info['name']}{check}", callback_data=f"set_hf_model:{model_id}"))
    return markup

@bot.message_handler(commands=['image_models'])
def show_hf_models(message):
    bot.send_message(message.chat.id, "🎨 Модель генерации:", reply_markup=build_hf_models_keyboard(message.chat.id))

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_hf_model:'))
def callback_set_hf_model(call):
    chat_id = call.message.chat.id
    model_id = call.data.split(':', 1)[1]
    user, db = get_db_user(chat_id)
    user.hf_model = model_id
    db.commit()
    db.close()
    info = hf_model_info(model_id)
    bot.answer_callback_query(call.id, info['name'])
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"✅ {info['name']}")

# === GEMINI MODELS ===
def build_models_keyboard(chat_id):
    user, db = get_db_user(chat_id)
    current_model = user.model
    db.close()
    markup = InlineKeyboardMarkup(row_width=1)
    for model_id, info in MODEL_INFO.items():
        check = " ✅" if model_id == current_model else ""
        markup.add(InlineKeyboardButton(text=f"{info['name']}{check}", callback_data=f"model:{model_id}"))
    return markup

@bot.message_handler(commands=['models'])
def show_models(message):
    bot.send_message(message.chat.id, "🧠 Модель:", reply_markup=build_models_keyboard(message.chat.id))

@bot.callback_query_handler(func=lambda call: call.data.startswith('model:'))
def callback_model_info(call):
    chat_id = call.message.chat.id
    model_id = call.data.split(':', 1)[1]
    info = MODEL_INFO.get(model_id)
    if not info:
        bot.answer_callback_query(call.id, "Не найдено")
        return
    markup = InlineKeyboardMarkup(row_width=1)
    markup.add(InlineKeyboardButton("✅ Выбрать", callback_data=f"confirm_model:{model_id}"))
    markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="back_to_models"))
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"*{info['name']}*\n{info['desc']}", reply_markup=markup, parse_mode="Markdown")
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith('confirm_model:'))
def callback_confirm_model(call):
    chat_id = call.message.chat.id
    model_id = call.data.split(':', 1)[1]
    user, db = get_db_user(chat_id)
    user.model = model_id
    db.commit()
    db.close()
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"✅ {model_info(model_id)['name']}")
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data == 'back_to_models')
def callback_back(call):
    chat_id = call.message.chat.id
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="🧠 Модель:", reply_markup=build_models_keyboard(chat_id))
    bot.answer_callback_query(call.id)

# === НАСТРОЙКИ ===
def build_settings_keyboard(chat_id):
    user, db = get_db_user(chat_id)
    markup = InlineKeyboardMarkup(row_width=1)
    markup.add(InlineKeyboardButton(text=f"📏 Контекст: {user.history_len}", callback_data="settings:history"))
    markup.add(InlineKeyboardButton(text=f"🎲 Температура: {user.temperature}", callback_data="settings:temperature"))
    markup.add(InlineKeyboardButton(text=f"📝 Токенов: {user.max_tokens}", callback_data="settings:max_tokens"))
    markup.add(InlineKeyboardButton(text="🔄 По умолчанию", callback_data="settings:reset"))
    db.close()
    return markup

@bot.message_handler(commands=['settings'])
def show_settings(message):
    bot.send_message(message.chat.id, "⚙️ Настройки:", reply_markup=build_settings_keyboard(message.chat.id))

@bot.callback_query_handler(func=lambda call: call.data.startswith('settings:'))
def callback_settings(call):
    chat_id = call.message.chat.id
    action = call.data.split(':', 1)[1]
    user, db = get_db_user(chat_id)

    if action == "history":
        current = user.history_len
        markup = InlineKeyboardMarkup(row_width=1)
        for opt in HISTORY_OPTIONS:
            check = " ✅" if opt == current else ""
            markup.add(InlineKeyboardButton(text=f"{opt}{check}", callback_data=f"set_history:{opt}"))
        markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:back"))
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="📏 Контекст:", reply_markup=markup)
    elif action == "temperature":
        current = user.temperature
        markup = InlineKeyboardMarkup(row_width=1)
        labels = {0.0: "0.0 — строго", 0.3: "0.3", 0.7: "0.7 — баланс", 1.0: "1.0 — креативно", 1.5: "1.5 — макс"}
        for opt in TEMPERATURE_OPTIONS:
            check = " ✅" if opt == current else ""
            markup.add(InlineKeyboardButton(text=f"{labels[opt]}{check}", callback_data=f"set_temperature:{opt}"))
        markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:back"))
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="🎲 Температура:", reply_markup=markup)
    elif action == "max_tokens":
        current = user.max_tokens
        markup = InlineKeyboardMarkup(row_width=1)
        for opt in MAX_TOKENS_OPTIONS:
            check = " ✅" if opt == current else ""
            markup.add(InlineKeyboardButton(text=f"{opt}{check}", callback_data=f"set_tokens:{opt}"))
        markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:back"))
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="📝 Макс. токенов:", reply_markup=markup)
    elif action == "reset":
        user.history_len = DEFAULT_HISTORY_LEN
        user.temperature = DEFAULT_TEMPERATURE
        user.max_tokens = DEFAULT_MAX_TOKENS
        db.commit()
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="✅ Сброшено", reply_markup=build_settings_keyboard(chat_id))
    elif action == "back":
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="⚙️ Настройки:", reply_markup=build_settings_keyboard(chat_id))

    db.close()
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_history:'))
def callback_set_history(call):
    chat_id = call.message.chat.id
    val = int(call.data.split(':', 1)[1])
    user, db = get_db_user(chat_id)
    user.history_len = val
    db.commit()
    db.close()
    trim_history(chat_id)
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="⚙️ Настройки:", reply_markup=build_settings_keyboard(chat_id))
    bot.answer_callback_query(call.id, f"Контекст: {val}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_temperature:'))
def callback_set_temp(call):
    chat_id = call.message.chat.id
    val = float(call.data.split(':', 1)[1])
    user, db = get_db_user(chat_id)
    user.temperature = val
    db.commit()
    db.close()
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="⚙️ Настройки:", reply_markup=build_settings_keyboard(chat_id))
    bot.answer_callback_query(call.id, f"Температура: {val}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_tokens:'))
def callback_set_tokens(call):
    chat_id = call.message.chat.id
    val = int(call.data.split(':', 1)[1])
    user, db = get_db_user(chat_id)
    user.max_tokens = val
    db.commit()
    db.close()
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="⚙️ Настройки:", reply_markup=build_settings_keyboard(chat_id))
    bot.answer_callback_query(call.id, f"Токенов: {val}")

# === ПРОФИЛЬ ===
@bot.message_handler(commands=['profile'])
def profile_command(message):
    chat_id = message.chat.id
    user, db = get_db_user(chat_id)
    xp_needed = int(100 * (1.5 ** user.level))
    ach_count = db.query(Achievement).filter(Achievement.user_id == user.id).count()
    text = (
        f"👤 **Профиль**\n"
        f"🏆 Уровень: {user.level}\n"
        f"✨ XP: {user.xp}/{xp_needed}\n"
        f"💰 Баланс: {user.balance:.2f}\n"
        f"🏅 Достижения: {ach_count}/{len(ACHIEVEMENTS)}"
    )
    bot.reply_to(message, text, parse_mode="Markdown")
    db.close()

# === ДОСТИЖЕНИЯ ===
@bot.message_handler(commands=['achievements'])
def achievements_command(message):
    chat_id = message.chat.id
    user, db = get_db_user(chat_id)
    unlocked = {a.code for a in db.query(Achievement).filter(Achievement.user_id == user.id).all()}
    lines = [f"🏆 Достижения ({len(unlocked)}/{len(ACHIEVEMENTS)})\n"]
    for code, info in ACHIEVEMENTS.items():
        mark = "✅" if code in unlocked else "🔒"
        lines.append(f"{mark} {info['name']} — {info['desc']}")
    bot.reply_to(message, "\n".join(lines), reply_markup=get_main_keyboard())
    db.close()

@bot.message_handler(func=lambda m: m.text == "🏆 Достижения")
def achievements_button(message):
    achievements_command(message)

# === НЕДЕЛЯ ===
@bot.message_handler(commands=['week'])
def week_command(message):
    chat_id = message.chat.id
    user, db = get_db_user(chat_id)
    week_ago = now_local() - timedelta(days=7)

    xp_total = db.query(func.sum(XPLog.amount)).filter(
        XPLog.user_id == user.id, XPLog.date >= week_ago
    ).scalar() or 0

    tasks_done = db.query(Task).filter(
        Task.user_id == user.id, Task.last_completed >= week_ago
    ).count()

    money_in = db.query(func.sum(Transaction.amount)).filter(
        Transaction.user_id == user.id, Transaction.amount > 0, Transaction.date >= week_ago
    ).scalar() or 0

    money_out = db.query(func.sum(Transaction.amount)).filter(
        Transaction.user_id == user.id, Transaction.amount < 0, Transaction.date >= week_ago
    ).scalar() or 0

    top_cats = db.query(
        Transaction.category, func.sum(Transaction.amount).label("total")
    ).filter(
        Transaction.user_id == user.id, Transaction.amount < 0, Transaction.date >= week_ago
    ).group_by(Transaction.category).order_by(func.sum(Transaction.amount)).limit(3).all()

    ach_week = db.query(Achievement).filter(
        Achievement.user_id == user.id, Achievement.unlocked_at >= week_ago
    ).count()

    lines = [
        "📊 **7 дней**\n",
        f"✨ XP: **{xp_total}**",
        f"✅ Задач: **{tasks_done}**",
        f"💰 Доход: **+{money_in:.0f}**",
        f"💸 Расход: **{money_out:.0f}**",
        f"📈 Итог: **{money_in + money_out:+.0f}**",
        f"🏆 Достижений: **{ach_week}**",
    ]
    if top_cats:
        lines.append("\n**Топ расходов:**")
        for cat, total in top_cats:
            lines.append(f"• {cat}: {total:.0f}")

    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown", reply_markup=get_main_keyboard())
    db.close()

@bot.message_handler(func=lambda m: m.text == "📊 Неделя")
def week_button(message):
    week_command(message)

# === СБРОС СТАТИСТИКИ ===
def build_reset_menu_keyboard():
    markup = InlineKeyboardMarkup(row_width=1)
    markup.add(InlineKeyboardButton("💰 Финансы", callback_data="rstats:money"))
    markup.add(InlineKeyboardButton("✨ XP и уровень", callback_data="rstats:xp"))
    markup.add(InlineKeyboardButton("🏆 Достижения", callback_data="rstats:ach"))
    markup.add(InlineKeyboardButton("📋 Задачи", callback_data="rstats:tasks"))
    markup.add(InlineKeyboardButton("💣 Всё", callback_data="rstats:all"))
    markup.add(InlineKeyboardButton("❌ Отмена", callback_data="rstats:cancel"))
    return markup

def build_reset_confirm_keyboard(action):
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("✅ Да", callback_data=f"rstats:do:{action}"),
        InlineKeyboardButton("❌ Нет", callback_data="rstats:menu"),
    )
    return markup

@bot.message_handler(commands=['reset_stats'])
def reset_stats_command(message):
    bot.send_message(message.chat.id, "🗑 Что сбросить?", reply_markup=build_reset_menu_keyboard())

@bot.message_handler(func=lambda m: m.text == "🗑 Сброс статистики")
def reset_stats_button(message):
    reset_stats_command(message)

@bot.callback_query_handler(func=lambda call: call.data == 'rstats:menu')
def cb_rstats_menu(call):
    try:
        bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text="🗑 Что сбросить?", reply_markup=build_reset_menu_keyboard())
    except Exception:
        pass
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data == 'rstats:cancel')
def cb_rstats_cancel(call):
    try:
        bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text="❌ Отменено")
    except Exception:
        pass
    bot.answer_callback_query(call.id, "Отменено")

@bot.callback_query_handler(func=lambda call: call.data in ('rstats:money', 'rstats:xp', 'rstats:ach', 'rstats:tasks', 'rstats:all'))
def cb_rstats_confirm(call):
    action = call.data.split(':', 1)[1]
    title, desc = RESET_TITLES.get(action, ("?", "?"))
    try:
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text=f"⚠️ *{title}*\n{desc}\n\nТочно?",
            parse_mode="Markdown",
            reply_markup=build_reset_confirm_keyboard(action)
        )
    except Exception:
        pass
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith('rstats:do:'))
def cb_rstats_do(call):
    action = call.data.split(':', 2)[2]
    chat_id = call.message.chat.id
    user, db = get_db_user(chat_id)
    try:
        if action == "money":
            db.query(Transaction).filter(Transaction.user_id == user.id).delete()
            user.balance = 0.0
            msg = "✅ Финансы сброшены"
        elif action == "xp":
            db.query(XPLog).filter(XPLog.user_id == user.id).delete()
            user.xp = 0
            user.level = 0
            msg = "✅ XP сброшен"
        elif action == "ach":
            db.query(Achievement).filter(Achievement.user_id == user.id).delete()
            msg = "✅ Достижения сброшены"
        elif action == "tasks":
            db.query(Task).filter(Task.user_id == user.id).delete()
            msg = "✅ Задачи удалены"
        elif action == "all":
            db.query(Transaction).filter(Transaction.user_id == user.id).delete()
            db.query(XPLog).filter(XPLog.user_id == user.id).delete()
            db.query(Achievement).filter(Achievement.user_id == user.id).delete()
            db.query(Task).filter(Task.user_id == user.id).delete()
            user.xp = 0
            user.level = 0
            user.balance = 0.0
            msg = "💣 Всё сброшено"
        else:
            msg = "❌ Неизвестное действие"
        db.commit()
    except Exception as e:
        db.rollback()
        msg = f"❌ Ошибка: {e}"
    finally:
        db.close()
    try:
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=msg)
    except Exception:
        pass
    bot.answer_callback_query(call.id, "Готово")

# === ЗАДАЧИ ===
def format_task_line(task, idx=None):
    emoji = {"one_time": "🎯", "daily": "🔁", "weekly": "📅"}.get(task.task_type, "📌")
    prefix = f"{idx}. " if idx else ""
    streak = f" 🔥{task.streak}" if task.streak > 0 else ""
    time_part = ""
    if task.task_type == "one_time" and task.time_str:
        try:
            dt = datetime.strptime(task.time_str, "%Y-%m-%d %H:%M")
            time_part = f" — {dt.strftime('%d.%m %H:%M')}"
        except Exception:
            time_part = f" — {task.time_str}"
    elif task.time_str:
        time_part = f" — {task.time_str}"
        if task.task_type == "weekly" and task.days:
            days_ru = ", ".join(WEEKDAYS_RU.get(WEEKDAYS_MAP.get(d.strip().lower(), -1), d) for d in task.days.split(","))
            time_part = f" — {days_ru} {task.time_str}"
    return f"{prefix}{emoji} {task.description}{time_part}{streak}"

def build_tasks_keyboard(tasks):
    markup = InlineKeyboardMarkup(row_width=2)
    for task in tasks:
        markup.add(
            InlineKeyboardButton(text=f"✅ {task.id}", callback_data=f"task_done:{task.id}"),
            InlineKeyboardButton(text=f"🗑 {task.id}", callback_data=f"task_del:{task.id}")
        )
    return markup

def get_active_tasks(user_db_id, db):
    return db.query(Task).filter(Task.user_id == user_db_id, Task.is_active == True).order_by(Task.id).all()

@bot.message_handler(commands=['tasks'])
def tasks_command(message):
    chat_id = message.chat.id
    user, db = get_db_user(chat_id)
    tasks = get_active_tasks(user.id, db)
    if not tasks:
        bot.reply_to(message,
            "📋 Задач нет.\n\n"
            "Примеры:\n"
            "• «Завтра в 15:00 позвонить врачу»\n"
            "• «Каждый день в 8:00 выпить воду»\n"
            "• «Каждый Пн и Ср в 19:00 читать»",
            reply_markup=get_main_keyboard())
        db.close()
        return
    lines = ["📋 **Задачи:**\n"]
    for i, t in enumerate(tasks, 1):
        lines.append(format_task_line(t, i))
    lines.append("\n✅ <id> — выполнить, 🗑 <id> — удалить")
    bot.reply_to(message, "\n".join(lines), parse_mode="Markdown", reply_markup=build_tasks_keyboard(tasks))
    db.close()

@bot.message_handler(func=lambda m: m.text == "📋 Задачи")
def tasks_button(message):
    tasks_command(message)

@bot.callback_query_handler(func=lambda call: call.data.startswith('task_done:'))
def callback_task_done(call):
    chat_id = call.message.chat.id
    task_id = int(call.data.split(':', 1)[1])
    user, db = get_db_user(chat_id)
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()
    if not task:
        bot.answer_callback_query(call.id, "Не найдено")
        db.close()
        return

    task.last_completed = now_local()
    if task.task_type in ("daily", "weekly"):
        task.streak = (task.streak or 0) + 1
        task.reminder_sent = None
        streak_msg = f" 🔥{task.streak}"
    else:
        task.is_active = False
        streak_msg = ""

    xp_for_task = 10 if task.task_type == "one_time" else 5
    level_up = add_xp(user, xp_for_task, "task", db)

    new_achievements = check_task_achievements(user, db)
    ach_msg, bonus_xp = format_achievements_msg(new_achievements, user) if new_achievements else ("", 0)

    bonus_level_up = None
    if bonus_xp > 0:
        bonus_level_up = add_xp(user, bonus_xp, "streak_bonus", db)

    db.commit()
    bot.answer_callback_query(call.id, "✅")

    remaining = get_active_tasks(user.id, db)
    if remaining:
        lines = ["📋 **Задачи:**\n"]
        for i, t in enumerate(remaining, 1):
            lines.append(format_task_line(t, i))
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="\n".join(lines), parse_mode="Markdown", reply_markup=build_tasks_keyboard(remaining))
        except Exception:
            pass
    else:
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="🎉 Все задачи выполнены")
        except Exception:
            pass

    msg = f"✅ {task.description}{streak_msg}\n+{xp_for_task} XP"
    if level_up:
        msg += f"\n{level_up}"
    if bonus_level_up:
        msg += f"\n{bonus_level_up}"
    if ach_msg:
        msg += ach_msg
    bot.send_message(chat_id, msg, parse_mode="Markdown")
    db.close()

@bot.callback_query_handler(func=lambda call: call.data.startswith('task_del:'))
def callback_task_del(call):
    chat_id = call.message.chat.id
    task_id = int(call.data.split(':', 1)[1])
    user, db = get_db_user(chat_id)
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()
    if not task:
        bot.answer_callback_query(call.id, "Не найдено")
        db.close()
        return
    task.is_active = False
    db.commit()
    bot.answer_callback_query(call.id, "🗑")
    remaining = get_active_tasks(user.id, db)
    if remaining:
        lines = ["📋 **Задачи:**\n"]
        for i, t in enumerate(remaining, 1):
            lines.append(format_task_line(t, i))
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="\n".join(lines), parse_mode="Markdown", reply_markup=build_tasks_keyboard(remaining))
        except Exception:
            pass
    else:
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="📋 Пусто")
        except Exception:
            pass
    db.close()

# === КОМАНДЫ ===
@bot.message_handler(commands=['start', 'reset'])
def send_welcome(message):
    chat_id = message.chat.id
    clear_history(chat_id)
    text = (
        "👋 Привет! Я твой полезный ассистент.\n\n"
        "**Что умею:**\n"
        "• 💬 Отвечаю на вопросы, помогаю с текстом и кодом\n"
        "• 🖼 Генерирую картинки — `/image кот в космосе`\n"
        "• 🎤 Распознаю голосовые и описываю фото\n"
        "• 📋 Веду задачи и напоминаю о них\n"
        "• 💰 Записываю траты и доходы\n"
        "• 🏆 Считаю XP, уровень и достижения\n\n"
        "**Примеры:**\n"
        "• «Завтра в 15:00 позвонить врачу»\n"
        "• «Каждый день в 8:00 выпить воду»\n"
        "• «Потратил 500 на еду»\n"
        "• «Установи баланс 5000»\n\n"
        "Полный список команд — `/help`"
    )
    bot.send_message(chat_id, text, parse_mode="Markdown", reply_markup=get_main_keyboard())

@bot.message_handler(commands=['help'])
def help_command(message):
    help_text = (
        "📚 **Помощь**\n\n"
        "**💬 Общение**\n"
        "• Просто пиши текст — отвечу на вопрос\n"
        "• Отправь фото — опишу\n"
        "• Отправь голосовое — расшифрую\n\n"
        "**🖼 Картинки**\n"
        "• `/image кот в космосе` — сгенерировать\n"
        "• `/image_models` — выбрать модель\n"
        "• `/format` — квадрат / широкий / вертикальный\n\n"
        "**📋 Задачи**\n"
        "• «Завтра в 15:00 позвонить врачу» — разовая\n"
        "• «Каждый день в 8:00 выпить воду» — ежедневная\n"
        "• «Каждый Пн и Ср в 19:00 читать» — еженедельная\n"
        "• `/tasks` — список задач\n"
        "Напомню за 5 минут до времени.\n\n"
        "**💰 Финансы**\n"
        "• «Потратил 500 на еду» — трата (−500)\n"
        "• «Зарплата 50000» — доход (+50000)\n"
        "• «Установи баланс 3327» — заменить баланс\n"
        "• «Мой баланс 5000» — заменить баланс\n"
        "• Спрашивай: «сколько потратил на еду за месяц?»\n\n"
        "**🏆 Прогресс**\n"
        "• `/profile` — уровень, XP, баланс\n"
        "• `/achievements` — достижения\n"
        "• `/week` — статистика за 7 дней\n\n"
        "**⚙️ Настройки**\n"
        "• `/models` — модель ответов\n"
        "• `/settings` — контекст, температура\n"
        "• `/stats` — текущие настройки\n"
        "• `/reset_stats` — сброс статистики\n"
        "• `/reset` — сброс истории чата"
    )
    bot.reply_to(message, help_text, parse_mode="Markdown", reply_markup=get_main_keyboard())

@bot.message_handler(commands=['stats'])
def stats_command(message):
    chat_id = message.chat.id
    history = get_history(chat_id)
    total_chars = sum(len(str(msg["content"])) for msg in history)
    user, db = get_db_user(chat_id)
    bot.reply_to(message,
                 f"📊 **Статус**\n"
                 f"🧠 {model_info(user.model)['name']}\n"
                 f"⚙️ {THINKING_LEVELS[user.thinking]}\n"
                 f"📏 {user.history_len}\n"
                 f"🎲 {user.temperature}\n"
                 f"📝 {user.max_tokens}\n"
                 f"🎨 {hf_model_info(user.hf_model)['name']}\n"
                 f"📐 {format_info(user.format)['name']}\n\n"
                 f"История: {len(history)} сообщ., {total_chars} симв.",
                 reply_markup=get_main_keyboard(), parse_mode="Markdown")
    db.close()

@bot.message_handler(func=lambda m: m.text == "👤 Профиль")
def profile_button(message):
    profile_command(message)

@bot.message_handler(func=lambda m: m.text == "🧠 Модели")
def models_button(message):
    show_models(message)

@bot.message_handler(func=lambda m: m.text == "⚙️ Настройки")
def settings_button(message):
    show_settings(message)

@bot.message_handler(func=lambda m: m.text == "🎨 Нарисовать")
def image_button(message):
    bot.reply_to(message, "🎨 Напиши: `/image кот в космосе`", parse_mode="Markdown")

@bot.message_handler(func=lambda m: m.text == "🎨 Модель картинок")
def hf_models_button(message):
    show_hf_models(message)

@bot.message_handler(func=lambda m: m.text == "📐 Формат")
def format_button(message):
    show_formats(message)

@bot.message_handler(func=lambda m: m.text == "🔄 Сбросить историю")
def reset_button(message):
    clear_history(message.chat.id)
    bot.reply_to(message, "✅ История очищена", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text == "ℹ️ Помощь")
def help_button(message):
    help_command(message)

@bot.message_handler(func=lambda m: m.text == "📊 Статус")
def status_button(message):
    stats_command(message)

# === ГЕНЕРАЦИЯ КАРТИНОК ===
@bot.message_handler(commands=['image'])
def generate_image(message):
    chat_id = message.chat.id
    prompt = message.text.replace('/image', '', 1).strip()
    if not prompt:
        bot.reply_to(message, "🖼 `/image кот в космосе`", parse_mode="Markdown")
        return
    if not hf_client:
        bot.reply_to(message, "❌ Не задан HF_TOKEN", reply_markup=get_main_keyboard())
        return

    user, db = get_db_user(chat_id)
    info = hf_model_info(user.hf_model)
    fmt = format_info(user.format)
    db.close()

    bot.send_message(chat_id, f"🎨 Генерирую ({info['name']})...")
    english_prompt = translate_to_english(prompt)

    try:
        image = hf_client.text_to_image(english_prompt, model=info["id"], width=fmt['width'], height=fmt['height'])
        buff = BytesIO()
        image.save(buff, format="PNG")
        buff.seek(0)
        bot.send_photo(chat_id, buff, reply_to_message_id=message.message_id)
    except Exception as e:
        error_str = str(e)
        print(f"Ошибка HF: {error_str}", flush=True)
        if "503" in error_str:
            bot.reply_to(message, "⏳ Модель загружается, попробуй через 20-30 сек.", reply_markup=get_main_keyboard())
        elif "429" in error_str:
            bot.reply_to(message, "⏳ Слишком много запросов, подожди минуту.", reply_markup=get_main_keyboard())
        elif "402" in error_str or "Payment Required" in error_str:
            bot.reply_to(message, "💳 Лимиты закончились. Смени модель.", reply_markup=get_main_keyboard())
        else:
            bot.reply_to(message, f"❌ Ошибка: {error_str[:200]}", reply_markup=get_main_keyboard())

# === ОБРАБОТКА ТЕКСТА ===
@bot.message_handler(content_types=['text'])
def reply_text(message):
    user_text = message.text
    if user_text.startswith('/'):
        return

    chat_id = message.chat.id
    update_history(chat_id, "user", user_text)
    bot.send_chat_action(chat_id, 'typing')

    user, db = get_db_user(chat_id)
    now = now_local()

    system_prompt = build_system_prompt(user, db)
    clean_history = [m for m in get_history(chat_id) if m["role"] != "system"]
    messages = [{"role": "system", "content": system_prompt}] + clean_history

    for attempt in range(3):
        try:
            raw_reply = call_gemini(messages, user.model, user.temperature, user.max_tokens)
            print(f"🤖 TXT RAW: {raw_reply[:600]}", flush=True)

            parsed = parse_gemini_json(raw_reply)
            if parsed is None:
                send_long_message(chat_id, raw_reply, message.message_id)
                update_history(chat_id, "assistant", raw_reply)
                break

            query = parsed.get("query")
            if query and isinstance(query, dict):
                print(f"🔍 {query}", flush=True)
                query_result = execute_query(user, db, query)
                print(f"📊 {str(query_result)[:300]}", flush=True)

                second_messages = messages + [
                    {"role": "assistant", "content": raw_reply},
                    {"role": "user", "content": (
                        f"Результат запроса: {json.dumps(query_result, ensure_ascii=False, default=str)}\n"
                        "Сформулируй финальный ответ. JSON: {\"reply\": \"...\", \"xp\": 0, \"money\": 0, \"set_balance\": null, \"category\": \"\", \"task\": null, \"query\": null}"
                    )}
                ]
                raw_reply = call_gemini(second_messages, user.model, user.temperature, user.max_tokens)
                print(f"🤖 TXT RAW2: {raw_reply[:600]}", flush=True)
                parsed = parse_gemini_json(raw_reply)
                if parsed is None:
                    send_long_message(chat_id, raw_reply, message.message_id)
                    update_history(chat_id, "assistant", raw_reply)
                    break

            final_text = handle_parsed_response(parsed, user, db, user_text, chat_id, message.message_id, now)
            db.commit()
            send_long_message(chat_id, final_text, message.message_id)
            update_history(chat_id, "assistant", raw_reply)
            break

        except Exception as e:
            print(f"Ошибка TXT ({attempt + 1}): {e}", flush=True)
            if attempt == 2:
                bot.reply_to(message, "❌ Ошибка", reply_markup=get_main_keyboard())
            else:
                time.sleep(2 * (attempt + 1))
    db.close()

# === ОБРАБОТКА ФОТО ===
@bot.message_handler(content_types=['photo'])
def reply_photo(message):
    chat_id = message.chat.id
    bot.send_chat_action(chat_id, 'typing')

    user, db = get_db_user(chat_id)
    now = now_local()

    for attempt in range(3):
        try:
            file_id = message.photo[-1].file_id
            file_info = bot.get_file(file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            image = Image.open(BytesIO(downloaded_file))
            buff = BytesIO()
            image.save(buff, format="JPEG")
            base64_image = base64.b64encode(buff.getvalue()).decode('utf-8')

            system_prompt = build_system_prompt(user, db)
            caption = message.caption or "Опиши фото. Если это чек или скрин — извлеки сумму и категорию."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": caption},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]

            raw_reply = call_gemini(messages, user.model, user.temperature, user.max_tokens)
            print(f"🤖 PHOTO RAW: {raw_reply[:600]}", flush=True)

            parsed = parse_gemini_json(raw_reply)
            if parsed is None:
                send_long_message(chat_id, raw_reply, message.message_id)
                update_history(chat_id, "assistant", raw_reply)
                break

            user_text = caption[:100]
            final_text = handle_parsed_response(parsed, user, db, user_text, chat_id, message.message_id, now)
            db.commit()
            send_long_message(chat_id, final_text, message.message_id)
            update_history(chat_id, "assistant", raw_reply)
            break

        except Exception as e:
            print(f"Ошибка PHOTO ({attempt + 1}): {e}", flush=True)
            if attempt == 2:
                bot.reply_to(message, "❌ Ошибка", reply_markup=get_main_keyboard())
            else:
                time.sleep(2 * (attempt + 1))
    db.close()

# === ОБРАБОТКА ГОЛОСОВЫХ ===
@bot.message_handler(content_types=['voice'])
def reply_voice(message):
    chat_id = message.chat.id
    bot.send_chat_action(chat_id, 'typing')

    user, db = get_db_user(chat_id)
    now = now_local()

    try:
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        audio = AudioSegment.from_file(BytesIO(downloaded_file), format="ogg")
        wav_buffer = BytesIO()
        audio.export(wav_buffer, format="wav")
        base64_audio = base64.b64encode(wav_buffer.getvalue()).decode('utf-8')

        system_prompt = build_system_prompt(user, db)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Расшифруй голосовое и выполни то, что просят."},
                {"type": "input_audio", "input_audio": {"data": base64_audio, "format": "wav"}}
            ]}
        ]

        raw_reply = call_gemini(messages, user.model, user.temperature, user.max_tokens)
        print(f"🤖 VOICE RAW: {raw_reply[:600]}", flush=True)

        parsed = parse_gemini_json(raw_reply)
        if parsed is None:
            send_long_message(chat_id, raw_reply, message.message_id)
            update_history(chat_id, "assistant", raw_reply)
            db.close()
            return

        user_text = "голосовое"
        final_text = handle_parsed_response(parsed, user, db, user_text, chat_id, message.message_id, now)
        db.commit()
        send_long_message(chat_id, final_text, message.message_id)
        update_history(chat_id, "assistant", raw_reply)

    except Exception as e:
        print(f"Ошибка VOICE: {e}", flush=True)
        bot.reply_to(message, "❌ Ошибка", reply_markup=get_main_keyboard())
    finally:
        db.close()

# === НАПОМИНАНИЯ ===
def send_task_reminder(task, user):
    try:
        bot.send_message(user.telegram_id, f"⏰ Через 5 мин: {task.description}")
    except Exception as e:
        print(f"Ошибка напоминания: {e}", flush=True)

def check_tasks():
    db = SessionLocal()
    try:
        now = now_local()
        now_min = now.replace(second=0, microsecond=0)
        current_weekday = now.weekday()

        tasks = db.query(Task).filter(Task.is_active == True).all()
        for task in tasks:
            try:
                user = db.query(User).filter(User.id == task.user_id).first()
                if not user:
                    continue

                reminder_dt = None
                scheduled = None

                if task.task_type == "one_time":
                    if not task.time_str:
                        continue
                    try:
                        scheduled = datetime.strptime(task.time_str, "%Y-%m-%d %H:%M")
                    except ValueError:
                        continue
                    reminder_dt = scheduled - timedelta(minutes=5)
                    if reminder_dt < now_min:
                        if (now_min - reminder_dt).total_seconds() > 120:
                            if scheduled < now - timedelta(hours=1):
                                task.is_active = False
                            continue

                elif task.task_type == "daily":
                    if not task.time_str:
                        continue
                    try:
                        t_time = datetime.strptime(task.time_str, "%H:%M").time()
                    except ValueError:
                        continue
                    scheduled = datetime.combine(now.date(), t_time)
                    reminder_dt = scheduled - timedelta(minutes=5)

                elif task.task_type == "weekly":
                    if not task.time_str or not task.days:
                        continue
                    task_days = []
                    for d in task.days.split(","):
                        d_clean = d.strip().lower()
                        if d_clean in WEEKDAYS_MAP:
                            task_days.append(WEEKDAYS_MAP[d_clean])
                    if current_weekday not in task_days:
                        continue
                    try:
                        t_time = datetime.strptime(task.time_str, "%H:%M").time()
                    except ValueError:
                        continue
                    scheduled = datetime.combine(now.date(), t_time)
                    reminder_dt = scheduled - timedelta(minutes=5)

                if reminder_dt is None:
                    continue

                if reminder_dt.replace(second=0, microsecond=0) == now_min:
                    marker = reminder_dt.strftime("%Y-%m-%d %H:%M")
                    if task.reminder_sent == marker:
                        continue
                    send_task_reminder(task, user)
                    task.reminder_sent = marker

                    if task.task_type in ("daily", "weekly") and task.last_completed:
                        days_since = (now.date() - task.last_completed.date()).days
                        if task.task_type == "daily" and days_since > 1:
                            task.streak = 0
                        elif task.task_type == "weekly" and days_since > 7:
                            task.streak = 0
            except Exception as e:
                print(f"Ошибка задачи {task.id}: {e}", flush=True)

        db.commit()
    except Exception as e:
        print(f"Ошибка планировщика: {e}", flush=True)
    finally:
        db.close()

# === УТРЕННИЙ ОТЧЁТ ===
def send_morning_reports():
    db = SessionLocal()
    try:
        now = now_local()
        today_str = now.strftime("%Y-%m-%d")
        weekday = now.weekday()

        users = db.query(User).all()
        for user in users:
            try:
                if user.last_report_date == today_str:
                    continue

                tasks = db.query(Task).filter(Task.user_id == user.id, Task.is_active == True).all()
                if not tasks:
                    user.last_report_date = today_str
                    continue

                today_tasks = []
                for t in tasks:
                    if t.task_type == "one_time":
                        if t.time_str:
                            try:
                                dt = datetime.strptime(t.time_str, "%Y-%m-%d %H:%M")
                                if dt.date() == now.date():
                                    today_tasks.append(f"🎯 {t.description} — {dt.strftime('%H:%M')}")
                            except Exception:
                                pass
                    elif t.task_type == "daily":
                        today_tasks.append(f"🔁 {t.description} — {t.time_str or '—'}")
                    elif t.task_type == "weekly":
                        if t.days:
                            task_days = []
                            for d in t.days.split(","):
                                d_clean = d.strip().lower()
                                if d_clean in WEEKDAYS_MAP:
                                    task_days.append(WEEKDAYS_MAP[d_clean])
                            if weekday in task_days:
                                today_tasks.append(f"📅 {t.description} — {t.time_str or '—'}")

                if not today_tasks:
                    user.last_report_date = today_str
                    continue

                text = "☀️ **Сегодня:**\n" + "\n".join(today_tasks)
                text += f"\n\nXP: {user.xp} | 💰 {user.balance:.0f}"

                try:
                    bot.send_message(user.telegram_id, text, parse_mode="Markdown")
                    user.last_report_date = today_str
                except Exception as e:
                    print(f"Отчёт не ушёл {user.telegram_id}: {e}", flush=True)
            except Exception as e:
                print(f"Ошибка отчёта {user.id}: {e}", flush=True)

        db.commit()
    except Exception as e:
        print(f"Ошибка утренних отчётов: {e}", flush=True)
    finally:
        db.close()

# === Веб-сервер ===
if os.environ.get("PORT"):
    from flask import Flask
    app = Flask(__name__)

    @app.route('/')
    def health():
        return "Bot is alive"

    def run_web():
        app.run(host='0.0.0.0', port=int(os.environ["PORT"]))

    threading.Thread(target=run_web, daemon=True).start()

# === ЗАПУСК ===
if __name__ == "__main__":
    scheduler.add_job(check_tasks, 'interval', minutes=1)
    scheduler.add_job(send_morning_reports, 'cron', hour=9, minute=0)
    scheduler.start()
    print("✅ Бот запущен!", flush=True)
    bot.infinity_polling()
