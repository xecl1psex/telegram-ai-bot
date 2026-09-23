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

from database import SessionLocal, User, Task, Transaction

load_dotenv()

TOKEN = os.environ.get("BOT_TOKEN")
API_KEY = os.environ.get("AI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not TOKEN:
    raise ValueError("Не задан BOT_TOKEN в переменных окружения")
if not API_KEY:
    raise ValueError("Не задан AI_API_KEY в переменных окружения")

AI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None

HF_MODELS = {
    "sdxl": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "name": "🎨 SDXL Base 1.0",
        "desc": "Мощная, стабильно бесплатная.",
    },
    "sdxl_turbo": {
        "id": "stabilityai/sdxl-turbo",
        "name": "⚡ SDXL Turbo",
        "desc": "Быстрая, 1-4 шага.",
    },
    "sd15": {
        "id": "runwayml/stable-diffusion-v1-5",
        "name": "🖼 Stable Diffusion 1.5",
        "desc": "Классика. Быстрая, менее детальная.",
    },
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

# === ЧАСОВОЙ ПОЯС ===
TIMEZONE = "Europe/Moscow"
TZ = ZoneInfo(TIMEZONE)

def now_local():
    """Текущее время в Москве без tzinfo (naive). Использовать везде вместо datetime.now()."""
    return datetime.now(TZ).replace(tzinfo=None)

WEEKDAYS_RU = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
WEEKDAYS_MAP = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
    "пн": 0, "вт": 1, "ср": 2, "чт": 3, "пт": 4, "сб": 5, "вс": 6,
}

chat_history = {}

MODEL_INFO = {
    "gemini-3.8-flash": {"name": "🌟 Gemini 3.8 Flash", "desc": "Самая умная Flash-модель."},
    "gemini-3.7-flash": {"name": "⚡ Gemini 3.7 Flash", "desc": "Быстрая и мощная."},
    "gemini-3.6-flash": {"name": "🚀 Gemini 3.6 Flash", "desc": "Надёжная рабочая лошадка."},
    "gemini-3.5-flash-lite": {"name": "🍃 Gemini 3.5 Flash-Lite", "desc": "Самая быстрая."},
}

THINKING_LEVELS = {
    "low": "⚡ Быстрый (low)",
    "medium": "🧠 Сбалансированный (medium)",
    "high": "🔬 Глубокий (high)",
}

HISTORY_OPTIONS = [3, 6, 10, 20, 50]
TEMPERATURE_OPTIONS = [0.0, 0.3, 0.7, 1.0, 1.5]
MAX_TOKENS_OPTIONS = [512, 1024, 2048, 4096, 8192]

IMAGE_FORMATS = {
    "square": {"name": "⬛ Квадрат (1:1)", "desc": "1024×1024.", "width": 1024, "height": 1024},
    "wide": {"name": "🖼 Широкий (16:9)", "desc": "1344×768.", "width": 1344, "height": 768},
    "portrait": {"name": "📱 Вертикальный (9:16)", "desc": "768×1344.", "width": 768, "height": 1344},
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

def get_main_keyboard():
    markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    markup.add(
        KeyboardButton("👤 Профиль"), KeyboardButton("📋 Задачи"),
        KeyboardButton("🧠 Модели"), KeyboardButton("⚙️ Настройки"),
        KeyboardButton("🎨 Нарисовать"), KeyboardButton("🎨 Модель картинок"),
        KeyboardButton("📐 Формат"), KeyboardButton("🔄 Сбросить историю"),
        KeyboardButton("ℹ️ Помощь"), KeyboardButton("📊 Статус"),
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
        content = content[:MAX_SAVED_ANSWER_LEN] + "... (обрезано)"
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
            print(f"⚠️ HTML не прошёл: {e}", flush=True)
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
                "content": f"Translate the following text to English. Output ONLY the English translation: {text}"
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

# === МЕНЮ ФОРМАТОВ ===
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
    bot.send_message(message.chat.id, "📐 *Выбери формат:*", reply_markup=build_formats_keyboard(message.chat.id), parse_mode="Markdown")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_format:'))
def callback_set_format(call):
    chat_id = call.message.chat.id
    fmt_id = call.data.split(':', 1)[1]
    user, db = get_db_user(chat_id)
    user.format = fmt_id
    db.commit()
    db.close()
    info = format_info(fmt_id)
    bot.answer_callback_query(call.id, f"Формат: {info['name']}")
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"✅ Формат: *{info['name']}*\n\n{info['desc']}", parse_mode="Markdown")

# === МЕНЮ МОДЕЛЕЙ HF ===
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
    bot.send_message(message.chat.id, "🎨 *Выбери модель для генерации:*", reply_markup=build_hf_models_keyboard(message.chat.id), parse_mode="Markdown")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_hf_model:'))
def callback_set_hf_model(call):
    chat_id = call.message.chat.id
    model_id = call.data.split(':', 1)[1]
    user, db = get_db_user(chat_id)
    user.hf_model = model_id
    db.commit()
    db.close()
    info = hf_model_info(model_id)
    bot.answer_callback_query(call.id, f"Выбрана: {info['name']}")
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"✅ Модель: *{info['name']}*\n\n{info['desc']}", parse_mode="Markdown")

# === МЕНЮ МОДЕЛЕЙ GEMINI ===
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
    bot.send_message(message.chat.id, "🧠 *Выбери модель:*", reply_markup=build_models_keyboard(message.chat.id), parse_mode="Markdown")

@bot.callback_query_handler(func=lambda call: call.data.startswith('model:'))
def callback_model_info(call):
    chat_id = call.message.chat.id
    model_id = call.data.split(':', 1)[1]
    info = MODEL_INFO.get(model_id)
    if not info:
        bot.answer_callback_query(call.id, "Модель не найдена")
        return
    markup = InlineKeyboardMarkup(row_width=1)
    markup.add(InlineKeyboardButton("✅ Выбрать эту модель", callback_data=f"confirm_model:{model_id}"))
    markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="back_to_models"))
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"*{info['name']}*\n\n{info['desc']}\n\nВыбрать?", reply_markup=markup, parse_mode="Markdown")
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith('confirm_model:'))
def callback_confirm_model(call):
    chat_id = call.message.chat.id
    model_id = call.data.split(':', 1)[1]
    user, db = get_db_user(chat_id)
    user.model = model_id
    db.commit()
    current_thinking = user.thinking
    db.close()
    markup = InlineKeyboardMarkup(row_width=1)
    for level_id, level_name in THINKING_LEVELS.items():
        check = " ✅" if level_id == current_thinking else ""
        markup.add(InlineKeyboardButton(text=f"{level_name}{check}", callback_data=f"thinking:{level_id}"))
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"Модель *{model_info(model_id)['name']}* выбрана!\n\nТеперь выбери *режим размышлений*:", reply_markup=markup, parse_mode="Markdown")
    bot.answer_callback_query(call.id, "Модель выбрана!")

@bot.callback_query_handler(func=lambda call: call.data.startswith('thinking:'))
def callback_thinking(call):
    chat_id = call.message.chat.id
    level = call.data.split(':', 1)[1]
    user, db = get_db_user(chat_id)
    user.thinking = level
    model_id = user.model
    db.commit()
    db.close()
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"✅ *Сохранено!*\n\n🧠 Модель: {model_info(model_id)['name']}\n⚙️ Режим: {THINKING_LEVELS[level]}", parse_mode="Markdown")
    bot.answer_callback_query(call.id, "Сохранено!")

@bot.callback_query_handler(func=lambda call: call.data == 'back_to_models')
def callback_back(call):
    chat_id = call.message.chat.id
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="🧠 *Выбери модель:*", reply_markup=build_models_keyboard(chat_id), parse_mode="Markdown")
    bot.answer_callback_query(call.id)

# === НАСТРОЙКИ ===
def build_settings_keyboard(chat_id):
    user, db = get_db_user(chat_id)
    markup = InlineKeyboardMarkup(row_width=1)
    markup.add(InlineKeyboardButton(text=f"📏 Длина контекста: {user.history_len}", callback_data="settings:history"))
    markup.add(InlineKeyboardButton(text=f"🎲 Температура: {user.temperature}", callback_data="settings:temperature"))
    markup.add(InlineKeyboardButton(text=f"📝 Макс. токенов: {user.max_tokens}", callback_data="settings:max_tokens"))
    markup.add(InlineKeyboardButton(text="🔄 Сбросить по умолчанию", callback_data="settings:reset"))
    db.close()
    return markup

@bot.message_handler(commands=['settings'])
def show_settings(message):
    bot.send_message(message.chat.id, "⚙️ *Настройки контекста*", reply_markup=build_settings_keyboard(message.chat.id), parse_mode="Markdown")

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
            markup.add(InlineKeyboardButton(text=f"{opt} сообщений{check}", callback_data=f"set_history:{opt}"))
        markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:back"))
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="📏 *Выбери длину контекста:*", reply_markup=markup, parse_mode="Markdown")
    elif action == "temperature":
        current = user.temperature
        markup = InlineKeyboardMarkup(row_width=1)
        labels = {0.0: "0.0 — Строго", 0.3: "0.3 — Умеренно", 0.7: "0.7 — Баланс", 1.0: "1.0 — Креативно", 1.5: "1.5 — Максимум"}
        for opt in TEMPERATURE_OPTIONS:
            check = " ✅" if opt == current else ""
            markup.add(InlineKeyboardButton(text=f"{labels[opt]}{check}", callback_data=f"set_temperature:{opt}"))
        markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:back"))
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="🎲 *Выбери температуру:*", reply_markup=markup, parse_mode="Markdown")
    elif action == "max_tokens":
        current = user.max_tokens
        markup = InlineKeyboardMarkup(row_width=1)
        for opt in MAX_TOKENS_OPTIONS:
            check = " ✅" if opt == current else ""
            markup.add(InlineKeyboardButton(text=f"{opt} токенов{check}", callback_data=f"set_tokens:{opt}"))
        markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:back"))
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="📝 *Выбери макс. длину ответа:*", reply_markup=markup, parse_mode="Markdown")
    elif action == "reset":
        user.history_len = DEFAULT_HISTORY_LEN
        user.temperature = DEFAULT_TEMPERATURE
        user.max_tokens = DEFAULT_MAX_TOKENS
        db.commit()
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="✅ *Сброшено!*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")
    elif action == "back":
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="⚙️ *Настройки контекста*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")

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
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="⚙️ *Настройки*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")
    bot.answer_callback_query(call.id, f"Контекст: {val}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_temperature:'))
def callback_set_temp(call):
    chat_id = call.message.chat.id
    val = float(call.data.split(':', 1)[1])
    user, db = get_db_user(chat_id)
    user.temperature = val
    db.commit()
    db.close()
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="⚙️ *Настройки*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")
    bot.answer_callback_query(call.id, f"Температура: {val}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_tokens:'))
def callback_set_tokens(call):
    chat_id = call.message.chat.id
    val = int(call.data.split(':', 1)[1])
    user, db = get_db_user(chat_id)
    user.max_tokens = val
    db.commit()
    db.close()
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="⚙️ *Настройки*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")
    bot.answer_callback_query(call.id, f"Макс. токенов: {val}")

# === ПРОФИЛЬ ===
@bot.message_handler(commands=['profile'])
def profile_command(message):
    chat_id = message.chat.id
    user, db = get_db_user(chat_id)
    xp_needed = int(100 * (1.5 ** user.level))
    text = (
        f"👤 **Профиль**\n\n"
        f"🏆 Уровень: {user.level}\n"
        f"✨ XP: {user.xp} / {xp_needed}\n"
        f"💰 Баланс: {user.balance:.2f}\n\n"
        f"📊 Прогресс: {user.xp}/{xp_needed}"
    )
    bot.reply_to(message, text, parse_mode="Markdown")
    db.close()

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
            "📋 У тебя нет активных задач.\n\n"
            "Просто напиши, например:\n"
            "• «Завтра в 15:00 позвонить врачу» — разовая\n"
            "• «Каждый день в 8:00 выпить воду» — ежедневная\n"
            "• «Каждый Пн и Ср в 19:00 читать» — еженедельная",
            reply_markup=get_main_keyboard())
        db.close()
        return
    lines = ["📋 **Твои задачи:**\n"]
    for i, t in enumerate(tasks, 1):
        lines.append(format_task_line(t, i))
    lines.append("\n✅ <id> — выполнить, 🗑 <id> — удалить.")
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
        bot.answer_callback_query(call.id, "Задача не найдена")
        db.close()
        return

    task.last_completed = now_local()
    if task.task_type in ("daily", "weekly"):
        task.streak = (task.streak or 0) + 1
        task.reminder_sent = None
        streak_msg = f" 🔥 Серия: {task.streak}"
    else:
        task.is_active = False
        streak_msg = ""

    xp_for_task = 10 if task.task_type == "one_time" else 5
    user.xp += xp_for_task
    xp_needed = int(100 * (1.5 ** user.level))
    level_up_msg = ""
    while user.xp >= xp_needed:
        user.xp -= xp_needed
        user.level += 1
        xp_needed = int(100 * (1.5 ** user.level))
        level_up_msg = f"\n\n🎉 Уровень повышен! Теперь Level {user.level}!"

    db.commit()
    bot.answer_callback_query(call.id, "Выполнено! ✅")

    remaining = get_active_tasks(user.id, db)
    if remaining:
        lines = ["📋 **Твои задачи:**\n"]
        for i, t in enumerate(remaining, 1):
            lines.append(format_task_line(t, i))
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="\n".join(lines), parse_mode="Markdown", reply_markup=build_tasks_keyboard(remaining))
        except Exception:
            pass
    else:
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="🎉 Все задачи выполнены!")
        except Exception:
            pass

    bot.send_message(chat_id, f"✅ Выполнено: {task.description}{streak_msg}\n✨ +{xp_for_task} XP{level_up_msg}", parse_mode="Markdown")
    db.close()

@bot.callback_query_handler(func=lambda call: call.data.startswith('task_del:'))
def callback_task_del(call):
    chat_id = call.message.chat.id
    task_id = int(call.data.split(':', 1)[1])
    user, db = get_db_user(chat_id)
    task = db.query(Task).filter(Task.id == task_id, Task.user_id == user.id).first()
    if not task:
        bot.answer_callback_query(call.id, "Задача не найдена")
        db.close()
        return
    task.is_active = False
    db.commit()
    bot.answer_callback_query(call.id, "Удалено 🗑")
    remaining = get_active_tasks(user.id, db)
    if remaining:
        lines = ["📋 **Твои задачи:**\n"]
        for i, t in enumerate(remaining, 1):
            lines.append(format_task_line(t, i))
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="\n".join(lines), parse_mode="Markdown", reply_markup=build_tasks_keyboard(remaining))
        except Exception:
            pass
    else:
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="📋 Список пуст.")
        except Exception:
            pass
    db.close()

# === КОМАНДЫ ===
@bot.message_handler(commands=['start', 'reset'])
def send_welcome(message):
    chat_id = message.chat.id
    clear_history(chat_id)
    user, db = get_db_user(chat_id)
    model_name = model_info(user.model)["name"]
    hf_model_name = hf_model_info(user.hf_model)["name"]
    format_name = format_info(user.format)["name"]
    db.close()

    bot.send_message(chat_id,
                     f"Привет! Я бот на нейросети Gemini.\n"
                     f"🧠 Модель: {model_name}\n"
                     f"🎨 Модель картинок: {hf_model_name}\n"
                     f"📐 Формат: {format_name}\n"
                     f"Умею: текст, фото, голосовые, картинки, задачи.\n"
                     f"Команды: /models /settings /image_models /format /image /profile /tasks",
                     reply_markup=get_main_keyboard())

@bot.message_handler(commands=['help'])
def help_command(message):
    help_text = (
        "📚 Помощь:\n"
        "• Отправь текст – отвечу.\n"
        "• Отправь фото – опишу.\n"
        "• Отправь голосовое – расшифрую.\n"
        "• 🎨 /image <описание> – генерация картинок.\n"
        "• 📐 /format – формат картинок.\n"
        "• 🧠 /models – модель Gemini.\n"
        "• ⚙️ /settings – контекст, температура.\n"
        "• 👤 /profile – уровень, XP, баланс.\n"
        "• 📋 /tasks – список задач.\n\n"
        "📝 Как ставить задачи:\n"
        "• «Завтра в 15:00 позвонить врачу»\n"
        "• «Каждый день в 8:00 выпить воду»\n"
        "• «Каждый Пн и Ср в 19:00 читать»\n"
        "Напомню за 5 минут до времени."
    )
    bot.reply_to(message, help_text, reply_markup=get_main_keyboard())

@bot.message_handler(commands=['stats'])
def stats_command(message):
    chat_id = message.chat.id
    history = get_history(chat_id)
    total_chars = sum(len(str(msg["content"])) for msg in history)
    user, db = get_db_user(chat_id)
    bot.reply_to(message,
                 f"📊 *Статус:*\n\n"
                 f"🧠 Модель: {model_info(user.model)['name']}\n"
                 f"⚙️ Режим: {THINKING_LEVELS[user.thinking]}\n"
                 f"📏 Контекст: {user.history_len}\n"
                 f"🎲 Температура: {user.temperature}\n"
                 f"📝 Макс. токенов: {user.max_tokens}\n"
                 f"🎨 Картинки: {hf_model_info(user.hf_model)['name']}\n"
                 f"📐 Формат: {format_info(user.format)['name']}\n\n"
                 f"Сообщений в истории: {len(history)}\n"
                 f"Размер: {total_chars} симв.",
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
    bot.reply_to(message, "✅ История очищена!", reply_markup=get_main_keyboard())

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
        bot.reply_to(message, "🖼 Напиши: `/image кот в космосе`", parse_mode="Markdown")
        return
    if not hf_client:
        bot.reply_to(message, "❌ Не задан HF_TOKEN", reply_markup=get_main_keyboard())
        return

    user, db = get_db_user(chat_id)
    info = hf_model_info(user.hf_model)
    fmt = format_info(user.format)
    db.close()

    bot.send_message(chat_id, f"🎨 Генерирую ({info['name']})...\n15-30 секунд.")
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
            bot.reply_to(message, "⏳ Модель загружается. Попробуй через 20-30 сек.", reply_markup=get_main_keyboard())
        elif "429" in error_str:
            bot.reply_to(message, "⏳ Слишком много запросов. Подожди минуту.", reply_markup=get_main_keyboard())
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
    today_str = now.strftime("%Y-%m-%d")
    weekday_str = WEEKDAYS_RU[now.weekday()]
    time_str = now.strftime("%H:%M")

    system_prompt = (
        "Ты — полезный ассистент. Отвечай кратко и по делу.\n\n"
        f"Сегодня: {today_str} ({weekday_str}), сейчас {time_str} (по Москве).\n\n"
        "Дополнительно веди учёт:\n"
        "- Пользователь сделал что-то полезное → xp (5-100).\n"
        "- Упомянул трату/доход → money (минус для трат), category.\n"
        "- Просит напоминание → task:\n"
        "  • one_time: description, date (YYYY-MM-DD), time (HH:MM)\n"
        "  • daily: description, time (HH:MM)\n"
        "  • weekly: description, time (HH:MM), days (Mon,Wed)\n"
        "- Обычный вопрос → xp=0, money=0, task=null.\n\n"
        "Отвечай СТРОГО одним JSON (без markdown):\n"
        "{\"reply\": \"...\", \"xp\": 0, \"money\": 0, \"category\": \"\", \"task\": null}\n\n"
        "task: {\"description\": \"...\", \"type\": \"one_time|daily|weekly\", \"date\": \"YYYY-MM-DD\", \"time\": \"HH:MM\", \"days\": \"Mon,Wed\"}"
    )

    clean_history = [m for m in get_history(chat_id) if m["role"] != "system"]
    messages = [{"role": "system", "content": system_prompt}] + clean_history

    for attempt in range(3):
        try:
            payload = {
                "model": user.model,
                "messages": messages,
                "max_tokens": user.max_tokens,
                "temperature": user.temperature,
            }
            response = requests.post(AI_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
            data = response.json()

            if 'choices' not in data or not data['choices']:
                raise ValueError(f"Нет choices: {str(data)[:300]}")

            raw_reply = data['choices'][0]['message']['content'].strip()
            print(f"🤖 RAW: {raw_reply[:600]}", flush=True)

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

            if parsed is None:
                send_long_message(chat_id, raw_reply, message.message_id)
                update_history(chat_id, "assistant", raw_reply)
                break

            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}

            if not isinstance(parsed, dict):
                send_long_message(chat_id, raw_reply, message.message_id)
                update_history(chat_id, "assistant", raw_reply)
                break

            reply_text = (
                parsed.get("reply") or parsed.get("response") or parsed.get("message")
                or parsed.get("text") or parsed.get("content")
            )
            if not reply_text:
                for v in parsed.values():
                    if isinstance(v, str) and v.strip():
                        reply_text = v
                        break
                if not reply_text:
                    reply_text = raw_reply

            try:
                xp_gain = int(parsed.get("xp", 0) or 0)
            except (ValueError, TypeError):
                xp_gain = 0

            try:
                money_change = float(parsed.get("money", 0) or 0)
            except (ValueError, TypeError):
                money_change = 0.0

            category = parsed.get("category") or "Разное"
            task_data = parsed.get("task")

            if xp_gain > 0:
                user.xp += xp_gain
                xp_needed = int(100 * (1.5 ** user.level))
                while user.xp >= xp_needed:
                    user.xp -= xp_needed
                    user.level += 1
                    xp_needed = int(100 * (1.5 ** user.level))
                    reply_text += f"\n\n🎉 Уровень повышен! Теперь Level {user.level}!"
                reply_text += f"\n\n✨ +{xp_gain} XP"

            if money_change != 0:
                user.balance += money_change
                db.add(Transaction(user_id=user.id, amount=money_change, category=category, description=user_text[:50]))
                sign = "+" if money_change > 0 else ""
                reply_text += f"\n💰 Баланс: {sign}{money_change} ({category}). Текущий: {user.balance:.2f}"

            if task_data and isinstance(task_data, dict):
                try:
                    t_type = task_data.get("type", "one_time")
                    t_time = task_data.get("time")
                    t_days = task_data.get("days")
                    t_desc = task_data.get("description", "Без названия")

                    stored_time = t_time
                    if t_type == "one_time":
                        t_date = task_data.get("date")
                        if t_date and t_time:
                            stored_time = f"{t_date} {t_time}"
                        elif t_time:
                            stored_time = f"{now.strftime('%Y-%m-%d')} {t_time}"

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
                        reply_text += f"\n\n📝 Разовая: {t_desc} — {stored_time}"
                    elif t_type == "daily":
                        reply_text += f"\n\n🔁 Ежедневно: {t_desc} в {t_time}"
                    elif t_type == "weekly":
                        reply_text += f"\n\n📅 По {t_days}: {t_desc} в {t_time}"
                except Exception as e:
                    print(f"Ошибка задачи: {e}", flush=True)

            db.commit()
            send_long_message(chat_id, reply_text, message.message_id)
            update_history(chat_id, "assistant", raw_reply)
            break

        except Exception as e:
            print(f"Ошибка (попытка {attempt + 1}): {e}", flush=True)
            if attempt == 2:
                bot.reply_to(message, "❌ Не удалось получить ответ.", reply_markup=get_main_keyboard())
            else:
                time.sleep(2 * (attempt + 1))
    db.close()

# === ФОТО ===
@bot.message_handler(content_types=['photo'])
def reply_photo(message):
    chat_id = message.chat.id
    bot.send_chat_action(chat_id, 'typing')
    user, db = get_db_user(chat_id)
    model = user.model
    temperature = user.temperature
    max_tokens = user.max_tokens
    db.close()

    for attempt in range(3):
        try:
            file_id = message.photo[-1].file_id
            file_info = bot.get_file(file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            image = Image.open(BytesIO(downloaded_file))
            buff = BytesIO()
            image.save(buff, format="JPEG")
            base64_image = base64.b64encode(buff.getvalue()).decode('utf-8')

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Что на фото? Опиши на русском."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            response = requests.post(AI_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
            data = response.json()
            if 'choices' not in data or not data['choices']:
                raise ValueError(f"Нет choices: {str(data)[:300]}")
            reply = data['choices'][0]['message']['content'].strip()
            send_long_message(chat_id, reply, message.message_id)
            update_history(chat_id, "assistant", reply)
            break
        except Exception as e:
            print(f"Ошибка фото: {e}", flush=True)
            if attempt == 2:
                bot.reply_to(message, "❌ Не удалось обработать фото.", reply_markup=get_main_keyboard())
            else:
                time.sleep(2 * (attempt + 1))

# === ГОЛОС ===
@bot.message_handler(content_types=['voice'])
def reply_voice(message):
    chat_id = message.chat.id
    bot.send_chat_action(chat_id, 'typing')
    user, db = get_db_user(chat_id)
    model = user.model
    temperature = user.temperature
    max_tokens = user.max_tokens
    db.close()

    try:
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        audio = AudioSegment.from_file(BytesIO(downloaded_file), format="ogg")
        wav_buffer = BytesIO()
        audio.export(wav_buffer, format="wav")
        base64_audio = base64.b64encode(wav_buffer.getvalue()).decode('utf-8')

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Расшифруй голосовое и ответь на русском."},
                {"type": "input_audio", "input_audio": {"data": base64_audio, "format": "wav"}}
            ]}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(AI_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
        data = response.json()
        if 'choices' not in data or not data['choices']:
            raise ValueError(f"Нет choices: {str(data)[:300]}")
        reply = data['choices'][0]['message']['content'].strip()
        send_long_message(chat_id, reply, message.message_id)
        update_history(chat_id, "assistant", reply)
    except Exception as e:
        print(f"Ошибка голосового: {e}", flush=True)
        bot.reply_to(message, "❌ Не удалось обработать голосовое.", reply_markup=get_main_keyboard())

# === НАПОМИНАНИЯ ===
def send_task_reminder(task, user):
    try:
        bot.send_message(user.telegram_id, f"⏰ *Напоминание через 5 минут:*\n{task.description}", parse_mode="Markdown")
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
    scheduler.start()
    print("✅ Бот запущен!", flush=True)
    bot.infinity_polling()
