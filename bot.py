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
from apscheduler.schedulers.background import BackgroundScheduler

# Импорт из нашего файла БД
from database import SessionLocal, User, Task, Transaction

# === Загрузка переменных окружения ===
load_dotenv()

TOKEN = os.environ.get("BOT_TOKEN")
API_KEY = os.environ.get("AI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not TOKEN:
    raise ValueError("Не задан BOT_TOKEN в переменных окружения")
if not API_KEY:
    raise ValueError("Не задан AI_API_KEY в переменных окружения")

# === Настройки API ===
AI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# === Настройки Hugging Face ===
hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None

HF_MODELS = {
    "flux_schnell": {
        "id": "black-forest-labs/FLUX.1-schnell",
        "name": "⚡ FLUX.1 Schnell",
        "desc": "Быстрая и качественная. Оптимизирована для 4 шагов генерации. Рекомендуется.",
    },
    "flux_dev": {
        "id": "black-forest-labs/FLUX.1-dev",
        "name": "🎨 FLUX.1 Dev",
        "desc": "Мощная модель с высокой детализацией. Может быть медленнее.",
    },
    "sd3_medium": {
        "id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "name": "🖼 Stable Diffusion 3 Medium",
        "desc": "Классическая модель от Stability AI. Хороший баланс качества и скорости.",
    },
}

bot = telebot.TeleBot(TOKEN)

# === Настройки по умолчанию ===
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_THINKING = "medium"
DEFAULT_HISTORY_LEN = 6
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_FORMAT = "square"
DEFAULT_HF_MODEL = "flux_schnell"

MAX_SAVED_ANSWER_LEN = 800
TIMEOUT = 600

# === Память пользователей (история чата оставлена в памяти для простоты) ===
chat_history = {}

# === Описания моделей Gemini ===
MODEL_INFO = {
    "gemini-3.8-flash": {
        "name": "🌟 Gemini 3.8 Flash",
        "desc": "Самая умная Flash-модель. Сложные задачи, рассуждения, анализ изображений.",
    },
    "gemini-3.7-flash": {
        "name": "⚡ Gemini 3.7 Flash",
        "desc": "Быстрая и мощная. Кодинг, работа с видео, агентные задачи.",
    },
    "gemini-3.6-flash": {
        "name": "🚀 Gemini 3.6 Flash",
        "desc": "Надёжная рабочая лошадка. Баланс скорости и качества.",
    },
    "gemini-3.5-flash-lite": {
        "name": "🍃 Gemini 3.5 Flash-Lite",
        "desc": "Самая быстрая. Простые вопросы, перевод, большие объёмы.",
    },
}

THINKING_LEVELS = {
    "low": "⚡ Быстрый (low)",
    "medium": "🧠 Сбалансированный (medium)",
    "high": "🔬 Глубокий (high)",
}

HISTORY_OPTIONS = [3, 6, 10, 20, 50]
TEMPERATURE_OPTIONS = [0.0, 0.3, 0.7, 1.0, 1.5]
MAX_TOKENS_OPTIONS = [512, 1024, 2048, 4096, 8192]

# === Форматы изображений ===
IMAGE_FORMATS = {
    "square": {
        "name": "⬛ Квадрат (1:1)",
        "desc": "1024×1024. Универсальный формат.",
        "width": 1024,
        "height": 1024,
    },
    "wide": {
        "name": "🖼 Широкий (16:9)",
        "desc": "1344×768. Для пейзажей, баннеров, обоев.",
        "width": 1344,
        "height": 768,
    },
    "portrait": {
        "name": "📱 Вертикальный (9:16)",
        "desc": "768×1344. Для портретов, сторис, мобильных обоев.",
        "width": 768,
        "height": 1344,
    },
}

# === Планировщик задач (APScheduler) ===
scheduler = BackgroundScheduler(timezone="Europe/Moscow")  # Укажи свой часовой пояс

# === Вспомогательные функции для работы с БД ===
def get_db_user(chat_id):
    """Возвращает объект пользователя из БД и сессию. Не забудь закрыть сессию!"""
    db = SessionLocal()
    user = db.query(User).filter(User.telegram_id == chat_id).first()
    if not user:
        user = User(telegram_id=chat_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user, db

# === Клавиатура ===
def get_main_keyboard():
    markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn_profile = KeyboardButton("👤 Профиль")
    btn_models = KeyboardButton("🧠 Модели")
    btn_settings = KeyboardButton("⚙️ Настройки")
    btn_image = KeyboardButton("🎨 Нарисовать")
    btn_hf = KeyboardButton("🎨 Модель картинок")
    btn_format = KeyboardButton("📐 Формат")
    btn_reset = KeyboardButton("🔄 Сбросить историю")
    btn_help = KeyboardButton("ℹ️ Помощь")
    btn_status = KeyboardButton("📊 Статус")
    markup.add(btn_profile, btn_models, btn_settings, btn_image, btn_hf, btn_format, btn_reset, btn_help, btn_status)
    return markup

# === История ===
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
        system_msg = None
        if history and history[0]["role"] == "system":
            system_msg = history[0]
        non_system = [msg for msg in history if msg["role"] != "system"]
        if len(non_system) > limit:
            non_system = non_system[-limit:]
        new_history = []
        if system_msg:
            new_history.append(system_msg)
        new_history.extend(non_system)
        chat_history[chat_id] = new_history

def update_history(chat_id, role, content):
    history = get_history(chat_id)
    if history and history[-1]["role"] == role and history[-1]["content"] == content:
        return
    if role == "assistant" and len(content) > MAX_SAVED_ANSWER_LEN:
        content = content[:MAX_SAVED_ANSWER_LEN] + "... (обрезано)"
    history.append({"role": role, "content": content})
    trim_history(chat_id)

def clear_history(chat_id):
    # Не добавляем system-сообщение — свой промпт мы добавляем в reply_text
    chat_history[chat_id] = []

# === Форматирование Markdown -> HTML для Telegram ===
def format_plain_text(text):
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', escaped)
    return escaped

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
            print(f"⚠️ HTML не прошёл, отправляю как plain: {e}", flush=True)
            if i == 0:
                bot.send_message(chat_id, chunk, reply_to_message_id=reply_to_message_id)
            else:
                bot.send_message(chat_id, chunk)
        if i < len(chunks) - 1:
            time.sleep(0.5)

# === ПЕРЕВОД ПРОМПТА через Gemini 3.5 Flash-Lite ===
def translate_to_english(text):
    try:
        payload = {
            "model": "gemini-3.5-flash-lite",
            "messages": [{
                "role": "user",
                "content": f"Translate the following text to English. Output ONLY the English translation, without quotes, explanations, or extra words: {text}"
            }],
            "max_tokens": 300,
            "temperature": 0.1
        }
        r = requests.post(AI_URL, json=payload, headers=HEADERS, timeout=30)
        data = r.json()
        if 'choices' in data and data['choices']:
            english = data['choices'][0]['message']['content'].strip()
            english = english.strip('"\'')
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
    chat_id = message.chat.id
    bot.send_message(chat_id, "📐 *Выбери формат изображения:*", reply_markup=build_formats_keyboard(chat_id), parse_mode="Markdown")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_format:'))
def callback_set_format(call):
    chat_id = call.message.chat.id
    fmt_id = call.data.split(':', 1)[1]

    user, db = get_db_user(chat_id)
    user.format = fmt_id
    db.commit()
    db.close()

    info = IMAGE_FORMATS[fmt_id]
    bot.answer_callback_query(call.id, f"Формат: {info['name']}")
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"✅ Формат: *{info['name']}*\n\n{info['desc']}", parse_mode="Markdown")

# === МЕНЮ МОДЕЛЕЙ HUGGING FACE ===
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
    chat_id = message.chat.id
    bot.send_message(chat_id, "🎨 *Выбери модель для генерации картинок:*\n\nВсе модели работают через Hugging Face — бесплатно, без водяных знаков.", reply_markup=build_hf_models_keyboard(chat_id), parse_mode="Markdown")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_hf_model:'))
def callback_set_hf_model(call):
    chat_id = call.message.chat.id
    model_id = call.data.split(':', 1)[1]

    user, db = get_db_user(chat_id)
    user.hf_model = model_id
    db.commit()
    db.close()

    info = HF_MODELS[model_id]
    bot.answer_callback_query(call.id, f"Выбрана модель: {info['name']}")
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
    chat_id = message.chat.id
    bot.send_message(chat_id, "🧠 *Выбери модель:*\n\nНажми на модель, чтобы увидеть её описание.", reply_markup=build_models_keyboard(chat_id), parse_mode="Markdown")

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
    markup.add(InlineKeyboardButton("⬅️ Назад к списку", callback_data="back_to_models"))
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"*{info['name']}*\n\n{info['desc']}\n\nВыбрать эту модель?", reply_markup=markup, parse_mode="Markdown")
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
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"Модель *{MODEL_INFO[model_id]['name']}* выбрана!\n\nТеперь выбери *режим размышлений*:", reply_markup=markup, parse_mode="Markdown")
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

    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text=f"✅ *Настройки сохранены!*\n\n🧠 Модель: {MODEL_INFO[model_id]['name']}\n⚙️ Режим: {THINKING_LEVELS[level]}", parse_mode="Markdown")
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
    markup.add(InlineKeyboardButton(text=f"📏 Длина контекста: {user.history_len} сообщений", callback_data="settings:history"))
    markup.add(InlineKeyboardButton(text=f"🎲 Температура: {user.temperature}", callback_data="settings:temperature"))
    markup.add(InlineKeyboardButton(text=f"📝 Макс. токенов: {user.max_tokens}", callback_data="settings:max_tokens"))
    markup.add(InlineKeyboardButton(text="🔄 Сбросить всё по умолчанию", callback_data="settings:reset"))
    db.close()
    return markup

@bot.message_handler(commands=['settings'])
def show_settings(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, "⚙️ *Настройки контекста*\n\n📏 *Длина контекста* — сколько сообщений бот помнит.\n🎲 *Температура* — креативность (0.0 — строго, 1.5 — творчески).\n📝 *Макс. токенов* — длина ответа.", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")

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
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="✅ *Настройки сброшены!*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")
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
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="⚙️ *Настройки контекста*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")
    bot.answer_callback_query(call.id, f"Контекст: {val}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_temperature:'))
def callback_set_temp(call):
    chat_id = call.message.chat.id
    val = float(call.data.split(':', 1)[1])
    user, db = get_db_user(chat_id)
    user.temperature = val
    db.commit()
    db.close()
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="⚙️ *Настройки контекста*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")
    bot.answer_callback_query(call.id, f"Температура: {val}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_tokens:'))
def callback_set_tokens(call):
    chat_id = call.message.chat.id
    val = int(call.data.split(':', 1)[1])
    user, db = get_db_user(chat_id)
    user.max_tokens = val
    db.commit()
    db.close()
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id, text="⚙️ *Настройки контекста*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")
    bot.answer_callback_query(call.id, f"Макс. токенов: {val}")

# === RPG: ПРОФИЛЬ ===
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
        f"📊 Прогресс до следующего уровня: {user.xp}/{xp_needed}"
    )
    bot.reply_to(message, text, parse_mode="Markdown")
    db.close()

# === КОМАНДЫ ===
@bot.message_handler(commands=['start', 'reset'])
def send_welcome(message):
    chat_id = message.chat.id
    clear_history(chat_id)
    user, db = get_db_user(chat_id)

    model_name = MODEL_INFO[user.model]["name"]
    hf_model_name = HF_MODELS[user.hf_model]["name"]
    format_name = IMAGE_FORMATS[user.format]["name"]
    db.close()

    bot.send_message(chat_id,
                     f"Привет! Я бот на нейросети Gemini.\n"
                     f"🧠 Модель: {model_name}\n"
                     f"🎨 Модель картинок: {hf_model_name}\n"
                     f"📐 Формат: {format_name}\n"
                     f"Умею: текст, фото, голосовые и генерацию картинок.\n"
                     f"Используй кнопки внизу или /models, /settings, /image_models, /format, /image, /profile",
                     reply_markup=get_main_keyboard())

@bot.message_handler(commands=['help'])
def help_command(message):
    help_text = (
        "📚 Помощь:\n"
        "• Отправь текст – отвечу.\n"
        "• Отправь фото – опишу.\n"
        "• Отправь голосовое – расшифрую и отвечу.\n"
        "• 🎨 Нарисовать или /image <описание> – генерация картинок.\n"
        "• 🎨 Модель картинок или /image_models – выбрать модель генерации.\n"
        "• 📐 Формат или /format – выбрать формат (квадрат/широкий/вертикальный).\n"
        "• 🧠 Модели – выбрать модель Gemini.\n"
        "• ⚙️ Настройки – контекст, температура, токены.\n"
        "• 👤 Профиль – посмотреть свой уровень и баланс.\n"
        "• 🔄 Сбросить историю – очистить память.\n"
        "• 📊 Статус – текущие настройки.\n"
        "• Команды: /start, /reset, /help, /stats, /models, /settings, /test, /image, /image_models, /format, /profile"
    )
    bot.reply_to(message, help_text, reply_markup=get_main_keyboard())

@bot.message_handler(commands=['stats'])
def stats_command(message):
    chat_id = message.chat.id
    history = get_history(chat_id)
    total_chars = sum(len(str(msg["content"])) for msg in history)
    user, db = get_db_user(chat_id)

    bot.reply_to(message,
                 f"📊 *Текущий статус:*\n\n"
                 f"🧠 Модель Gemini: {MODEL_INFO[user.model]['name']}\n"
                 f"⚙️ Режим: {THINKING_LEVELS[user.thinking]}\n"
                 f"📏 Контекст: {user.history_len}\n"
                 f"🎲 Температура: {user.temperature}\n"
                 f"📝 Макс. токенов: {user.max_tokens}\n"
                 f"🎨 Модель картинок: {HF_MODELS[user.hf_model]['name']}\n"
                 f"📐 Формат: {IMAGE_FORMATS[user.format]['name']}\n\n"
                 f"Сообщений в истории: {len(history)}\n"
                 f"Размер: {total_chars} символов (~{total_chars//4} токенов)",
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
    bot.reply_to(message, "🎨 Напиши, что нарисовать, командой:\n`/image кот в космосе`\n\nПромпт можно на русском — я переведу на английский автоматически.", parse_mode="Markdown")

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

# === ГЕНЕРАЦИЯ КАРТИНОК (Hugging Face) ===
@bot.message_handler(commands=['image'])
def generate_image(message):
    chat_id = message.chat.id
    prompt = message.text.replace('/image', '', 1).strip()

    if not prompt:
        bot.reply_to(message, "🖼 Напиши, что нарисовать: `/image кот в космосе`", parse_mode="Markdown")
        return

    if not hf_client:
        bot.reply_to(message, "❌ Не задан токен Hugging Face. Добавь переменную `HF_TOKEN` на Render.", parse_mode="Markdown")
        return

    user, db = get_db_user(chat_id)
    model_info = HF_MODELS[user.hf_model]
    fmt = IMAGE_FORMATS[user.format]
    db.close()

    bot.send_message(chat_id, f"🎨 Генерирую через Hugging Face ({model_info['name']})...\nЭто может занять 15-30 секунд.")

    english_prompt = translate_to_english(prompt)
    print(f"🎨 Итоговый промпт: {english_prompt}", flush=True)
    print(f"📐 Формат: {fmt['width']}×{fmt['height']}", flush=True)

    try:
        image = hf_client.text_to_image(english_prompt, model=model_info["id"], width=fmt['width'], height=fmt['height'])
        buff = BytesIO()
        image.save(buff, format="PNG")
        buff.seek(0)
        bot.send_photo(chat_id, buff, reply_to_message_id=message.message_id)
    except Exception as e:
        error_str = str(e)
        print(f"Ошибка Hugging Face: {error_str}", flush=True)
        if "503" in error_str:
            bot.reply_to(message, "⏳ Модель загружается на сервере. Попробуй ещё раз через 20-30 секунд.", reply_markup=get_main_keyboard())
        elif "429" in error_str:
            bot.reply_to(message, "⏳ Слишком много запросов. Подожди минуту и попробуй снова.", reply_markup=get_main_keyboard())
        elif "402" in error_str or "Payment Required" in error_str:
            bot.reply_to(message, "💳 Эта модель требует платный доступ. Выбери другую в меню 🎨 Модель картинок.", reply_markup=get_main_keyboard())
        elif "width" in error_str.lower() or "height" in error_str.lower():
            bot.reply_to(message, "⚠️ Эта модель не поддерживает выбранный формат. Попробуй квадрат или другую модель.", reply_markup=get_main_keyboard())
        else:
            bot.reply_to(message, f"❌ Ошибка Hugging Face: {error_str[:200]}", reply_markup=get_main_keyboard())

# === ОБРАБОТКА ТЕКСТА (С RPG-ЛОГИКОЙ) ===
@bot.message_handler(content_types=['text'])
def reply_text(message):
    user_text = message.text
    if user_text.startswith('/'):
        return

    chat_id = message.chat.id
    update_history(chat_id, "user", user_text)
    bot.send_chat_action(chat_id, 'typing')

    user, db = get_db_user(chat_id)

    system_prompt = (
        "Ты — Cyberpunk-lite ассистент в Telegram. Ты общаешься с пользователем как обычная нейросеть, "
        "но параллельно отслеживаешь его прогресс в RPG-стиле.\n"
        "ВСЕГДА отвечай СТРОГО одним JSON-ОБЪЕКТОМ (не массивом, без markdown-обёрток, без ```json):\n"
        "{\n"
        "  \"reply\": \"Твой текстовый ответ пользователю\",\n"
        "  \"xp\": 0,\n"
        "  \"money\": 0,\n"
        "  \"category\": \"\",\n"
        "  \"task\": null\n"
        "}\n"
        "Правила:\n"
        "1. Обычный вопрос / болтовня — просто ответь в 'reply', остальное по нулям.\n"
        "2. Пользователь выполнил задачу — xp от 5 до 100.\n"
        "3. Упоминает трату/доход — money (минус для трат), category — категория.\n"
        "4. Просит напоминание — заполни task: {\"description\": \"...\", \"type\": \"one_time|daily|weekly\", \"time\": \"HH:MM\", \"days\": \"Mon,Wed\"}."
    )

    # Формируем messages: один system + история без system-сообщений
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
            print(f"🤖 RAW ответ Gemini: {raw_reply[:600]}", flush=True)

            # Убираем markdown-обёртки ```json ... ```
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

            # Если JSON не получился вообще — отдаём как обычный текст
            if parsed is None:
                print(f"⚠️ JSON не распарсился, отправляю как plain", flush=True)
                send_long_message(chat_id, raw_reply, message.message_id)
                update_history(chat_id, "assistant", raw_reply)
                break

            # Если Gemini вернул массив — берём первый элемент
            if isinstance(parsed, list):
                print(f"⚠️ Gemini вернул list, беру первый элемент", flush=True)
                parsed = parsed[0] if parsed else {}

            # Если всё ещё не словарь — fallback
            if not isinstance(parsed, dict):
                print(f"⚠️ JSON не dict: {type(parsed)}. Отправляю raw.", flush=True)
                send_long_message(chat_id, raw_reply, message.message_id)
                update_history(chat_id, "assistant", raw_reply)
                break

            # Пробуем разные ключи для текста ответа
            reply_text = (
                parsed.get("reply")
                or parsed.get("response")
                or parsed.get("message")
                or parsed.get("text")
                or parsed.get("content")
            )

            if not reply_text:
                for k, v in parsed.items():
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

            # 1. XP и уровень
            if xp_gain > 0:
                user.xp += xp_gain
                xp_needed = int(100 * (1.5 ** user.level))
                while user.xp >= xp_needed:
                    user.xp -= xp_needed
                    user.level += 1
                    xp_needed = int(100 * (1.5 ** user.level))
                    reply_text += f"\n\n🎉 **Уровень повышен!** Теперь ты Level {user.level}!"
                reply_text += f"\n\n✨ +{xp_gain} XP"

            # 2. Баланс
            if money_change != 0:
                user.balance += money_change
                db.add(Transaction(
                    user_id=user.id,
                    amount=money_change,
                    category=category,
                    description=user_text[:50]
                ))
                sign = "+" if money_change > 0 else ""
                reply_text += f"\n💰 Баланс: {sign}{money_change} ({category}). Текущий: {user.balance:.2f}"

            # 3. Задача
            if task_data and isinstance(task_data, dict):
                new_task = Task(
                    user_id=user.id,
                    description=task_data.get('description', 'Без названия'),
                    task_type=task_data.get('type', 'one_time'),
                    time_str=task_data.get('time'),
                    days=task_data.get('days')
                )
                db.add(new_task)
                reply_text += f"\n\n📝 Задача добавлена: {task_data.get('description')} в {task_data.get('time')}"

            db.commit()
            send_long_message(chat_id, reply_text, message.message_id)
            update_history(chat_id, "assistant", raw_reply)
            break

        except Exception as e:
            print(f"Ошибка (попытка {attempt + 1}): {e}", flush=True)
            if attempt == 2:
                bot.reply_to(message, "❌ Не удалось получить ответ. Напиши /test.", reply_markup=get_main_keyboard())
            else:
                time.sleep(2 * (attempt + 1))
    db.close()

# === ОБРАБОТКА ФОТО ===
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

            user_text = "Что изображено на этом фото? Опиши подробно на русском."

            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
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
            print(f"Ошибка при фото (попытка {attempt + 1}): {e}", flush=True)
            if attempt == 2:
                bot.reply_to(message, "❌ Не удалось обработать фото.", reply_markup=get_main_keyboard())
            else:
                time.sleep(2 * (attempt + 1))

# === ОБРАБОТКА ГОЛОСОВЫХ ===
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
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Расшифруй это голосовое сообщение и ответь на него по существу на русском языке."},
                    {"type": "input_audio", "input_audio": {"data": base64_audio, "format": "wav"}}
                ]}
            ],
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
        print(f"Ошибка при голосовом: {e}", flush=True)
        bot.reply_to(message, "❌ Не удалось обработать голосовое.", reply_markup=get_main_keyboard())

# === ФОНОВАЯ ПРОВЕРКА ЗАДАЧ (APScheduler) ===
def check_tasks():
    """Запускается каждую минуту. Пока заглушка — реальную логику напоминаний добавим позже."""
    db = SessionLocal()
    try:
        tasks = db.query(Task).filter(Task.is_active == True).all()
        # Логика напоминаний будет добавлена на следующем шаге
        for task in tasks:
            pass
    except Exception as e:
        print(f"Ошибка планировщика: {e}", flush=True)
    finally:
        db.close()

# === Веб-сервер для Render ===
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
