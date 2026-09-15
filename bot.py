import os
import base64
import time
import threading
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
    "sdxl": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "name": "🚀 SDXL Base 1.0",
        "desc": "Проверенная модель с широкими возможностями настройки.",
    },
}

bot = telebot.TeleBot(TOKEN)

# === Настройки по умолчанию ===
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_THINKING = "medium"
DEFAULT_HISTORY_LEN = 6
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_FORMAT = "square"
DEFAULT_HF_MODEL = "flux_schnell"

MAX_SAVED_ANSWER_LEN = 800
TIMEOUT = 600

# === Память пользователей ===
user_models = {}
user_thinking = {}
user_history_len = {}
user_temperature = {}
user_max_tokens = {}
user_hf_model = {}
user_format = {}
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

# === Клавиатура ===
def get_main_keyboard():
    markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn_models = KeyboardButton("🧠 Модели")
    btn_settings = KeyboardButton("⚙️ Настройки")
    btn_image = KeyboardButton("🎨 Нарисовать")
    btn_hf = KeyboardButton("🎨 Модель картинок")
    btn_format = KeyboardButton("📐 Формат")
    btn_reset = KeyboardButton("🔄 Сбросить историю")
    btn_help = KeyboardButton("ℹ️ Помощь")
    btn_status = KeyboardButton("📊 Статус")
    markup.add(btn_models, btn_settings, btn_image, btn_hf, btn_format, btn_reset, btn_help, btn_status)
    return markup

# === История ===
def get_history(chat_id):
    if chat_id not in chat_history:
        chat_history[chat_id] = []
    return chat_history[chat_id]

def trim_history(chat_id):
    history = get_history(chat_id)
    limit = user_history_len.get(chat_id, DEFAULT_HISTORY_LEN)
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
    chat_history[chat_id] = [{"role": "system", "content": "Ты — полезный ИИ-ассистент. Отвечай кратко и по делу."}]

# === Длинные сообщения ===
def send_long_message(chat_id, text, reply_to_message_id=None):
    if not text:
        return
    if len(text) <= 4096:
        bot.send_message(chat_id, text, reply_to_message_id=reply_to_message_id)
        return
    parts = []
    while text:
        if len(text) <= 4096:
            parts.append(text)
            break
        split_pos = text[:4096].rfind('. ')
        if split_pos == -1:
            split_pos = text[:4096].rfind(' ')
        if split_pos == -1:
            split_pos = 4096
        else:
            split_pos += 1
        parts.append(text[:split_pos])
        text = text[split_pos:]
    for i, part in enumerate(parts):
        if i == 0:
            bot.send_message(chat_id, part, reply_to_message_id=reply_to_message_id)
        else:
            bot.send_message(chat_id, part)
        time.sleep(0.5)

# === ПЕРЕВОД ПРОМПТА ===
def translate_to_english(text):
    # Google Translate через clients5 (работает с облачных серверов)
    try:
        r = requests.get(
            "https://clients5.google.com/translate_a/t",
            params={"client": "dict-chrome-ex", "sl": "auto", "tl": "en", "q": text},
            timeout=15
        )
        data = r.json()
        if isinstance(data, list) and data and isinstance(data[0], list):
            english = "".join(part[0] for part in data[0] if part)
            if english and "MYMEMORY WARNING" not in english:
                print(f"✅ Google: '{text}' → '{english}'", flush=True)
                return english.strip()
    except Exception as e:
        print(f"⚠️ Google (clients5) ошибка: {e}", flush=True)

    # Резерв
    try:
        r = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "auto", "tl": "en", "dt": "t", "q": text},
            timeout=15
        )
        data = r.json()
        english = "".join(part[0] for part in data[0] if part[0])
        if english and "MYMEMORY" not in english:
            print(f"✅ Google (gtx): '{text}' → '{english}'", flush=True)
            return english.strip()
    except Exception as e:
        print(f"⚠️ Google (gtx) ошибка: {e}", flush=True)

    print(f"❌ Перевод не удался: '{text}'", flush=True)
    return text

# === МЕНЮ ФОРМАТОВ ===
def build_formats_keyboard(chat_id):
    markup = InlineKeyboardMarkup(row_width=1)
    current = user_format.get(chat_id, DEFAULT_FORMAT)
    for fmt_id, info in IMAGE_FORMATS.items():
        check = " ✅" if fmt_id == current else ""
        markup.add(InlineKeyboardButton(
            text=f"{info['name']}{check}",
            callback_data=f"set_format:{fmt_id}"
        ))
    return markup

@bot.message_handler(commands=['format'])
def show_formats(message):
    chat_id = message.chat.id
    bot.send_message(
        chat_id,
        "📐 *Выбери формат изображения:*",
        reply_markup=build_formats_keyboard(chat_id),
        parse_mode="Markdown"
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_format:'))
def callback_set_format(call):
    chat_id = call.message.chat.id
    fmt_id = call.data.split(':', 1)[1]
    user_format[chat_id] = fmt_id
    info = IMAGE_FORMATS[fmt_id]
    bot.answer_callback_query(call.id, f"Формат: {info['name']}")
    bot.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text=f"✅ Формат: *{info['name']}*\n\n{info['desc']}",
        parse_mode="Markdown"
    )

# === МЕНЮ МОДЕЛЕЙ HUGGING FACE ===
def build_hf_models_keyboard(chat_id):
    markup = InlineKeyboardMarkup(row_width=1)
    current = user_hf_model.get(chat_id, DEFAULT_HF_MODEL)
    for model_id, info in HF_MODELS.items():
        check = " ✅" if model_id == current else ""
        markup.add(InlineKeyboardButton(
            text=f"{info['name']}{check}",
            callback_data=f"set_hf_model:{model_id}"
        ))
    return markup

@bot.message_handler(commands=['image_models'])
def show_hf_models(message):
    chat_id = message.chat.id
    bot.send_message(
        chat_id,
        "🎨 *Выбери модель для генерации картинок:*\n\n"
        "Все модели работают через Hugging Face — бесплатно, без водяных знаков.",
        reply_markup=build_hf_models_keyboard(chat_id),
        parse_mode="Markdown"
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_hf_model:'))
def callback_set_hf_model(call):
    chat_id = call.message.chat.id
    model_id = call.data.split(':', 1)[1]
    user_hf_model[chat_id] = model_id
    info = HF_MODELS[model_id]
    bot.answer_callback_query(call.id, f"Выбрана модель: {info['name']}")
    bot.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text=f"✅ Модель: *{info['name']}*\n\n{info['desc']}",
        parse_mode="Markdown"
    )

# === МЕНЮ МОДЕЛЕЙ GEMINI ===
def build_models_keyboard(chat_id):
    markup = InlineKeyboardMarkup(row_width=1)
    current_model = user_models.get(chat_id, DEFAULT_MODEL)
    for model_id, info in MODEL_INFO.items():
        check = " ✅" if model_id == current_model else ""
        button = InlineKeyboardButton(
            text=f"{info['name']}{check}",
            callback_data=f"model:{model_id}"
        )
        markup.add(button)
    return markup

@bot.message_handler(commands=['models'])
def show_models(message):
    chat_id = message.chat.id
    bot.send_message(
        chat_id,
        "🧠 *Выбери модель:*\n\nНажми на модель, чтобы увидеть её описание.",
        reply_markup=build_models_keyboard(chat_id),
        parse_mode="Markdown"
    )

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

    bot.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text=f"*{info['name']}*\n\n{info['desc']}\n\nВыбрать эту модель?",
        reply_markup=markup,
        parse_mode="Markdown"
    )
    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith('confirm_model:'))
def callback_confirm_model(call):
    chat_id = call.message.chat.id
    model_id = call.data.split(':', 1)[1]
    user_models[chat_id] = model_id

    markup = InlineKeyboardMarkup(row_width=1)
    current_thinking = user_thinking.get(chat_id, DEFAULT_THINKING)
    for level_id, level_name in THINKING_LEVELS.items():
        check = " ✅" if level_id == current_thinking else ""
        markup.add(InlineKeyboardButton(
            text=f"{level_name}{check}",
            callback_data=f"thinking:{level_id}"
        ))

    bot.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text=f"Модель *{MODEL_INFO[model_id]['name']}* выбрана!\n\nТеперь выбери *режим размышлений*:",
        reply_markup=markup,
        parse_mode="Markdown"
    )
    bot.answer_callback_query(call.id, "Модель выбрана!")

@bot.callback_query_handler(func=lambda call: call.data.startswith('thinking:'))
def callback_thinking(call):
    chat_id = call.message.chat.id
    level = call.data.split(':', 1)[1]
    user_thinking[chat_id] = level

    model_id = user_models.get(chat_id, DEFAULT_MODEL)
    bot.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text=f"✅ *Настройки сохранены!*\n\n"
             f"🧠 Модель: {MODEL_INFO[model_id]['name']}\n"
             f"⚙️ Режим: {THINKING_LEVELS[level]}",
        parse_mode="Markdown"
    )
    bot.answer_callback_query(call.id, "Сохранено!")

@bot.callback_query_handler(func=lambda call: call.data == 'back_to_models')
def callback_back(call):
    chat_id = call.message.chat.id
    bot.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text="🧠 *Выбери модель:*",
        reply_markup=build_models_keyboard(chat_id),
        parse_mode="Markdown"
    )
    bot.answer_callback_query(call.id)

# === НАСТРОЙКИ ===
def build_settings_keyboard(chat_id):
    markup = InlineKeyboardMarkup(row_width=1)
    hist = user_history_len.get(chat_id, DEFAULT_HISTORY_LEN)
    temp = user_temperature.get(chat_id, DEFAULT_TEMPERATURE)
    tokens = user_max_tokens.get(chat_id, DEFAULT_MAX_TOKENS)

    markup.add(InlineKeyboardButton(text=f"📏 Длина контекста: {hist} сообщений", callback_data="settings:history"))
    markup.add(InlineKeyboardButton(text=f"🎲 Температура: {temp}", callback_data="settings:temperature"))
    markup.add(InlineKeyboardButton(text=f"📝 Макс. токенов: {tokens}", callback_data="settings:max_tokens"))
    markup.add(InlineKeyboardButton(text="🔄 Сбросить всё по умолчанию", callback_data="settings:reset"))
    return markup

@bot.message_handler(commands=['settings'])
def show_settings(message):
    chat_id = message.chat.id
    bot.send_message(
        chat_id,
        "⚙️ *Настройки контекста*\n\n"
        "📏 *Длина контекста* — сколько сообщений бот помнит.\n"
        "🎲 *Температура* — креативность (0.0 — строго, 1.5 — творчески).\n"
        "📝 *Макс. токенов* — длина ответа.",
        reply_markup=build_settings_keyboard(chat_id),
        parse_mode="Markdown"
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith('settings:'))
def callback_settings(call):
    chat_id = call.message.chat.id
    action = call.data.split(':', 1)[1]

    if action == "history":
        current = user_history_len.get(chat_id, DEFAULT_HISTORY_LEN)
        markup = InlineKeyboardMarkup(row_width=1)
        for opt in HISTORY_OPTIONS:
            check = " ✅" if opt == current else ""
            markup.add(InlineKeyboardButton(text=f"{opt} сообщений{check}", callback_data=f"set_history:{opt}"))
        markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:back"))
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id,
                              text="📏 *Выбери длину контекста:*", reply_markup=markup, parse_mode="Markdown")

    elif action == "temperature":
        current = user_temperature.get(chat_id, DEFAULT_TEMPERATURE)
        markup = InlineKeyboardMarkup(row_width=1)
        labels = {0.0: "0.0 — Строго", 0.3: "0.3 — Умеренно", 0.7: "0.7 — Баланс",
                  1.0: "1.0 — Креативно", 1.5: "1.5 — Максимум"}
        for opt in TEMPERATURE_OPTIONS:
            check = " ✅" if opt == current else ""
            markup.add(InlineKeyboardButton(text=f"{labels[opt]}{check}", callback_data=f"set_temperature:{opt}"))
        markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:back"))
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id,
                              text="🎲 *Выбери температуру:*", reply_markup=markup, parse_mode="Markdown")

    elif action == "max_tokens":
        current = user_max_tokens.get(chat_id, DEFAULT_MAX_TOKENS)
        markup = InlineKeyboardMarkup(row_width=1)
        for opt in MAX_TOKENS_OPTIONS:
            check = " ✅" if opt == current else ""
            markup.add(InlineKeyboardButton(text=f"{opt} токенов{check}", callback_data=f"set_tokens:{opt}"))
        markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:back"))
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id,
                              text="📝 *Выбери макс. длину ответа:*", reply_markup=markup, parse_mode="Markdown")

    elif action == "reset":
        user_history_len[chat_id] = DEFAULT_HISTORY_LEN
        user_temperature[chat_id] = DEFAULT_TEMPERATURE
        user_max_tokens[chat_id] = DEFAULT_MAX_TOKENS
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id,
                              text="✅ *Настройки сброшены!*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")

    elif action == "back":
        bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id,
                              text="⚙️ *Настройки контекста*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")

    bot.answer_callback_query(call.id)

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_history:'))
def callback_set_history(call):
    chat_id = call.message.chat.id
    val = int(call.data.split(':', 1)[1])
    user_history_len[chat_id] = val
    trim_history(chat_id)
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id,
                          text="⚙️ *Настройки контекста*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")
    bot.answer_callback_query(call.id, f"Контекст: {val}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_temperature:'))
def callback_set_temp(call):
    chat_id = call.message.chat.id
    val = float(call.data.split(':', 1)[1])
    user_temperature[chat_id] = val
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id,
                          text="⚙️ *Настройки контекста*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")
    bot.answer_callback_query(call.id, f"Температура: {val}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_tokens:'))
def callback_set_tokens(call):
    chat_id = call.message.chat.id
    val = int(call.data.split(':', 1)[1])
    user_max_tokens[chat_id] = val
    bot.edit_message_text(chat_id=chat_id, message_id=call.message.message_id,
                          text="⚙️ *Настройки контекста*", reply_markup=build_settings_keyboard(chat_id), parse_mode="Markdown")
    bot.answer_callback_query(call.id, f"Макс. токенов: {val}")

# === ТЕСТ ===
@bot.message_handler(commands=['test'])
def test_gemini(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, "🧪 Тестирую Gemini...")
    model = user_models.get(chat_id, DEFAULT_MODEL)
    try:
        payload = {"model": model, "messages": [{"role": "user", "content": "Скажи: работает"}], "max_tokens": 50}
        r = requests.post(AI_URL, json=payload, headers=HEADERS, timeout=60)
        bot.send_message(chat_id, f"📡 Модель: {model}\nСтатус: {r.status_code}\n\n{r.text[:1500]}")
    except Exception as e:
        bot.send_message(chat_id, f"💥 Ошибка: {type(e).__name__}: {e}")

# === КОМАНДЫ ===
@bot.message_handler(commands=['start', 'reset'])
def send_welcome(message):
    chat_id = message.chat.id
    clear_history(chat_id)
    model_name = MODEL_INFO[user_models.get(chat_id, DEFAULT_MODEL)]["name"]
    hf_model_name = HF_MODELS[user_hf_model.get(chat_id, DEFAULT_HF_MODEL)]["name"]
    format_name = IMAGE_FORMATS[user_format.get(chat_id, DEFAULT_FORMAT)]["name"]
    bot.send_message(chat_id,
                     f"Привет! Я бот на нейросети Gemini.\n"
                     f"🧠 Модель: {model_name}\n"
                     f"🎨 Модель картинок: {hf_model_name}\n"
                     f"📐 Формат: {format_name}\n"
                     f"Умею: текст, фото, голосовые и генерацию картинок.\n"
                     f"Используй кнопки внизу или /models, /settings, /image_models, /format, /image",
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
        "• 🔄 Сбросить историю – очистить память.\n"
        "• 📊 Статус – текущие настройки.\n"
        "• Команды: /start, /reset, /help, /stats, /models, /settings, /test, /image, /image_models, /format"
    )
    bot.reply_to(message, help_text, reply_markup=get_main_keyboard())

@bot.message_handler(commands=['stats'])
def stats_command(message):
    chat_id = message.chat.id
    history = get_history(chat_id)
    total_chars = sum(len(str(msg["content"])) for msg in history)
    model = user_models.get(chat_id, DEFAULT_MODEL)
    thinking = user_thinking.get(chat_id, DEFAULT_THINKING)
    hist_len = user_history_len.get(chat_id, DEFAULT_HISTORY_LEN)
    temp = user_temperature.get(chat_id, DEFAULT_TEMPERATURE)
    tokens = user_max_tokens.get(chat_id, DEFAULT_MAX_TOKENS)
    hf_model_id = user_hf_model.get(chat_id, DEFAULT_HF_MODEL)
    fmt = user_format.get(chat_id, DEFAULT_FORMAT)
    bot.reply_to(message,
                 f"📊 *Текущий статус:*\n\n"
                 f"🧠 Модель Gemini: {MODEL_INFO[model]['name']}\n"
                 f"⚙️ Режим: {THINKING_LEVELS[thinking]}\n"
                 f"📏 Контекст: {hist_len}\n"
                 f"🎲 Температура: {temp}\n"
                 f"📝 Макс. токенов: {tokens}\n"
                 f"🎨 Модель картинок: {HF_MODELS[hf_model_id]['name']}\n"
                 f"📐 Формат: {IMAGE_FORMATS[fmt]['name']}\n\n"
                 f"Сообщений в истории: {len(history)}\n"
                 f"Размер: {total_chars} символов (~{total_chars//4} токенов)",
                 reply_markup=get_main_keyboard(), parse_mode="Markdown")

@bot.message_handler(func=lambda m: m.text == "🧠 Модели")
def models_button(message):
    show_models(message)

@bot.message_handler(func=lambda m: m.text == "⚙️ Настройки")
def settings_button(message):
    show_settings(message)

@bot.message_handler(func=lambda m: m.text == "🎨 Нарисовать")
def image_button(message):
    bot.reply_to(
        message,
        "🎨 Напиши, что нарисовать, командой:\n"
        "`/image кот в космосе`\n\n"
        "Промпт можно на русском — я переведу на английский автоматически.",
        parse_mode="Markdown"
    )

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

# === ГЕНЕРАЦИЯ КАРТИНОК (только Hugging Face) ===
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

    hf_model_id = user_hf_model.get(chat_id, DEFAULT_HF_MODEL)
    model_info = HF_MODELS[hf_model_id]
    fmt_id = user_format.get(chat_id, DEFAULT_FORMAT)
    fmt = IMAGE_FORMATS[fmt_id]

    bot.send_message(chat_id, f"🎨 Генерирую через Hugging Face ({model_info['name']})...\nЭто может занять 15-30 секунд.")

    english_prompt = translate_to_english(prompt)
    print(f"🎨 Итоговый промпт: {english_prompt}", flush=True)
    print(f"📐 Формат: {fmt['width']}×{fmt['height']}", flush=True)

    try:
        image = hf_client.text_to_image(
            english_prompt,
            model=model_info["id"],
            width=fmt['width'],
            height=fmt['height'],
        )
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
        elif "width" in error_str.lower() or "height" in error_str.lower():
            bot.reply_to(message, "⚠️ Эта модель не поддерживает выбранный формат. Попробуй квадрат или другую модель.", reply_markup=get_main_keyboard())
        else:
            bot.reply_to(message, f"❌ Ошибка Hugging Face: {error_str[:200]}", reply_markup=get_main_keyboard())

# === ОБРАБОТКА ТЕКСТА ===
@bot.message_handler(content_types=['text'])
def reply_text(message):
    user_text = message.text
    if user_text.startswith('/'):
        return

    chat_id = message.chat.id
    update_history(chat_id, "user", user_text)
    bot.send_chat_action(chat_id, 'typing')

    model = user_models.get(chat_id, DEFAULT_MODEL)
    temperature = user_temperature.get(chat_id, DEFAULT_TEMPERATURE)
    max_tokens = user_max_tokens.get(chat_id, DEFAULT_MAX_TOKENS)

    for attempt in range(3):
        try:
            payload = {
                "model": model,
                "messages": get_history(chat_id),
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            response = requests.post(AI_URL, json=payload, headers=HEADERS, timeout=TIMEOUT)
            data = response.json()

            if 'choices' not in data or not data['choices']:
                raise ValueError(f"Нет choices: {str(data)[:300]}")

            reply = data['choices'][0]['message']['content'].strip()
            if not reply:
                raise ValueError("Пустой ответ")

            send_long_message(chat_id, reply, message.message_id)
            update_history(chat_id, "assistant", reply)
            break
        except Exception as e:
            print(f"Ошибка (попытка {attempt + 1}): {e}", flush=True)
            if attempt == 2:
                bot.reply_to(message, "❌ Не удалось получить ответ. Напиши /test.", reply_markup=get_main_keyboard())
            else:
                time.sleep(2 * (attempt + 1))

# === ОБРАБОТКА ФОТО ===
@bot.message_handler(content_types=['photo'])
def reply_photo(message):
    chat_id = message.chat.id
    bot.send_chat_action(chat_id, 'typing')

    model = user_models.get(chat_id, DEFAULT_MODEL)
    temperature = user_temperature.get(chat_id, DEFAULT_TEMPERATURE)
    max_tokens = user_max_tokens.get(chat_id, DEFAULT_MAX_TOKENS)

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

    model = user_models.get(chat_id, DEFAULT_MODEL)
    temperature = user_temperature.get(chat_id, DEFAULT_TEMPERATURE)
    max_tokens = user_max_tokens.get(chat_id, DEFAULT_MAX_TOKENS)

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

print("✅ Бот запущен!", flush=True)
bot.infinity_polling()
