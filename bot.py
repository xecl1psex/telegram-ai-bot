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

# === Загрузка переменных окружения ===
load_dotenv()

TOKEN = os.environ.get("BOT_TOKEN")
API_KEY = os.environ.get("AI_API_KEY")

if not TOKEN:
    raise ValueError("Не задан BOT_TOKEN в переменных окружения")
if not API_KEY:
    raise ValueError("Не задан AI_API_KEY в переменных окружения")

# === Настройки API (Gemini через OpenAI-совместимый эндпоинт) ===
AI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

bot = telebot.TeleBot(TOKEN)

# === Настройки по умолчанию ===
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_THINKING = "medium"
DEFAULT_HISTORY_LEN = 6
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096

MAX_SAVED_ANSWER_LEN = 800
TIMEOUT = 600

# === Словари для хранения настроек пользователей ===
user_models = {}
user_thinking = {}
user_history_len = {}
user_temperature = {}
user_max_tokens = {}
chat_history = {}

# === Описания моделей ===
MODEL_INFO = {
    "gemini-3.8-flash": {
        "name": "🌟 Gemini 3.8 Flash",
        "desc": "Самая умная Flash-модель. Отлично подходит для сложных задач, рассуждений и анализа изображений.",
    },
    "gemini-3.7-flash": {
        "name": "⚡ Gemini 3.7 Flash",
        "desc": "Быстрая и мощная. Хороша для кодинга, работы с видео и агентных задач.",
    },
    "gemini-3.6-flash": {
        "name": "🚀 Gemini 3.6 Flash",
        "desc": "Надёжная рабочая лошадка. Баланс скорости, качества и эффективности.",
    },
    "gemini-3.5-flash-lite": {
        "name": "🍃 Gemini 3.5 Flash-Lite",
        "desc": "Самая быстрая и лёгкая. Идеальна для простых вопросов, перевода и больших объёмов.",
    },
}

THINKING_LEVELS = {
    "low": "⚡ Быстрый (low)",
    "medium": "🧠 Сбалансированный (medium)",
    "high": "🔬 Глубокий (high)",
}

# Варианты для настроек контекста
HISTORY_OPTIONS = [3, 6, 10, 20, 50]
TEMPERATURE_OPTIONS = [0.0, 0.3, 0.7, 1.0, 1.5]
MAX_TOKENS_OPTIONS = [512, 1024, 2048, 4096, 8192]

# === Клавиатура ===
def get_main_keyboard():
    markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn_reset = KeyboardButton("🔄 Сбросить историю")
    btn_help = KeyboardButton("ℹ️ Помощь")
    btn_status = KeyboardButton("📊 Статус")
    btn_models = KeyboardButton("🧠 Модели")
    btn_settings = KeyboardButton("⚙️ Настройки")
    markup.add(btn_models, btn_settings, btn_reset, btn_help)
    return markup

# === Работа с историей ===
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

# === Отправка длинных сообщений ===
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

# === МЕНЮ МОДЕЛЕЙ ===
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
        "🧠 *Выбери модель:*\n\n"
        "Нажми на модель, чтобы увидеть её описание и выбрать.",
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
    confirm_btn = InlineKeyboardButton("✅ Выбрать эту модель", callback_data=f"confirm_model:{model_id}")
    back_btn = InlineKeyboardButton("⬅️ Назад к списку", callback_data="back_to_models")
    markup.add(confirm_btn, back_btn)

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
        button = InlineKeyboardButton(
            text=f"{level_name}{check}",
            callback_data=f"thinking:{level_id}"
        )
        markup.add(button)

    bot.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text=f"Модель *{MODEL_INFO[model_id]['name']}* выбрана!\n\n"
             f"Теперь выбери *режим размышлений*:\n\n"
             f"• ⚡ `low` — быстрые ответы, минимум затрат\n"
             f"• 🧠 `medium` — баланс скорости и качества\n"
             f"• 🔬 `high` — глубокий анализ, сложные задачи",
        reply_markup=markup,
        parse_mode="Markdown"
    )
    bot.answer_callback_query(call.id, f"Модель выбрана!")

@bot.callback_query_handler(func=lambda call: call.data.startswith('thinking:'))
def callback_thinking(call):
    chat_id = call.message.chat.id
    level = call.data.split(':', 1)[1]
    user_thinking[chat_id] = level

    model_id = user_models.get(chat_id, DEFAULT_MODEL)
    model_name = MODEL_INFO[model_id]["name"]

    bot.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text=f"✅ *Настройки сохранены!*\n\n"
             f"🧠 Модель: {model_name}\n"
             f"⚙️ Режим: {THINKING_LEVELS[level]}\n\n"
             f"Теперь можешь задавать вопросы!",
        parse_mode="Markdown"
    )
    bot.answer_callback_query(call.id, "Настройки сохранены!")

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

# === МЕНЮ НАСТРОЕК КОНТЕКСТА ===
def build_settings_keyboard(chat_id):
    markup = InlineKeyboardMarkup(row_width=1)
    hist = user_history_len.get(chat_id, DEFAULT_HISTORY_LEN)
    temp = user_temperature.get(chat_id, DEFAULT_TEMPERATURE)
    tokens = user_max_tokens.get(chat_id, DEFAULT_MAX_TOKENS)

    markup.add(InlineKeyboardButton(
        text=f"📏 Длина контекста: {hist} сообщений",
        callback_data="settings:history"
    ))
    markup.add(InlineKeyboardButton(
        text=f"🎲 Температура: {temp}",
        callback_data="settings:temperature"
    ))
    markup.add(InlineKeyboardButton(
        text=f"📝 Макс. токенов ответа: {tokens}",
        callback_data="settings:max_tokens"
    ))
    markup.add(InlineKeyboardButton(
        text="🔄 Сбросить всё к значениям по умолчанию",
        callback_data="settings:reset"
    ))
    return markup

@bot.message_handler(commands=['settings'])
def show_settings(message):
    chat_id = message.chat.id
    bot.send_message(
        chat_id,
        "⚙️ *Настройки контекста*\n\n"
        "Здесь можно настроить, как нейросеть работает с диалогом:\n\n"
        "📏 *Длина контекста* — сколько последних сообщений бот помнит.\n"
        "🎲 *Температура* — насколько креативные ответы (0.0 — строго, 1.5 — творчески).\n"
        "📝 *Макс. токенов* — максимальная длина одного ответа.",
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
            markup.add(InlineKeyboardButton(
                text=f"{opt} сообщений{check}",
                callback_data=f"set_history:{opt}"
            ))
        markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:back"))
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=call.message.message_id,
            text="📏 *Выбери длину контекста:*\n\n"
                 "Чем больше — тем лучше бот помнит диалог, но тем больше токенов тратит.",
            reply_markup=markup,
            parse_mode="Markdown"
        )

    elif action == "temperature":
        current = user_temperature.get(chat_id, DEFAULT_TEMPERATURE)
        markup = InlineKeyboardMarkup(row_width=1)
        labels = {
            0.0: "0.0 — Строгие, точные ответы",
            0.3: "0.3 — Умеренно строгие",
            0.7: "0.7 — Баланс (рекомендуется)",
            1.0: "1.0 — Креативные",
            1.5: "1.5 — Максимально творческие",
        }
        for opt in TEMPERATURE_OPTIONS:
            check = " ✅" if opt == current else ""
            markup.add(InlineKeyboardButton(
                text=f"{labels[opt]}{check}",
                callback_data=f"set_temperature:{opt}"
            ))
        markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:back"))
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=call.message.message_id,
            text="🎲 *Выбери температуру:*\n\n"
                 "Влияет на «креативность» ответов. Чем ниже — тем более предсказуемые и точные ответы.",
            reply_markup=markup,
            parse_mode="Markdown"
        )

    elif action == "max_tokens":
        current = user_max_tokens.get(chat_id, DEFAULT_MAX_TOKENS)
        markup = InlineKeyboardMarkup(row_width=1)
        for opt in MAX_TOKENS_OPTIONS:
            check = " ✅" if opt == current else ""
            markup.add(InlineKeyboardButton(
                text=f"{opt} токенов{check}",
                callback_data=f"set_tokens:{opt}"
            ))
        markup.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:back"))
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=call.message.message_id,
            text="📝 *Выбери максимальную длину ответа:*\n\n"
                 "512 токенов ≈ 1-2 абзаца, 8192 ≈ длинная статья.",
            reply_markup=markup,
            parse_mode="Markdown"
        )

    elif action == "reset":
        user_history_len[chat_id] = DEFAULT_HISTORY_LEN
        user_temperature[chat_id] = DEFAULT_TEMPERATURE
        user_max_tokens[chat_id] = DEFAULT_MAX_TOKENS
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=call.message.message_id,
            text="✅ *Настройки сброшены к значениям по умолчанию!*",
            reply_markup=build_settings_keyboard(chat_id),
            parse_mode="Markdown"
        )

    elif action == "back":
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=call.message.message_id,
            text="⚙️ *Настройки контекста*",
            reply_markup=build_settings_keyboard(chat_id),
            parse_mode="Markdown"
        )

    bot.answer_callback_query(call.id)

# Обработчики выбора значений
@bot.callback_query_handler(func=lambda call: call.data.startswith('set_history:'))
def callback_set_history(call):
    chat_id = call.message.chat.id
    val = int(call.data.split(':', 1)[1])
    user_history_len[chat_id] = val
    trim_history(chat_id)
    bot.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text="⚙️ *Настройки контекста*",
        reply_markup=build_settings_keyboard(chat_id),
        parse_mode="Markdown"
    )
    bot.answer_callback_query(call.id, f"Контекст: {val} сообщений")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_temperature:'))
def callback_set_temp(call):
    chat_id = call.message.chat.id
    val = float(call.data.split(':', 1)[1])
    user_temperature[chat_id] = val
    bot.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text="⚙️ *Настройки контекста*",
        reply_markup=build_settings_keyboard(chat_id),
        parse_mode="Markdown"
    )
    bot.answer_callback_query(call.id, f"Температура: {val}")

@bot.callback_query_handler(func=lambda call: call.data.startswith('set_tokens:'))
def callback_set_tokens(call):
    chat_id = call.message.chat.id
    val = int(call.data.split(':', 1)[1])
    user_max_tokens[chat_id] = val
    bot.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text="⚙️ *Настройки контекста*",
        reply_markup=build_settings_keyboard(chat_id),
        parse_mode="Markdown"
    )
    bot.answer_callback_query(call.id, f"Макс. токенов: {val}")

# === ТЕСТ ===
@bot.message_handler(commands=['test'])
def test_gemini(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, "🧪 Тестирую Gemini, подожди...")
    model = user_models.get(chat_id, DEFAULT_MODEL)
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Скажи только слово: работает"}],
            "max_tokens": 50
        }
        r = requests.post(AI_URL, json=payload, headers=HEADERS, timeout=60)
        bot.send_message(chat_id, f"📡 Модель: {model}\nСтатус: {r.status_code}\n\nОтвет:\n{r.text[:1500]}")
    except Exception as e:
        bot.send_message(chat_id, f"💥 Ошибка: {type(e).__name__}: {e}")

# === Команды ===
@bot.message_handler(commands=['start', 'reset'])
def send_welcome(message):
    chat_id = message.chat.id
    clear_history(chat_id)
    model_name = MODEL_INFO[user_models.get(chat_id, DEFAULT_MODEL)]["name"]
    bot.send_message(chat_id,
                     f"Привет! Я бот на нейросети Gemini.\n"
                     f"Сейчас активна модель: {model_name}\n"
                     f"Умею считать, переводить и видеть картинки! ✨🧠\n"
                     f"Используй кнопки внизу или /models и /settings.",
                     reply_markup=get_main_keyboard())

@bot.message_handler(commands=['help'])
def help_command(message):
    help_text = (
        "📚 Помощь:\n"
        "• Отправь мне текст – я отвечу.\n"
        "• Отправь фото – я опишу его.\n"
        "• Кнопка '🧠 Модели' – выбрать модель и режим.\n"
        "• Кнопка '⚙️ Настройки' – настроить контекст, температуру, длину ответа.\n"
        "• Кнопка '🔄 Сбросить историю' – очищает память.\n"
        "• Кнопка '📊 Статус' – показывает текущие настройки.\n"
        "• Команды: /start, /reset, /help, /stats, /models, /settings, /test"
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
    bot.reply_to(message,
                 f"📊 *Текущий статус:*\n\n"
                 f"🧠 Модель: {MODEL_INFO[model]['name']}\n"
                 f"⚙️ Режим: {THINKING_LEVELS[thinking]}\n"
                 f"📏 Контекст: {hist_len} сообщений\n"
                 f"🎲 Температура: {temp}\n"
                 f"📝 Макс. токенов: {tokens}\n\n"
                 f"Сообщений в истории: {len(history)}\n"
                 f"Примерный размер: {total_chars} символов (~{total_chars//4} токенов)",
                 reply_markup=get_main_keyboard(),
                 parse_mode="Markdown")

@bot.message_handler(func=lambda m: m.text == "🧠 Модели")
def models_button(message):
    show_models(message)

@bot.message_handler(func=lambda m: m.text == "⚙️ Настройки")
def settings_button(message):
    show_settings(message)

@bot.message_handler(func=lambda m: m.text == "🔄 Сбросить историю")
def reset_button(message):
    clear_history(message.chat.id)
    bot.reply_to(message, "✅ История очищена! Можете задавать новый вопрос.", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda m: m.text == "ℹ️ Помощь")
def help_button(message):
    help_command(message)

@bot.message_handler(func=lambda m: m.text == "📊 Статус")
def status_button(message):
    stats_command(message)

# === Обработка текста ===
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
                bot.reply_to(message, "❌ Не удалось получить ответ. Напиши /test чтобы увидеть причину.", reply_markup=get_main_keyboard())
            else:
                time.sleep(2)

# === Обработка фото ===
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
                time.sleep(2)

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
