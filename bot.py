import os
import base64
import time
import threading
from io import BytesIO
from PIL import Image
import requests
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
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
AI_MODEL = "gemini-3.8-flash"  # стабильная бесплатная модель
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

bot = telebot.TeleBot(TOKEN)

chat_history = {}

# === Настройки ===
MAX_HISTORY_LENGTH = 6
MAX_SAVED_ANSWER_LEN = 800
MAX_TOKENS = 4096
TIMEOUT = 600

# === Клавиатура ===
def get_main_keyboard():
    markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn_reset = KeyboardButton("🔄 Сбросить историю")
    btn_help = KeyboardButton("ℹ️ Помощь")
    btn_status = KeyboardButton("📊 Статус")
    markup.add(btn_reset, btn_help, btn_status)
    return markup

# === Работа с историей ===
def get_history(chat_id):
    if chat_id not in chat_history:
        chat_history[chat_id] = []
    return chat_history[chat_id]

def trim_history(chat_id):
    history = get_history(chat_id)
    if len(history) > MAX_HISTORY_LENGTH + 1:
        system_msg = None
        if history and history[0]["role"] == "system":
            system_msg = history[0]
        non_system = [msg for msg in history if msg["role"] != "system"]
        if len(non_system) > MAX_HISTORY_LENGTH:
            non_system = non_system[-MAX_HISTORY_LENGTH:]
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

# === ТЕСТ GEMINI ===
@bot.message_handler(commands=['test'])
def test_gemini(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, "🧪 Тестирую Gemini, подожди...")
    try:
        payload = {
            "model": AI_MODEL,
            "messages": [{"role": "user", "content": "Скажи только слово: работает"}],
            "max_tokens": 50
        }
        r = requests.post(AI_URL, json=payload, headers=HEADERS, timeout=60)
        bot.send_message(chat_id, f"📡 Статус: {r.status_code}\n\nОтвет:\n{r.text[:1500]}")
    except Exception as e:
        bot.send_message(chat_id, f"💥 Ошибка: {type(e).__name__}: {e}")

# === Команды ===
@bot.message_handler(commands=['start', 'reset'])
def send_welcome(message):
    chat_id = message.chat.id
    clear_history(chat_id)
    bot.send_message(chat_id,
                     "Привет! Я бот на нейросети Gemini.\n"
                     "Умею считать, переводить и видеть картинки! ✨🧠\n"
                     "Используй кнопки внизу для управления.",
                     reply_markup=get_main_keyboard())

@bot.message_handler(commands=['help'])
def help_command(message):
    help_text = (
        "📚 Помощь:\n"
        "• Отправь мне текст – я отвечу.\n"
        "• Отправь фото – я опишу его.\n"
        "• Кнопка '🔄 Сбросить историю' – очищает память.\n"
        "• Кнопка '📊 Статус' – показывает информацию о текущем диалоге.\n"
        "• Команды: /start, /reset, /help, /stats, /test"
    )
    bot.reply_to(message, help_text, reply_markup=get_main_keyboard())

@bot.message_handler(commands=['stats'])
def stats_command(message):
    chat_id = message.chat.id
    history = get_history(chat_id)
    total_chars = sum(len(str(msg["content"])) for msg in history)
    bot.reply_to(message,
                 f"📊 Статус диалога:\n"
                 f"Сообщений в истории: {len(history)}\n"
                 f"Примерный размер: {total_chars} символов\n"
                 f"(лимит ~8000 токенов, сейчас ~{total_chars//4} токенов)",
                 reply_markup=get_main_keyboard())

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

    for attempt in range(3):
        try:
            payload = {
                "model": AI_MODEL,
                "messages": get_history(chat_id),
                "max_tokens": MAX_TOKENS,
                "temperature": 0.7
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
                "model": AI_MODEL,
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.7
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
