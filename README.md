# Telegram AI Bot (RPG Edition)

Бот на Gemini с поддержкой текста, изображений и RPG-механик.

## Запуск локально
1. `pip install -r requirements.txt`
2. Скопируйте `.env.example` в `.env` и вставьте свои ключи (BOT_TOKEN, AI_API_KEY, HF_TOKEN, DATABASE_URL).
3. Создайте базу данных PostgreSQL (например, на Neon) и укажите ссылку в `DATABASE_URL`.
4. `python bot.py`

## Важное примечание для Render
Бесплатный тариф Render "засыпает" через 15 минут бездействия. Это значит, что APScheduler (напоминания) не будет работать, пока бот спит. 
Чтобы напоминания приходили вовремя, рекомендую использовать внешний сервис (например, [cron-job.org](https://cron-job.org/)), который будет пинговать ваш Render URL каждые 5-10 минут, чтобы бот не засыпал.
