"""Telegram-бот поверх RAG-конвейера.

Запуск:
    export TELEGRAM_BOT_TOKEN="токен от @BotFather"
    .venv/bin/python src/bot.py

Ключевая деталь — блокирующий вызов в асинхронном коде.

rag.answer() работает ~10 секунд и блокирует поток: внутри синхронные HTTP-
запросы к ollama. Библиотека бота асинхронная и крутит один event loop на всех
пользователей. Если позвать блокирующую функцию прямо в обработчике, event loop
встанет, и пока один пользователь ждёт ответа, бот не ответит НИКОМУ.

Поэтому вызов уходит в отдельный поток через asyncio.to_thread(). Обработчик
на время ожидания освобождает loop, и остальные пользователи обслуживаются
параллельно. Это общее правило: любую синхронную операцию дольше пары
миллисекунд в async-коде нужно уводить в поток или переписывать на async.
"""
import asyncio
import logging
import os

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from rag import answer

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Лимит Telegram на одно сообщение — 4096 символов
MAX_MESSAGE_LEN = 4000

GREETING = (
    "Справочник по соционике: 16 типов и Модель А.\n\n"
    "Спроси, например:\n"
    "• кто такой Дон Кихот\n"
    "• какая болевая функция у Габена\n"
    "• какие типы в квадре Дельта\n"
    "• кто создаёт уют и избегает конфликтов\n\n"
    "Отвечаю только по своей базе: если данных нет — так и скажу, "
    "а не буду выдумывать. К каждому ответу прилагаю источники."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(GREETING)


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отвечает на вопрос пользователя через RAG-конвейер."""
    question = (update.message.text or "").strip()
    if not question:
        return

    # Ответ идёт ~10 секунд, поэтому показываем "печатает" — иначе пользователь
    # решит, что бот умер. Индикатор живёт ~5 секунд, обновляем его в фоне.
    typing = asyncio.create_task(_keep_typing(update, context))
    try:
        # Блокирующий вызов уводим в отдельный поток, чтобы не вставал event loop
        result = await asyncio.to_thread(answer, question)
    except Exception:
        logger.exception("Ошибка при ответе на вопрос: %s", question)
        await update.message.reply_text(
            "Что-то сломалось на моей стороне. Попробуй ещё раз чуть позже."
        )
        return
    finally:
        typing.cancel()

    await update.message.reply_text(_format_reply(result))
    logger.info("вопрос=%r источников=%d за %.1fс",
                question, len(result["sources"]), result["elapsed_sec"])


async def _keep_typing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Держит индикатор «печатает», пока считается ответ."""
    try:
        while True:
            await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


def _format_reply(result: dict) -> str:
    """Собирает текст ответа, перечисляя ТОЛЬКО процитированные источники.

    Перечислять все найденные документы нельзя: модель могла ответить из общих
    знаний, не воспользовавшись ни одним. Подпись "Источники" под таким ответом
    придаёт выдумке ложную достоверность — это хуже, чем совсем без ссылок.
    """
    text = result["answer"]
    cited = [s for s in result["sources"] if s["cited"]]
    if cited:
        listing = ", ".join(f"[{s['n']}] {s['name']}" for s in cited)
        text += f"\n\nИсточники: {listing}"
    elif result["sources"]:
        text += ("\n\n⚠️ Ответ не опирается на мою базу — возможно, "
                 "это общие знания модели, проверь их отдельно.")
    return text[:MAX_MESSAGE_LEN]


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "Не задан TELEGRAM_BOT_TOKEN.\n"
            "Получи токен у @BotFather и положи его в переменную окружения:\n"
            '    export TELEGRAM_BOT_TOKEN="..."'
        )

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    logger.info("Бот запущен, ждёт сообщения")
    # Long polling: бот сам опрашивает Telegram методом getUpdates. Альтернатива —
    # webhook, когда Telegram стучится к нам; он требует публичный HTTPS-адрес,
    # поэтому для локального запуска выбран polling.
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
