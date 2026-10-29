"""Полный RAG-конвейер: вопрос -> поиск -> промпт -> ответ со ссылками.

Запуск:  .venv/bin/python src/rag.py "кто такой Дон Кихот"
         .venv/bin/python src/rag.py --prompt loose "у кого болевая этика отношений"
         .venv/bin/python src/rag.py --show-context "какие типы в квадре Дельта"

Здесь сходятся оба слоя: ретривал (что нашли) и генерация (как рассказали).
Модуль намеренно тонкий — он только оркестрирует и ничего не решает сам.
"""
import re
import sys
import time

from config import TOP_K, DEFAULT_GENERATION, DEFAULT_PROMPT
from generate import complete
from prompts import SYSTEM_PROMPTS, build_user_message
from retrieve import search

NO_CONTEXT_ANSWER = "В моей базе нет данных по этому вопросу."


def answer(question: str, k: int = TOP_K, prompt: str = DEFAULT_PROMPT,
           generation: str = DEFAULT_GENERATION, **search_kwargs) -> dict:
    """Отвечает на вопрос по корпусу и возвращает ответ вместе с источниками.

    Возвращаем не только текст, но и найденные документы: без этого ответ
    невозможно проверить, а RAG без проверяемости теряет главное преимущество
    перед голой моделью.
    """
    started = time.perf_counter()
    hits = search(question, k=k, **search_kwargs)

    # Пустой ретривал — не повод звать LLM: без контекста она может только
    # выдумать. Дешевле и честнее ответить отказом сразу.
    if not hits:
        return {"question": question, "answer": NO_CONTEXT_ANSWER,
                "sources": [], "elapsed_sec": time.perf_counter() - started}

    text = complete(
        system=SYSTEM_PROMPTS[prompt],
        user=build_user_message(question, hits),
        profile=generation,
    )
    # Какие источники модель РЕАЛЬНО процитировала. Показывать все найденные
    # безусловно нельзя: если модель ответила из общих знаний и не сослалась
    # ни на что, подпись "Источники" придаёт выдумке ложную достоверность —
    # это хуже, чем отсутствие ссылок вообще.
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    return {
        "question": question,
        "answer": text,
        "sources": [{"n": i, "id": h["id"], "name": h["name"],
                     "matched_slot": h.get("matched_slot", ""), "cited": i in cited}
                    for i, h in enumerate(hits, start=1)],
        "grounded": bool(cited),
        "context": hits,
        "elapsed_sec": time.perf_counter() - started,
    }


def main() -> None:
    args = sys.argv[1:]
    prompt, show_context = DEFAULT_PROMPT, False
    while args and args[0].startswith("--"):
        flag, args = args[0], args[1:]
        if flag == "--prompt":
            prompt, args = args[0], args[1:]
        elif flag == "--show-context":
            show_context = True
    if not args:
        print('Использование: python src/rag.py [--prompt strict|loose] '
              '[--show-context] "вопрос"')
        raise SystemExit(1)

    result = answer(" ".join(args), prompt=prompt)

    print(f"ВОПРОС: {result['question']}\n")
    if show_context and result.get("context"):
        print("--- контекст, отданный модели ---")
        for i, hit in enumerate(result["context"], start=1):
            print(f"[{i}] {hit['name']} (совпал чанк: {hit.get('matched_slot') or '-'})")
        print()
    print(f"ОТВЕТ:\n{result['answer']}\n")
    cited = [s for s in result["sources"] if s["cited"]]
    if cited:
        print("ИСТОЧНИКИ: " + ", ".join(f"[{s['n']}] {s['name']}" for s in cited))
    elif result["sources"]:
        print("⚠️  Модель не сослалась ни на один источник — ответ не подтверждён базой")
    print(f"({result['elapsed_sec']:.1f} с, промпт: {prompt})")


if __name__ == "__main__":
    main()
