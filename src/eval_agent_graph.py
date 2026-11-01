"""Замер: свой агентный цикл (agent.py) против того же цикла на LangGraph (agent_graph.py).

Запуск:  .venv/bin/python src/eval_agent_graph.py

Честное сравнение — меняется ОДНА вещь, оркестратор. Одинаковые: MCP-сервер,
промпт, модель (qwen2.5-7b, temperature 0), лимит шагов, repair, 6 контрольных
вопросов. Если граф перенёс логику без искажений, при temperature 0 траектории
(какие инструменты с какими аргументами) должны совпасть, а с ними и итог.

Критерии зачёта — по смыслу вопроса, проверяются по тексту ответа и трассе:
  1 Цезарь                         — назван СЭЭ / Наполеон
  2 интроверт-логик-сенсорик-иррац — назван СЛИ / Габен
  3 ENTP против «силового давления» — запрошена карточка ENTP (план полный) И назван
                                      Робеспьер или Достоевский (болевая ЧС)
  4 Илон Маск                      — тип НЕ назван (типирования людей в базе нет)
  5 ISTK                           — предложены ISTJ и ISTP
  6 уют и без конфликтов           — назван Дюма или Гюго
"""
import time

import agent
from agent import CONTROL_QUESTIONS, connect, run_agent
from agent_graph import run_graph
from eval_people import named_types

ENTP_NAMES = ("entp", "дон кихот", "илэ")


def passed(index: int, result: dict) -> bool:
    answer = result["answer"] or ""
    named = named_types(answer)
    if index == 0:
        return "see" in named
    if index == 1:
        return "sli" in named
    if index == 2:
        asked_entp = any(t["tool"] == "get_type" and str(t["args"].get("name", "")).lower() in ENTP_NAMES
                         for t in result["trace"])
        return asked_entp and bool(named & {"lii", "eii"})
    if index == 3:
        return result["answer"] is not None and not named
    if index == 4:
        return "ISTJ" in answer and "ISTP" in answer
    return bool(named & {"sei", "ese"})


def path(result: dict) -> list[tuple]:
    return [(t["tool"], tuple(sorted(t["args"].items()))) for t in result["trace"]]


def main() -> None:
    calls = {"n": 0}
    original = agent.llm

    def counted(messages, tools):
        calls["n"] += 1
        return original(messages, tools)

    agent.llm = counted
    client, tools = connect()
    try:
        runners = {"свой цикл": lambda q, i: run_agent(q, client, tools),
                   "LangGraph": lambda q, i: run_graph(q, client, tools, thread_id=f"eval-{i}")}
        results = {}
        for name, run in runners.items():
            calls["n"], start = 0, time.perf_counter()
            results[name] = [run(q, i) for i, q in enumerate(CONTROL_QUESTIONS)]
            results[name + ":stats"] = (calls["n"], time.perf_counter() - start)

        own, graph = results["свой цикл"], results["LangGraph"]
        print(f"{'#':>2} {'вопрос':<46} {'свой':>5} {'граф':>5} {'путь':>6}")
        for i, q in enumerate(CONTROL_QUESTIONS):
            same = "=" if path(own[i]) == path(graph[i]) and own[i]["answer"] == graph[i]["answer"] else "≠"
            print(f"{i + 1:>2} {q[:46]:<46} {'✓' if passed(i, own[i]) else '✗':>5} "
                  f"{'✓' if passed(i, graph[i]) else '✗':>5} {same:>6}")
            if same == "≠":
                print(f"     свой: {path(own[i])}\n     граф: {path(graph[i])}")
        for name in runners:
            score = sum(passed(i, r) for i, r in enumerate(results[name]))
            n_calls, seconds = results[name + ":stats"]
            print(f"{name}: {score}/6, вызовов модели {n_calls}, {seconds:.0f} с")
        print("\nОтветы:")
        for i, q in enumerate(CONTROL_QUESTIONS):
            print(f"{i + 1}. {q}\n   свой: {(own[i]['answer'] or own[i].get('stopped'))[:200]}\n"
                  f"   граф: {(graph[i]['answer'] or graph[i].get('stopped'))[:200]}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
