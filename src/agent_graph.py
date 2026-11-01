"""Тот же агентный цикл, что в agent.py, но графом LangGraph.

Запуск:  .venv/bin/python src/agent_graph.py                         # контрольные вопросы
         .venv/bin/python src/agent_graph.py "кто такой Цезарь?"     # один вопрос
         .venv/bin/python src/agent_graph.py --approve "..."         # пауза перед каждым инструментом
         .venv/bin/python src/agent_graph.py --crash-at 2 --thread t1 "..."  # «падение» на шаге 2
         .venv/bin/python src/agent_graph.py --resume --thread t1             # продолжить t1

Логика шага не изменилась: модель, инструменты MCP-сервера, лимит шагов, repair
вызова, напечатанного текстом. Меняется только то, КТО ведёт цикл:

    agent.py:        for step in range(MAX_STEPS): ...          — порядок в коде
    agent_graph.py:  узлы model / tools / limit + условные рёбра  — порядок в графе

Что граф даёт сверх этого (и чего у agent.py нет):
  - checkpoint: после каждого узла состояние пишется в SQLite. Процесс упал —
    тот же thread_id продолжает с последнего узла, прошлые вызовы модели не
    повторяются;
  - interrupt: узел tools может остановить граф и ждать решения человека; ответ
    приходит хоть через неделю, потому что состояние уже на диске.

Чего граф НЕ даёт: шаги внутри узлов те же синхронные функции. Граф решает,
какой шаг следующий, а не как ждать сети внутри шага (см. asyncio_graph_demo.py).
"""
import argparse
import json
import operator
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import agent
from agent import (CONTROL_QUESTIONS, GIVE_UP_ANSWER, MAX_REPAIRS, MAX_STEPS,
                   REPAIR_MESSAGE, SYSTEM, call_tool, connect, looks_like_leaked_call)

CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "demo_state" / "graph_checkpoints.sqlite"


class AgentState(TypedDict):
    # Reducer operator.add: узел возвращает только НОВЫЕ сообщения, граф дописывает их
    # в конец. Без reducer'а значение поля заменялось бы целиком.
    messages: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]
    step: int
    repairs: int
    answer: str | None
    stopped: str | None


def build_graph(client, tools: list[dict], *, approve: bool = False, crash_at: int | None = None):
    """Граф поверх уже подключённого MCP-клиента. Узлы — замыкания над client/tools."""

    def model(state: AgentState) -> dict:
        step = state["step"] + 1
        if crash_at is not None and step == crash_at:
            raise RuntimeError(f"имитация падения процесса перед шагом {step}")
        message = agent.llm(state["messages"], tools)
        if message.get("tool_calls"):
            return {"messages": [message], "step": step}

        text = (message.get("content") or "").strip()
        if not looks_like_leaked_call(text):
            return {"messages": [message], "step": step, "answer": text}

        # Вызов напечатан текстом: не исполняем (текст — не команда), даём переделать.
        broken = {"step": step, "tool": "(битый вызов)", "args": {}, "error": True, "result": text[:200]}
        if state["repairs"] >= MAX_REPAIRS:
            return {"messages": [message], "step": step, "trace": [broken],
                    "answer": GIVE_UP_ANSWER, "stopped": "malformed_tool_call"}
        return {"messages": [message, {"role": "user", "content": REPAIR_MESSAGE}],
                "step": step, "trace": [broken], "repairs": state["repairs"] + 1}

    def tools_node(state: AgentState) -> dict:
        calls = state["messages"][-1]["tool_calls"]
        if approve:
            # interrupt() ДО любого вызова и один на весь узел: после resume узел
            # выполняется заново с начала. Спроси мы разрешение между вызовами,
            # уже сделанные вызовы повторились бы — для платного инструмента это
            # двойная оплата.
            decision = interrupt({"calls": [{"tool": c["function"]["name"],
                                             "args": c["function"].get("arguments")} for c in calls]})
            if decision != "yes":
                refusal = "Пользователь не разрешил вызов инструмента. Ответь без него или объясни, что нужно."
                return {"messages": [{"role": "tool", "tool_name": c["function"]["name"], "content": refusal}
                                     for c in calls]}
        messages, trace = [], []
        for call in calls:
            fn = call["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):  # как в agent.py: OpenAI-совместимые API отдают строку JSON
                args = json.loads(args)
            result, is_error = call_tool(client, fn["name"], args)
            trace.append({"step": state["step"], "tool": fn["name"], "args": args,
                          "error": is_error, "result": result})
            messages.append({"role": "tool", "tool_name": fn["name"], "content": result})
        return {"messages": messages, "trace": trace}

    def limit(state: AgentState) -> dict:
        return {"answer": None, "stopped": "max_steps"}

    def after_model(state: AgentState) -> str:
        if state["answer"] is not None or state["stopped"]:
            return END
        if state["messages"][-1].get("tool_calls"):
            return "tools"
        return "model" if state["step"] < MAX_STEPS else "limit"  # после repair

    def after_tools(state: AgentState) -> str:
        return "model" if state["step"] < MAX_STEPS else "limit"

    graph = StateGraph(AgentState)
    graph.add_node("model", model)
    graph.add_node("tools", tools_node)
    graph.add_node("limit", limit)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", "model", "limit", END])
    graph.add_conditional_edges("tools", after_tools, ["model", "limit"])
    graph.add_edge("limit", END)
    return graph


def initial_state(question: str) -> AgentState:
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
            "trace": [], "step": 0, "repairs": 0, "answer": None, "stopped": None}


def to_result(state: dict) -> dict:
    """Итог графа в формате agent.run_agent — чтобы сравнивать и печатать одинаково."""
    result = {"answer": state["answer"], "trace": state["trace"], "steps": state["step"]}
    if state.get("stopped"):
        result["stopped"] = state["stopped"]
    return result


def run_graph(question: str, client, tools: list[dict], *, checkpointer=None, thread_id: str = "eval") -> dict:
    """Один вопрос через граф без пауз — для замера рядом с agent.run_agent."""
    app = build_graph(client, tools).compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return to_result(app.invoke(initial_state(question), config))


def _checkpointer() -> SqliteSaver:
    CHECKPOINT_DB.parent.mkdir(exist_ok=True)
    # check_same_thread=False: граф может вызывать сохранение не из главного потока.
    return SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--approve", action="store_true", help="пауза перед каждым вызовом инструмента")
    parser.add_argument("--crash-at", type=int, help="имитировать падение перед этим шагом модели")
    parser.add_argument("--resume", action="store_true", help="продолжить поток --thread с последнего checkpoint")
    parser.add_argument("--thread", default="cli")
    args = parser.parse_args()

    calls = {"n": 0}
    original_llm = agent.llm

    def counted_llm(messages, tools):
        calls["n"] += 1
        return original_llm(messages, tools)

    agent.llm = counted_llm  # счётчик вызовов модели — видно, что resume их не повторяет

    client, tools = connect()
    try:
        app = build_graph(client, tools, approve=args.approve, crash_at=args.crash_at).compile(
            checkpointer=_checkpointer())
        config = {"configurable": {"thread_id": args.thread}}

        if args.resume:
            saved = app.get_state(config)
            print(f"продолжаю поток {args.thread}: следующий узел {saved.next}, шагов уже сделано {saved.values['step']}")
            pending = None  # None = «продолжи с того места, где остановился»
        else:
            questions = [" ".join(args.question)] if args.question else CONTROL_QUESTIONS
            if len(questions) > 1 and (args.approve or args.crash_at):
                sys.exit("--approve и --crash-at — для одного вопроса")
            if len(questions) > 1:
                for i, q in enumerate(questions):
                    agent.show(q, to_result(app.invoke(initial_state(q), {"configurable": {"thread_id": f"{args.thread}-{i}"}})))
                print(f"\nвызовов модели: {calls['n']}")
                return
            pending = initial_state(questions[0])

        try:
            state = app.invoke(pending, config)
            while "__interrupt__" in state:  # граф остановился и ждёт человека
                for item in state["__interrupt__"]:
                    print("Агент хочет вызвать:", item.value["calls"])
                decision = input("Разрешить? [yes/no]: ").strip()
                state = app.invoke(Command(resume=decision), config)
        except RuntimeError as exc:
            print(f"ПАДЕНИЕ: {exc}. Состояние сохранено; продолжить: --resume --thread {args.thread}")
            print(f"вызовов модели до падения: {calls['n']}")
            return
        agent.show(app.get_state(config).values["messages"][1]["content"], to_result(state))
        print(f"вызовов модели в этом запуске: {calls['n']}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
