"""Инварианты filter_types с параметрами-словами.

Тесты на КЛАСС: перебираются все 16 типов и все допустимые значения, а не
один удобный пример. Сеть и коллекция не нужны — инструмент читает types.json.
"""
import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import mcp_server

# Параметр -> (признак в данных, значение для true, значение для false)
PARAMS = {"energy": ("is_extravert", "extravert", "introvert"),
          "judgment": ("is_logic", "logic", "ethics"),
          "perception": ("is_intuitive", "intuitive", "sensing"),
          "lifestyle": ("is_rational", "rational", "irrational")}


def words_for(type_card: dict) -> dict:
    return {param: (yes if type_card[feature] else no)
            for param, (feature, yes, no) in PARAMS.items()}


@pytest.mark.parametrize("type_id", sorted(mcp_server._types()))
def test_four_words_find_exactly_that_type(type_id):
    """Любой тип находится по своим четырём словам — и только он."""
    card = mcp_server._types()[type_id]
    assert [t["id"] for t in mcp_server.filter_types(**words_for(card))] == [type_id]


@pytest.mark.parametrize("param", PARAMS)
def test_each_word_selects_its_half(param):
    """Каждое значение — ровно та половина типов, которую называет слово.

    Главное — значения «за false» (introvert, ethics, sensing, irrational):
    именно их модель путала, пока параметры были boolean.
    """
    feature, yes, no = PARAMS[param]
    for word, expected in ((yes, True), (no, False)):
        found = mcp_server.filter_types(**{param: word})
        assert len(found) == 8
        assert all(mcp_server._types()[t["id"]][feature] is expected for t in found)


@pytest.mark.parametrize("bad", ["", "не логик", "sensor", "true"])
def test_value_outside_enum_is_an_error_with_allowed_values(bad):
    """Значение вне списка -> ошибка с подсказкой, а не молча расширенный фильтр.

    Ollama схему не проверяет, поэтому проверяет сервер. Идём через call_tool,
    как настоящий клиент: проверка живёт в схеме, а не в теле функции.
    """
    with pytest.raises(ToolError) as error:
        asyncio.run(mcp_server.server.call_tool("filter_types", {"judgment": bad}))
    assert "'logic'" in str(error.value) and "'ethics'" in str(error.value)


def test_no_features_points_to_search():
    with pytest.raises(ToolError, match="search_types"):
        mcp_server.filter_types()
