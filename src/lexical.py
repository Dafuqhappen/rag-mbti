"""Лексический поиск (BM25) — вторая половина гибридного ретривала.

Зачем он нужен рядом с векторным поиском: эмбеддинг ищет по смыслу и потому
плохо ловит редкие точные токены — имена, коды типов, аббревиатуры. Запрос
"Цезарь" семантически близок ко многим текстам про власть, но нам нужен
документ, где это слово стоит буквально. BM25 работает ровно наоборот:
он ценит редкое точное совпадение. Отсюда идея гибрида — сложить оба сигнала.
"""
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

# Грубый стеммер: обрезаем токен до N символов. Русский словоизменяет окончания
# ("Габена" -> "габен", "Максима Горького" -> "макси горьк"), и такой обрез
# склеивает словоформы без словаря и без внешних зависимостей.
STEM_LEN = 5

K1 = 1.5   # насыщение частоты термина: выше -> повторы слова важнее
B = 0.75   # нормализация по длине документа: 0 -> длину игнорируем, 1 -> учитываем полностью


def tokenize(text: str) -> list[str]:
    """Текст -> список нормализованных, обрезанных до основы токенов."""
    return [t.lower()[:STEM_LEN] for t in TOKEN_RE.findall(text)]


class BM25Index:
    """Классический BM25 поверх готовых документов.

    Хранит частоты термов и IDF. IDF — ключевая часть: слово, встречающееся
    во всех документах ("болевая", "функции"), получает вес около нуля,
    а редкое ("цезарь", "бета") — высокий. Именно это и различает документы.
    """

    def __init__(self, ids: list[str], documents: list[str]):
        self.ids = ids
        self.docs = [tokenize(d) for d in documents]
        self.freqs = [Counter(d) for d in self.docs]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.docs)

        n_docs = len(self.docs)
        doc_freq: Counter[str] = Counter()
        for tokens in self.docs:
            doc_freq.update(set(tokens))
        self.idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score(self, query: str) -> dict[str, float]:
        """Считает BM25-скор каждого документа для запроса."""
        q_tokens = tokenize(query)
        scores: dict[str, float] = {}
        for i, doc_id in enumerate(self.ids):
            freq, length = self.freqs[i], self.lengths[i]
            total = 0.0
            for term in q_tokens:
                if term not in freq:
                    continue
                tf = freq[term]
                norm = tf + K1 * (1 - B + B * length / self.avg_len)
                total += self.idf.get(term, 0.0) * tf * (K1 + 1) / norm
            scores[doc_id] = total
        return scores

    def top(self, query: str, k: int) -> list[str]:
        """Возвращает id документов, отсортированные по убыванию скора."""
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [doc_id for doc_id, score in ranked[:k] if score > 0]
