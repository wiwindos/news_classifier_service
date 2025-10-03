#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOEX ISS → извлечь все тикеры (SECID) с TQBR и сохранить в txt (один тикер на строку).
Зависимости: только стандартная библиотека Python.
"""

import sys
import json
import urllib.request


DEFAULT_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
DEFAULT_OUT = "tickers.txt"


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def extract_secids(payload: dict) -> list[str]:
    """
    Универсальный разбор структуры ISS:
    ожидаем блок 'securities' с полями 'columns' и 'data'.
    """
    def from_block(block: dict) -> list[str]:
        cols = block.get("columns", [])
        data = block.get("data", [])
        if "SECID" not in cols:
            return []
        i = cols.index("SECID")
        out = []
        for row in data:
            if isinstance(row, list) and len(row) > i and row[i]:
                out.append(str(row[i]).strip())
        return out

    secids = []
    sec = payload.get("securities")
    if isinstance(sec, dict):
        secids.extend(from_block(sec))
    elif isinstance(sec, list):
        for b in sec:
            if isinstance(b, dict):
                secids.extend(from_block(b))

    # Уникализируем и сортируем для удобства
    return sorted(set(s for s in secids if s))


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    # Первичная попытка
    data = fetch_json(url)
    secids = extract_secids(data)

    # Fallback: пробуем "расширенный" формат, если вдруг SECID не найден
    if not secids:
        url2 = url + ("&" if "?" in url else "?") + "iss.json=extended&iss.meta=off"
        data2 = fetch_json(url2)
        secids = extract_secids(data2)

    if not secids:
        print("Не удалось найти тикеры (SECID) в ответе ISS MOEX.", file=sys.stderr)
        sys.exit(2)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(secids))

    print(f"Сохранено {len(secids)} тикеров в файл: {out_path}")


if __name__ == "__main__":
    main()
