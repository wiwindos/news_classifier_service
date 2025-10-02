#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplest logic per request + title deduplication + emoji cleanup in title:

- HEADLINE: concatenate message 'text' parts strictly BEFORE the first entity with type == "hashtag".
- If there is NO hashtag entity at all -> SKIP message (строго "до хэштега").
- If any TICKER from tickers.txt is found in HEADLINE -> write CSV row.
- Before saving: strip emojis/variation selectors/ZWJ from the TITLE ONLY.
- Deduplicate by FULL cleaned title (single-line). If a title already written, skip.

CSV columns (strict order): url,title,text,topic,tags,date
  url   = ""
  title = HEADLINE (emoji-free, single-line, trimmed)
  text  = full message text (flattened); newlines escaped as '\n' by default
  topic = "экономика"
  tags  = ""
  date  = message["date"]

Usage:
  python extract_issuer_titles.py --messages rdv_sample.json --tickers tickers.txt --out-csv titles.csv
  # optional:
  #   --sep ;
  #   --text-newlines keep|space|escape  (default: escape)
  #   --no-bom
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from typing import Any, Dict, List

# -----------------------
# Loaders
# -----------------------

def load_tickers(path: str) -> List[str]:
    """Read tickers; take only the first '|' column; return UPPERCASE unique list."""
    tickers: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tk = line.split("|", 1)[0].strip()
            if not tk:
                continue
            tk_u = tk.upper()
            if tk_u not in seen:
                seen.add(tk_u)
                tickers.append(tk_u)
    return tickers

def load_messages(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    msgs = data.get("messages") if isinstance(data, dict) else None
    if isinstance(msgs, list):
        return msgs
    return data if isinstance(data, list) else []

# -----------------------
# Text utils
# -----------------------

def flatten_text(msg_text: Any) -> str:
    """Flatten Telethon 'text' (string or entities list) to a plain string."""
    if isinstance(msg_text, str):
        return msg_text
    parts: List[str] = []
    if isinstance(msg_text, list):
        for t in msg_text:
            if isinstance(t, str):
                parts.append(t)
            elif isinstance(t, dict):
                parts.append(t.get("text", ""))
    return "".join(parts)

def headline_before_hashtag(msg_text: Any) -> str:
    """
    Concatenate ONLY the parts before the first entity with type == "hashtag".
    If there is no hashtag entity at all -> return "" (strict rule "до хэштега").
    """
    if isinstance(msg_text, list):
        out: List[str] = []
        seen_hashtag = False
        for t in msg_text:
            if isinstance(t, dict) and t.get("type") == "hashtag":
                seen_hashtag = True
                break
            if isinstance(t, dict):
                out.append(t.get("text", ""))
            elif isinstance(t, str):
                out.append(t)
        return "".join(out).strip() if seen_hashtag else ""
    # Plain string -> no hashtag boundary -> skip
    return ""

# -----------------------
# Emoji cleanup (TITLE ONLY)
# -----------------------
# Emoji ranges + variation selectors + ZWJ/ZWSP/LRM/RLM
EMOJI_TITLE_RE = re.compile(
    r"[\u2600-\u26FF\u2700-\u27BF\U0001F300-\U0001FAFF\uFE0E\uFE0F\u200D\u200B\u200C\u200E\u200F]"
)

def strip_emojis(s: str) -> str:
    """Remove emojis/variation selectors/ZWJ from title."""
    return EMOJI_TITLE_RE.sub("", s)

# -----------------------
# Ticker matching in headline
# -----------------------

def find_tickers_in_headline(headline: str, tickers: List[str]) -> List[str]:
    """
    Return list of tickers found in HEADLINE (case-insensitive).
    - For len>=3: word-boundary match.
    - For short tickers (len<3): require parentheses "(TK)" to reduce noise.
    """
    s_raw = headline
    s_low = s_raw.lower()
    found: List[str] = []
    seen = set()
    for tk in tickers:
        tk_low = tk.lower()
        ok = False
        if len(tk) >= 3:
            if re.search(rf"\b{re.escape(tk_low)}\b", s_low):
                ok = True
        else:
            if re.search(rf"\(\s*{re.escape(tk)}\s*\)", s_raw):
                ok = True
        if ok and tk not in seen:
            seen.add(tk)
            found.append(tk)
    return found

# -----------------------
# CSV helpers
# -----------------------

def normalize_title(title: str) -> str:
    """Make title a single line: collapse all whitespace to single spaces; trim."""
    return re.sub(r"\s+", " ", title).strip()

def normalize_text_for_csv(s: str, mode: str = "escape") -> str:
    """
    mode = 'escape' (default): replace newlines with literal '\\n' (1 row per record for simple viewers)
    mode = 'keep'            : keep real newlines (CSV quoting preserves them)
    mode = 'space'           : collapse all whitespace runs to single spaces
    """
    if mode == "keep":
        return s
    if mode == "space":
        return re.sub(r"\s+", " ", s).strip()
    # escape
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.replace("\n", "\\n")

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Extract CSV rows: url,title,text,topic,tags,date (headline up to hashtag, dedup by title, emoji-free title)")
    ap.add_argument("--messages", required=True, help="Path to Telethon export JSON")
    ap.add_argument("--tickers", required=True, help="Path to tickers.txt (first column = TICKER)")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--sep", choices=[",",";"], default=",", help="CSV delimiter (default: ,)")
    ap.add_argument("--text-newlines", choices=["escape","keep","space"], default="escape",
                    help="How to write newlines in 'text' column (default: escape)")
    ap.add_argument("--no-bom", action="store_true", help="Write CSV without BOM (default: with BOM for Excel)")
    args = ap.parse_args()

    # Safer console printing on Windows
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    tickers = load_tickers(args.tickers)
    if not tickers:
        raise SystemExit("tickers.txt is empty or invalid")

    messages = load_messages(args.messages)

    encoding = "utf-8" if args.no_bom else "utf-8-sig"  # BOM helps Excel on Windows
    rows_written = 0
    seen_titles: set[str] = set()  # dedup by FULL cleaned title

    with open(args.out_csv, "w", encoding=encoding, newline="") as f:
        writer = csv.writer(f, delimiter=args.sep, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(["url", "title", "text", "topic", "tags", "date"])

        for m in messages:
            if m.get("type") != "message":
                continue

            msg_text = m.get("text", "")
            headline_raw = headline_before_hashtag(msg_text)
            if not headline_raw:
                continue  # no hashtag or empty headline -> skip (strict rule)

            # Clean emojis ONLY in title, then collapse whitespace and trim
            headline_clean = normalize_title(strip_emojis(headline_raw))
            if not headline_clean:
                continue

            # Dedup check: FULL cleaned title
            if headline_clean in seen_titles:
                continue

            # Match tickers in the same cleaned headline
            matched = find_tickers_in_headline(headline_clean, tickers)
            if not matched:
                continue  # no ticker in HEADLINE -> skip

            # Prepare CSV fields
            url = ""
            full_text = flatten_text(msg_text) or ""
            text = normalize_text_for_csv(full_text, args.text_newlines)
            topic = "экономика"
            tags = ""
            date_iso = str(m.get("date") or "")

            # Write exactly ONE row per unique cleaned title
            writer.writerow([url, headline_clean, text, topic, tags, date_iso])
            seen_titles.add(headline_clean)
            rows_written += 1

    print(f"Saved {rows_written} unique-title rows to {args.out_csv}")

if __name__ == "__main__":
    main()
