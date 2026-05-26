"""
Scraper - Arquivo Pessoa (arquivopessoa.net)
============================================
Strategy: brute force over numeric IDs (ID_MIN..ID_MAX).

Filters (all optional, combined with AND logic):
  --authors     List of author keys to accept (e.g. CAEIRO CAMPOS)
  --works       List of work terms to accept (e.g. "Mensagem" "Odes")
  --type        Text type: "verso", "prosa", or "ambos" (default: verso)
  --language    List of language codes to accept (e.g. pt en fr)

Examples:
  # Original behaviour — 3 heteronyms, verse only, Portuguese only
  python3 scraper.py --authors CAEIRO CAMPOS REIS --type verso --language pt

  # All authors, verse only, Portuguese only
  python3 scraper.py --type verso --language pt

  # Campos in English and Portuguese
  python3 scraper.py --authors CAMPOS --language pt en

  # Everything, no filters
  python3 scraper.py --type ambos

  # Test on a small range
  python3 scraper.py --start 1440 --end 1510 --authors CAEIRO --language pt

Dependencies:
  pip install requests beautifulsoup4 lingua-language-detector
"""

import re
import json
import time
import os
import argparse
import requests
from bs4 import BeautifulSoup

# Language Detection

try:
    from lingua import Language, LanguageDetectorBuilder

    _SUPPORTED_LANGUAGES = [
        Language.PORTUGUESE,
        Language.ENGLISH,
        Language.FRENCH,
        Language.SPANISH,
        Language.LATIN,
    ]
    _DETECTOR = LanguageDetectorBuilder.from_languages(*_SUPPORTED_LANGUAGES).build()
    _LANGUAGE_TO_CODE = {
        Language.PORTUGUESE: "pt",
        Language.ENGLISH:    "en",
        Language.FRENCH:     "fr",
        Language.SPANISH:    "es",
        Language.LATIN:      "la",
    }
    LANGUAGE_AVAILABLE = True
except ImportError:
    LANGUAGE_AVAILABLE = False


# Configuration
BASE_URL    = "http://arquivopessoa.net"
HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
ID_MIN      = 1
ID_MAX      = 4600
DELAY       = 0.5
DELAY_RETRY = 0.5
OUTPUT_DIR  = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Known authors
KNOWN_AUTHORS = {
    "Alberto Caeiro":   "CAEIRO",
    "Álvaro de Campos": "CAMPOS",
    "Ricardo Reis":     "REIS",
    "Fernando Pessoa":  "PESSOA",
}

# HTTP
def fetch(url: str):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code in (404, 403):
                return None
            r.raise_for_status()
            r.encoding = "utf-8"
            return BeautifulSoup(r.text, "html.parser")
        except requests.exceptions.Timeout:
            print(f"    [timeout] attempt {attempt + 1}")
            time.sleep(DELAY_RETRY)
        except Exception as e:
            print(f"    [error] {e} — attempt {attempt + 1}")
            time.sleep(DELAY_RETRY)
    return None


# Language detection
def detect_language(text: str) -> str | None:
    """
    Returns the ISO code of the detected language ('pt', 'en', 'fr', 'es', 'la').
    Returns None if the library is unavailable or detection fails.
    """
    if not LANGUAGE_AVAILABLE:
        return None
    language = _DETECTOR.detect_language_of(text)
    if language is None:
        return None
    return _LANGUAGE_TO_CODE.get(language)

def language_accepted(text: str, language_filter: set) -> bool:
    """
    Returns True if the text is in the desired language.
    In case of doubt (failed detection), always accepts.
    """
    language = detect_language(text)
    if language is None:
        return True   # doubt → accept
    return language in language_filter


# Extraction
def identify_author(soup):
    div = soup.find("div", class_="autor")
    if not div:
        return None, None
    name = div.get_text(strip=True)
    key = KNOWN_AUTHORS.get(name, name.upper().replace(" ", "_"))
    return name, key

def extract_title(soup):
    h1 = soup.find("h1", class_="titulo-texto")
    return h1.get_text(strip=True) if h1 else ""

def extract_verse_body(soup):
    div = soup.find("div", class_="texto-poesia")
    if not div:
        return ""
    lines = [p.get_text(strip=True) for p in div.find_all("p")]
    body = "\n".join(lines)
    body = re.sub(r"\n{2,}", "\n\n", body)
    body = re.sub(r"^\s*[IVXLCDM]+\s*\n+", "", body)
    return body.strip()

def extract_prose_body(soup):
    div = soup.find("div", class_="texto-prosa")
    if not div:
        return ""
    body = div.get_text("\n", strip=True)
    body = re.sub(r"\n{2,}", "\n\n", body)
    return body.strip()

def detect_type(soup):
    if soup.find("div", class_="texto-poesia"):
        return "verso"
    if soup.find("div", class_="texto-prosa"):
        return "prosa"
    return None

def work_referenced(soup, works_filter):
    text = soup.get_text(" ", strip=True)
    return any(w.lower() in text.lower() for w in works_filter)


# Formatting and writing
def format_block(poem, token_map):
    parts = []
    token = token_map.get(poem["autor"]) if token_map else None
    if token:
        parts.append(token)
    parts.append(poem["titulo"])
    parts.append("")
    parts.append(poem["corpo"])
    return "\n".join(parts)

def write_corpus(path, blocks):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(blocks))
        f.write("\n")
    kb = os.path.getsize(path) / 1024
    print(f"  -> {path}  ({kb:.1f} KB, {len(blocks)} texts)")


# Main
def main(args):
    author_filter   = {a.upper() for a in args.authors} if args.authors else None
    works_filter    = args.works if args.works else None
    type_filter     = args.type
    language_filter = set(args.language) if args.language else None

    if language_filter and not LANGUAGE_AVAILABLE:
        print("WARNING: --language was specified but 'lingua-language-detector' is not installed.")
        print("         Install with: pip install lingua-language-detector")
        print("         Ignoring language filter.\n")
        language_filter = None

    token_map = {}
    poems     = []
    ignored   = 0

    print(f"Scraping IDs {args.start}..{args.end}")
    if author_filter:
        print(f"  Author filter   : {', '.join(sorted(author_filter))}")
    if works_filter:
        print(f"  Works filter    : {', '.join(works_filter)}")
    print(f"  Type filter     : {type_filter}")
    if language_filter:
        print(f"  Language filter : {', '.join(sorted(language_filter))}")
    print("=" * 60)

    for page_id in range(args.start, args.end + 1):
        url  = f"{BASE_URL}/textos/{page_id}"
        soup = fetch(url)
        if soup is None:
            continue

        # Filter 1: Author
        author_name, key = identify_author(soup)
        if key is None:
            ignored += 1
            continue

        if author_filter and key not in author_filter:
            ignored += 1
            time.sleep(DELAY)
            continue

        if key not in token_map:
            token_map[key] = f"<{key}>"

        # Filter 2: Type
        text_type = detect_type(soup)
        if text_type is None:
            ignored += 1
            continue

        if type_filter == "verso" and text_type != "verso":
            ignored += 1
            time.sleep(DELAY)
            continue

        if type_filter == "prosa" and text_type != "prosa":
            ignored += 1
            time.sleep(DELAY)
            continue

        # Filter 3: Work
        if works_filter and not work_referenced(soup, works_filter):
            print(f"  [{page_id}] {key} — work not accepted, skipping")
            ignored += 1
            time.sleep(DELAY)
            continue

        # Extraction
        title = extract_title(soup)
        body  = extract_verse_body(soup) if text_type == "verso" else extract_prose_body(soup)

        if not body:
            print(f"  [{page_id}] {key} — empty body, skipping")
            ignored += 1
            time.sleep(DELAY)
            continue

        # Filter 4: Language
        if language_filter and not language_accepted(body, language_filter):
            print(f"  [{page_id}] {key} — language not accepted ({detect_language(body)}), skipping")
            ignored += 1
            time.sleep(DELAY)
            continue

        detected_language = detect_language(body) if LANGUAGE_AVAILABLE else None

        poem = {
            "id":         page_id,
            "autor":      key,
            "nome_autor": author_name,
            "tipo":       text_type,
            "lingua":     detected_language,
            "titulo":     title,
            "corpo":      body,
            "url":        url,
            "n_chars":    len(body),
            "n_linhas":   len(body.splitlines()),
        }
        poems.append(poem)
        language_label = f" [{detected_language}]" if detected_language else ""
        print(f"  [{page_id}] {key}{language_label} — {title[:50]}")

        time.sleep(DELAY)

    # Individual files per author
    print("\n" + "=" * 60)
    print("Writing files...")

    found_authors = sorted({p["autor"] for p in poems})

    for key in found_authors:
        subset = [p for p in poems if p["autor"] == key]
        blocks = [format_block(p, {}) for p in subset]
        write_corpus(os.path.join(OUTPUT_DIR, f"{key.lower()}.txt"), blocks)

    blocks_without = [format_block(p, {}) for p in poems]
    write_corpus(os.path.join(OUTPUT_DIR, "corpus_sem_token.txt"), blocks_without)

    blocks_with = [format_block(p, token_map) for p in poems]
    write_corpus(os.path.join(OUTPUT_DIR, "corpus_com_token.txt"), blocks_with)

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(poems, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key in found_authors:
        subset = [p for p in poems if p["autor"] == key]
        chars  = sum(p["n_chars"] for p in subset)
        print(f"  {key:12s}: {len(subset):4d} texts   {chars/1024:6.1f} KB")
    total = sum(p["n_chars"] for p in poems)
    print(f"  {'TOTAL':12s}: {len(poems):4d} texts   {total/1024:6.1f} KB")
    print(f"  Ignored: {ignored}")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arquivo Pessoa scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start",    type=int, default=ID_MIN,
                        help=f"Start ID (default: {ID_MIN})")
    parser.add_argument("--end",      type=int, default=ID_MAX,
                        help=f"End ID (default: {ID_MAX})")
    parser.add_argument("--authors",  nargs="+", default=None,
                        help="Author keys to accept (e.g. CAEIRO CAMPOS REIS PESSOA).")
    parser.add_argument("--works",    nargs="+", default=None,
                        help='Work terms to accept (e.g. "Mensagem" "Odes").')
    parser.add_argument("--type",     default="verso",
                        choices=["verso", "prosa", "ambos"],
                        help="Text type to accept (default: verso)")
    parser.add_argument("--language", nargs="+", default=None,
                        metavar="CODE",
                        help="ISO language codes to accept (e.g. pt en fr). "
                             "Requires: pip install lingua-language-detector")
    args = parser.parse_args()
    main(args)
