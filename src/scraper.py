"""
Scraper — Arquivo Pessoa (arquivopessoa.net)
============================================
Estratégia: força bruta sobre IDs numéricos (ID_MIN..ID_MAX).
Cada página é aceite se:
  1. Tiver <div class="autor"> com um dos 3 heterónimos
  2. Tiver <div class="texto-poesia"> (é um poema, não prosa)
  3. O rodapé bibliográfico referenciar uma obra poética aceite

Produz:
  output/caeiro.txt
  output/campos.txt
  output/reis.txt
  output/corpus_sem_token.txt
  output/corpus_com_token.txt
  output/metadata.json

Uso:
  pip install requests beautifulsoup4
  python3 scraper.py
  python3 scraper.py --inicio 1400 --fim 1600
"""

import re
import json
import time
import os
import argparse
import requests
from bs4 import BeautifulSoup

# Configuração 
BASE_URL    = "http://arquivopessoa.net"
HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
ID_MIN      = 1
ID_MAX      = 5000
DELAY       = 0.1
DELAY_RETRY = 1.0
OUTPUT_DIR  = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Autores
AUTORES = {
    "Alberto Caeiro":   "CAEIRO",
    "Álvaro de Campos": "CAMPOS",
    "Ricardo Reis":     "REIS",
}

TOKEN_MAP = {
    "CAEIRO": "<CAEIRO>",
    "CAMPOS": "<CAMPOS>",
    "REIS":   "<REIS>",
}

OBRAS_ACEITES = {
    "CAEIRO": [
        "Guardador de Rebanhos",
        "Pastor Amoroso",
        "Poemas Inconjuntos",
        "Poemas de Alberto Caeiro",
        "Fragmentos",
    ],
    "CAMPOS": [
        "Poesias de Álvaro de Campos",
        "Poemas de Álvaro de Campos",
        "Livro de Versos",
    ],
    "REIS": [
        "Odes de Ricardo Reis",
        "Poemas de Ricardo Reis",
        "Odes",
    ],
}

# HTTP
def fetch(url: str):
    for tentativa in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code in (404, 403):
                return None
            r.raise_for_status()
            r.encoding = "utf-8"
            return BeautifulSoup(r.text, "html.parser")
        except requests.exceptions.Timeout:
            print(f"    [timeout] tentativa {tentativa + 1}")
            time.sleep(DELAY_RETRY)
        except Exception as e:
            print(f"    [erro] {e} — tentativa {tentativa + 1}")
            time.sleep(DELAY_RETRY)
    return None

# Extracção
def identificar_autor(soup):
    """Lê <div class='autor'> e devolve chave do autor ou None."""
    div = soup.find("div", class_="autor")
    if not div:
        return None
    nome = div.get_text(strip=True)
    return AUTORES.get(nome)


def obra_aceite(soup, chave):
    """Verifica referência bibliográfica no texto completo da página."""
    texto = soup.get_text(" ", strip=True)
    return any(t.lower() in texto.lower() for t in OBRAS_ACEITES[chave])


def extrair_titulo(soup):
    """Lê <h1 class='titulo-texto'>."""
    h1 = soup.find("h1", class_="titulo-texto")
    return h1.get_text(strip=True) if h1 else ""


def extrair_corpo(soup):
    """
    Lê <div class='texto-poesia'>.
    Cada <p> é um verso; <p> vazio é separador de estrofe.
    Devolve texto com estrofes separadas por linha em branco.
    """
    div = soup.find("div", class_="texto-poesia")
    if not div:
        return ""

    paragrafos = div.find_all("p")
    linhas = []
    for p in paragrafos:
        texto = p.get_text(strip=True)
        linhas.append(texto)   # string vazia para <p> vazio = linha em branco

    # Colapsa mais de 1 linha em branco consecutiva para exactamente 1
    corpo = "\n".join(linhas)
    corpo = re.sub(r"\n{2,}", "\n\n", corpo)

    # Remove o numeral romano isolado que o site repete no início
    # (ex: "I\n\nEu nunca guardei..." → "Eu nunca guardei...")
    corpo = re.sub(r"^\s*[IVXLCDM]+\s*\n+", "", corpo)

    return corpo.strip()

# Formatação
def formatar_bloco(poema, com_token):
    partes = []
    if com_token:
        partes.append(TOKEN_MAP[poema["autor"]])
    partes.append(poema["titulo"])
    partes.append("")
    partes.append(poema["corpo"])
    return "\n".join(partes)

# Escrita

def escrever_corpus(caminho, blocos):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(blocos))
        f.write("\n")
    kb = os.path.getsize(caminho) / 1024
    print(f"  -> {caminho}  ({kb:.1f} KB, {len(blocos)} poemas)")

# Main

def main(id_inicio, id_fim):
    poemas = []
    ignoradas = 0

    print(f"Scraping IDs {id_inicio}..{id_fim}")
    print("=" * 60)

    for id_pagina in range(id_inicio, id_fim + 1):
        url  = f"{BASE_URL}/textos/{id_pagina}"
        soup = fetch(url)
        if soup is None:
            continue

        # 1. Autor
        chave = identificar_autor(soup)
        if chave is None:
            ignoradas += 1
            continue

        # 2. Tem div.texto-poesia? (filtra prosa e páginas de índice)
        if not soup.find("div", class_="texto-poesia"):
            print(f"  [{id_pagina}] {chave} — sem texto-poesia, a ignorar")
            ignoradas += 1
            time.sleep(DELAY)
            continue

        # 3. Obra aceite?
        if not obra_aceite(soup, chave):
            print(f"  [{id_pagina}] {chave} — obra não aceite, a ignorar")
            ignoradas += 1
            time.sleep(DELAY)
            continue

        # 4. Extrair
        titulo = extrair_titulo(soup)
        corpo  = extrair_corpo(soup)

        if not corpo:
            print(f"  [{id_pagina}] {chave} — corpo vazio, a ignorar")
            ignoradas += 1
            time.sleep(DELAY)
            continue

        poema = {
            "id":       id_pagina,
            "autor":    chave,
            "titulo":   titulo,
            "corpo":    corpo,
            "url":      url,
            "n_chars":  len(corpo),
            "n_linhas": len(corpo.splitlines()),
        }
        poemas.append(poema)
        print(f"  [{id_pagina}] {chave} — {titulo[:55]}")

        time.sleep(DELAY)

    # Ficheiros individuais
    print("\n" + "=" * 60)
    print("A escrever ficheiros...")

    for chave in ("CAEIRO", "CAMPOS", "REIS"):
        subset = [p for p in poemas if p["autor"] == chave]
        if not subset:
            continue
        blocos = [formatar_bloco(p, com_token=False) for p in subset]
        escrever_corpus(os.path.join(OUTPUT_DIR, f"{chave.lower()}.txt"), blocos)

    blocos_sem = [formatar_bloco(p, com_token=False) for p in poemas]
    escrever_corpus(os.path.join(OUTPUT_DIR, "corpus_sem_token.txt"), blocos_sem)

    blocos_com = [formatar_bloco(p, com_token=True) for p in poemas]
    escrever_corpus(os.path.join(OUTPUT_DIR, "corpus_com_token.txt"), blocos_com)

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(poemas, f, ensure_ascii=False, indent=2)

    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    for chave in ("CAEIRO", "CAMPOS", "REIS"):
        subset = [p for p in poemas if p["autor"] == chave]
        chars  = sum(p["n_chars"] for p in subset)
        print(f"  {chave:8s}: {len(subset):4d} poemas   {chars/1024:6.1f} KB")
    total = sum(p["n_chars"] for p in poemas)
    print(f"  {'TOTAL':8s}: {len(poemas):4d} poemas   {total/1024:6.1f} KB")
    print(f"  Ignoradas: {ignoradas}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inicio", type=int, default=ID_MIN)
    parser.add_argument("--fim",    type=int, default=ID_MAX)
    args = parser.parse_args()
    main(args.inicio, args.fim)