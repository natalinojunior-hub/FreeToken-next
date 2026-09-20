#!/usr/bin/env python3
"""validate-crossref.py — Valida referências cruzadas entre docs/dev/*.md e root/*.md"""

import re
import sys
from pathlib import Path

DOCS_DIR = Path("docs/dev")
ROOT_DIR = Path(".")
MD_FILES = list(DOCS_DIR.glob("*.md")) + list(ROOT_DIR.glob("*.md"))

# Padrões de referência
REF_PATTERN = re.compile(r"`([A-Z][A-Z0-9_-]*\.md)`")
FILE_REF_PATTERN = re.compile(r"([A-Z][A-Z0-9_-]*\.md)")


def extract_refs(content):
    """Extrai todas as referências a arquivos .md no conteúdo"""
    refs = set()
    # Backtick references
    refs.update(REF_PATTERN.findall(content))
    # Bare references (heuristic: uppercase start, ends with .md)
    for match in FILE_REF_PATTERN.finditer(content):
        fname = match.group(1)
        if fname[0].isupper() and fname.endswith(".md"):
            refs.add(fname)
    return refs


def main():
    all_files = {f.name for f in MD_FILES}
    missing = []

    for md_file in MD_FILES:
        content = md_file.read_text(encoding="utf-8")
        refs = extract_refs(content)

        for ref in refs:
            if ref not in all_files:
                missing.append(f"{md_file.name} -> {ref} (não existe)")

    if missing:
        print("❌ Referências quebradas encontradas:")
        for m in missing:
            print(f"  {m}")
        return 1

    print(f"✅ Todas as {len(all_files)} referências cruzadas válidas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
