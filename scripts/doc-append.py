#!/usr/bin/env python3
"""doc-append.py — Auto-atualização de documentação viva pelo modelo

Uso:
  ./scripts/doc-append.py --errors "MOE-005 | OOM ao carregar expert bank 12 | bank_idx overflow em get_expert | bounds check em expert_banks.py | validador em moe/offload_cache.py | moe/expert_banks.py"
  ./scripts/doc-append.py --lessons "Cache leak em replay MTP -> _prepare_batch skip_alloc=False -> skip_alloc=True em replay paths"
  ./scripts/doc-append.py --runbooks "8. Cache corruption diagnóstico" "Sintoma: VRAM ledger mismatch..." "Diagnóstico: ..." "Fix: ..."
  ./scripts/doc-append.py --experiments "EXP-049" "Tuning accept-rate k=2" "Setup: ..." "Resultado: ..." "Verdict: KEEP"
  ./scripts/doc-append.py --checkpoints "Ornith-35B" "MoE 35B" "8" "Q3_K" "GGUF" "/models/Ornith-35B-GGUF" "Phase 7 ready"
"""

import argparse
import sys
from datetime import date
from pathlib import Path

DOCS_DIR = Path("docs/dev")


def append_errors(entry: str):
    """Append to ERRORS.md - format: CÓDIGO | SINTOMA | CAUSA RAIZ | FIX | PREVENÇÃO | ARQUIVO"""
    path = DOCS_DIR / "ERRORS.md"
    content = path.read_text(encoding="utf-8")
    # Find the component section or append at end before EOF
    lines = content.split("\n")
    # Insert before last line (usually empty)
    insert_idx = len(lines) - 1
    while insert_idx > 0 and not lines[insert_idx].strip():
        insert_idx -= 1
    lines.insert(insert_idx + 1, f"| {entry} |")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Appended to ERRORS.md")


def append_lessons(entry: str):
    """Append to LESSONS.md - format: sintoma -> causa -> fix"""
    path = DOCS_DIR / "LESSONS.md"
    content = path.read_text(encoding="utf-8")
    today = date.today().isoformat()
    lines = content.split("\n")
    # Find the component section (## CUDA, ## MTP, etc.) or append at end
    # Simple: append at end before final newline
    insert_idx = len(lines) - 1
    while insert_idx > 0 and not lines[insert_idx].strip():
        insert_idx -= 1
    lines.insert(insert_idx + 1, f"- **{entry}** ({today})")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Appended to LESSONS.md")


def append_runbooks(title: str, symptom: str, diagnosis: str, fix: str):
    """Append to RUNBOOKS.md - new section"""
    path = DOCS_DIR / "RUNBOOKS.md"
    content = path.read_text(encoding="utf-8")
    today = date.today().isoformat()
    section = f"""
## {title}

**Adicionado:** {today} (auto)

### Sintoma
{symptom}

### Diagnóstico
{diagnosis}

### Fix
{fix}

---
"""
    # Append before last line
    lines = content.split("\n")
    insert_idx = len(lines) - 1
    while insert_idx > 0 and not lines[insert_idx].strip():
        insert_idx -= 1
    lines.insert(insert_idx + 1, section.rstrip())
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Appended to RUNBOOKS.md")


def append_experiments(exp_id: str, focus: str, setup: str, result: str, verdict: str):
    """Append to EXPERIMENTS.md - new row in table before '### TurboKV/QSA' section"""
    path = DOCS_DIR / "EXPERIMENTS.md"
    content = path.read_text(encoding="utf-8")
    row = f"| {exp_id} | {focus} | {setup} | {result} | {verdict} |"
    # Check if already exists
    if exp_id in content:
        print(f"⚠️ {exp_id} already exists in EXPERIMENTS.md, skipping")
        return
    lines = content.split("\n")
    # Find the "### TurboKV/QSA Integration" section header
    insert_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("### TurboKV/QSA"):
            insert_idx = i
            break
    if insert_idx == -1:
        print(f"⚠️ Could not find TurboKV/QSA section, skipping")
        return
    lines.insert(insert_idx, row)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Appended to EXPERIMENTS.md")


def append_checkpoints(
    model: str,
    arch: str,
    experts: str,
    active: str,
    quant: str,
    fmt: str,
    path_str: str,
    status: str,
):
    """Append to CHECKPOINTS.md - new row in table"""
    path = DOCS_DIR / "CHECKPOINTS.md"
    content = path.read_text(encoding="utf-8")
    row = f"| {model} | {arch} | {experts} | {active} | {quant} | {fmt} | {path_str} | {status} |"
    # Check if already exists
    if model in content:
        print(f"⚠️ {model} already exists in CHECKPOINTS.md, skipping")
        return
    lines = content.split("\n")
    # Find the main checkpoints table
    insert_idx = -1
    for i, line in enumerate(lines):
        if "| Modelo |" in line and "Arquitetura" in line:
            # Find next separator line
            for j in range(i + 1, len(lines)):
                if "|---" in lines[j]:
                    insert_idx = j
                    break
            break
    if insert_idx == -1:
        print(f"⚠️ Could not find checkpoints table, skipping")
        return
    lines.insert(insert_idx, row)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Appended to CHECKPOINTS.md")


def main():
    ap = argparse.ArgumentParser(description="Auto-append to living docs")
    ap.add_argument("--errors", help="Append to ERRORS.md (pipe-separated fields)")
    ap.add_argument("--lessons", help="Append to LESSONS.md (sintoma -> causa -> fix)")
    ap.add_argument(
        "--runbooks",
        nargs=4,
        metavar=("TITLE", "SYMPTOM", "DIAGNOSIS", "FIX"),
        help="Append to RUNBOOKS.md",
    )
    ap.add_argument(
        "--experiments",
        nargs=5,
        metavar=("ID", "FOCUS", "SETUP", "RESULT", "VERDICT"),
        help="Append to EXPERIMENTS.md",
    )
    ap.add_argument(
        "--checkpoints",
        nargs=8,
        metavar=("MODEL", "ARCH", "EXPERTS", "ACTIVE", "QUANT", "FMT", "PATH", "STATUS"),
        help="Append to CHECKPOINTS.md",
    )
    args = ap.parse_args()

    if args.errors:
        append_errors(args.errors)
    elif args.lessons:
        append_lessons(args.lessons)
    elif args.runbooks:
        append_runbooks(*args.runbooks)
    elif args.experiments:
        append_experiments(*args.experiments)
    elif args.checkpoints:
        append_checkpoints(*args.checkpoints)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
