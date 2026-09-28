"""Extract the ggml i-quant codebook tables into a host-includable C++ header.

The CUDA header declares them `static const __device__`, so a plain C++ TU cannot include
it. cpu_moe_ext.cpp already inlines iq3s_grid by hand; this generates the four remaining
grids plus the sign/mask tables verbatim so the CPU W4A8 kernels can index them.
"""

import re
import sys

SRC = "/models/desenvolvimento/freetoken-next/python/freetoken/kernel/csrc/gguf/ggml-common.h"
DST = "/models/desenvolvimento/freetoken-next/python/freetoken/kernel/csrc/cpu_moe/gguf_iquant_tables.h"

WANT = [
    ("uint64_t", "iq2xxs_grid", 256),
    ("uint64_t", "iq2xs_grid", 512),
    ("uint64_t", "iq2s_grid", 1024),
    ("uint32_t", "iq3xxs_grid", 256),
    ("uint8_t", "ksigns_iq2xs", 128),
]

src = open(SRC).read()
out = [
    "// Generated from kernel/csrc/gguf/ggml-common.h by scripts/gen_iquant_tables.py --",
    "// do not edit by hand. Host (non-__device__) copies of the ggml i-quant codebooks the",
    "// CPU MoE W4A8 kernels index. Values are verbatim; MIT, ggml authors.",
    "#ifndef FREETOKEN_CPU_MOE_GGUF_IQUANT_TABLES_H_",
    "#define FREETOKEN_CPU_MOE_GGUF_IQUANT_TABLES_H_",
    "",
    "#include <cstdint>",
    "",
]

for ctype, name, count in WANT:
    m = re.search(
        r"static const __device__ %s %s\[%d\] = \{(.*?)\};" % (ctype, name, count),
        src,
        re.S,
    )
    if not m:
        sys.exit(f"FAILED to find {name}[{count}]")
    body = m.group(1)
    vals = re.findall(r"0[xX][0-9a-fA-F]+[uUlL]*|\b\d+[uUlL]*", body)
    if len(vals) != count:
        sys.exit(f"{name}: parsed {len(vals)} values, expected {count}")
    ints = [int(v.rstrip("uUlL"), 0) for v in vals]
    # byte-max check: these feed VPDPBUSD as the UNSIGNED operand, so every byte of every
    # grid word must be a non-negative magnitude.
    bpe = {"uint64_t": 8, "uint32_t": 4, "uint8_t": 1}[ctype]
    mx = max((v >> (8 * b)) & 0xFF for v in ints for b in range(bpe)) if bpe > 1 else max(ints)
    print(f"{name:16s} {ctype:9s} n={count:5d} max_byte={mx:3d} unsigned_operand_ok={mx < 128}")
    out.append(f"static const {ctype} {name}[{count}] = {{")
    for i in range(0, count, 4):
        out.append("    " + ", ".join(vals[i : i + 4]) + ",")
    out.append("};")
    out.append("")

out += ["#endif  // FREETOKEN_CPU_MOE_GGUF_IQUANT_TABLES_H_", ""]
open(DST, "w").write("\n".join(out))
print(f"\nwrote {DST}: {len(out)} lines")
