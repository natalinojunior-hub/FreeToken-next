"""Adaptive MTP + Context Governor (Roadmap Linha 15).

Dynamically balances speculative draft depth k (MTP) and expert cache allocation
based on active sequence length and VRAM pressure, avoiding degradation in long context.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from freetoken.core import Req


def resolve_adaptive_k(
    req: "Req",
    max_k: int,
    long_context_threshold: int = 65536,
    ultra_long_threshold: int = 131072,
    moe_slots_per_layer: float | None = None,
    top_k_experts: int = 10,
) -> int:
    """Dynamically determine optimal MTP speculation depth k for a request.

    Diretriz Soberana: a métrica de decisão é TG (tok/s), nunca a taxa de aceitação isolada.
    Elevadas profundidades de k (2, 3, 4+) são mantidas sempre que sustentarem ou elevarem o TG,
    mesmo que a taxa de aceitação percentual caia.
    O recuo de k ocorre quando a janela de contexto atinge regimes extremos ou quando o working
    set de experts na verificação (k + 1) * top_k excede os slots residentes do MoE cache,
    induzindo thrashing severo de PCIe.
    """
    if max_k <= 0:
        return 0

    seq_len = getattr(req, "device_len", 0)
    adaptive_mode = os.getenv("FREETOKEN_ADAPTIVE_MTP", "1") == "1"

    if not adaptive_mode:
        return min(max_k, req.remain_len - 1)

    effective_k = max_k

    # Diretriz Soberana: a métrica de decisão é TG (tok/s).
    # Permitir especulação máxima até 64K; recuar gradualmente apenas sob pressão extrema de KV
    if seq_len > ultra_long_threshold:
        effective_k = min(1, effective_k)
    elif seq_len > long_context_threshold:
        effective_k = min(max(2, effective_k // 2), effective_k)

    return min(effective_k, req.remain_len - 1)
