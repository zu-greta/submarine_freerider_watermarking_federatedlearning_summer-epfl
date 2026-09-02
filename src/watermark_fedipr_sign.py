"""FedIPR feature-based (WHITE-BOX) sign watermark -- embedded across one or MORE
normalization layers (server-decided), read white-box from the weights.

================================================================================
 FedIPR hides a bit-string in the SIGNS of normalization scale weights W_gamma and
 reads it white-box. 
 Each chosen layer i carries its own bit-slice B_i via its own secret matrix E_i:
        embed :  minimise  sum_i  mean_j max( margin - b'_{i,j} * (gamma_i . E_i)_j , 0 )
        read  :  B_hat_i = sign(gamma_i . E_i)  ;  BER = total Hamming / total bits.

 SERVER CHOOSES HOW MANY LAYERS (config.fedipr_sign_carrier / fedipr_sign_layers):
   * fedipr_sign_layers = 1  (+ carrier "auto_last_bn")  -> ONLY the output-layer scale
     (net.layer4.1.bn2.weight, inside head2). 
   * fedipr_sign_layers = N (>1)  -> the N output-most normalization scales. The extra
     carriers live in the BODY, OUTSIDE the head2 scope. 
   * carrier = "all_bn"  -> every normalization scale (full-depth FedIPR, most robust).
   * carrier = "name1,name2,..."  -> an explicit list (server forces exact locations).
================================================================================
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Carrier -- which normalization scales carry the mark.
# ---------------------------------------------------------------------------
def list_bn_scale_names(model) -> list:
    """All 1-D normalization scale weights (BN/norm/downsample-BN), in MODEL order
    (input -> output). The last entry is the output-block scale."""
    names = []
    for n, p in model.named_parameters():
        if p.ndim == 1 and n.endswith("weight") and (
                "bn" in n or "norm" in n or "downsample.1" in n):
            names.append(n)
    if not names:                                     # fallback: any 1-D '*.weight'
        names = [n for n, p in model.named_parameters()
                 if p.ndim == 1 and n.endswith("weight")]
    return names


def resolve_carrier_names(model, carrier: str = "auto_last_bn", n_layers: int = 1) -> list:
    """Return the ordered list of carrier param names (output-most first).

    carrier="auto_last_bn" -> the last `n_layers` normalization scales (from the output
                              backward). n_layers=1 == the single output-layer scale.
    carrier="all_bn"       -> every normalization scale (deepest robustness).
    carrier="a,b,c"        -> exactly these parameter names (server-forced locations).
    """
    named = dict(model.named_parameters())
    if carrier and carrier not in ("auto_last_bn", "all_bn"):
        want = [c.strip() for c in carrier.split(",") if c.strip()]
        for c in want:
            if c not in named:
                raise ValueError(
                    f"fedipr_sign_carrier '{c}' is not a model parameter. 1-D scale "
                    f"weights: {list_bn_scale_names(model)}")
        return want
    bn_out_first = list(reversed(list_bn_scale_names(model)))   # output-most first
    if carrier == "all_bn":
        return bn_out_first
    n = max(1, int(n_layers))
    return bn_out_first[:n]


def per_carrier_channels(model, names) -> list:
    d = dict(model.named_parameters())
    return [int(d[n].numel()) for n in names]


def plan_bits(channels, bits_per_layer: int, n_clients: int) -> list:
    """Bits carried by each layer. Clamp to <= channels // n_clients so that K clients
    sharing one layer's gamma can all embed (FedIPR Thm.1 capacity, K*bits <= C). >= 1."""
    out = []
    for C in channels:
        cap = max(1, int(C) // max(1, int(n_clients)))
        out.append(max(1, min(int(bits_per_layer), cap)))
    return out


# ---------------------------------------------------------------------------
# Per-client secret keys (E_i) + target bits (B_i), one pair per carrier layer.
# ---------------------------------------------------------------------------
def _make_bits(n_bits: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed) + 7919)
    if n_bits < 4:
        return torch.randint(0, 2, (n_bits,), generator=g).long()
    half = n_bits // 2
    base = torch.tensor([1] * half + [0] * (n_bits - half))
    return base[torch.randperm(n_bits, generator=g)].long()


def build_client_signsets(cids, channels, bits_list, seed) -> dict:
    """cid -> {'E': [E_i in R^{C_i x N_i}], 'bits': [B_i in {0,1}^{N_i}]} (one per carrier)."""
    out = {}
    for cid in sorted(set(int(c) for c in cids)):
        Es, Bs = [], []
        for i, (C, nb) in enumerate(zip(channels, bits_list)):
            s = int(seed) + 1000 * int(cid) + 3 + 7 * i
            Es.append(torch.randn(C, nb, generator=torch.Generator().manual_seed(s)))
            Bs.append(_make_bits(nb, s))
        out[cid] = {"E": Es, "bits": Bs}
    return out


# ---------------------------------------------------------------------------
# Projection / embed loss / extraction / detection  (per carrier, then summed)
# ---------------------------------------------------------------------------
def _project(gamma: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """gamma [C] . E [C,N] -> z [N]."""
    return gamma.reshape(1, -1).matmul(E.to(gamma.device)).reshape(-1)


def sign_embed_loss(gammas, Es, bits_list, margin: float = 0.1) -> torch.Tensor:
    """Mean over carriers of FedIPR hinge sign-loss (Eq. 19). Differentiable in the
    trainable gammas; frozen carriers contribute a constant (no gradient)."""
    total = None
    for g, E, b in zip(gammas, Es, bits_list):
        z = _project(g, E)
        bb = b.to(g.device).float() * 2.0 - 1.0
        term = torch.clamp(margin - bb * z, min=0.0).mean()
        total = term if total is None else total + term
    return total / max(1, len(gammas))


@torch.no_grad()
def sign_ber(gammas, Es, bits_list) -> float:
    """Total per-bit BER over ALL carriers = (sum wrong bits) / (sum bits).
    Honest (all carriers embedded) ~0; a free-rider that only re-embedded some carriers
    keeps the others at chance -> ber ~ (untrained bits / total bits) * 0.5."""
    wrong = tot = 0
    for g, E, b in zip(gammas, Es, bits_list):
        bh = (_project(g, E) >= 0).long()
        wrong += int((bh.cpu() != b.cpu()).sum())
        tot += int(len(b))
    return wrong / max(1, tot)


def gather_gammas_params(model, names) -> list:
    """Live (grad-enabled) carrier tensors from the model, in `names` order."""
    d = dict(model.named_parameters())
    return [d[n] for n in names]


@torch.no_grad()
def sign_ber_from_state(state: dict, names, Es, bits_list, device="cpu") -> float | None:
    """WHITE-BOX read: pull every carrier scale from a submitted state_dict and compute
    the total BER. Returns None if a carrier is missing from the state."""
    gammas = []
    for n in names:
        if n not in state:
            return None
        gammas.append(state[n].to(device))
    return sign_ber(gammas, Es, bits_list)