"""FedIPR feature-based (white-box) sign watermark 

================================================================================
 FedIPR white box hides bit string in signs of norm scale weights + reads white box
 Each chosen layer i carries its own bit-slice B_i via its own secret matrix E_i:
        embed :  minimise  sum_i  mean_j max( margin - b'_{i,j} * (gamma_i . E_i)_j , 0 )
        read  :  B_hat_i = sign(gamma_i . E_i)  ;  BER = total Hamming / total bits.

 server config num layers and which ones (config.fedipr_sign_carrier / fedipr_sign_layers):
   * fedipr_sign_layers = 1  (+ carrier "auto_last_bn")  -> only output-layer scale
     (net.layer4.1.bn2.weight, inside head2). 
   * fedipr_sign_layers = N (>1)  -> N output-most normalization scales
   * carrier = "all_bn"  -> every normalization scale (full-depth FedIPR, most robust).
   * carrier = "name1,name2,..."  -> an explicit list (server forces exact locations).
   * carrier = "scatter"  -> no layer at all: fedipr_sign_scatter_n single scalar weights at
     random positions in every parameter tensor form one virtual carrier vector.
================================================================================
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Carrier -- which normalization scales carry the mark.
# ---------------------------------------------------------------------------
def list_bn_scale_names(model) -> list:
    """All 1-D normalization scale weights (BN/norm/downsample-BN)"""
    names = []
    for n, p in model.named_parameters():
        if p.ndim == 1 and n.endswith("weight") and (
                "bn" in n or "norm" in n or "downsample.1" in n):
            names.append(n)
    if not names:                                     # fallback: any 1-D '*.weight'
        names = [n for n, p in model.named_parameters()
                 if p.ndim == 1 and n.endswith("weight")]
    return names


def scatter_carrier(model, n_weights: int, seed: int) -> dict:
    """carrier="scatter": {param name: sorted flat positions} of `n_weights` scalar weights,
    an equal share in every parameter tensor at random positions inside it.
    Positions are the server's choice, shared by all clients (each keeps its own E / B)."""
    named = list(model.named_parameters())
    sizes = [int(p.numel()) for _, p in named]
    left = min(max(1, int(n_weights)), sum(sizes))
    quota = [0] * len(named)
    while left > 0:                                   # equal share; small tensors cap at their size
        open_ = [i for i in range(len(named)) if quota[i] < sizes[i]]
        share = max(1, left // len(open_))
        for i in open_:
            take = min(share, sizes[i] - quota[i], left)
            quota[i] += take
            left -= take
            if left == 0:
                break
    g = torch.Generator().manual_seed(int(seed) + 31337)
    return {n: torch.randperm(sizes[i], generator=g)[:quota[i]].sort().values
            for i, (n, _) in enumerate(named) if quota[i] > 0}


def resolve_carrier_names(model, carrier: str = "auto_last_bn", n_layers: int = 1,
                          scatter_n: int = 512, seed: int = 0) -> list:
    """ordered list of carriers: a param name (whole tensor) or a scatter dict

    carrier="auto_last_bn" -> the last `n_layers` normalization scales (from output backward). n_layers=1 == the single output-layer scale.
    carrier="all_bn"       -> every normalization scale
    carrier="a,b,c"        -> exactly these parameter names
    carrier="scatter"      -> ONE virtual carrier: `scatter_n` scalar weights spread over every tensor
    """
    if carrier == "scatter":
        return [scatter_carrier(model, scatter_n, seed)]
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


def _carrier_vec(tensors: dict, c) -> torch.Tensor:
    """One carrier's weights out of a name -> tensor mapping (live params or a state_dict):
    the whole tensor for a param name, the scalars at its positions for a scatter carrier."""
    if isinstance(c, dict):
        return torch.cat([tensors[n].reshape(-1)[idx.to(tensors[n].device)]
                          for n, idx in c.items()])
    return tensors[c]


def per_carrier_channels(model, names) -> list:
    d = dict(model.named_parameters())
    return [int(_carrier_vec(d, c).numel()) for c in names]


def describe_carriers(carriers) -> list:
    """Log/JSON-safe carrier names (a scatter carrier is summarised, not dumped)."""
    return [f"scatter[{sum(len(i) for i in c.values())} weights / {len(c)} tensors]"
            if isinstance(c, dict) else c for c in carriers]


def carriers_to(carriers, device) -> list:
    """Same carriers with the scatter positions on `device` (no transfer per training step)."""
    return [{n: i.to(device) for n, i in c.items()} if isinstance(c, dict) else c
            for c in carriers]


def carrier_masks(model, carriers) -> dict:
    """param name -> bool mask of the scalars that carry the mark (all True for a whole-tensor
    carrier). Params with no entry carry nothing. Used by the scope="wm" free-rider."""
    d = dict(model.named_parameters())
    masks = {}
    for c in carriers:
        for n, idx in (c.items() if isinstance(c, dict) else [(c, None)]):
            m = masks.setdefault(n, torch.zeros_like(d[n], dtype=torch.bool))
            if idx is None:
                m.fill_(True)
            else:
                m.view(-1)[idx.to(m.device)] = True
    return masks


def plan_bits(channels, bits_per_layer: int, n_clients: int) -> list:
    """Bits carried by each layer"""
    out = []
    for C in channels:
        cap = max(1, int(C) // max(1, int(n_clients)))
        out.append(max(1, min(int(bits_per_layer), cap)))
    return out


# ---------------------------------------------------------------------------
# Per-client secret keys (E_i) + target bits (B_i), one pair per carrier layer
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
    """Mean over carriers of FedIPR sign-loss."""
    total = None
    for g, E, b in zip(gammas, Es, bits_list):
        z = _project(g, E)
        bb = b.to(g.device).float() * 2.0 - 1.0
        term = torch.clamp(margin - bb * z, min=0.0).mean()
        total = term if total is None else total + term
    return total / max(1, len(gammas))


@torch.no_grad()
def sign_ber(gammas, Es, bits_list) -> float:
    """Total per-bit BER over all carriers = (sum wrong bits) / (sum bits)
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
    return [_carrier_vec(d, c) for c in names]


@torch.no_grad()
def sign_ber_from_state(state: dict, names, Es, bits_list, device="cpu") -> float | None:
    """white box read: pull every carrier scale from a submitted state_dict and compute
    the total BER. Returns None if a carrier is missing from the state."""
    gammas = []
    for c in names:
        if any(n not in state for n in (c if isinstance(c, dict) else [c])):
            return None
        gammas.append(_carrier_vec(state, c).to(device))
    return sign_ber(gammas, Es, bits_list)