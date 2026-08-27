"""FedIPR backdoor (black-box, output-read) watermark -- second output-layer scheme.

FedIPR paper's *backdoor* watermark (its Algorithm-3 `alpha * L_T` term)

Mechanism (FedIPR, Li et al. 2022; repo purp1eHaze/FedIPR utils/datasets.py):
  * registration : each client k owns a private trigger set T_k = {(X_T, y_T)} of
                   `num_trigger` out-of-distribution images, each carrying a secret
                   target label y_T. (repo: prepare_wm / prepare_wm_new load an image
                   folder + assigned labels and hand each backdoored client a disjoint
                   `wm_iid` slice.)
  * embedding    : the client adds L_T = CE(y_T, f(X_T)) to its task loss (alpha=1).
  * verification : detection rate eta_T = mean( argmax f(X_T) == y_T ). Watermark
                   present iff eta_T >= 1 - eps_B.
  * free-rider   : a fabricated / untrained model classifies triggers at chance
                   (~1/num_classes) -> eta_T collapses -> flagged.

-------------------------------------------------------------------------------
FedIPR statistic as

        ber_fedipr := 1 - trigger_set_accuracy

Honest client -> trigger acc ~1 -> ber ~0.  Free-rider -> trigger acc ~1/C ->
ber ~ (1 - 1/C).  Nothing downstream needs to know how the scalar was produced.
-------------------------------------------------------------------------------
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F

# same per-dataset normalization the task inputs use (src/datasets.py:_NORM), so
# an OOD trigger image is presented to the model in the same input space it was
# trained in -- this is exactly what the repo's prepare_wm does (CIFAR mean/std).
_NORM = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def _normalize(x: torch.Tensor, dataset: str) -> torch.Tensor:
    """Apply the task dataset's channel normalization to trigger images in [0,1]."""
    mean, std = _NORM.get(dataset.lower(), (None, None))
    if mean is None:
        return x
    m = torch.tensor(mean).view(1, -1, 1, 1)
    s = torch.tensor(std).view(1, -1, 1, 1)
    if x.shape[1] != m.shape[1]:                      # channel mismatch guard
        m = m.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1, -1)
        s = s.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1, -1)
    return (x - m) / s


# ---------------------------------------------------------------------------
# Trigger POOL (the OOD images), before per-client slicing
# ---------------------------------------------------------------------------
def _pool_noise(n_total, in_channels, hw, seed) -> torch.Tensor:
    """Self-contained, reproducible OOD triggers: fixed pseudo-random RGB images.
    Needs no network and no external files -- the safe default on any cluster."""
    g = torch.Generator().manual_seed(int(seed) + 987654321)
    # low-frequency-ish noise so it isn't pure static: random blobs upsampled
    small = torch.rand((n_total, in_channels, max(4, hw // 4), max(4, hw // 4)),
                       generator=g)
    x = F.interpolate(small, size=(hw, hw), mode="bilinear", align_corners=False)
    return x.clamp(0, 1)


def _pool_svhn(n_total, in_channels, hw, seed, data_root) -> torch.Tensor:
    """Faithful OOD triggers: real unrelated images (SVHN), the analogue of the
    FedIPR repo's `trigger/pics` folder. Downloaded like src/datasets.py pulls
    CIFAR (download=True)."""
    import torchvision
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((hw, hw)), transforms.ToTensor()])
    root = os.path.join(data_root, "svhn")
    try:
        ds = torchvision.datasets.SVHN(root=root, split="test",
                                       download=True, transform=tf)
    except Exception as e:
        raise RuntimeError(
            f"FedIPR trigger source 'svhn' could not be loaded from {root!r} "
            f"({type(e).__name__}: {e}). The pod has no SVHN and could not download it. "
            f"Fix: pre-stage test_32x32.mat into {root}/ (needs scipy in the image), OR "
            f"run with FEDIPR_TRIGGER_SOURCE=noise (self-contained, no download).") from e
    g = torch.Generator().manual_seed(int(seed) + 424242)
    idx = torch.randperm(len(ds), generator=g)[:n_total].tolist()
    xs = []
    for i in idx:
        x, _ = ds[i]
        if x.shape[0] != in_channels:                 # e.g. grayscale task
            x = x.mean(dim=0, keepdim=True).expand(in_channels, -1, -1)
        xs.append(x)
    return torch.stack(xs).clamp(0, 1)


def _pool_folder(n_total, in_channels, hw, seed, folder) -> torch.Tensor:
    """Exact FedIPR faithfulness: point at a real trigger image folder
    (torchvision ImageFolder), matching repo prepare_wm_new."""
    import torchvision
    from torchvision import transforms
    tf = transforms.Compose([transforms.CenterCrop(hw) if hw else transforms.Lambda(lambda z: z),
                             transforms.Resize((hw, hw)), transforms.ToTensor()])
    ds = torchvision.datasets.ImageFolder(folder, tf)
    g = torch.Generator().manual_seed(int(seed) + 111)
    idx = torch.randperm(len(ds), generator=g)[:n_total].tolist()
    xs = []
    for i in idx:
        x, _ = ds[i]
        if x.shape[0] != in_channels:
            x = x.mean(dim=0, keepdim=True).expand(in_channels, -1, -1)
        xs.append(x)
    return torch.stack(xs).clamp(0, 1)


def _pool_indist(n_total, in_channels, hw, seed, data_root, dataset) -> torch.Tensor:
    """FedIPR IN-DISTRIBUTION triggers (repo prepare_wm_indistribution): real task-set
    images relabeled to a secret target"""
    import torchvision
    from torchvision import transforms
    name = (dataset or "cifar100").lower()
    tf = transforms.Compose([transforms.ToTensor()])
    root = data_root or "."
    if name == "cifar100":
        ds = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=tf)
    elif name == "cifar10":
        ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf)
    elif name == "mnist":
        ds = torchvision.datasets.MNIST(root, train=False, download=True, transform=tf)
    else:
        return _pool_noise(n_total, in_channels, hw, seed)
    g = torch.Generator().manual_seed(int(seed) + 20240607)
    idx = torch.randperm(len(ds), generator=g)[:n_total].tolist()
    xs = []
    for i in idx:
        x, _ = ds[i]                                   # [C,H,W] in [0,1], true label discarded
        if x.shape[-1] != hw:
            x = F.interpolate(x.unsqueeze(0), size=(hw, hw), mode="bilinear",
                              align_corners=False).squeeze(0)
        if x.shape[0] != in_channels:
            x = x.mean(0, keepdim=True).expand(in_channels, -1, -1)
        xs.append(x)
    return torch.stack(xs).clamp(0, 1)


def build_trigger_pool(source, n_total, in_channels, hw, seed,
                       data_root=None, folder=None, dataset=None) -> torch.Tensor:
    src = (source or "indist").lower()
    if src == "indist":
        return _pool_indist(n_total, in_channels, hw, seed, data_root or ".", dataset)
    if src == "noise":
        return _pool_noise(n_total, in_channels, hw, seed)
    if src == "svhn":
        return _pool_svhn(n_total, in_channels, hw, seed, data_root or ".")
    if src == "folder":
        if not folder:
            raise ValueError("fedipr_trigger_source='folder' needs fedipr_trigger_dir=<path>")
        return _pool_folder(n_total, in_channels, hw, seed, folder)
    raise ValueError(f"unknown fedipr_trigger_source '{source}' "
                     f"(use 'indist' | 'noise' | 'svhn' | 'folder').")


# ---------------------------------------------------------------------------
# Per-client trigger sets (disjoint wm_iid slices + secret target labels)
# ---------------------------------------------------------------------------
def _target_label(cid, num_classes, mode, seed) -> int:
    mode = (mode or "cid").lower()
    if mode == "fixed":
        return 5 % num_classes                        # FedIPR in-distribution default
    if mode == "random":
        g = torch.Generator().manual_seed(int(seed) + 7 * int(cid) + 13)
        return int(torch.randint(0, num_classes, (1,), generator=g).item())
    return int(cid) % num_classes                     # "cid" (default): distinct per client


def build_client_triggersets(cids, num_trigger, num_classes, dataset,
                             in_channels, hw, seed, *, source="noise",
                             target_mode="cid", data_root=None, folder=None):
    """cid -> {'x': [n,C,H,W] normalized, 'y': [n] target label, 'target': int}.

    `cids` is the list of clients that carry a trigger set
    """
    cids = sorted(set(int(c) for c in cids))
    per = max(1, int(num_trigger))
    pool = build_trigger_pool(source, per * len(cids), in_channels, hw, seed,
                              data_root=data_root, folder=folder, dataset=dataset)
    pool = _normalize(pool, dataset)
    out = {}
    for j, cid in enumerate(cids):
        x = pool[j * per:(j + 1) * per].clone()
        tgt = _target_label(cid, num_classes, target_mode, seed)
        y = torch.full((x.shape[0],), tgt, dtype=torch.long)
        out[cid] = {"x": x, "y": y, "target": tgt}
    return out


# ---------------------------------------------------------------------------
# Embedding loss + detection statistic
# ---------------------------------------------------------------------------
def embed_loss(logits: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """FedIPR backdoor loss L_T = CE(y_T, f(X_T))  (alpha = 1, faithful)."""
    return F.cross_entropy(logits, y_target)


@torch.no_grad()
def trigger_accuracy(model, x: torch.Tensor, y: torch.Tensor, device) -> float:
    """eta_T = fraction of trigger images classified as their target label."""
    if x is None or len(x) == 0:
        return float("nan")
    model.eval()
    pred = model(x.to(device)).argmax(dim=1).cpu()
    return float((pred == y.cpu()).float().mean().item())


def ber_from_acc(acc: float) -> float:
    """Map detection rate to the pipeline's [0,1] 'ber' (low = mark present)."""
    if acc is None or acc != acc:                     # NaN guard
        return None
    return float(1.0 - acc)


@torch.no_grad()
def detect_ber(model, x, y, device) -> float:
    """ber_fedipr = 1 - trigger_set_accuracy, the drop-in for wm.bit_error_rate."""
    return ber_from_acc(trigger_accuracy(model, x, y, device))