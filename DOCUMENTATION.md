# Output-Layer Watermarking is Free-Rideable — Technical Documentation

**Project:** Adapting FedIPR's backdoor watermark to an output-layer setting (à la FareMark),
and demonstrating that **output-layer / box-free watermarking cannot reliably detect free-riders**.

**Audience:** someone with *no prior knowledge* of this codebase who needs to pick it up, run it,
understand every moving part, and write the paper. Read this end-to-end before touching the code.

---

## 0. TL;DR — what this project claims and how it proves it

Three federated-learning (FL) watermarking schemes place their mark **in the output layer** — the first two
in the output *behaviour* (what the net predicts on trigger inputs), the third in the output-layer *weights*:

| Scheme | Verify | Where the mark lives | Detection statistic | "no mark" value |
|---|---|---|---|---|
| **FareMark** (Li et al., IEEE IoT J. 2025) | black-box | softmax probabilities on **trigger-class** images, projected through a secret ±1 key into `m` bits | **BER** vs registered bits | `0.5` |
| **FedIPR backdoor** (Li et al., TPAMI 2022) | black-box | argmax label on a private **trigger set** of images carrying a secret target label | `ber = 1 − trigger_set_accuracy` | `1 − 1/C` (≈`0.99` on CIFAR-100) |
| **FedIPR sign, forced to output layer** (this project, §4b) | **white-box** | the **signs** of an **output-layer scale vector** (last BN γ, in `head2`), read from the weights | `ber = Hamming(sign(γ·E), B)/N` | `0.5` |

The third scheme exists to show the fragility is about the mark's **location** (confined to the output layer),
**not** about black-box vs white-box verification — see §4b. Its free-rider forges it by training only `head2`,
exactly like the black-box marks.

Both schemes assume the same thing: **a free-rider who does not train cannot reproduce the mark**, so a
model whose mark is missing (high BER) is a free-rider. Their published free-rider is a *fabricated* model
(re-sent old global weights, or global + Gaussian noise) that never trains — and is trivially caught
(BER ≈ chance).

**This project's thesis:** because the mark is a pure **output-layer** phenomenon, a free-rider only has to
train the **last few layers** on a **tiny amount of data** for a **few rounds** to re-embed it. The result is
a model whose BER sits **inside the honest band** (BER ≈ 0), indistinguishable from an honest client by any
BER threshold — while the free-rider spends a small fraction of honest compute and never contributes a real
model. We build three such attackers (`reduced`, `graftblock`, `adaptive_tap`/"submarine") and show they
defeat **both** FareMark and FedIPR.

The uploaded result figures already show this:
- **positive control** (`H5`/`F_H5` previous-models free-rider): BER pinned near the "no-mark" line → **caught** (as it should be).
- **graftblock free-rider** (`L1`/`L5`, `F_L1`/`F_L5`, head2-scope training on `cpc=5` data): BER ≈ 0, buried in the honest cloud → **evades**. This holds for FedIPR too (`F_*` families).

---

## 1. Repository map & the two-repo split

**Important:** the code you run on the cluster is **not** these uploaded files. `submit_experiment.sh`
clones and runs a *different* git repo:

```
GIT_REPO="https://github.com/zu-greta/submarine_freerider_watermarking_federatedlearning_summer-epfl.git"
GIT_BRANCH="main"
SCRIPT="scripts/run_experiment.py"
```

The uploaded files are the `src/` package + `scripts/` + shell orchestration of that repo. When you edit
locally you must **push to that GitHub repo** before submitting, or the pods run stale code. (The pod does
`git clone --depth 1` every job; there is no local-code path.)

### 1.1 Files provided (what each does)

| File | Layer | Responsibility |
|---|---|---|
| `src/config.py` | config | `ExpConfig` dataclass, the `CONFIGS` list, every knob's default + doc |
| `src/datasets.py` | data | load CIFAR/MNIST, IID + Dirichlet non-IID partition into per-client loaders |
| `src/clients.py` | clients | honest `Client`, `WatermarkClient` (both schemes), and **all three attackers** |
| `src/watermark.py` | scheme A | **FareMark** box-free softmax-projection scheme (Eq. 1–16) |
| `src/watermark_fedipr.py` | scheme B | **FedIPR** backdoor trigger-set scheme (black-box) |
| `src/watermark_fedipr_sign.py` | scheme C | **FedIPR feature-based sign** watermark, forced into the output-layer scale (**white-box**); see §4b |
| `src/wm_verify.py` | server | registry + per-round verification hook for **both** schemes |
| `src/server.py` | server | FedAvg aggregation + round loop |
| `src/compute_meter.py` | metering | per-client samples / GPU-ms / FLOPs accounting (the "free-riding is cheap" evidence) |
| `scripts/run_experiment.py` | entry | one `(config, seed)` run → `result.json` |
| `scripts/plots.py` | analysis | every figure family |
| `src/utils.py` | util | seeding, logging, accuracy |
| `run_now.sh` | orchestration | builds `jobs.tsv` (the experiment manifest) for groups A,T,D,E,EA,H,K,Y,Z,L,**F** |
| `runbook.sh` | orchestration | phase driver: probe → manifest → submit → monitor → plot → grade |
| `submit_experiment.sh` | orchestration | one RunAI/Kubernetes job submission |

### 1.2 Files provided in the second batch

- `src/models.py` — `build_model`; **ResNet-18** (CIFAR stem: 3×3 stride-1 conv1, `maxpool→Identity`) +
  `SmallCNN` for smoke tests. 62 named parameter tensors — the number the attack scopes (`head2`=last 5,
  `block2`=last 20) count from; see the ATTACK-SURFACE NOTE added to the file header.
- `src/fast_data.py` — `wrap_build_data` / `FastLoader`: the whole dataset is held as one uint8 tensor on the
  GPU and normalised/augmented on-device (removes DataLoader fork storms). It self-tests against torchvision
  augmentation in its `__main__`. Enabled by `FAST_DATA=1`. Behaviour is statistically identical to the CPU
  loaders; it only changes speed.

### 1.3 Files referenced but **not uploaded** (confirm they exist in the run repo)

`src/manifest.py` (`build_manifest`), `src/runlog.py` (log tables), `scripts/resultio.py`,
`scripts/paper_check.py`, `submit_pool.sh` (the pod pool worker). Nothing here contradicts them, but their
internals are inferred, not verified.

---

## 2. Datasets — full explanation (asked for explicitly)

### 2.1 The task datasets

`src/datasets.py` wraps torchvision. Supported: **MNIST** (10 cls, 1×28×28), **CIFAR-10** (10 cls,
3×32×32), **CIFAR-100** (100 cls, 3×32×32), and — **Food-101**
(`food101`, ETH-Zürich, 101 cls), natural photos resized to `FOOD_SIZE`² (default 64×64) with ImageNet normalisation.
Train aug for CIFAR is `RandomCrop(32, padding=4)` + `RandomHorizontalFlip`; for both food sets it is
`RandomResizedCrop(FOOD_SIZE)` + flip (test = `Resize`+`CenterCrop`).

### 2.1b Switching datasets (the one-line switch) + Food-101 download

**Switch.** `runbook.sh` and `run_now.sh` each have a `DATASET` switch at the very top
(`export DATASET="${DATASET:-cifar100}"`, choices `cifar100 | cifar10 | mnist | food101 | food100`). Flip it
once and the whole suite (every group, every family) runs on that dataset — it is exported, so
`submit_experiment.sh` adds `--dataset $DATASET`, which overrides config 14's dataset. Ready-made configs also
exist: **15** = `attack_base_resnet18_food101` (Food-101)

**No new plumbing needed** — `--dataset` was already an overridable arg; this change only adds Food-101 to
`datasets.py`/`fast_data.py`, the top-of-file switch, and a per-dataset results path.

**Results never collide.** `cifar100` keeps the original flat path `…/results/<RUN_TAG>` (back-compat with your
existing runs); any other dataset writes to `…/results/<dataset>/<RUN_TAG>`, and `runbook.sh` points `RES`
there for plotting. **Family tags keep their `_c100` label regardless of dataset** (they name the free-rider
ids, not the dataset); the true dataset is in the path and inside `result.json`. So plot one dataset at a time.

**Recalibrate η.** Honest BER floors are dataset-specific, so re-freeze `ETA_T*/ETA_L*` (and the FedIPR
`ETA_*_FI` / sign `ETA_*_WS`) from the new honest baseline (`A1_honest` / `F_A1` / `G_A1`) before quoting numbers.

**Downloading Food-101 (`food101`).** Automatic: torchvision `Food101(..., download=True)` pulls a ~5 GB
tarball to `DATA_ROOT/food-101` on first use (750 train + 250 test per class). `DATA_ROOT` is on the PVC
(`$MOUNT/home/zu/data`), so **the first pod downloads it once and every later run reuses it** — that pod needs
internet egress and a few GB of PVC space. Air-gapped: pre-stage the extracted `food-101/` folder.

**`FAST_DATA` is a no-op for both food sets** (file-based, no fixed `.data` array), so those runs use the CPU
DataLoaders — set `NUM_WORKERS` > 0 for food runs to keep the GPU fed.

**The experiments here run almost entirely on CIFAR-100** (`config_idx 14`, `attack_base_resnet18_cifar100`,
ResNet-18, `num_clients=10`, 50 rounds, `local_epochs=5`, `lr=0.01`, `batch_size=16`, SGD momentum 0.9,
weight-decay 5e-4). CIFAR-100 = 50 000 train / 10 000 test images.

### 2.2 How data is split across the 10 clients

- **IID** (`partition=iid`, the default and most groups): shuffle all 50 000 training indices with a seeded
  RNG, `np.array_split` into 10 near-equal shards of ~5 000 images. Every client sees all 100 classes roughly
  uniformly.
- **Non-IID** (`partition=dirichlet`, groups E/EA): for each class draw a `Dirichlet(alpha)` vector over the
  10 clients and hand out that class's images in those proportions (Hsu et al. 2019 label-skew). `alpha=0.5`
  is the standard FL non-IID benchmark; `alpha=0.1` is severe skew; `alpha=1.0` is milder. `datasets.py`
  reshuffles each shard afterwards. Small α means a client may hold **very few or zero** images of its
  assigned trigger class — this is the "starvation" the non-IID groups probe.

### 2.3 Trigger data — two very different notions, do not confuse them

1. **FareMark trigger class** — *not extra data*. Each client is assigned one existing task class as its
   "trigger class" (default `cid % num_classes`, so cids 0–9 → classes 0–9 on CIFAR-100). The mark is embedded
   by shaping the softmax **on images of that class that already live in the client's shard**. Verification
   uses a held-out bank of test-set images of that class (`build_trigger_bank`), or per-client variants
   (`wm_trigger_mode` = `class` | `client` | `client_train`, mirroring FareMark §V-F3).

2. **FedIPR trigger set** — *separate, out-of-distribution (or patched) images* with a secret **target
   label**. Built once per client in `watermark_fedipr.build_client_triggersets`, 40 images/client by
   default, disjoint per client. Sources (`fedipr_trigger_source`):
   - `indist` (**config default**): real CIFAR test images, each stamped with a per-client solid-colour
     **BadNets patch** in a corner (`_stamp_patch_`), then normalised. The patch is the separable feature the
     backdoor keys on.
   - `svhn`: real OOD images (SVHN test split, downloaded like CIFAR). The faithful analogue of the FedIPR
     repo's `trigger/pics` folder.
   - `noise`: self-contained low-frequency RGB noise (no download; safe default on an air-gapped pod).
   - `folder`: `torchvision.ImageFolder` at `fedipr_trigger_dir` (exact-repo faithfulness path).

   Target label (`fedipr_target_mode`): `cid` (=`cid % num_classes`, **default**, distinct per client so 10
   clients' patch→label maps don't collide), `fixed` (=5, matches the FedIPR repo's in-distribution default),
   or `random`.

### 2.4 The free-rider's "reduced shard"

When an attacker "taps" (trains a little), it does **not** use its full shard. `_SimpleFRMixin._prepare`
builds a reduced loader = **all its trigger-class images** + **`cpc` images per common class** (`cpc` =
`autop_common_per_class` / `tap_data_cpc`, default 5). So `cpc=5` on CIFAR-100 = trigger-class images + 5×99 ≈
495 common images, a fraction of the ~5 000-image honest shard. `cpc=-1` = full shard; `cpc=0` = trigger
images only. A slice of trigger images is optionally **held out** as the attacker's self-BER "probe"
(`tap_probe_holdout`, default 16) so it can measure its own mark strength without the server.

### 2.5 Dataset-related gotcha in the naming

The family tags are **not** dataset names. In `A3_reduced_c100_c36`: `c100` = CIFAR-**100**; `c36` = the
free-rider **client ids are 3 and 6** (their trigger classes are therefore 3 and 6). `c17` = cids 1,7.
`aXX` = Dirichlet α (e.g. `a01`=0.1, `a10`=1.0). This overloading of "c" bites everyone once.

---

## 3. FareMark scheme (`src/watermark.py`) — mechanism & paper mapping

FareMark is **box-free**: verification reads only the model's **softmax outputs**, never its weights.

**Pipeline (paper equation numbers):**
1. Take the `n`-dim softmax `P` on a trigger-class image. Use only the first `m·l` outputs, split into `m`
   groups of size `l = n // m` (`grouping`). On CIFAR-100 with `m = n//10 = 10`, `l = 10`.
2. **Smooth** each probability with `f()` so the argmax doesn't dominate the projection: `f(p)=(p+ε)^α`,
   `0<α<1` (default `α=0.4`), Eq. 7–9 (`smooth`). Cross-entropy makes softmax steep (one class ≈1); smoothing
   amplifies the tail so it can carry bits without hurting the true-class prediction.
3. **Project** each group onto a per-client pseudo-random ±1 key row `M`: `z_k = Σ_j f(p_{k,j})·M_{k,j}`
   (Eq. 1/13, `project_logits`).
4. **Bit** `b_k = 1 if z_k ≥ 0 else 0` (Eq. 2, `extract_bits` after averaging over `N_T` samples, Eq. 15).
5. **Embed** by adding `L_wm = BCE(z, target_bits)` to the task loss: `L = L_cl + λ·L_wm` (Eq. 11–12,
   `watermark_loss`, `λ=5`). Only trigger-class samples in a batch contribute `L_wm`.
6. **Memory-enhanced update** (Eq. 14): `W_new = β(memory + δ) + (1−β)·W_global`, `δ = W_sgd − W_global`,
   `β=0.6` (`WatermarkClient._memory_update`). Counteracts the averaging that erodes the mark.
7. **Detect**: `BER = (1/m)Σ|b̂_k − b_k|`; benign iff `BER < η` (Eq. 16). Paper's threshold `η = μ + 3σ` over
   benign BER (`calibrate_eta`).

**Honest client → BER ≈ 0** (mark embedded). **Fabricated free-rider → BER ≈ 0.5** (random bits).

### 3.1 Faithfulness to the FareMark paper — and deviations

**Faithful:** the grouping/first-`m·l`-outputs rule, power-smoothing `x^α`, ±1 random key (`make_key`,
`balanced=False` = paper's random `M`), full-softmax projection (no trigger-class column excluded by default),
BCE embed loss, memory update Eq. 14 with β=0.6, λ=5, `η=μ+3σ`, and the two paper free-rider baselines
(`PreviousModelsFreeRider` = `2W_t − W_{t−1}` Eq. 17; `GaussianNoiseFreeRider` = `W_t + N(0,σ²)` Eq. 18).

**Deviations (all justified, but list them in the paper):**
- **η is frozen offline**, not recomputed live. `wm_verify.make_verifier` uses a pre-calibrated constant
  `WM_ETA_FIXED` = `μ+3σ` of honest BER over the honest baseline runs (last 20 rounds, pooled over the honest
  seeds); the live `μ+3σ` code path is present but commented out. This makes the threshold a stable, fair
  fixed operating point instead of a moving target, but it is a methodological choice the reader must know.
- **Two frozen η's are reported**, not one: `η_tight = 0.064` (μ+3σ over per-round *means*) and
  `η_loose = 0.264` (μ+3σ over per-*client* BERs). This is the project's own "threshold dilemma" framing, not
  in the paper. Non-IID uses `0.161 / 0.576`.
- **`label_smoothing=0.1`** on the task CE and **`clip_grad_norm_(…,5.0)`** are added for stability; not in the
  paper.
- **Unembeddable-bit guard** (`unembeddable_fraction`, `wm_balanced_keys`): a random ±1 key row that is all
  +1 or all −1 forces `z_k` to a fixed sign for every input (because `f(p)≥0`), so that bit can never be
  embedded and honest BER floors near `0.5·frac`. The paper's random-`M` description silently ignores this;
  the code detects it and offers sign-balanced rows. Keep `balanced=False` for paper-faithful runs, but
  **report the floor**.
- **`sin` smoothing is effectively disabled** at the paper's α. `smoothing_gain("sin", 0.4) ≈ 1.01` (i.e. no
  smoothing at all); the code raises unless `gain ≥ 1.10`. This is a genuine issue with FareMark's Eq. 9
  alternative — worth a sentence in the paper's "we reproduced and found…" discussion.
- **Datasets/models actually run:** CIFAR-100/ResNet-18 only in these groups (the paper also does
  MNIST/CIFAR-10/Food100, AlexNet/ShuffleNet/GoogLeNet). Not a bug, just scope.

---

## 4. FedIPR backdoor scheme (`src/watermark_fedipr.py`) — mechanism & paper mapping

FedIPR's paper has **two** watermarks: a **feature-based** one (a binary string embedded in *normalization
scale weights*, read white-box via `sign()`), and a **backdoor** one (a private trigger set → target label,
read black-box). **This project implements only the backdoor half**, on purpose, so it is directly comparable
to FareMark's output-layer, box-free setting.

**Mechanism (paper Alg. 2/3/4):**
- **Registration** `G()`: each backdoored client owns a private trigger set `T_k = {(X_T, y_T)}` of
  `num_trigger` images, all carrying a secret target label.
- **Embedding** `E()`: add `L_T = CE(f(X_T), y_T)` to the task loss with `α=1` via **batch poisoning** —
  trigger samples are *concatenated into each normal training batch*, not trained in a separate pass
  (paper Alg. 3 lines 8–14: "Batch poisoning approach is adopted, thus α_l=1, L_c = L_D + L_T").
- **Verification** `V_B()`: detection rate `η_T = mean(argmax f(X_T) == y_T)`; mark present iff
  `η_T ≥ 1 − ε_B` (Eq. 4/8). This project maps it to the shared pipeline via **`ber_fedipr = 1 − η_T`**
  (`detect_ber` / `ber_from_acc`), so every downstream plot/threshold path is reused unchanged.
- **Free-rider**: an untrained/fabricated model classifies triggers at chance (`1/C`) → `η_T` collapses →
  `ber ≈ 1 − 1/C` → flagged.

### 4.1 Faithfulness to the original FedIPR repo (`purp1eHaze/FedIPR`) — and deviations

I read the repo's `utils/datasets.py` (`prepare_wm`, `prepare_wm_indistribution`, `prepare_wm_new`) and the
paper's Alg. 2–4.

**Faithful:**
- Per-client private trigger set of `num_trigger` images (repo default 40 in `prepare_wm_new` /
  `prepare_wm_indistribution`; this project defaults to 40), disjoint slices across clients (`wm_iid` ↔ this
  project's per-client slicing).
- Backdoor loss = plain `CE(logits, target)` (`embed_loss`), **α=1 batch-poisoning by concatenation** — an
  exact match to Alg. 3. The code even documents *why* mixing (not a trigger-only pass) is required: a
  trigger-only pass computes BatchNorm stats from triggers alone and the mark collapses under eval-mode
  running stats. This is a correctness fix that keeps the implementation faithful in practice.
- Detection = trigger-set accuracy thresholded (`η_T ≥ 1 − ε_B`), OOD/real-image triggers normalised with the
  task's mean/std exactly as the repo's `prepare_wm` does.
- `folder` source (`ImageFolder`) ↔ repo `prepare_wm_new`; `fixed` target label 5 ↔ repo
  `prepare_wm_indistribution` ("all trigger samples → label 5").

**Deviations (state these clearly in the paper):**
1. **Feature-based (white-box, sign-loss / passport-layer) watermark is NOT implemented.** FedIPR's headline
   robustness story lives there; this project drops it deliberately to stay output-layer/box-free. So "FedIPR"
   here means "FedIPR-backdoor," not the full scheme.
2. **Triggers are OOD/patched images, not PGD adversarial samples.** The FedIPR paper (§5.2) generates
   triggers as PGD adversarial examples of real data; the repo's shipped path uses an external `trigger/pics`
   folder / in-distribution images. This project uses SVHN/noise/folder (OOD) or in-distribution CIFAR + a
   **BadNets colour patch**. Conceptually equivalent (memorised input→label mapping) but **not** the paper's
   PGD triggers. The patch is this project's addition to make in-distribution triggers separable.
3. **Target label default = `cid % C` (distinct per client)**, vs the repo's hardcoded `5`. Chosen so plots
   group by class and each free-rider owns a distinct label; `fixed` reproduces the repo.
4. **All clients aggregated every round**; FedIPR's Alg. 3 randomly selects `cK` of `K`. Matches FareMark's
   setup, diverges from FedIPR.
5. **≤4 trigger samples mixed per task batch** (`K = min(4, len)`); the paper samples `t` per batch without
   fixing the count. Minor.
6. Architecture is this project's ResNet-18/AlexNet (no passport conv layers). Fine for the backdoor half.
7. **Provisional η** for FedIPR (`ETA_T_FI=0.20`, `ETA_L_FI=0.50`) — must be **recalibrated** from the
   `F_A1_honest` runs (μ+3σ of honest `ber_fedipr`) before the paper. `runbook.sh` says as much; do it.

## 4b. FedIPR feature-based SIGN watermark, forced into the output layer (scheme C, WHITE-BOX)

**File:** `src/watermark_fedipr_sign.py` (+ `fedipr_sign` branches in `clients.py`, `wm_verify.py`,
`config.py`, `run_experiment.py`). **Group G** in `run_now.sh` / `runbook.sh`. Select with
`WM_SCHEME=fedipr_sign`.

**Why it exists.** FedIPR's *feature-based* watermark hides a secret bit-string in the **signs of a
normalization scale vector** (`W_γ`), read **white-box** straight from the weights (FedIPR Eq. 18–19, Alg. 4).
In real FedIPR that carrier is spread through the body, which is exactly why a head-only free-rider can't
forge it — correctly out of scope (§9). Here we do the thing a naive "improver" of FedIPR might do: **force
the whole sign-watermark into one carrier in the output layer** and show the same head-only free-rider forges
it. This isolates the real cause of the fragility: **the mark's *location* (confined to the output layer),
not black-box vs white-box verification.**

**Mechanism (faithful to FedIPR feature-based).**
- *Carrier* = a scale vector chosen by the **server** (`fedipr_sign_carrier`, default `auto_last_bn` → the
  last BatchNorm scale, `net.layer4.1.bn2.weight`, a 512-dim γ that sits **inside `head2`**). Setting a body
  layer instead gives the robust contrast — that switch *is* "the server forcing output-layer watermarking".
- *Registration* = each client gets a secret embedding matrix `E_k ∈ ℝ^{C×N}` and target bits `B_k ∈ {0,1}^N`
  (`N` = `fedipr_sign_bits`, default 40; keep `K·N ≤ C` so 10·40 ≤ 512 stays feasible — FedIPR Thm. 1 capacity).
- *Embedding* = task CE **+** `λ·L_sign`, the FedIPR hinge sign-loss `mean_j max(margin − b'_j·(γ·E)_j, 0)`
  (`clients._local_train_fedipr_sign`). Honest clients train the full model; the loss drives `sign(γ·E) → B_k`.
- *Verification* = **white-box**: read `γ` from the submitted weights (no forward pass), extract
  `sign(γ·E)`, `ber = Hamming/N` (`wm_verify.py` `fedipr_sign` branch + `watermark_fedipr_sign.sign_ber`).
  Honest → `ber ≈ 0`; random/fabricated weights → `ber ≈ 0.5` (per-bit coin flip — **same chance as FareMark**,
  so it drops into the identical threshold/plot pipeline; `0.5`, not `1 − 1/C`).

**The free-rider.** The very same `graftblock` / `adaptive_tap` attackers — no new attacker code. Because the
carrier lives in `head2`, freezing everything else and training only `head2` (with the client's **own** `E_k`,
`B_k`) re-embeds its own sign mark on reduced data, `ber → 0`, inside the honest band → evades. The one rule:
**the carrier must be inside the free-rider's scope** — `head2`/`block2`/`full` include the last BN γ; a bare
`head` (fc-only) scope does **not**, and then the mark can't be forged (a nice built-in ablation).

**Validated here.** A CPU end-to-end smoke (ResNet-18, 3 clients, 1 graftblock free-rider) gave, per round:
honest BER `0.21 → 0.15 → 0.05 → 0.04` (full-model embed) while the **head2-only free-rider** went
`0.125 → 0.075 → 0.000 → 0.000` — it forges its own white-box mark and is unflagged from round 2. Same
outcome as the black-box schemes.

**Faithfulness / deviations to state.** Faithful: the sign-of-`W_γ` carrier, the hinge sign-loss, white-box
extraction, per-client secret `(E_k, B_k)`, FedAvg (no memory update). Deviations: (1) the carrier is *forced
into one output-layer scale* rather than spread across body normalization layers — this is the whole point,
so frame it as *"a stress-test construction, not a scheme FedIPR ships"*; (2) `N=40` bits, `margin=0.1`,
`λ=1.0` are defaults that may need light tuning (watch the honest floor in `G_A1`); (3) provisional η
(`ETA_T_WS=0.20`, `ETA_L_WS=0.50`) — recalibrate from `G_A1_honest` (μ+3σ of honest sign BER) before quoting.

**Paper framing.** With scheme C you can upgrade the thesis from *"output-behaviour watermarks are fragile"*
to *"output-**layer** watermarks are fragile, black-box or white-box — the vulnerability is the location, plus
the free-rider holding its own key."* The real FedIPR feature-based mark (in the body) staying robust is the
control that proves location is the axis.

---

## 4c. Unified theory — why all three schemes fail the same way

The three schemes differ on almost every axis, yet break identically. That invariance is the paper's core
argument, so state it as one claim rather than three separate weaknesses.

### The three schemes, side by side

| | **1. FareMark** | **2. FedIPR backdoor** | **3. FedIPR sign (output-layer)** |
|---|---|---|---|
| What carries the mark | softmax probabilities on **trigger-class** images | argmax label on a private **trigger set** (40 images) | the **signs of an output-layer scale vector** (last BN γ) |
| Verify modality | black-box (read outputs) | black-box (read outputs) | **white-box** (read weights) |
| Honest embeds by training | full model + projection loss on trigger-class samples | full model + backdoor CE on the trigger set | full model + hinge sign-loss on the carrier |
| What the FR needs to forge it | trigger-class images + a little task data | its trigger set + a little task data | **only its secret key** + a little task data (no watermark data) |
| Chance / "no-mark" BER | 0.5 | 1 − 1/C | 0.5 |

### Per-scheme statement (precise)

1. **FareMark.** The mark is a loss shaping the softmax on a client's *trigger class*, read box-free from the
   output. Forgeable: the free-rider trains only the output layers (`head2`) on a reduced shard focused on
   trigger-class samples and re-embeds it.
2. **FedIPR backdoor.** The mark is **not stored in any layer's weights** — it is an *output behavior*
   ("trigger set → secret label"), read black-box (argmax on the triggers). It is forgeable because (a) that
   decision is controlled by the last layers acting on the body's features, and (b) the free-rider inherits
   the honest feature-extracting **body for free** via FedAvg, so retraining only the head on its trigger set
   reinstalls the behavior. ("Focused on the output layer" describes the *free-rider's training*; the
   *watermark itself* is output behavior, not an output-layer object.)
3. **FedIPR sign (forced to the output layer).** The mark is the signs of an output-layer scale, read
   white-box from the weights. Forgeable: the free-rider trains only `head2` — which *contains* the carrier —
   using **its own key**, needing **no watermark data at all** (the sign-loss depends only on the key applied
   to the head weights). White-box makes the forgery *easier* to state, not harder.

### The theorem

> **An output-layer watermark is forgeable by a free-rider when three conditions hold:**
> 1. **Location** — the mark's evidence is confined to, or controlled by, the output layer (the last few tensors).
> 2. **Free body** — the expensive feature-extracting body is handed to the free-rider for free by FedAvg aggregation.
> 3. **Own key** — the free-rider is a registered client, so it holds its own secret (trigger class / trigger set / sign key) and can re-embed *its own* valid mark.
>
> Then a head-only free-rider, training on a fraction of the data, produces a model with a valid watermark
> that lands **inside the honest BER band** → no threshold η separates honest from free-rider without
> unacceptable false positives.

### Why this is the strong claim

The failure is **invariant** to (a) the mark mechanism (projected bits vs backdoor label vs sign bits),
(b) the "no-mark" value (0.5 vs 1 − 1/C), and (c) the verification modality (**black-box vs white-box**). It
**depends on** the mark's **location** (the output layer) plus the free-rider holding its own key. Three
schemes that differ on every one of those first three axes yet all break the same way is what promotes the
claim from *"FareMark has a weakness"* to *"output-layer watermarking for free-rider detection is impossible
in general."* The **control** that proves location is the axis: the real FedIPR feature-based mark, embedded
across the **body**, stays robust to the identical head-only attack (violating condition 1).

---

## 5. The attackers (`src/clients.py` §3) — how free-riding is done

All attackers subclass `WatermarkClient` (so they *can* embed either mark) via `_SimpleFRMixin`, and share a
schedule: **honest warmup** `[1, W)`, a **calibration window** (last `K` warmup rounds), then **free-ride**
from round `W` (`autop_honest_until=W=12`, `autop_calib_rounds=K=4`).

### 5.1 Paper baselines (must be CAUGHT — positive controls)
- `PreviousModelsFreeRider` (Eq. 17): resend `2W_t − W_{t−1}`, never trains. BER ≈ chance. Groups **H5 / F_H5**.
- `GaussianNoiseFreeRider` (Eq. 18): `W_t + N(0,σ²)`. Groups **H6 / F_H6**.
These validate the detector works at all. In the figures they sit on the "no-mark" line.

### 5.2 `reduced` (group A/D/E) — honest-but-lazy
After warmup, train exactly like an honest client **but on the reduced shard** (`cpc` images/class). Re-embeds
the mark every round at a fraction of the data. Simplest evasion; shows the mark survives on tiny data.

### 5.3 `graftblock` (groups **L**, **F_L**) — the clean output-layer attack (your main figures)
After warmup, every round: switch to the reduced shard **and freeze all but the last few parameter tensors**
(`tap_scope`), then train. Scope map (`_SCOPE_KEEP`): `head2` = last 5 tensors (softmax `fc` + the conv before
it, ~2.4M params ≈ 21% of ResNet-18); `block2` = last 20 tensors (~80%); `head`=2; `full`=all. Optional
`graft`: after training, overwrite the body with the *exact current global* and keep only the freshly-trained
head (so the body is provably the honest global; only the head moved). **This is the cleanest proof that the
mark is output-layer-only:** training just the head (`head2`) re-embeds it, BER→0, evading detection. The
uploaded `L1/L5` and `F_L1/F_L5` figures are exactly this.

### 5.4 `adaptive_tap` (the "submarine", groups K/Y/E/EA/F_K) — the stealthy attack
Free-rides between "taps," training only when it must:
- **Threshold estimation** (`tap_eta_source`): `oracle` (handed the true η) or **`self`** — estimate
  `η̂ = μ + k·σ` (`tap_eta_k=3`) over its *own* calib-window self-probe BERs. It never sees the server η.
- **Tap target** (`tap_margin_mode`): `fixed` → `target = η̂ − margin`; `derived` → `margin = k·σ(own probe
  BER)`, so the safety gap widens when its own estimate is noisy.
- **Decision each round** (`tap_when=threshold`): probe the model it *would* submit if it coasts; if that BER
  > target, **tap** (train head/block on reduced shard); else **coast** (submit without training). Safety cap
  `tap_max_coast` forces a periodic tap.
- **Coasting** (`tap_coast_mode`): `decay` = resubmit its own last-tapped weights (mark frozen, decays under
  aggregation); `graft` = fresh global body + its frozen mark-head (body tracks global, head frozen),
  optionally blended toward global by `tap_graft_decay` to kill tail spikes.
- **Dynamic warmup** (`tap_warmup_mode=dynamic`): instead of a fixed defect round, defect once its own probe
  BER has *converged* (flat within `tap_conv_eps` for `tap_conv_patience+1` rounds), bounded to
  `[tap_honest_min, tap_warmup_cap]`. Makes the attack self-timing and un-scheduled.

Net effect: the submarine keeps BER just under η with the **minimum** number of cheap head-only taps, so it is
cheap *and* stays inside the honest band. `K4`=block2 scope, `K9`=head2 scope; `J4`(group Y)=oracle-η ablation
of `K4`.

### 5.5 What the compute meter proves (`src/compute_meter.py`)
Every training client accumulates `samples`, `gpu_ms` (CUDA events), `opt_steps`, estimated `flops`, and a
`duty_cycle` (fraction of rounds it actually trained). The result: `effort_ratio_samples` /
`effort_ratio_gpu` = free-rider effort ÷ honest effort, and the `gpu_savings` plots. **This is the axis
FareMark/FedIPR never measure** — a free-rider obtains the honest global model for a small fraction of the
compute while remaining undetected. `samples` is the safe cross-run cost axis (`gpu_ms` is only comparable at
concurrency 1 and on the same GPU model — the code records `gpu_concurrency` / `gpu_name` so you don't compare
apples to oranges).

---

## 6. Server, verification, and the round loop

- `src/server.py`: plain weighted **FedAvg** (`Aggregator.aggregate`, weight = client sample count; integer
  buffers like BN `num_batches_tracked` are copied, not averaged). Each round: collect every client's
  `produce_update(global, prev_global, r)`, run the `verify_hook`, then aggregate and evaluate global test
  accuracy. `prev_global_state` is kept so the previous-models baseline can extrapolate.
- `src/wm_verify.py`: registers every client's mark; each round loads each submitted model into a throwaway
  `verify_model`, extracts the mark (FareMark: project+sign+BER; FedIPR: trigger-set accuracy → `1−acc`),
  applies the frozen η, and emits per-client `{ber, flagged, trig_acc, pmax, entropy, dominance}` plus
  aggregates (`wm_benign_ber`, `wm_fr_ber`, `wm_fpr`, `wm_fr_recall`, `wm_detect_acc`, percentile band
  `p90/max`). **The FedIPR branch and the FareMark branch produce the same `ber` field**, which is the whole
  point of the unification.

Everything a run produces lands in one `result.json` (schema v2): `manifest`, flat `summary`, `compute`
(per-client effort), and `history` (per-round metrics incl. `wm_per_client` BER lists). `plots.py` consumes
globs of these.

---

## 7. How to run it (phases)

`runbook.sh` is the driver. Env knobs: `BATCH` (which groups), `PODS`, `WORKERS`, `RES`, `OUT`, `FAST_DATA`,
`DETERMINISM`.

```
# on the cluster (needs submit_experiment.sh + .env with PROJECT/IMAGE/PVC/...):
./runbook.sh probe            # 0.  FareMark embedding sanity (ber_h drops, pmax not nan)
BATCH=F ./runbook.sh manifest # 1.  build jobs.tsv for group F (FedIPR); tokens: A T D E EA H K Y Z L F
WORKERS=6 PODS=2 ./runbook.sh submit   # 2. run the pod pool over jobs.tsv
./runbook.sh monitor          # 3.  runai list jobs + digest
# ... wait ...

# locally, pointing at pulled results:
RES=~/local/results ./runbook.sh plot   # 4. all figures into $RES/figs
RES=~/local/results ./runbook.sh grade  # 5. optional paper-repro tables
```

`run_now.sh` is what `manifest` calls; each group is a block guarded by `has <TOKEN>`. `DRYRUN=1` makes
`submit_experiment.sh` append one manifest row per run instead of submitting; the pool replays `jobs.tsv`.
Because `RUN_TAG` is deterministic (`<FAMILY>_rep<seed>`), a restarted pod skips runs whose `result.json`
already exists.

### 7.1 Experiment groups (from `run_now.sh`)

| Token | What | Free-rider ids | Purpose |
|---|---|---|---|
| **A** | IID honest baseline + reduced FR | 1,7 / 3,6 | honest floor + first evasion |
| **T** | honest band across CIFAR-100 decades (0–9…90–99) | none | show the band isn't an artifact of only using classes 0–9 |
| **D** | reduced +N data-budget spectrum | 6,8 | how little data is enough to evade |
| **E** | non-IID (Dirichlet) honest/reduced + α sweep + submarine | 3,6 / 1,7 | starvation & amplification |
| **EA** | non-IID **distribution-aware** trigger assignment | 3,6 / 1,7 | fairness fix + attack under it |
| **H** | **positive controls**: previous-models, gaussian | 3,6 | must be CAUGHT |
| **K** | the submarine (self-η, derived margin, dynamic warmup) | 3,6 / 1,7 | headline stealth attack (block2/head2) |
| **Y** | oracle-η ablation (J4) of K4 | 3,6 / 1,7 | how much self-estimation costs |
| **Z** | no-watermark control (λ=0, verifier on) | none | proves trig_acc≈0 is caused by the mark |
| **L** | **graftblock** head2 (main clean attack) | 3,6 / 1,7 | mark is output-layer-only |
| **F** | **FedIPR** mirror of A/H/K/L under `WM_SCHEME=fedipr` | 3,6 / 1,7 | same story for the 2nd scheme |
| **G** | **FedIPR sign (white-box)** mirror of A/H/K/L under `WM_SCHEME=fedipr_sign` | 3,6 / 1,7 | 3rd scheme: mark in the output-layer γ, read white-box (§4b) |

---

## 8. Results & analysis (what the figures mean, and what to write)

### 8.1 The "money plot": overlap of honest band and free-rider operating points
`plots.py overlap` (FareMark: `overlap_band_vs_fr`; FedIPR: `F_overlap_band_vs_fr`) puts the honest per-class
BER **band** (pooled over the honest decade runs) on one axis with every attacker's converged BER. **The
finding: the graftblock/submarine free-riders sit *inside* the honest band.** No horizontal threshold η
separates honest from free-rider without either missing free-riders (η loose) or flagging honest hard-class
clients (η tight). That is the detector failing.

### 8.2 The uploaded figures, read correctly
- **`F_H5_prevmodel_c100_fi`** (FedIPR positive control): free-rider BER pinned ≈ 1.0 (trigger acc ≈ chance),
  honest ≈ 0. Detector works on a *non-training* free-rider. ✔ expected.
- **`F_L1_graftblock_head2_c36_fi`** and **`F_L5_…_c17_fi`** (2 seeds): free-rider BER ≈ 0, on top of the
  honest line. A free-rider that trains only `head2` on `cpc=5` data **re-embeds the FedIPR backdoor** and is
  invisible. ✔ thesis. Same shape as FareMark's `L1/L5`.
- The tiny blue/orange bumps (~round 32, 44) are honest hard-class noise, well below either η — they underline
  the band/threshold tension.

### 8.3 Threshold dilemma, quantified
Report both operating points: `η_tight` (0.064 IID) catches lightly-trained free-riders but flags honest
hard-class clients (FPR↑); `η_loose` (0.264) keeps FPR≈0 but lets every graftblock/submarine free-rider
through (recall↓). There is **no** η with both FPR≈0 and recall≈1 once the adversary trains a little. Use the
`roc` plot (`roc_iid_vs_niid`) for the ROC/threshold-dilemma curve.

### 8.4 Cost of evasion
From `gpu_savings_*` / `effort_ratio_samples`: state the free-rider's fraction of honest samples and GPU-ms
per family (e.g. head2 + cpc5 taps + coasts → single-digit-percent duty cycle). The pitch: *undetected, and
nearly free.*

### 8.5 Non-IID amplification
Groups E/EA: under Dirichlet α the honest band widens (starvation) and the attack surface grows; the
`starvation_vs_alpha`, `savings_vs_alpha`, and `iso_compare` panels quantify how much easier evasion gets as α
shrinks, and whether distribution-aware assignment (EA) helps honesty without closing the attack.

### 8.6 FareMark vs FedIPR — the two-scheme comparison (your uploaded figures)

This is the heart of the paper: the *same* conclusion on two mechanically different output-layer schemes.
Read the FareMark figures (`A1`, `H5`, `J4_*`, `K4_*`, `K7_*`) against the FedIPR figures (`F_H5`,
`F_L1`, `F_L5`).

**(a) Both detectors work on a non-training free-rider — the controls pass.**
`H5_prevmodel_c100` (FareMark): free-rider mean BER ≈ 0.50–0.58, honest ≈ 0.03 — a clean gap, FR far above
`η_loose` (0.264). `F_H5_prevmodel_c100_fi` (FedIPR): free-rider BER ≈ 1.0, honest ≈ 0. The two "no-mark"
values differ (**0.5** for FareMark's bit-projection, **1 − 1/C ≈ 0.99** for FedIPR's trigger accuracy) but
the meaning is identical: a fabricated, never-trained model is trivially caught by both. **Lead with these** —
they prove the detectors are real before you break them.

**(b) A head-only / adaptive free-rider evades in *both* — but with two different geometries.**
- **FedIPR (`F_L1`/`F_L5`, graftblock head2, cpc5):** free-rider BER ≈ **0**, sitting exactly on the honest
  line (honest floor at the FR classes = 0.00). The backdoor is a memorization task, so training just the head
  (fc + one conv) on ~500 images drives trigger accuracy → 1 → ber → 0. The free-rider becomes **literally
  indistinguishable** from an honest client: honest = 0, FR = 0. This is the *degenerate* failure — **no
  η > 0 separates them, and η = 0 flags everyone.**
- **FareMark (`K4`/`J4`/`K7`, submarine block2, cpc5–10):** the free-rider lands **inside the honest band**
  rather than exactly on the mean. On **easy** classes (`K4_alldyn_block2_c17`) its self-estimated threshold is
  `η̂ = 0.000` — the attack discovers it needs essentially *no* training — so it coasts almost every round and
  its BER hugs 0–0.06, **below `η_tight` and below the honest mean (~0.08)**. On **hard** classes
  (`K4_alldyn_block2_c36`, `η̂=0.200`; `K7`, `η̂=0.155`) it floats at BER ≈ 0.11–0.17 — which is *exactly the
  honest hard-class floor* (see next point). Either way it is buried in the honest cloud.

**(c) The FareMark honest band already breaks `η_tight` with zero attackers — the strongest single fact.**
`A1_honest_c100` (6 seeds) reports per-class honest floors: cls6 = **0.110**, cls4 = **0.088**, cls7 = 0.056,
… down to cls9 = 0.001. **cls6 and cls4 sit above `η_tight` = 0.064 with no attack at all**, so the tight
threshold has honest false positives before any free-rider appears. And the submarine on hard classes parks at
BER ≈ 0.11 — *the same value as honest cls6*. Honest-cls6 and the free-rider are provably at the same operating
point: **no horizontal line separates them.** FedIPR reaches the same wall from the opposite side — its honest
band is tight (~0), but the free-rider is tight too (~0), so the gap is still zero.

**(d) The threshold dilemma, made numeric, for each scheme.**
- *FareMark:* `η_tight` (0.064) already flags honest cls6/cls4 → FPR > 0 with no attack; `η_loose` (0.264)
  gives FPR ≈ 0 but every submarine/graftblock free-rider (BER ≤ 0.26, mostly < 0.17) passes → recall ≈ 0.
  **No η gives both low FPR and high recall.**
- *FedIPR:* honest ≈ 0 and free-rider ≈ 0 → the separable region has **zero width**; the dilemma isn't a
  trade-off, it's a contradiction.

**(e) Adaptivity + cost.** The FareMark submarine plots show mostly **COAST** markers with scattered **TAP**s:
the free-rider trains only when its self-probe nears `η̂`, and `η̂` itself tracks class difficulty
(0.000 easy → 0.200 hard). So it spends the *minimum* effort to stay in the band — undetected **and** cheap
(quantify with `gpu_savings_*`, §8.4). FedIPR's graftblock doesn't even need the adaptivity: head2 training
trivially pins ber to 0 every round on a tiny shard.

**What to write from this.** The result is not "FareMark has a bug" or "FedIPR has a bug." Two schemes with
*different* mark mechanics (secret-key bit projection vs backdoor accuracy), *different* no-mark values
(0.5 vs ≈1.0), and *different* honest-band widths (wide vs ~0) **fail the same way**: a free-rider that trains
only the output head on a fraction of the data lands where honest clients already are, and no BER threshold
separates them without unacceptable honest FPR. That the failure is invariant to the scheme's internals is the
evidence that it is a property of **placing the watermark in the output layer**, not of any one construction —
which is exactly your hypothesis, generalised.

---

## 9. FareMark — flaws & missing pieces you should confront (you asked)

These are the gaps your paper stands on; also the ones a reviewer will push on for *both* schemes.

1. **Threat model is too narrow.** FareMark's (and FedIPR's) only free-rider does *zero* training (Eq. 17/18).
   Neither paper models a "lazy contributor" who trains a little. Your whole result is that this gap is fatal.
   Say it plainly and cite Eq. 17/18 as the modelled adversary.
2. **The mark is output-layer-only, so it is cheap to forge.** FareMark claims "the free rider cannot embed
   the watermark unless he contributes to the training." `graftblock/head2` refutes it: training ~21% of the
   tensors (the fc + one conv) on ~500 images re-embeds the mark. This is the single strongest sentence.
3. **No compute axis.** FareMark reports detection acc/FPR but never *how much compute* a free-rider must
   spend. Your compute meter shows evasion is cheap — a missing, decisive metric.
4. **Threshold is unfalsifiable as presented.** μ+3σ over benign BER assumes the benign distribution is known
   and clean; a lightly-trained free-rider *is* in that distribution. Report the ROC and show no η works.
5. **IID assumption hides starvation.** FareMark evaluates the detector on evenly-split data where every
   client has ample trigger-class images. Under non-IID a client can hold ~0 of its trigger class → honest BER
   floors high → η must widen → detection collapses *before* any attack. FareMark never tests this for
   detection.
6. **Blind trigger assignment.** `cid % n` round-robin is unfair under non-IID (some clients starved). Your
   distribution-aware assignment (EA) is a fix FareMark lacks — but it also shows the assignment is a free
   design lever, not a security boundary.
7. **Capacity / oversubscription.** When #clients > #classes (FareMark Table IX), clients share a trigger
   class and their marks are only separated by the secret key `M`/bits `B`, not by behaviour → same-class
   non-separability. Your `trigger_class_map` / `num_clients` knobs probe this; note that oversubscription
   makes an honest twin and a free-rider look identical on the shared class.
8. **Unembeddable-bit artifact.** Random ±1 keys have same-sign rows with non-zero probability → those bits
   can't be embedded → honest BER floors at ~`0.5·frac`. FareMark's random-`M` never accounts for it; your
   `unembeddable_fraction` + balanced-key option does. Include the honest-floor caveat.
9. **`sin` smoothing (Eq. 9) gives ~no smoothing** at the paper's α range (gain ≈ 1.0). Minor, but it's a
   reproducibility finding.
10. **Robustness ≠ evasion.** FareMark's robustness tests (fine-tune/prune/DP) are about an *owner* removing a
    mark from a stolen model — a different threat than a *contributor* evading detection. Don't let a reviewer
    conflate them; state the threat models side by side.

**FedIPR-specific — how to frame it (not gaps, just scoping):**
- You implement the **backdoor** half only, and that is correct for this thesis. FedIPR's feature-based
  (normalization-sign, white-box) watermark is a **weight-space** mark, not an output-layer one, and it is
  *designed* to survive exactly the head-only training your attack uses — so it is a different category, out
  of scope by construction. **State the claim as: output-layer / output-behaviour (black-box) watermarking
  cannot separate honest from adaptive free-riders.** One sentence saying the feature-based mark is out of
  scope pre-empts the reviewer question; you do **not** need to implement it.
- Trigger source (OOD/patch vs PGD) and target-label default (`cid` vs fixed 5) are **thesis-neutral** — they
  change how the backdoor embeds, not whether a head-only-trained model can re-embed it (see §4.1). If a
  reviewer wants exact-repo faithfulness, add one run with `FEDIPR_TARGET_MODE=fixed` +
  `FEDIPR_TRIGGER_SOURCE=folder`; it will not change the conclusion.
- **Recalibrate FedIPR η** from `F_A1_honest` before quoting numbers (currently provisional 0.20/0.50) — same
  `μ+3σ` of honest `ber_fedipr` rule as FareMark's η.
- Sanity-check that `F_H5`/`F_H6` truly sit near `1 − 1/C` (≈0.99 on CIFAR-100), not lower — if a fabricated
  model accidentally hits the target label often, your positive control is weak.

---

## 10. Glossary (every term you'll hit)

- **FL / FedAvg** — federated learning; server averages client weights (sample-weighted) each round.
- **Free-rider** — a client that wants the global model without contributing real training (Lin et al. 2019).
- **Watermark / mark** — a secret, verifiable signal the server checks to confirm a client trained.
- **Box-free / black-box** — verification reads only model *outputs* (FareMark, FedIPR-backdoor), never weights.
- **White-box / feature-based** — verification reads *weights* (FedIPR's normalization-sign mark; not implemented here).
- **BER (bit-error-rate)** — fraction of watermark bits recovered wrong. Honest ≈ 0, no-mark ≈ 0.5 (FareMark).
- **Trigger class (FareMark)** — the one task class whose softmax carries a client's bits.
- **Trigger set (FedIPR)** — private OOD/patched images with a secret target label.
- **η (eta)** — detection threshold on BER; benign iff BER < η. `tight`/`loose` = two frozen operating points.
- **η_T** — FedIPR trigger-set accuracy; `ber_fedipr = 1 − η_T`.
- **Smoothing f(), α** — `(p+ε)^α`, amplifies tail softmax probs so bits can be shaped (FareMark Eq. 7–9).
- **Memory-enhanced update, β** — Eq. 14 client-side momentum vs aggregation erosion.
- **λ (lambda)** — weight of the watermark loss in `L = L_cl + λ·L_wm`.
- **Warmup (W) / calib (K)** — honest rounds before defection; last K calibrate the attacker's self-η.
- **Tap / coast** — submarine trains (tap) vs resubmits without training (coast).
- **Scope (head2/block2/full)** — which trailing parameter tensors an attacker trains/keeps.
- **cpc / `tap_data_cpc` / `autop_common_per_class`** — images per common class in the reduced shard.
- **Graft** — replace the model body with the exact current global, keep only the trained head.
- **Distribution-aware assignment** — server gives each client a trigger class it actually holds a lot of (non-IID fairness).
- **Dirichlet(α)** — non-IID label-skew partition; small α = severe skew.
- **cXX in family names** — free-rider client ids (c36 = cids 3,6; c17 = cids 1,7), **not** a dataset.
- **c100** — CIFAR-100. **aXX** — Dirichlet α. **rep<seed>** — repeat/seed.
- **duty cycle** — fraction of rounds an attacker actually trained (evasion-cost signal).

---

## 11. Before you write the paper — checklist

- [ ] **Recalibrate FedIPR η** from `F_A1_honest` (μ+3σ of honest `ber_fedipr`); replace the provisional 0.20/0.50.
- [ ] **Pin the GPU** (`RUNAI_EXTRA="--node-pools a100-80"`) so `gpu_ms` is comparable; otherwise report `samples`.
- [ ] **Seeds:** honest baselines use 6 seeds (0–5), attacks 3 (0–2). Say so; report mean ± std (the aggregated figures already do "over N seeds").
- [ ] **State both threat models** (owner-removal vs contributor-evasion) and scope your claim to output-behaviour marks.
- [ ] **Report the compute axis** (effort_ratio / duty_cycle) next to detection — it's your novel metric.
- [ ] **Show the ROC / no-η-works figure**, not just single-threshold timelines.
- [ ] **Positive controls first** (H/F_H at the no-mark line) so the reader trusts the detector before you break it.
- [ ] **List the deviations** in §3.1 and §4.1 in a "Reproducibility & Faithfulness" paragraph.
- [ ] **Scope the claim to output-layer / output-behaviour (black-box) watermarks** and say the feature-based, white-box FedIPR mark is out of scope by construction (one sentence — do not implement it).
- [ ] **Confirm the un-uploaded files** (`manifest.py`, `runlog.py`, `resultio.py`, `paper_check.py`, `submit_pool.sh`) are in the run repo and match these assumptions.
- [ ] *(optional, only if a reviewer asks)* one exact-repo FedIPR run (`FEDIPR_TARGET_MODE=fixed`, `FEDIPR_TRIGGER_SOURCE=folder`) to show the trigger/label defaults don't change the result.