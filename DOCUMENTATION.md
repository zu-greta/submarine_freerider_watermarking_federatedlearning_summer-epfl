# Watermarking is an Effort - output layer watermaking weaknesses — Technical Documentation

**Project:** Adapting FedIPR/Faremark output layer watermark schemes and demonstrating that **output-layer / box-free watermarking cannot reliably detect free-riders**.

---

## 0. TL;DR — what this project claims and how it proves it

Three federated-learning (FL) watermarking schemes place their mark **in the output layer** — the first two
in the output *behaviour* (what the net predicts on trigger inputs), the third in the output-layer *weights*:

| Scheme | Verify | Where the mark lives | Detection statistic | "no mark" value |
|---|---|---|---|---|
| **FareMark** (Li et al., IEEE IoT J. 2025) | black-box | softmax probabilities on **trigger-class** images, projected through a secret ±1 key into `m` bits | **BER** vs registered bits | `0.5` |
| **FedIPR backdoor** (Li et al., TPAMI 2022) | black-box | argmax label on a private **trigger set** of images carrying a secret target label | `ber = 1 − trigger_set_accuracy` | `1 − 1/C` (≈`0.99` on CIFAR-100) |
| **FedIPR sign, forced to output layer** (this project, §4b) | **white-box** | the **signs** of an **output-layer scale vector** (last BN γ, in `head2`), read from the weights | `ber = Hamming(sign(γ·E), B)/N` | `0.5` |

**Project's thesis:** because the mark is a pure **output-layer** phenomenon, a free-rider only has to train the last layers on a reduced amount to re-embed it. The result is a model whose BER sits inside the honest band (BER ≈ 0), indistinguishable from an honest client by any BER threshold — while the free-rider spends a small fraction of honest compute and never contributes a real model. We build three such attackers (`reduced`, `graftblock`, `adaptive_tap`/"submarine") and show they defeat both FareMark and FedIPR watermarking schemes.

---

## 1 Files in this project (not limited)

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
| `src/models.py` | model | ResNet-18 (CIFAR stem) + SmallCNN; the 62 named tensors the attack scopes count from |

---

## 2. Datasets — full explanation

### 2.1 The task datasets

`src/datasets.py` wraps torchvision. Supported: **MNIST** (10 cls, 1×28×28), **CIFAR-10** (10 cls, 3×32×32), **CIFAR-100** (100 cls, 3×32×32), and — **Food-101** (`food101`, ETH-Zürich, 101 cls), natural photos resized to `FOOD_SIZE`² (default 64×64) with ImageNet normalisation.
Train aug for CIFAR is `RandomCrop(32, padding=4)` + `RandomHorizontalFlip`; for both food sets it is `RandomResizedCrop(FOOD_SIZE)` + flip (test = `Resize`+`CenterCrop`).

### 2.1b Switching datasets (the one-line switch) + Food-101 download

**Switch.** `runbook.sh` and `run_now.sh` each have a `DATASET` switch at the very top
(`export DATASET="${DATASET:-cifar100}"`, choices `cifar100 | cifar10 | mnist | food101 | food100`). Flip it
once and the whole suite (every group, every family) runs on that dataset — it is exported, so
`submit_experiment.sh` adds `--dataset $DATASET`, which overrides config 14's dataset. Ready-made configs also
exist: **15** = `attack_base_resnet18_food101` (Food-101)

**Recalibrate η.** Honest BER floors are dataset-specific, so re-freeze `ETA_T*/ETA_L*` (and the FedIPR
`ETA_*_FI` / sign `ETA_*_WS`) from the new honest baseline (`A1_honest` / `F_A1` / `G_A1`) before quoting numbers.

**Downloading Food-101 (`food101`).** Automatic: torchvision `Food101(..., download=True)` pulls a ~5 GB
tarball to `DATA_ROOT/food-101` on first use (750 train + 250 test per class). `DATA_ROOT` is on the PVC
(`$MOUNT/home/zu/data`), so **the first pod downloads it once and every later run reuses it** — that pod needs
internet egress and a few GB of PVC space. Air-gapped: pre-stage the extracted `food-101/` folder.

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

### 2.3 Trigger data 

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

When an attacker "taps" (trains a little), it does **not** use its full shard. `_SimpleFRMixin._prepare` builds a reduced loader = **all its trigger-class images** + **`cpc` images per common class** (`cpc` = `autop_common_per_class` / `tap_data_cpc`, default 5). So `cpc=5` on CIFAR-100 = trigger-class images + 5×99 ≈ 495 common images, a fraction of the ~5 000-image honest shard. `cpc=-1` = full shard; `cpc=0` = trigger images only.

### 2.5 Naming notes

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

### 3.1 FareMark paper deviations

**Faithful:** the grouping/first-`m·l`-outputs rule, power-smoothing `x^α`, ±1 random key (`make_key`,
`balanced=False` = paper's random `M`), full-softmax projection (no trigger-class column excluded by default),
BCE embed loss, memory update Eq. 14 with β=0.6, λ=5, `η=μ+3σ`, and the two paper free-rider baselines
(`PreviousModelsFreeRider` = `2W_t − W_{t−1}` Eq. 17; `GaussianNoiseFreeRider` = `W_t + N(0,σ²)` Eq. 18).

**Deviations:**
- **η is frozen offline**, not recomputed live. `wm_verify.make_verifier` uses a pre-calibrated constant
  `WM_ETA_FIXED` = `μ+3σ` of honest BER over the honest baseline runs (last 20 rounds, pooled over the honest
  seeds); the live `μ+3σ` code path is present but commented out. This makes the threshold a stable, fair
  fixed operating point instead of a moving target, but it is a methodological choice the reader must know.
- **Two frozen η's are reported**, not one: `η_tight = 0.064` (μ+3σ over per-round *means*) and
  `η_loose = 0.264` (μ+3σ over per-*client* BERs). This is the project's own "threshold dilemma" framing, not
  in the paper. Non-IID uses `0.161 / 0.576`.
- **`label_smoothing=0.1`** on the task CE and **`clip_grad_norm_(…,5.0)`** are added for stability; not in the
  paper.
- **Datasets/models actually run:** CIFAR-100/ResNet-18 only in these groups (the paper also does
  MNIST/CIFAR-10/Food100, AlexNet/ShuffleNet/GoogLeNet). 

---

## 4. FedIPR backdoor scheme (`src/watermark_fedipr.py`) — mechanism & paper mapping

FedIPR's paper has *2 watermarks: a **feature-based** one (a binary string embedded in *normalization scale weights*, read white-box via `sign()`), and a **backdoor** one (a private trigger set → target label, read black-box). 

**Mechanism:**
- **Registration** `G()`: each backdoored client owns a private trigger set `T_k = {(X_T, y_T)}` of
  `num_trigger` images, all carrying a secret target label.
- **Embedding** `E()`: add `L_T = CE(f(X_T), y_T)` to the task loss with `α=1` via **batch poisoning** —
  trigger samples are *concatenated into each normal training batch*, not trained in a separate pass
- **Verification** `V_B()`: detection rate `η_T = mean(argmax f(X_T) == y_T)`; mark present iff
  `η_T ≥ 1 − ε_B` (Eq. 4/8). This project maps it to the shared pipeline via **`ber_fedipr = 1 − η_T`**
  (`detect_ber` / `ber_from_acc`), so every downstream plot/threshold path is reused unchanged.
- **Free-rider**: an untrained/fabricated model classifies triggers at chance (`1/C`) → `η_T` collapses →
  `ber ≈ 1 − 1/C` → flagged.

### 4.1 FedIPR repo (`purp1eHaze/FedIPR`) deviations

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

**Deviations:**
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

## 4b. FedIPR feature-based sign watermark — 1+ normalization layers (scheme C, WHITE-BOX)

**Goal.** FedIPR's *feature-based* watermark hides a bit-string in the **signs of normalization scale weights** (`W_γ`), read **white-box** from the weights. The server decides **how many layers** carry the mark (`fedipr_sign_layers` / `fedipr_sign_carrier`), and this is the whole experiment:

- **`fedipr_sign_layers = 1`** (output layer only, `net.layer4.1.bn2.weight`, inside `head2`) → the **fragile**
  case: the head2 free-rider trains that one layer and forges its bits → `ber ≈ 0` → **evades**.
- **`fedipr_sign_layers = N > 1`** → the `N` output-most normalization scales; the extra `N−1` live in the
  **body, outside `head2`**. 

**Mechanism**
- *Carriers* = an ordered list chosen by the **server**: `auto_last_bn` + `fedipr_sign_layers=N` → the last
  `N` normalization scales (output→body); `all_bn` → every scale; or an explicit `"a,b,c"` name list.
- *Registration* = per **carrier** `i`, each client gets a secret matrix `E_{k,i} ∈ ℝ^{C_i×N_i}` and bits
  `B_{k,i}`. `N_i` = `fedipr_sign_bits` per layer, **auto-clamped** to `C_i // K` so all `K` clients can embed
  in that shared layer (FedIPR Thm. 1 capacity) — `plan_bits`.
- *Embedding* = task CE **+** `λ·L_sign`, where `L_sign` = **mean over carriers** of the hinge sign-loss
  `mean_j max(margin − b'_j·(γ_i·E_i)_j, 0)` (`clients._local_train_fedipr_sign`). Honest trains the full model
  → all carriers move → every bit embeds. A scope-frozen free-rider only moves carriers in its scope.
- *Verification* = **white-box**: read every `γ_i` from the submitted weights (no forward pass), extract
  `sign(γ_i·E_i)` per layer, `ber = (Σ wrong bits)/(Σ bits)` (`wm_verify.py` `fedipr_sign` branch +
  `watermark_fedipr_sign.sign_ber`). Honest → `ber ≈ 0`; fabricated/untouched → `ber ≈ 0.5` (per-bit chance —
  **same as FareMark**, so it reuses the identical threshold/plot pipeline; `0.5`, not `1 − 1/C`).

**The free-rider.** The very same `graftblock` / `adaptive_tap` attackers — no new attacker code. `head2`
contains only the last normalization scale, so it can re-embed just carrier 0; every deeper carrier is
frozen at the honest global. The rule the server exploits: place carriers OUTSIDE the free-rider's scope.
(`head2` = last norm scale; `block2` reaches further; `full` = all — so against a `block2` free-rider the
server needs carriers deeper still.)

---

## 4c. theory

### all 3 schemes are forgeable by a head-only free-rider

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
   to the head weights). 
   
### The theorem

> **An output-layer watermark is forgeable by a free-rider when three conditions hold:**
> 1. **Location** — the mark's evidence is confined to, or controlled by, the output layer (the last few tensors).
> 2. **Free body** — the expensive feature-extracting body is handed to the free-rider for free by FedAvg aggregation.
> 3. **Own key** — the free-rider is a registered client, so it holds its own secret (trigger class / trigger set / sign key) and can re-embed *its own* valid mark.
>
> Then a head-only free-rider, training on a fraction of the data, produces a model with a valid watermark
> that lands **inside the honest BER band** → no threshold η separates honest from free-rider without
> unacceptable false positives.

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

### 5.3 `graftblock` (groups **L**, **F_L**) — the clean output-layer attack (our attack)
After warmup, every round: switch to the reduced shard **and freeze all but the last few parameter tensors**
(`tap_scope`), then train. Scope map (`_SCOPE_KEEP`): `head2` = last 5 tensors (softmax `fc` + the conv before
it, ~2.4M params ≈ 21% of ResNet-18); `block2` = last 20 tensors (~80%); `head`=2; `full`=all. Optional
`graft`: after training, overwrite the body with the *exact current global* and keep only the freshly-trained
head (so the body is provably the honest global; only the head moved). **This is the cleanest proof that the
mark is output-layer-only:** training just the head (`head2`) re-embeds it, BER→0, evading detection. The
uploaded `L1/L5` and `F_L1/F_L5` figures are exactly this.

### 5.4 `adaptive_tap` (the "submarine", groups K/Y/E/EA/F_K) — appendix ignore for now
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
concurrency 1 and on the same GPU model — the code records `gpu_concurrency` / `gpu_name` 

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
  `p90/max`). 

---

## 7. Run instructions

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
---

## 8. Glossary 

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