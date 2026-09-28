# DOCUMENTATION 
# Head-Only Watermarking Cannot Stop Free-Riders in Federated Learning — Technical Documentation

**Project:** We adapt the FareMark and FedIPR FL watermark schemes and show that **head-only watermarking**, where the mark lives in a small designated subset of parameters (the *head*), cannot reliably detect free-riders. A free-rider that re-trains only the head, on very little data, passes verification at a fraction of the honest cost.

---

## 0. TL;DR 

Three federated-learning (FL) watermarking schemes place their mark in the head of the network. The first two put it in the output behaviour (what the net predicts on trigger inputs), the third in the output-layer weights:

| Scheme | Verify | Where the mark lives | Detection statistic | "no mark" value | Attack |
|---|---|---|---|---|---|
| **FareMark** | box-free | softmax probabilities on trigger-class images, projected through a secret ±1 key into `m` bits | BER vs registered bits | `0.5` | head-only FR attack |
| **FedIPR white-box, forced to output layer** | white-box | the signs of an output-layer scale vector (last BN γ, in `head2`), read from the weights; optionally spread over N BN layers | `ber = Hamming(sign(γ·E), B)/N` | `0.5` | head2 FR attack; layer sweep as potential solution |
| **FedIPR black-box** | black-box | argmax label on a private trigger set of images carrying a secret target label | `ber = 1 − trigger_set_accuracy` | `1 − 1/C` (≈`0.99` on CIFAR-100) | appendix - head-only FR attack |

**Goal:** prove that because a head-only watermark is checked on the head alone, and the head is a tiny part of the model, a free-rider therefore only has to train the head, on a reduced shard, to re-embed its own mark, while FedAvg hands it the expensive body for free. The resulting model has a BER inside the honest band (≈ 0 for the white-box sign, at the honest per-class floor for FareMark). No BER threshold separates it from honest clients, yet the free-rider spends ≈0.3× of the honest compute and never contributes a real model.

**Current draft numbers** (CIFAR-100 / ResNet-18, 3 seeds unless stated):
- Cost (Table I): FareMark `head` FR = **0.305×** honest samples / **0.325×** GPU; FedIPR-sign `head2` FR = **0.298×** / **0.297×**. Food-101 / ResNet-50 (Table II, appendix): 0.280× / 0.263× and 0.272× / 0.254×.
- Global accuracy: all-honest without watermark 73.5%; with watermark 73.25 ± 0.17%; with two head-only free-riders 71.70 ± 0.02% (easy 1,7) and 71.75 ± 0.12% (hard 3,6).
- Baselines (previous-models, Gaussian) sit at chance (≈0.5) and are caught; the head-only FR sits on the honest floor.
- Defence direction (`fig4_detect_cost`): spreading the sign mark over N = 1 / 6 / 20 BN layers catches a *fixed* head2 FR. An *adaptive* FR that widens its scope still evades, but its cost climbs to ≈0.53× on a reduced shard and ≈1.0× (honest cost) on full data.
- Appendix ROC (FareMark, pooled FRs vs honest): AUC 0.55, only 4% TPR at 5% FPR; class-normalized AUC 0.39.

---

## 1 Files in this project (not limited)

| File | Layer | Responsibility |
|---|---|---|
| `src/config.py` | config | `ExpConfig` dataclass, the `CONFIGS` list (0–15), every knob's default + doc; `MODEL` env overrides the backbone |
| `src/datasets.py` | data | load MNIST / CIFAR-10 / CIFAR-100 / Food-101, IID + Dirichlet non-IID partition into per-client loaders |
| `src/fast_data.py` | data | GPU-resident `FastLoader` (`FAST_DATA=1`); Food-101 JPEGs|
| `src/clients.py` | clients | honest `Client`, `WatermarkClient` (all three schemes), paper baselines, and the attackers (`reduced`, `graftblock` = **head-only**, `adaptive_tap` = submarine) |
| `src/watermark.py` | scheme A | **FareMark** box-free softmax-projection scheme (Eq. 1–16) |
| `src/watermark_fedipr_sign.py` | scheme C | **FedIPR feature-based sign** watermark in 1+ BN scales (**white-box**) |
| `src/watermark_fedipr.py` | scheme B | **FedIPR** backdoor trigger-set scheme (black-box) |
| `src/wm_verify.py` | server | registry + per-round verification hook for all 3 schemes |
| `src/server.py` | server | FedAvg aggregation + round loop (+ free-rider phase-change notes) |
| `src/compute_meter.py` | metering | per-client samples / GPU-ms / FLOPs accounting - cost |
| `src/runlog.py` | logging | `run.log` layout: run banner, config diffs vs defaults, data / watermark / free-rider setup blocks, per-round table, closing report |
| `src/models.py` | model | ResNet-18/34/50 (`ResNetX`, CIFAR or ImageNet stem) + SmallCNN; the named-tensor order the attack scopes count from |
| `src/utils.py` | util | seeding (+ determinism switch), logging, accuracy |
| `scripts/run_experiment.py` | entry | one `(config, seed)` run -> `result.json` |
| `scripts/to_pgfplots.py` | analysis | paper figures + Table I -> `export/data/*.dat` + `export/fig/*.tex` (+ `all_figures.tex` menu); `--appendix` for the appendix set |
| `scripts/paper_figs_mpl.py` | analysis | matplotlib (PNG) twins of the same paper/appendix figures + `tab1_costs.csv` |
| `scripts/plots.py` | analysis | legacy per-family diagnostics (`./runbook.sh plot-legacy`) |
| `infra/run_now.sh` | orchestration | builds `jobs.tsv` (the experiment manifest) for groups A, T, D, E, EA, H, K, Y, Z, L, F, G, FD, HS |
| `infra/runbook.sh` | orchestration | phase driver: manifest -> submit -> plot / appendix / appendix-food |
| `infra/submit_experiment.sh` | orchestration | one RunAI/Kubernetes job submission (or one manifest row with `DRYRUN=1`); writes `pod.log` |
| `infra/submit_pool.sh` | orchestration | replays `jobs.tsv` over `PODS × WORKERS` |

Per run, the output dir holds `pod.log` (machine, commit, flags), `run.log` (the experiment record) and `result.json` (the data).

---

## 2. Datasets 

### 2.1 The task datasets

`src/datasets.py` wraps torchvision. Supported: **MNIST** (10 cls, 1×28×28), **CIFAR-10** (10 cls, 3×32×32), **CIFAR-100** (100 cls, 3×32×32), and **Food-101** (`food101`, ETH-Zürich, 101 cls, 750 train + 250 test per class), which is natural photos resized to `FOOD_SIZE` (default 64×64) with ImageNet normalisation.
Train aug for CIFAR is `RandomCrop(32, padding=4)` + `RandomHorizontalFlip`; for Food-101 it is `RandomResizedCrop(FOOD_SIZE, scale=(0.7,1))` + flip (test = `Resize(1.15·FOOD_SIZE)` + `CenterCrop`). With `FAST_DATA=1` the same crop/flip is done on the GPU (Food-101 pad = `FOOD_FAST_PAD`, default 8).

**The main experiments run on CIFAR-100** (`config_idx 14`, `attack_base_resnet18_cifar100`,
ResNet-18, `num_clients=10`, 50 rounds, `local_epochs=5`, `lr=0.01`, `batch_size=16`, SGD momentum 0.9,
weight-decay 5e-4). CIFAR-100 = 50 000 train / 10 000 test images. Food-101 / ResNet-50 (config 15) is the
appendix generalisation check.

### 2.2 How data is split across the 10 clients

- **IID** (`partition=iid`, the default and most groups): shuffle all training indices with a seeded RNG and `np.array_split` them into 10 near-equal shards (~5 000 images on CIFAR-100). Every client sees all classes roughly uniformly.
- **Non-IID** (`partition=dirichlet`, groups E/EA, appendix): for each class, draw a `Dirichlet(alpha)` vector over the 10 clients and hand out that class's images in those proportions (Hsu et al. 2019 label skew). `alpha=0.5` is the standard FL non-IID benchmark, `alpha=0.1` is severe skew and `alpha=1.0` is milder. `datasets.py` reshuffles each shard afterwards. With small α a client may hold **very few or zero** images of its assigned trigger class. This is the "starvation" the non-IID groups probe. `run.log` warns when shard sizes vary by more than 3×.

### 2.3 Trigger data 

1. **FareMark trigger class**: Each client is assigned one existing task class as its
   "trigger class". The default is `cid % num_classes`, so cids 0–9 -> classes 0–9 on CIFAR-100. `TRIGGER_CLASS_MAP` overrides it (group T), and `wm_trigger_assign=distribution` makes the server give each client a class it holds a lot of (group EA).
   The mark is embedded by shaping the softmax on images of that class that already live in the client's shard.
   Verification uses a held-out bank of test-set images of that class (`build_trigger_bank`, `N_T=50`), or
   per-client variants (`wm_trigger_mode` = `class` | `client` | `client_train`, mirroring FareMark §V-F3).

2. **FedIPR sign (white-box)**: The mark is read from the weights 

3. **FedIPR backdoor trigger set**: separate images with a secret target label. Built once per client
   in `watermark_fedipr.build_client_triggersets`, 40 images/client by default, disjoint per client. Sources
   (`fedipr_trigger_source`):
   - `noise`: self-contained low-frequency RGB noise (no download). **This is the source used for all our FedIPR backdoor runs** and the `run_now.sh` default for groups F and HS.
   - `indist` (`ExpConfig` default): real CIFAR test images, each stamped with a per-client solid-colour
     **BadNets patch** in a corner (`_stamp_patch_`), then normalised.
   - `svhn`: real OOD images (SVHN test split), the analogue of the FedIPR repo's `trigger/pics` folder.
   - `folder`: `torchvision.ImageFolder` at `fedipr_trigger_dir` 

### 2.4 The free-rider's "reduced shard"

When an attacker trains, it does not use its full shard. `_SimpleFRMixin._prepare` builds a reduced loader
`D̂ = trig(D_i) ∪ (d imgs / common class)`: **all its trigger-class images** + **`cpc` images per common class**
(`cpc` = `autop_common_per_class` / `tap_data_cpc`, default 5). So `cpc=5` on CIFAR-100 = trigger-class images
+ 5×99 ≈ 495 common images, about 10% of the ~5 000-image honest shard. `cpc=-1` = full shard; `cpc=0` = trigger images only. `autop_n_common_classes=K` restricts the common images to K random classes.
- FedIPR backdoor (`_prepare_fedipr`): the client's own trigger set is split into an embed slice + a held-out probe slice.
- FedIPR sign (`_prepare_sign`): there are no trigger images; the shard is just `cpc` images per class, and with `cpc=0` the free-rider embeds its sign bits with no task data at all.

### 2.5 Naming notes

The family tags are not dataset names. In `A3_reduced_c100_c36`, `c100` = CIFAR-**100** and `c36` = the
free-rider **client ids are 3 and 6** (so their trigger classes are 3 and 6). `c17` = cids 1,7, which are the "easy" classes; 3,6 are the "hard" classes. **Every free-rider run uses cids 3,6 or 1,7.** `aXX` = Dirichlet α (`a01`=0.1, `a10`=1.0).
Suffixes: `_fi` = FedIPR backdoor, `_ws` = FedIPR sign, `_L<N>` = N sign carrier layers, `clsXXYY` = trigger-class decade (group T). 

---

## 3. FareMark scheme (`src/watermark.py`) 

FareMark is **box-free**: verification reads only the model's **softmax outputs**.

**Pipeline:**
1. Take the `n`-dim softmax `P` on a trigger-class image. Use only the first `m·l` outputs, split into `m`
   groups of size `l = n // m` (`grouping`). On CIFAR-100, `m = n//10 = 10` and `l = 10`.
2. **Smooth** each probability with `f()` so the argmax doesn't dominate the projection: `f(p)=(p+ε)^α` with
   `0<α<1` (default `α=0.4`, `ε=SMOOTH_EPS=1e-3`), Eq. 7–9 (`smooth`). `wm_f=sin` is Eq. 9 and is guarded to
   `0<α≤π/2` with a real smoothing gain. Cross-entropy makes the softmax steep (one class ≈1); smoothing amplifies the tail so it can carry bits.
3. **Project** each group onto a per-client pseudo-random ±1 key row `M`: `z_k = Σ_j f(p_{k,j})·M_{k,j}`
   (Eq. 1/13, `project_logits`).
4. **Bit** `b_k = 1 if z_k ≥ 0 else 0` (Eq. 2, `extract_bits` after averaging over `N_T` samples, Eq. 15).
5. **Embed** by adding `L_wm = BCE(z, target_bits)` to the task loss: `L = L_cl + λ·L_wm` (Eq. 11–12,
   `watermark_loss`, `λ=5`). Only trigger-class samples in a batch contribute `L_wm`.
6. **Memory-enhanced update** (Eq. 14): `W_new = β(memory + δ) + (1−β)·W_global`, `δ = W_sgd − W_global`,
   `β=0.6` (`WatermarkClient._memory_update`). Counteracts the averaging that erodes the mark.
7. **Detect**: `BER = (1/m)Σ|b̂_k − b_k|`; benign iff `BER < η` (Eq. 16). The paper's threshold is `η = μ + 3σ` over benign BER (`calibrate_eta`).

---

## 4. FedIPR backdoor scheme (`src/watermark_fedipr.py`) 

FedIPR's paper has two watermarks: a **feature-based** one (a binary string embedded in *normalization scale weights*, read white-box via `sign()`) and a **backdoor** one (a private trigger set ->target
label, read black-box), described in this section. The backdoor is fully implemented and runs through the shared pipeline, but its figures (`fig1_fedipr_timeline*`) are currently used for appendix only.

**Mechanism:**
- **Registration** `G()`: each backdoored client owns a private trigger set `T_k = {(X_T, y_T)}` of
  `num_trigger` images, all carrying a secret target label.
- **Embedding** `E()`: add `L_T = CE(f(X_T), y_T)` to the task loss with `α=1` via **batch poisoning**: 
  trigger samples are *concatenated into each normal training batch*, not trained in a separate pass.
- **Verification** `V_B()`: detection rate `η_T = mean(argmax f(X_T) == y_T)`; the mark is present iff
  `η_T ≥ 1 − ε_B` (Eq. 4/8). This project maps it to the shared pipeline via **`ber_fedipr = 1 − η_T`**
  (`detect_ber` / `ber_from_acc`), so every downstream plot/threshold path is reused unchanged.
- **Free-rider**: an untrained/fabricated model classifies triggers at chance (`1/C`), so `η_T` collapses,
  `ber ≈ 1 − 1/C`, and the model is flagged.

## 4b. FedIPR feature-based sign watermark — 1+ normalization layers (scheme C, WHITE-BOX)

**Goal.** FedIPR's *feature-based* watermark hides a bit string in the **signs of normalization scale weights** (`W_γ`), read **white-box** from the weights. The **server** decides **how many layers** carry the mark (`fedipr_sign_layers` / `fedipr_sign_carrier`), and that choice is the experiment:

- **`fedipr_sign_layers = 1`** (output layer only: `net.layer4.1.bn2.weight` on ResNet-18, `layer4.2.bn3.weight` on
  ResNet-50, inside `head2`) -> the **fragile** case, used in the main figures. The `head2` free-rider trains
  that one layer and forges its bits -> `ber ≈ 0` -> **evades**. (`head` = fc only does *not* contain the carrier, which is why the sign scheme is attacked with `head2` while FareMark is attacked with `head`.)
- **`fedipr_sign_layers = N > 1`** -> the `N` output-most normalization scales; the extra `N−1` live in the
  **body, outside `head2`**. The layer sweep uses N = 1 / 6 / 20 on ResNet-18, matching the free-rider scopes
  `head2` (covers 1), `block2` (covers 6) and `full` (covers all 20).

**Mechanism**
- *Carriers* = an ordered list chosen by the **server**: `auto_last_bn` + `fedipr_sign_layers=N` -> the last
  `N` normalization scales (output->body); `all_bn` -> every scale; or an explicit `"a,b,c"` name list
  (`resolve_carrier_names`).
- *Registration* = for each **carrier** `i`, each client gets a secret matrix `E_{k,i} ∈ ℝ^{C_i×N_i}` and bits
  `B_{k,i}`. `N_i` = `fedipr_sign_bits` (default 40) per layer, **auto-clamped** to `C_i // K` so all `K` clients can embed in that shared layer (FedIPR Thm. 1 capacity; `plan_bits`).
- *Embedding* = task CE **+** `λ·L_sign` (`fedipr_sign_lambda=1.0`), where `L_sign` = **mean over carriers**
  of the hinge sign-loss `mean_j max(margin − b'_j·(γ_i·E_i)_j, 0)` (`margin=0.1`, Eq. 19; `sign_embed_loss`,
  `clients._local_train_fedipr_sign`). An honest client trains the full model, so all carriers move and every bit embeds. A scope-frozen free-rider only moves the carriers inside its scope.
- *Verification* = **white-box**: read every `γ_i` from the submitted weights (no forward pass), extract
  `sign(γ_i·E_i)` per layer, `ber = (Σ wrong bits)/(Σ bits)` (`wm_verify.py` `fedipr_sign` branch +
  `watermark_fedipr_sign.sign_ber_from_state`). Honest -> `ber ≈ 0`; fabricated/untouched -> `ber ≈ 0.5` (per-bit chance, **same as FareMark**, so it reuses the identical threshold/plot pipeline). The verifier logs `trig_acc = 1 − ber` for this scheme.
- The white-box BER is much cleaner than FareMark's (no sampling noise, no softmax projection, no class 
  difficulty): honest and head2 free-rider both sit at ≈ 0.

---

## 5. The attackers (`src/clients.py`) 

All attackers subclass `WatermarkClient` (so they can embed any of the three marks) via `_SimpleFRMixin`, and share a schedule: **honest warmup** `[1, W)`, a **calibration window** (the last `K` warmup rounds), then **free-ride** from round `W` (`autop_honest_until=W=12`, `autop_calib_rounds=K=4`). Phase changes are logged in `run.log` (`free-rider cidX -> TAP/COAST/...`) and stored per round in the client's `trace`.

### 5.1 Paper baselines (positive controls)
- `PreviousModelsFreeRider` (Eq. 17): resend `2W_t − W_{t−1}` and never train. BER ≈ chance. Groups **H5 / F_H5 / G_H5**.
- `GaussianNoiseFreeRider` (Eq. 18): `W_t + N(0,σ²)` (`σ=0.1`, optional `noise_decay`). Groups **H6 / F_H6 / G_H6**.
These validate that the detector works at all. In figs 2–3 they sit at ≈ 0.4–0.7, well above the honest floor.

### 5.2 `reduced` (groups A/D/E/EA) 
After warmup, train exactly like an honest client (full model) **but on the reduced shard** (`cpc` images/class).
This re-embeds the mark every round on a fraction of the data. It is the simplest evasion and shows the mark survives on tiny data.
Group D sweeps `cpc` ∈ {−1, 0, 1, 2, 5, 10}.

### 5.3 `graftblock` (groups **L**, **G**, **HS**, **FD**, **F_L**) — head-only free-riding (our attack, paper Alg. 1)
After warmup, every round: switch to the reduced shard `D̂` (or the full shard with `cpc=-1`), **freeze all but the last
few parameter tensors** (`tap_scope`), then run the normal `WmUpdate` (task + watermark loss + memory update). The body stays
at the current global, so the submission tracks the moving global model while only the head is trained.

Scope map (`_SCOPE_KEEP`, counted from the end of `named_parameters()`; ResNet-18 has 62 tensors, ~11.2M scalars):

| `tap_scope` | tensors | ResNet-18 content | used for |
|---|---|---|---|
| `head` | 2 | `fc.weight, fc.bias` (softmax / output layer only) | **FareMark main attack** (L6/L7) |
| `head2` | 5 | `layer4.1.conv2.weight`, `layer4.1.bn2.{weight,bias}`, `fc.{weight,bias}` (~2.41M, ~21%) | **FedIPR-sign main attack** (G_L1), FareMark L1/L5 |
| `block` | 8 | last conv/BN pairs + fc | not used |
| `block2` | 20 | last ~2.5 residual blocks + fc (~9.04M, ~80%) | adaptive sign FR at N=6; submarine K4 |
| `full` | all | the whole model (= honest compute path) | adaptive sign FR at N=20 |


### 5.4 `adaptive_tap` (the "submarine", groups K/Y/E4/EA3) — appendix
It free-rides between "taps", training only when it must:
- **Threshold estimation** (`tap_eta_source`): `oracle` (handed the true η, group Y / J4) or **`self`**, which estimates `η̂ = μ + k·σ` (`tap_eta_k=3`) over its *own* calib-window self-probe BERs. In `self` mode it never sees the server η.
- **Tap target** (`tap_margin_mode`): `fixed` -> `target = η̂ − margin`; `derived` -> `margin = k·σ(own probe
  BER)`, so the safety gap widens when its own estimate is noisy.
- **Decision each round** (`tap_when=threshold`): probe the model it *would* submit if it coasts. If that BER is above the target, **tap** (train its scope on the reduced shard); otherwise **coast** (submit without training). The safety cap `tap_max_coast` forces a periodic tap.
- **Coasting** (`tap_coast_mode`): `decay` = resubmit its own last-tapped weights; `graft` = fresh global body + its frozen mark head, optionally blended toward the global by `tap_graft_decay` to kill tail spikes.
- **Dynamic warmup** (`tap_warmup_mode=dynamic`): instead of a fixed defect round, defect once its own probe
  BER has *converged* (flat within `tap_conv_eps` for `tap_conv_patience+1` rounds), bounded to 
  `[tap_honest_min, tap_warmup_cap]`.

---

## 6. Run instructions

`infra/runbook.sh` is the driver. Env knobs: `BATCH` (which groups), `PODS`, `WORKERS`, `MPS`, `RES`, `OUT`,
`FAST_DATA` (default 1), `DETERMINISM` (default 0), `DATASET`, `TAIL` (converged-tail window, default 20),
`EXPORT`, `FIGS`, `FOOD_RES`.

```
# locally (needs submit_experiment.sh + submit_pool.sh + .env with REPO/PROJECT/IMAGE/PVC/MOUNT/...):
BATCH="L G HS" ./runbook.sh manifest   # 1. build jobs.tsv (tokens: A T D E EA H K Y Z L F G FD HS)
WORKERS=6 PODS=2 ./runbook.sh submit   # 2. run the pod pool over jobs.tsv
runai list jobs                        #    monitor; results land in $RES/<RUN_TAG>/
# (all-submit = manifest + submit)

# locally, pointing at pulled results:
RES=~/local/results ./runbook.sh plot          # 3. paper figs + Table I: pgfplots -> $RES/export, PNG -> $RES/figs
RES=~/local/results ./runbook.sh appendix      # 3b. appendix set (class band, non-IID, submarine, ROC)
FOOD_RES=~/local/results/food101 ./runbook.sh appendix-food   # 3c. Food-101 set (food101_* files)
# appendix-all = 3b + 3c ; plot-legacy = old plots.py diagnostics ; paper = alias of plot
```

### 6.1 Experiment groups (from `run_now.sh`)

| Token | What | Free-rider ids | Purpose | Paper |
|---|---|---|---|---|
| **A** | IID honest baseline (6 seeds) + reduced FR | 1,7 / 3,6 | honest floor, η calibration, first evasion | main |
| **H** | **positive controls**: previous-models, gaussian | 3,6 | must be caught | main (fig 2) |
| **L** | **graftblock head-only**: L6/L7 = `head` (**FareMark headline**), L1/L5 = `head2` (L2–L4 commented out) | 3,6 / 1,7 | the mark is head-only | main (figs 1, 2, 4, Table I) |
| **G** | **FedIPR sign (white-box)**: honest + H controls + graftblock head2 (+ head), plus the **layer sweep** `G_A1_honest_c100_ws_L{N}` / `G_L1_graftblock_head2_c36_ws_L{N}`, N = 1/6/20 | 3,6 / 1,7 | 2nd scheme: mark in the output-layer γ | main (fig 3, Table I, fig 6) |
| **HS** | 1-seed bundle: (a) FareMark `head` FR, (b) backdoor `head` FR, (c) **adaptive** sign FR `G_Ladapt_c36_ws_L{N}` (scope head2/block2/full for N = 1/6/20), (d) honest + **fixed** head2 FR per N, (e) **adaptive** sign FR on the **full shard** `G_Lfull_c36_ws_L{N}` (`cpc=-1`) | 3,6 | fixed-vs-adaptive defender's dilemma | main (fig 6) |
| **Z** | no-watermark control (λ=0, verifier on) | none | accuracy cost of the mark; trig_acc sanity | main (text) |
| **T** | honest band across CIFAR-100 decades (10–19 … 90–99) | none | the class band is not an artifact of classes 0–9 | appendix (fig 7, ROC) |
| **D** | reduced +N data-budget spectrum, N ∈ {−1,0,1,2,5,10} (`D1_reduced_c100_c36_n{N}`) | 3,6 | how little data is enough to evade | appendix |
| **E** | non-IID (Dirichlet) honest/reduced + α sweep (0.1/0.5/1.0) + submarine E4 | 3,6 / 1,7 | starvation & amplification | appendix (fig 8) |
| **EA** | non-IID **distribution-aware** trigger assignment (+ submarine EA3) | 3,6 / 1,7 | fairness fix + attack under it | appendix (fig 8) |
| **K** | the submarine (self-η, derived margin, dynamic warmup): K9 = head2 (active), K4 = block2 (commented out) | 3,6 / 1,7 | adaptive stealth attack | appendix (figs 9–11) |
| **Y** | oracle-η submarine ablation (J4) | 3,6 / 1,7 | what self-estimation costs | appendix |
| **FD** | Food-101 / ResNet-50 (config 15, 1 seed): FareMark honest / prev-models / head2 / head, sign honest / prev-models / head2 / head | 3,6 | generalisation across dataset + model | appendix (figs 12–19, Table II) |
| **F** | **FedIPR backdoor** mirror: honest (6 seeds) + H controls + graftblock head2/head (submarine rows commented out) | 3,6 / 1,7 | 3rd scheme, black-box | implemented; not used |

### 6.2 Paper figures -> scripts -> families

Both `to_pgfplots.py` (pgfplots) and `paper_figs_mpl.py` (PNG) render the same registry; after each figure they print
which `result.json` files and seeds were read.

| Draft | Name | Families |
|---|---|---|
| Fig 1 | `fig1_faremark_timeline_head_c17` (+ `…_head` for 3,6) | `L7_graftblock_head_c17` (`L6_graftblock_head_c36`) |
| Table I | `tab1_costs` | FareMark: `L6` + `L7` pooled; sign: `G_L1_graftblock_head2_c36_ws` |
| Fig 2 | `fig2_attack_compare_head` | `A1_honest_c100`, `H5_prevmodel_c100`, `H6_gaussian_c100`, `L6_graftblock_head_c36` |
| Fig 3 | `fig2_sign_attack_compare` (+ `fig1_sign_timeline`) | `G_A1_honest_c100_ws`, `G_H5…_ws`, `G_H6…_ws`, `G_L1…_ws` |
| Fig 4 / 5 | `fig3a_class_ber_head` / `fig3b_class_entropy` (mpl: combined `fig3_class_difficulty`) | `A1_honest_c100`, `L6` (+ `L7` in mpl) |
| Fig 6 | `fig4_detect_cost` | `G_L1_graftblock_head2_c36_ws_L{N}`, `G_Ladapt_c36_ws_L{N}`, `G_Lfull_c36_ws_L{N}`, N = 1/6/20 |
| App. Fig 7 | `app_faremark_class_band` | `A1` + all `T*` decades |
| App. Fig 8 | `app_niid_reduced_timeline` | `E1`, `E2`, `EA2`, `E3_*`, `A1`, `A3` |
| App. Fig 9 | `app_submarine_timeline_k9` | `A1`, `K9_alldyn_head2_c17/c36` |
| App. Figs 10 / 11 | `app_roc_faremark` / `app_roc_faremark_norm` | honest `A1` + `T*`; FR `L6/L7`, `A2/A3`, `K9`, `K4` |
| App. Figs 12–19, Table II | same names with `food101_` prefix | the `FD` runs |

Note: captions and figs might be edited and changed in the overleaf and not updated in the scripts here.

---
---

## 7. Glossary 

- **FL / FedAvg**: federated learning; the server averages client weights (sample-weighted) each round.
- **Free-rider**: a client that wants the global model without contributing real training .
- **Watermark / mark**: a secret, verifiable signal the server checks as evidence that a client trained.
- **Head / body**: the paper's generic split `w = (θ, W)`: the head is the small watermark-carrying subset (the last layers), the body is the feature extractor. **Head-only watermarking** = the mark lives in / is controlled by the head.
- **Box-free / black-box**: verification reads only model *outputs* (FareMark, FedIPR backdoor), never weights.
- **White-box / feature-based**: verification reads *weights*.
- **DICTION framework**: a watermark = keyed extraction `Ext(·, κ_ext)` + keyed projection `Proj(·, κ_proj)` to bits.
- **BER (bit-error-rate)**: fraction of watermark bits recovered wrong. Honest ≈ 0 (FareMark: its class floor), no mark ≈ 0.5 (FareMark, sign) or ≈ 1 − 1/C (backdoor).
- **Trigger class (FareMark)**: the one task class whose softmax carries a client's bits.
- **Trigger set (FedIPR backdoor)**: private noise/OOD/patched images with a secret target label.
- **Carrier (FedIPR sign)**: a BN scale vector γ whose projected signs carry the bits; `N` = number of carrier layers.
- **η (eta)**: detection threshold on BER; benign iff BER < η. `tight`/`loose` = two frozen operating points.
- **η_T**: FedIPR trigger-set accuracy; `ber_fedipr = 1 − η_T`.
- **Smoothing f(), α**: `(p+ε)^α`, amplifies tail softmax probabilities so bits can be shaped (FareMark Eq. 7–9).
- **Memory-enhanced update, β**: Eq. 14 client-side momentum against aggregation erosion.
- **λ (lambda)**: weight of the watermark loss in `L = L_cl + λ·L_wm`.
- **Warmup (W) / calib (K)**: honest rounds before defection; the last K calibrate the attacker's self-η.
- **Tap / coast**: the submarine trains (tap) or resubmits without training (coast).
- **Scope (`head` / `head2` / `block2` / `full`)**: which trailing parameter tensors an attacker trains (2 / 5 / 20 / all on ResNet-18).
- **Fixed vs adaptive free-rider (fig 6)**: fixed keeps scope `head2` whatever N is; adaptive widens scope to cover all N carriers.
- **cpc / `tap_data_cpc` / `autop_common_per_class`**: images per common class in the reduced shard (−1 = full shard).
- **Graft**: replace the model body with the exact current global and keep only the trained head.
- **Distribution-aware assignment**: the server gives each client a trigger class it actually holds a lot of (non-IID fairness).
- **Dirichlet(α)**: non-IID label-skew partition; small α = severe skew.
- **cXX in family names**: free-rider client ids (c36 = cids 3,6 "hard"; c17 = cids 1,7 "easy"; the only two pairs used), **not** a dataset.
- **c100**: CIFAR-100. **aXX**: Dirichlet α. **rep<seed>**: repeat/seed. **_fi / _ws / _L<N>**: backdoor / sign / N carrier layers.
- **duty cycle**: fraction of rounds an attacker actually trained (evasion-cost signal).
- **effort ratio**: free-rider samples (or GPU time) ÷ honest; the cost axis of Table I.