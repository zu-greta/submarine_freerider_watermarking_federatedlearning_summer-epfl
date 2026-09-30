# Head-Only Watermarking Cannot Stop Free-Riders in Federated Learning

Summer@EPFL 2026 - SaCS lab project

**Project**:
Reproduction and limitations study of head (output layer) FL watermarking used as free-rider detection. The schemes studied are **FareMark** and **FedIPR**, both generalized theoretically to an output layer watermarking scheme as described in **DICTION**. All of them embed the watermark in a small, designated subset of parameters: the head of the network. We propose a **head-only free-riding** attack. The free-rider trains only the watermark-carrying parameters, on very little data, and leaves the body at the current global model. It passes verification like an honest client while spending ≈0.3× of the honest compute (up to ~70% saved). We also give a formal cost analysis and theoretical insights into why head-only watermarking cannot serve as a countermeasure.

## FareMark + FedIPR — reproduction + limitations study

Re-implementation and limitations analysis of three FL watermarks that live in the head:
- **FareMark** (Li et al., IEEE IoT-J 12(18), 2025): box-free softmax-projection **BER** scheme - attacked with the **`head`** free-rider (softmax fc only, last 2 tensors).
- **FedIPR sign, forced to the output layer**: FedIPR's feature-based sign watermark (**white-box**), placed in the last-BN scale γ (inside `head2`); `ber = Hamming(sign(γ·E), B)/N` - attacked with the **`head2`** free-rider (fc + the conv/BN before it, last 5 tensors), since the carrier BN lives in `head2`, not in `head`. The layer sweep (the mark spread over N = 1 / 6 / 20 BN layers) is the "potential defence" figure.
- **FedIPR backdoor**: private trigger set -> target label (black-box); `ber = 1 − trigger_set_accuracy`. Fully implemented, but not in the current paper figures (appendix)

Centralized FedAvg simulated on one GPU, with a per-client watermark loss, a memory-enhanced update (Eq. 14, FareMark) and server-side verification. Select the scheme per run with `WM_SCHEME=faremark|fedipr|fedipr_sign`. All three share the result/plot pipeline (each maps to the same `ber` field). A full technical walkthrough is in [`DOCUMENTATION.md`](DOCUMENTATION.md): every file, knob and dataset, plus extra notes, the theory and the figure map.

The **adaptive "submarine"** free-rider (`adaptive_tap`) still exists and is documented, but it is now appendix material. The main attack is the static head-only free-rider (`graftblock`).

---

## Layout

| Path | Role |
|---|---|
| `src/watermark.py` | **FareMark** box-free softmax-projection BER scheme (Eq. 1–16) |
| `src/watermark_fedipr.py` | **FedIPR backdoor** trigger-set scheme (black-box); `ber = 1 − trigger_acc` |
| `src/watermark_fedipr_sign.py` | **FedIPR feature-based sign** watermark (WHITE-BOX), 1+ BN carrier layers; `ber = Hamming(sign(γ·E), B)/N` |
| `src/wm_verify.py` | registry + per-round verification hook for all three schemes |
| `src/clients.py` | honest client, watermark client, the paper baselines and the attackers (`reduced`, `graftblock` = **head-only, our attack**, `adaptive_tap` = submarine) |
| `src/server.py` | FedAvg aggregation + round loop |
| `src/datasets.py` | MNIST / CIFAR-10 / CIFAR-100 / Food-101 loaders, IID + Dirichlet non-IID partition |
| `src/fast_data.py` | GPU-resident loaders (`FAST_DATA=1`; statistically identical, just faster; decodes and caches Food-101 once) |
| `src/models.py` | ResNet-18/34/50 (CIFAR or ImageNet stem) + SmallCNN; the named-tensor order the attack scopes count from |
| `src/compute_meter.py` | per-client samples / GPU-ms / FLOPs (the "free-riding is cheap" evidence) |
| `src/runlog.py` | human-readable `run.log` (setup blocks, per-round table, closing report) |
| `src/config.py` | `ExpConfig` + the `CONFIGS` list (config **14** = CIFAR-100/ResNet-18 attack base, **15** = Food-101/ResNet-50) |
| `src/utils.py` | seeding, logging, accuracy |
| `scripts/run_experiment.py` | one `(config, seed)` run → `result.json` |
| `scripts/to_pgfplots.py` | paper figures + table -> pgfplots `.dat` / `.tex` (Overleaf); `--appendix` for the appendix set |
| `scripts/paper_figs_mpl.py` | the same paper / appendix figures rendered with matplotlib (PNG) |
| `scripts/plots.py` | legacy per-family diagnostic plots (`./runbook.sh plot-legacy`) |
| `infra/run_now.sh` | builds `jobs.tsv` for groups `A T D E EA H K Y Z L F G GS FD HS` |
| `infra/runbook.sh` | phase driver: manifest -> submit -> plot (paper / appendix / Food-101) |
| `infra/submit_experiment.sh` | one RunAI/Kubernetes job submission (or one manifest row with `DRYRUN=1`) |
| `infra/submit_pool.sh` | runs the `jobs.tsv` queue over `PODS × WORKERS` |

---

## Standard setup

CIFAR-100, ResNet-18 (CIFAR stem), 10 clients (8 honest + 2 free-riders), 50 rounds, 5 local epochs, batch 16, lr 0.01, N_T=50, λ=5, β=0.6, α=0.4. Free-riders on cids **1,7** ("easy" trigger classes) or **3,6** ("hard"). Trigger class = `cid % n`. Honest warm-up W=12 rounds, then free-ride on a reduced shard of trigger images + 5 images per common class (`cpc=5`).

| dataset | m | l  | ceiling | paper reports |
|---|---|---|---|---|
| CIFAR-100 | 10 (code default) | 10 | 99.90% | 99.71 |

Threshold: `η = mean over seeds of (μ_s + 3σ_s)` over the per-round mean-over-clients honest BER, last 20 rounds. It is frozen and injected as `WM_ETA_FIXED`. This threshold is not used in the paper, only as a reference line based on the threshold describe in the Faremark paper.

**Headline numbers in the current draft** (CIFAR-100/ResNet-18, 3 seeds; Table I):

| Scheme | Free-rider | Samples / honest | GPU / honest |
|---|---|---|---|
| FareMark | `head` (L6 + L7) | 0.305 | 0.325 ± 0.012 |
| FedIPR sign | `head2` (G_L1) | 0.298 | 0.297 ± 0.004 |

Food-101/ResNet-50 (appendix, Table II): 0.280 / 0.263 (FareMark head), 0.272 / 0.254 (sign head2).

---

## Experiment groups (`infra/run_now.sh`)

**Main paper:**
`A` IID honest baseline (+ reduced FR) · `H` positive controls (previous-models, gaussian) ·
`L` **head-only graftblock (our attack)**: L6/L7 = `head` (FareMark headline), L1/L5 = `head2` ·
`G` **FedIPR sign (white-box)**: honest + controls + graftblock, plus the **layer sweep** N = 1/6/20 ·
`HS` 1-seed bundle: FareMark / backdoor `head` FR + the sign **fixed-vs-adaptive** sweep feeding draft Fig. 6 (`fig4_detect_cost`) ·
`Z` no-watermark control (λ=0).

**Appendix:**
`T` honest band across CIFAR-100 decades · `D` reduced +N data-budget spectrum ·
`E` non-IID (Dirichlet) + α sweep + submarine · `EA` non-IID distribution-aware assignment ·
`K` submarine (self-η, derived margin, dynamic warmup; K9 = head2 active, K4 = block2 commented out) ·
`Y` oracle-η submarine ablation (J4) · `FD` Food-101 / ResNet-50 basics (config 15, 1 seed) ·
`F` FedIPR **backdoor** mirror of A/H/L (implemented, not in the current figures).

**Exploratory (not in the draft):**
`GS` FedIPR sign with a **scattered carrier** (`FEDIPR_SIGN_CARRIER=scatter`): the mark sits in 512 scalar weights at random positions in every parameter tensor instead of a layer. Active row (1 seed): `GS_Lwm`, the `TAP_SCOPE=wm` FR that trains only those marked weights on the reduced shard, i.e. `G_L1_graftblock_head2_c36_ws` with the mark scattered. Commented out: honest floor, fixed `head2` FR, and the no-data variant `GS_Lwm0`.

`HS` also builds `G_Lfull_c36_ws_L{1,6,20}` (item e): the adaptive sign free-rider on the full shard, which is draft Fig. 6's purple "full-data cost" line. `BATCH=HS` therefore reproduces all of Fig. 6 (`SEEDS_HS="0 1 2"` for 3 seeds).

Family-tag decoder: `c100` = CIFAR-100; `c36`/`c17` = free-rider **client ids** 3,6 / 1,7 (**not** a dataset);
`aXX` = Dirichlet α; `rep<seed>` = seed; `_fi` = FedIPR backdoor; `_ws` = FedIPR sign white-box; `_L<N>` = number of sign carrier layers; `_s<N>` = N scattered sign carrier weights (group GS).
Food-101 runs reuse the CIFAR-100 family names (`…_c100…`) inside their own results folder.

---

## Quickstart

1. Pick the groups in [infra/run_now.sh](infra/run_now.sh) and build the manifest: `BATCH="L G HS" ./runbook.sh manifest` (whole tokens, space/comma separated).
2. Check the generated `jobs.tsv` and edit any row if needed.
3. Run the pool on the cluster: `WORKERS=6 PODS=2 ./runbook.sh submit`. Adjust `WORKERS`/`PODS` to your cluster capacity (`MPS=1` expects `nvidia-cuda-mps-control -d` in each pod). Monitor with `runai list jobs`.
4. Copy the results locally and build the figures:
   - `RES=<results dir> ./runbook.sh plot`: paper figures + Table I (pgfplots → `$RES/export`, PNG → `$RES/figs`)
   - `RES=<results dir> ./runbook.sh appendix`: appendix set (class band, non-IID, submarine, ROC)
   - `FOOD_RES=<food101 results dir> ./runbook.sh appendix-food`: Food-101 set with the `food101_` prefix
   - `appendix-all` = both appendix sets; `plot-legacy` = the old `plots.py` diagnostics

To run one thing directly (no manifest), export the knobs and call `submit_experiment.sh <config_idx> <seed>`:

```bash
# FareMark honest baseline (calibration source), CIFAR-100/ResNet-18, seed 0
ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 FAMILY=A1_honest_c100 ./submit_experiment.sh 14 0

# FareMark HEAD-ONLY free-rider (softmax fc only, cpc5), cids 3,6: evades at the honest floor
ATTACK=graftblock PARTITION=iid FAST_DATA=1 TAP_SCOPE=head TAP_COAST_MODE=decay \
  AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
  WM_ETA_FIXED=0.064 FREE_RIDER_IDS=3,6 ROUNDS=50 FAMILY=L6_graftblock_head_c36 ./submit_experiment.sh 14 0

# (FedIPR backdoor runs: add WM_SCHEME=fedipr FEDIPR_TRIGGER_SOURCE=noise FEDIPR_NUM_TRIGGER=40 FEDIPR_TARGET_MODE=cid)

# FedIPR SIGN white-box, mark forced into the output-layer BN scale: honest baseline, then
# a head2 graftblock free-rider that re-embeds its OWN sign bits and EVADES (ber ≈ 0)
WM_SCHEME=fedipr_sign FEDIPR_SIGN_BITS=40 FEDIPR_SIGN_CARRIER=auto_last_bn \
  ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 FAMILY=G_A1_honest_c100_ws ./submit_experiment.sh 14 0
WM_SCHEME=fedipr_sign FEDIPR_SIGN_BITS=40 ATTACK=graftblock TAP_SCOPE=head2 TAP_COAST_MODE=decay \
  AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 FAST_DATA=1 \
  WM_ETA_FIXED=0.20 FREE_RIDER_IDS=3,6 ROUNDS=50 FAMILY=G_L1_graftblock_head2_c36_ws ./submit_experiment.sh 14 0

# Food-101 / ResNet-50 (config 15, appendix)
DATASET=food101 ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 FAMILY=A1_honest_c100 ./submit_experiment.sh 15 0
```

---

## Reproducibility notes

- **Pods run a `git clone` of this repo, not your local files** (`submit_experiment.sh` -> `GIT_REPO=$REPO` from `.env`, branch `main`). Push before you submit or the pods run stale code.
- **Seeds:** honest baselines use 0–5 (6 seeds), attacks 0–2 (3 seeds). Report mean ± std (the aggregated figures already average over seeds; each plotting run prints which `result.json` files and seeds fed each figure).
- **Cost axis:** `samples` (cumulative (image, label) pairs through a forward+backward pass) is device-independent and is the primary cost; an honest client = 5 epochs × 5,000 × 50 rounds = 1,250,000 on CIFAR-100. **GPU time** (`gpu_ms`) is only comparable at concurrency 1 on the same card. Each `result.json` records `gpu_name` / `gpu_concurrency`.
- **Free-rider ids:** every free-rider run uses cids **3,6** (hard) or **1,7** (easy).

---