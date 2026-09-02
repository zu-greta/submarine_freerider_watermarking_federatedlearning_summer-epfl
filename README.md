# submarine_freerider_watermarking_federatedlearning_summer-epfl

Summer@EPFL 2026 - SaCS lab project

**Project**:
Reproduction and limitations study of **FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning** (Li et al., IEEE IoT-J 12(18), 2025) + new adapted free-rider attack design and experiments to prove that output layer watermarking in federated learning for free-rider detection is impossible in general.
Goal: Show experimentally that under the paper's own setup and under extensions (non-IID, adaptive free-riders, more clients than classes), no threshold separates honest clients from free-riders - ie. output layer watermarking in federated learning for free-rider detection is impossible in general through a newly designed adaptive free-rider attack (submarine attack) that exploits the paper's own limitations.

The same claim is then shown on a **second, mechanically different output-layer scheme** - the **FedIPR backdoor** (Li et al., TPAMI 2023) - so the fragility is a property of *output-layer* watermarking, not of any one construction. A **third scheme** (`fedipr_sign`) then forces FedIPR's *white-box* feature-based mark into the output-layer scale and shows the same head-only free-rider forges it too - proving the fragility is about the mark's **location**, not black-box vs white-box. (Real FedIPR's feature-based mark lives spread through the body and stays robust - that is the control.)

## FareMark + FedIPR — reproduction + limitations study

Re-implementation and limitations analysis of three output-layer FL watermarks:
- **FareMark** (Li et al., IEEE IoT-J 12(18), 2025) - box-free softmax-projection **BER** scheme (black-box).
- **FedIPR backdoor** (Li et al., TPAMI 2023) - private trigger-set → target-label (black-box); `ber = 1 − trigger_set_accuracy`.
- **FedIPR sign, forced to output layer** (this project) - FedIPR feature-based sign watermark (**white-box**) in the last-BN scale γ (inside head2); `ber = Hamming(sign(γ·E), B)/N`.

Centralized FedAvg simulated on one GPU, with a per-client output-layer watermark loss, a memory-enhanced update (Eq. 14, FareMark), and server-side verification. Select the scheme per run with `WM_SCHEME=faremark|fedipr|fedipr_sign`; all three share the result/plot pipeline (each maps to the same `ber` field). A full technical walkthrough (every file, knob, dataset, faithfulness note, and results analysis) is in [`DOCUMENTATION.md`](DOCUMENTATION.md) — the `fedipr_sign` scheme is §4b.

---

## Layout

| Path | Role |
|---|---|
| `src/watermark.py` | **FareMark** box-free softmax-projection BER scheme (Eq. 1–16) |
| `src/watermark_fedipr.py` | **FedIPR backdoor** trigger-set scheme (black-box); `ber = 1 − trigger_acc` |
| `src/watermark_fedipr_sign.py` | **FedIPR feature-based sign** watermark (WHITE-BOX) forced into the output-layer scale; `ber = Hamming(sign(γ·E), B)/N` |
| `src/wm_verify.py` | registry + per-round verification hook for **all three** schemes |
| `src/clients.py` | honest client, watermark client, and the attackers (`reduced`, `graftblock`, `adaptive_tap`=submarine) |
| `src/server.py` | FedAvg aggregation + round loop |
| `src/datasets.py` | CIFAR/MNIST loaders, IID + Dirichlet non-IID partition |
| `src/fast_data.py` | GPU-resident loaders (`FAST_DATA=1`; statistically identical, just faster) |
| `src/models.py` | ResNet-18 (CIFAR stem) + SmallCNN; the 62 named tensors the attack scopes count from |
| `src/compute_meter.py` | per-client samples / GPU-ms / FLOPs (the "evasion is cheap" evidence) |
| `src/config.py` | `ExpConfig` + the `CONFIGS` list (config 14 = CIFAR-100/ResNet-18 attack base) |
| `scripts/run_experiment.py` | one `(config, seed)` run → `result.json` |
| `scripts/plots.py` | all figure families |
| `infra/run_now.sh` | builds `jobs.tsv` for groups A T D E EA H K Y Z L F |
| `infra/runbook.sh` | phase driver: probe → manifest → submit → monitor → plot |
| `infra/submit_experiment.sh` | one RunAI/Kubernetes job submission |

---

## Standard setup

CIFAR-100, ResNet-18, 10 clients, 50 rounds, 5 local epochs, batch 16, N_T=50, λ=5,
β=0.6, α=0.4
Trigger class = `cid % n`

| dataset | m | l | stuck | ceiling | paper reports |
|---|---|---|---|---|---|
| CIFAR-100 | 10 (code default) | 10 | 0.20% | 99.90% | 99.71 |

Threshold: `η = mean over seeds of (μ_s + 3σ_s)` over per-round mean-over-clients honest BER, last 20 rounds; frozen and injected as `WM_ETA_FIXED`.
CIFAR-100 / 10 clients: **η_tight = 0.064**, **η_loose = 0.264** (per-seed 0.017–0.115, std ≈ 40%); non-IID **0.161 / 0.576**.
FedIPR uses the same `μ+3σ` rule on honest `ber_fedipr` (recalibrate from `F_A1_honest`; provisional 0.20 / 0.50).

> Note the tight threshold is *already* below the honest floor of the hardest CIFAR-100 classes
> (cls6 ≈ 0.110, cls4 ≈ 0.088 in `A1_honest_c100`), so `η_tight` false-positives honest clients with **no**
> attack — half of the threshold-dilemma argument, before any free-rider runs.

---

## Experiment groups (`infra/run_now.sh`)

`A` IID baseline + reduced FR · `T` honest band across CIFAR-100 decades · `D` reduced +N data-budget spectrum ·
`E` non-IID (Dirichlet) + α sweep + submarine · `EA` non-IID distribution-aware assignment ·
`H` positive controls (previous-models, gaussian) · `K` submarine (self-η, derived margin, dynamic warmup) ·
`Y` oracle-η ablation (J4) · `Z` no-watermark control (λ=0) · `L` graftblock (main clean attack) ·
`F` **FedIPR** mirror of A/H/K/L under `WM_SCHEME=fedipr` ·
`G` **FedIPR sign (white-box)** mirror of A/H/K/L under `WM_SCHEME=fedipr_sign` (mark in the output-layer γ).

Family-tag decoder: `c100` = CIFAR-100; `c36`/`c17` = free-rider **client ids** 3,6 / 1,7 (**not** a dataset);
`aXX` = Dirichlet α; `rep<seed>` = seed; `_fi` = FedIPR backdoor; `_ws` = FedIPR sign white-box.

---

## Quickstart

1. Update the commands in [infra/runbook.sh](infra/runbook.sh) and [infra/run_now.sh](infra/run_now.sh) and run `BATCH="<input the letters here following format from runbook file>" ./runbook.sh manifest`.
2. Check the `jobs.tsv` file created and modify any of the commands if needed.
3. Run the jobs on the cluster with `WORKERS=3 PODS=2 BATCH="K" ./runbook.sh submit`. Modify the `WORKERS` and `PODS` values to suit your cluster capacity and the `BATCH` value to the desired experiment batch.
4. When the jobs are done, copy the results to your local machine and run the plotting script to generate the figures using `RES=<path to results jsons folder> ./runbook.sh plot`

To run one thing directly (no manifest), export the knobs and call `submit_experiment.sh <config_idx> <seed>`:

```bash
# FedIPR honest baseline (calibration source), CIFAR-100/ResNet-18, seed 0
WM_SCHEME=fedipr FEDIPR_TRIGGER_SOURCE=indist FEDIPR_NUM_TRIGGER=40 FEDIPR_TARGET_MODE=cid \
  ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 FAMILY=F_A1_honest_c100_fi ./submit_experiment.sh 14 0

# FedIPR graftblock free-rider (head2, cpc5), cids 3,6 — should EVADE (ber ≈ 0)
WM_SCHEME=fedipr ATTACK=graftblock TAP_SCOPE=head2 TAP_COAST_MODE=decay \
  AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
  WM_ETA_FIXED=0.20 FREE_RIDER_IDS=3,6 ROUNDS=50 FAMILY=F_L1_graftblock_head2_c36_fi ./submit_experiment.sh 14 0

# FedIPR SIGN white-box, mark forced into the output-layer scale: honest baseline, then
# a head2-only graftblock free-rider that re-embeds its OWN sign bits and EVADES (ber ≈ 0)
WM_SCHEME=fedipr_sign FEDIPR_SIGN_BITS=40 FEDIPR_SIGN_CARRIER=auto_last_bn \
  ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 FAMILY=G_A1_honest_c100_ws ./submit_experiment.sh 14 0
WM_SCHEME=fedipr_sign FEDIPR_SIGN_BITS=40 ATTACK=graftblock TAP_SCOPE=head2 TAP_COAST_MODE=decay \
  AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
  WM_ETA_FIXED=0.20 FREE_RIDER_IDS=3,6 ROUNDS=50 FAMILY=G_L1_graftblock_head2_c36_ws ./submit_experiment.sh 14 0
```

---

## Reproducibility notes

- **Pods run a `git clone` of this repo, not your local files** (`submit_experiment.sh` → `GIT_REPO=zu-greta/...`, branch `main`). Push before you submit or the pods run stale code.
- **Seeds:** honest baselines use 0–5 (6 seeds), attacks 0–2 (3 seeds); report mean ± std (the aggregated figures already do "over N seeds").
- **GPU timing** (`gpu_ms`) is only comparable at concurrency 1 on the same card — pin with `RUNAI_EXTRA="--node-pools a100-80"`, or report `samples` (device-independent). Each `result.json` records `gpu_name` / `gpu_concurrency`.
- Exit code `2` = accuracy outside the config's expected band; **normal for attack runs**, `result.json` is already written.
- `FAST_DATA=1` (GPU-resident loaders) and `DETERMINISM=0` (cuDNN autotuner) only change speed; results are statistically identical over seeds.

---

## Papers

- **FareMark** — Li Li, Xinpeng Zhang, Hanzhou Wu, Guorui Feng, Weiming Zhang, "FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning Model," *IEEE IoT Journal*, 12(18), 2025.
- **FedIPR** — Bowen Li, Lixin Fan, Hanlin Gu, Jie Li, Qiang Yang, "FedIPR: Ownership Verification for Federated Deep Neural Network Models," *IEEE TPAMI*, 2023 (arXiv:2109.13236). Original code: `github.com/purp1eHaze/FedIPR` (the **backdoor** half is reproduced here; the feature-based, white-box mark is out of scope).