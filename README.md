# submarine_freerider_watermarking_federatedlearning_summer-epfl

Summer@EPFL 2026 - SaCS lab project

Project: 
Reproduction and limitations study of **FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning** (Li et al., IEEE IoT-J 12(18), 2025) + new adapted free-rider attack design and experiments to prove that output layer watermarking in federated learning for free-rider detection is impossible in general.
Goal: Show experimentally that under the paper's own setup and under extensions (non-IID, adaptive free-riders, more clients than classes), no threshold separates honest clients from free-riders - ie. output layer watermarking in federated learning for free-rider detection is impossible in general through a newly design adaptive free-rider attack (submarine attack) that exploits the paper's own limitations.

## FareMark — reproduction + limitations study

Re-implementation and limitations analysis of **FareMark: Model-Watermark-Driven Free-Rider Detection in Federated Learning** (Li et al., IEEE IoT-J 12(18), 2025).
Centralized FedAvg simulated on one GPU, with a per-client output-layer watermark loss, a memory-enhanced update (Eq. 14), and server-side verification (Eq. 15–16).

---

## Layout - TODO 


---

## Standard setup

CIFAR-100, ResNet-18, 10 clients, 50 rounds, 5 local epochs, batch 16, N_T=50, λ=5,
β=0.6, α=0.4
Trigger class = `cid % n`

| dataset | m | l | stuck | ceiling | paper reports | 
|---|---|---|---|---|---|---|
| CIFAR-100 | 10 (code default) | 10 | 0.20% | 99.90% | 99.71 | 

Threshold: `η = mean over seeds of (μ_s + 3σ_s)` over per-round mean-over-clients honest BER, last 20 rounds; frozen and injected as `WM_ETA_FIXED`.
CIFAR-100 / 10 clients: **η = 0.063** (per-seed 0.017–0.115, std ≈ 40%).

---

## Quickstart 

1. Update the commands in [infra/runbook.sh](infra/runbook.sh) and [infra/run_now.sh](infra/run_now.sh) and run `BATCH="<input the letters here following format from runbook file>" ./runbook.sh manifest`. 
2. Check the `jobs.tsv` file created and modify any of the commands if needed.
3. Run the jobs on the cluster with `WORKERS=3 PODS=2 BATCH="K" ./runbook.sh submit`. Modify the `WORKERS` and `PODS` values to suit your cluster capacity and the `BATCH` value to the desired experiment batch.
4. When the jobs are done, copy the results to your local machine and run the plotting script to generate the figures using `RES=<path to results jsons folder> ./runbook.sh plot`

---
---
---