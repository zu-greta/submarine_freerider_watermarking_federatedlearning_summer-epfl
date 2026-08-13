# DRAFT PROPOSAL — The Submarine Free-Rider: Evading Output-Layer Watermarking for Free-Rider Detection in Federated Learning

## goal/thesis
Output-layer (box-free) watermarking is being proposed as a free-rider detector in FL. We show the **approach itself** is broken, not just one paper: even when the server is *granted a best-case oracle detection threshold*, a cheap **adaptive ("submarine") free-rider** stays under it round after round while walking away with the full-quality global model — at a small fraction of honest compute. The reason is structural: honest clients train the whole network, but the watermark is read only from the **output layer**, which has very few parameters, so a free-rider only has to fool that thin layer — little work. Class difficulty and non-IID make the defender's job strictly harder still.

*Framing note (from meeting): do NOT sell the "no threshold is possible" impossibility as the headline. Assume a threshold exists and set it to an **oracle** (best case for the server/honest clients). The contribution is the generic detector + the attack that evades even that oracle.*

Two supporting results, in priority order:
- **Constructive attack (headline).** An adaptive free-rider evades an **oracle** per-round threshold while spending ≈25–30% of honest compute (≤~22% on the easiest classes), keeping honest-level accuracy.
- **Class-/heterogeneity-dependence (support).** The honest BER floor is a per-class band (≈0.001–0.114 IID; ≈0.008–0.34+ non-IID). This is *why the defender is forced to a loose threshold in the first place* — cover the submarine can hide in — rather than an independent "impossibility" claim.

---

## 1. Abstract

- FL intro + free-rider problem (+ definition).
- Output-layer watermarking as a proposed detector (FareMark as the concrete, best-documented instance; one-line method + decision rule).
- We abstract the family into a **generic reference detector** and grant it an **oracle threshold**.
- We show a cheap **submarine** free-rider evades it, keeping full model quality, at a fraction of honest compute.
- Non-IID amplifies the effect; the honest band is class-dependent.
- Conclude the failure is a property of the **approach** (output-layer watermarking for FR detection), not of any single scheme.

---

## 2. Introduction

- **Free-rider problem.** From Lin, Fraboni, et al. Define the free-rider formally; note our refinement is **effort-based** (a client minimizing samples/compute, →0, while still passing detection), which differs slightly from the classic "contributes nothing" definition.
- **Output-layer watermarking as a detector.** Watermarking in general → its use for FR detection → the per-round, per-client BER-vs-threshold decision rule. Define the threshold and state that **we grant the server an oracle** value (best case) rather than arguing no threshold exists.
- **Key intuition, stated up front.** Honest clients optimize the whole model; the watermark is decoded only from the output layer's softmax. The output layer is a tiny fraction of parameters, so preserving/forging the mark is cheap relative to honest training — the structural reason the detector is evadable.
- **Contributions.**
  1. A **generic reference detector** abstracted from output-layer watermarking FR-detection schemes (FareMark and the broader family), evaluated under an oracle threshold.
  2. The **submarine attack**: an adaptive, effort-minimizing free-rider (warm-up → calibrate → tap/coast with output-layer grafting).
  3. Empirical evasion on ResNet-18 / CIFAR-100 under the oracle threshold, with compute-cost accounting.
  4. Non-IID amplification (random vs. distribution-aware trigger assignment).
  5. A generality argument that the attack targets the approach, not one paper.

**PLOTS:** headline effort-vs-BER figure — honest band + reduced sweep + submarine + baseline free-riders, with the two η reference lines — showing the free-rider reaching the honest band at far less effort, under an oracle threshold.

---

## 3. Background & Preliminaries & Related Works

- FL / FedAvg recap.
- Free-rider definitions and our effort-based refinement.
- Introduce related works on output layer watermarking - justify the generic algo in section 4
   - **Output-layer watermarking in general** — the shared structure: per-client trigger key, a mark read from the output softmax, a per-round BER, a threshold decision.
   - **FareMark as the concrete instance** — BER on trigger-class softmax; μ+3σ rule (kept only as the canonical instantiation; we do not attack its specific σ estimate).

TODO — fill the family table (used to justify the *generic* detector in §4):
| Scheme | Metric | Decision rule | Free-rider tested? | Threshold specified? |
|---|---|---|---|---|
| **FareMark** (IoT-J'25) | BER on trigger-class softmax | hard, per-round, per-client: flag if `BER ≥ η = μ+3σ` | baseline (prev-model, Gaussian) + train-then-attack + trigger-only | μ, σ and η |
| **FedIPR** (TPAMI'23) | — | — | — | — |
| **FedTracker** (TDSC'24) | — | — | — | — |
| **Lansari review** (MAKE'23) | — | — | — | — |

---

## 4. A Generic Output-Layer Watermarking Detector (reference algorithm)

- Abstract the family into one detector: per-client trigger key on the output layer; per-round BER on a held-out trigger bank; a decision threshold η.
- **Grant η an oracle value** — the best separation the defender could hope for, computed with full knowledge of honest behavior. This is deliberately generous: it removes "you just tuned the threshold badly" as a reviewer escape hatch.
- Show FareMark / FedIPR / FedTracker as **parameterizations** of this detector (metric, key placement, threshold rule).
- State the claim: any attack that evades this reference detector evades the family, modulo the parameterization.

**PLOTS:** schematic of the generic detector (key → output softmax → BER → oracle η).

---

## 5. Threat Model

- **Setup.** K=10 clients, ResNet-18 / CIFAR-100, FedAvg, 50 rounds; each client assigned one trigger class, a secret key `M`, target bits `B`; server verifies every round on `N_T=50` held-out trigger-class images. (Note external-validity caveat: single model/dataset — see Limitations.)
- **Adversary (free-rider).** Goals, knowledge, capabilities, and constraints: participates like an honest client, sees the global model each round, controls its own submitted update, wants a full-quality model while minimizing its own samples/compute and never being flagged.
- **Success criterion (submarine).** Per-round BER stays ≤ η (the oracle) for all verified rounds AND cumulative samples ≪ honest, while final accuracy tracks honest.

**PLOTS:** threat-model / attack-loop diagram.

---

## 6. The Submarine Free-Rider Attack

- **Mechanism.** Forced-honest warm-up → short calibration window (estimate/receive η) → **tap/coast** loop: coast (no training) while BER stays safely under η; tap (a short, localized training step) when BER drifts up toward η.
- **Output-layer grafting (coast).** Copy the current global core for free; re-attach a preserved watermarked head (details + math in the Technical Appendix). This is the minimal-parameter realization of the §2 intuition.
- **Tap scope (block2).** A tap updates only the last block + head, where the BER gradient concentrates — re-embeds the mark at a fraction of the cost and leaves the copied core intact.
- **Effort accounting.** Cumulative samples / GPU-time vs. honest mean; report % saved.

---

## 7. Evaluation

**Q1 — Reproduction & sanity.** Baseline free-riders are caught by the reference detector; the no-watermark control confirms trigger-class accuracy →0 is caused by embedding, not class difficulty; watermark/free-riding does not degrade global FedAvg accuracy.
- Plots: baseline-FR timelines (H5 previous-models, H6 gaussian — both sit near BER 0.5, caught); A0 no-watermark trig_acc panel; global-accuracy attack-vs-honest.

**Q2 — Why the defender is forced to a loose (oracle) threshold: the class-dependent honest band.** Honest BER is a per-class band; hard classes overlap the region a cheap free-rider occupies. Framed as *cover for the submarine*, not as a standalone impossibility.
- Plots: A1 per-class honest floors + honest-per-round (trig_acc panel); **T1/T2** honest bands on other CIFAR-100 decades (classes 40–49, 90–99) to show the band is a class property, not a 0–9 artifact.

**Q3 — A cheap *static* free-rider already reaches the band (effort spectrum).** Reduced +N data-budget sweep; compute cost vs. BER; isolated same-class honest-vs-FR overlays.
- Plots: D1 spectrum + reduced timelines + gpu_savings; isolated per-class overlays.

**Q4 — The submarine evades the oracle threshold at low effort.** Per-free-rider tap/coast timelines (easy cid3 vs. hard cid6); effort side-by-side; scope (block2 vs. full = K5 ablation); **oracle-vs-self threshold (J4 = group Y) ablation**.
- Plots: tap_perfr / tap_perseed / tap_effort (K4 headline, K5 ablation); accuracy_K4/K5; iso overlays; the combined-FR `timeline` overview.

**Q5 — Non-IID amplifies everything.** Dirichlet α sweep; random (E) vs. distribution-aware (EA) trigger assignment; honest band widens; submarine effort drops.
- Plots: dirichlet_dist; E1/EA1 honest bands; E2/E3/EA2 timelines + iso + gpu_savings.

---

## 8. Generality & Discussion

- Extend the argument to the family via the §4 reference detector: the attack targets output-layer key preservation, which every instance shares.
- **Limitations** (state plainly): single model + dataset (ResNet-18 / CIFAR-100); simulated FL; the multi-scheme claim is architectural/argued unless second-scheme runs land in time.
- **No mitigation section** (meeting: pure attack paper).

---

## 9. Conclusion

- Restate: output-layer watermarking for FR detection fails under a best-case oracle threshold; the submarine evades cheaply while keeping model quality.
- Empirical scope is one model/dataset; the mechanism (few-parameter output layer) is architecture-general, so we expect it to transfer.
- Future work: second/third schemes, collusion, reputation-based detectors — as *attack surface*, not as proposed fixes.

---
---
---

# Experiments Plan (updated to the current run groups)

ResNet-18 on CIFAR-100. 

- **[done]** No-watermark control (all-honest, λ=0) — group **Z** (`A0_nowm_honest_c100`). Confirms trig_acc→0 is caused by the watermark.
- **Honest band** (per-class BER, class difficulty, oracle-threshold basis) — multiple seeds.
   - **[done]** Group **A** (`A1_honest_c100`), classes 0–9, 6 seeds.
   - **[TODO - group T]** Trigger-class generality: `T1_honest_c100_cls4049` (classes 40–49) and `T2_honest_c100_cls9099` (90–99), 3 seeds. Shows the band is a class property, not a 0–9 artifact.
- **[redo - group H]** Baseline free-riders (positive controls, must be caught): `H5_prevmodel_c100` + `H6_gaussian_c100`. 
- **[redo - dynamic?]** Static reduced +N spectrum — group **D** (`D1_reduced_c100_c36_n{-1,0,1,2,5,10}`).
- **Submarine** — group **K** (`K4_alldyn_block2_c36` headline; `K5_alldyn_full_c36` scope ablation) + group **Y** (`J4_*` = **oracle-threshold ablation**, was "reproduction"; single seed).
   - Decision: **K4 (block2 + graft + self-η) is the headline**; K5 (full scope) is the "why localize the tap" ablation; J4 (oracle η) is the "grant best-case threshold, still evades" ablation. (Rationale in the Technical Appendix.)
- **Non-IID** — group **E** (random assignment) + group **EA** (distribution-aware). **[TODO]** rerun `EA2_reduced_niid_distrib_c36` (it was missing from results, not a plotter bug) and regenerate comparison plots.

Plots for the paper:
- honest BER over seeds (class difficulty / oracle basis); baseline reproduction comparison; trig-class + global accuracy (quality preserved); reduced +N spectrum (timeline + savings + isolated overlays + threshold overlaps, multi-seed); submarine config table; submarine timelines + cumulative savings vs. honest (per seed tap/coast); submarine param/index/FR-count variations; non-IID random-vs-fair comparisons.

Extra sanity: no-watermark trig_acc (done, group Z).

---

meeting notes:

- Don't sell the threshold - assume one exists; set it to an **oracle** (best case for server/honest clients) and show the submarine evades it. The deliverable is a **generic reference algorithm** for output-layer watermarking (inspired by FareMark + all other output-layer watermarking papers) plus the attack that beats it.
- Check how FareMark classifies watermark accuracy and detects free-riders (is flagging a per-round misclassification whenever the threshold is passed?).
- Main priority: the **reference algorithm** built from generic output-layer watermarking papers, then the submarine evading it. We attack the **approach**, not each paper individually.
- **No solution proposal — pure attack paper.**
- Generalise and collect output-layer-watermarking papers for the family table / generality argument.

---
---
---

# Technical Appendix (working notes)

These are the mechanism explanations to fold into §6 (method) and §8 (discussion). 

## A. Grafting (the coast mechanism)

**Plain intuition.** The watermark is a secret handshake the server checks each round. Doing the handshake "for real" means training your whole model — expensive. The free-rider learns the handshake once, during the honest warm-up, and saves it. After that, every round it copies the *new* global model everyone else built (so it looks up-to-date and its accuracy is great), but it keeps its own hand frozen in the handshake shape and staples that hand onto the copied body. Copying the body is free; holding one hand still is almost free. It looks like a hard-working, watermarked client while doing almost nothing. "Graft" = stapling the saved handshake-hand onto the freshly copied body.

**Technical (paper-ready).** Split parameters into the output-layer head θ_H (the classifier layer whose softmax the watermark reads) and the core θ_C. At the end of the warm-up round W, the free-rider stores its watermarked head θ_H★ — the one producing BER below η. On a **coast** round t, instead of retraining it grafts:
- θ_C ← θ_C^(t)   (copied from the current global model — free)
- θ_H ← (1−γ)·θ_H★ + γ·θ_H^(t)   (preserved watermarked head, γ = `tap_graft_decay`)

With γ=0 it re-sends the frozen watermarked head verbatim; small γ>0 lets the head drift toward the global head so it stays consistent with the moving core (this is what removes the late-run BER spikes seen in the single-seed runs). Because the mark is decoded **only** from the head's softmax and the head is a tiny fraction of parameters, preserving it costs ≈0 samples while the copied core keeps task accuracy at honest level. On a **tap** round it does a short real training step (block2 scope) to re-lower BER when it drifts up toward η.

**Why graft, and not the alternatives.** Three of these are exactly the *baseline* free-riders the detector is designed to catch (our H5/H6 positive controls, which sit near BER 0.5 in our runs):
- *Re-send previous global model (previous_models / H5).* No head preservation → the mark isn't at the free-rider's key → BER ≈ 0.5 → **caught**.
- *Gaussian noise (H6).* Noise/noised averages destroy the mark → BER ≈ 0.5 → **caught**.
- *Previous-model subtraction / delta tricks.* Same failure: nothing maintains the head's key-specific softmax pattern → BER floats to chance.
- *Re-send own last full watermarked model verbatim (`coast_mode=decay`).* This preserves the mark but a stale full model desyncs from the moving global core, so BER/accuracy drift — strictly worse than grafting a fresh core under a preserved head.

Graft is the only option that (a) preserves the mark AND (b) tracks the moving global model for free. The others fail (a) or (b). This is a clean paragraph for §6 and directly instantiates the §2 intuition (only the low-parameter output layer must be fooled).

## B. Tap scope: block2 vs. full (K4 vs. K5)

**Plain intuition.** When the free-rider does decide to do a little real work (a "tap"), it chooses how much of the model to nudge. "Full" wiggles the entire model to redo the handshake — lots of effort, and it disturbs the good copied body. "block2" wiggles only the last little chunk (final ResNet block + head) — the part the handshake actually reads. Same handshake fixed, far less work, body untouched.

**Technical.** `block2` restricts the tap's backward pass to the last ~20 tensors (final residual block + FC head); `full` updates all parameters. The BER gradient that lowers the mark concentrates near the head (that's where the mark is decoded), so updating the core barely lowers BER but costs a full backward pass and perturbs task-relevant core features. block2 re-embeds at a fraction of the FLOPs/samples and preserves the copied core.

**Why K5 (full) is worse — and the data shows it.** In the K5 runs the free-rider taps **95–100%** of rounds and saves **less** compute (≈67–69% vs. K4's ≈70–75%). Full-scope taps change the whole model each tap, desyncing the free-rider's self-probe from the server BER; the probe stays pessimistically high → it thinks it must tap almost every round → it stops coasting → it stops saving. block2 keeps probe and server BER aligned, so it coasts most rounds. **block2 is the effort-minimizing scope; full defeats the purpose.** That is exactly the ablation: K4 = headline, K5 = "why the tap must be localized."

## C. Is the attack working? (reading the current results)

Short answer: **yes**, by the metric that matters — recalibrate the expectation about the *shape*.

- **It evades under the (oracle) loose threshold.** In K4 (6 seeds) both free-riders sit under η_loose=0.264 in the converged tail (cid3 ≈0.04, cid6 ≈0.21) while spending ≈25–30% of honest compute (≈70–75% saved). That is a successful low-effort evasion and supports the headline claim.
- **Why it looks "less dramatic" than expected.** (1) With m=10 bits, BER is **quantized to tenths**, so a single-seed run is a blocky square wave, not a smooth sawtooth — the jagged J4/Y single-seed plots are that, not a bug; the 6-seed K4 is the clean one. (2) On the **hard class (cid6)** the honest twin itself rides at ≈0.12–0.21, so the free-rider barely has to dive to blend in — the submarine has shallow water to hide in. That is the class-dependence (Q2) doing double duty as cover, not weak attack.
- **The genuine weak spot to report, not hide.** On the **easy class** the honest twin is near 0, so the free-rider must keep BER very low and taps a bit more → smaller saving there. Report both; the class-dependence *is* a finding.
- **For the cleanest figure:** use K4 at 6 seeds with the combined-FR `timeline` view (seed-averaging fills in the tenths; two free-riders averaged smooth the teeth).

**One thing to verify before quoting savings numbers:** confirm `tap_data_cpc=5` actually held in each K/J run (the config echo didn't surface it). If any run logged `tap_data_cpc=-1`, a tap trained on the full shard and its "% saved" is inflated — grep `result.json["config"]["tap_data_cpc"]` across the runs first.