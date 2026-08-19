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

\section{Storyline (to be removed later)}
\begin{itemize}
    \item output layer watermarking is a function of datasets and classes
    \item honest clients with difficult classes can be above threshold as opposed to freeriders with easy classes
    \item problem gets amplified for non-iid 
    \item create a reference algorithm for a generic output layer watermarking scheme 
    \item submarine attack is a way to generally attack these output layer attacks 
    \item test for other papers/watermarking schemes 
    \item prove theoretically if output-layer watermarking is BS (or not) 
\end{itemize}

Intuition of why the output-layer watermarking is ineffective: honest clients train the entire model while the free-riders can trick only the output-layer to pass the watermarking detection. As the output-layer as very few parameters compared to the entire model, it requires few work to achieve.
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
---
---

# Technical Appendix II — Model anatomy, the tap mechanism, and the new results

> New consolidated section (added after the working notes above). It gathers every
> mechanism explanation in one place: what the model *is*, what "freezing the head"
> literally does, how the self-probe/threshold-estimation/tap-decision work, why graft
> is optimal, and how to read the H / D / E / EA results. Where it refines an earlier
> note (esp. Appendix B on `block2`), that is flagged **[refines B]**.

## D. What a model *is*, and what "freezing the head" means (visual)

### D.1 A model is an ordered list of tensors

A **tensor** here is just one weight array with a name and a shape. ResNet‑18
(torchvision, `num_classes=100`) is an ordered list of **62 such tensors**, holding
**11,227,812 numbers** in total. `fc.weight` is one tensor of shape `(100, 512)` =
51,200 numbers; `fc.bias` is `(100,)` = 100 numbers; `layer4.1.conv2.weight` is
`(512,512,3,3)` = 2,359,296 numbers; and so on. (BatchNorm running mean/var are 60
**buffers**, *not* parameters — they carry no gradient and are copied, never trained.)

Think of the model as a stack you read **bottom → top**: raw pixels enter at the
bottom, each tensor transforms the signal, and the top tensor (`fc`) emits the 100
class logits that become the softmax the watermark is read from.

```
   INPUT image (3x32x32)
        │
        ▼
 ┌───────────────────────────── BODY (feature extractor) ─────────────────────────────┐
 │ idx  0  conv1.weight            (64,3,3,3)                                           │
 │ idx  1  bn1.weight  … bn1.bias                                                       │
 │ idx  … layer1.*  layer2.*  layer3.0.*  layer3.1.conv1/bn1   (generic edge/texture/  │
 │ idx 41  layer3.1.bn1.bias                                    shape features)         │
 └─────────────────────────────────────────────────────────────────────────────────────┘
 ══════════════ cut = 62 − 20 = index 42 ══════════ ("block2" keeps the last 20 tensors)
 ┌───────────────────────────── HEAD (what block2 taps) ──────────────────────────────┐
 │ idx 42  layer3.1.conv2.weight   (256,256,3,3)                                        │
 │ idx 45  layer4.0.conv1.weight   (512,256,3,3)   ← last residual STAGE (layer4)       │
 │ idx …   layer4.0/1.* conv+bn+downsample                                              │
 │ idx 60  fc.weight               (100,512)       ← THE CLASSIFIER                     │
 │ idx 61  fc.bias                 (100,)          ← the watermark is decoded from here │
 └─────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   100 logits ──softmax──► P  ──smooth f()──► project by key M ──sign──► m watermark bits
```

The watermark **B** is decoded *only* from that final softmax **P** (Eq. 13/15). So the
mark physically lives at the very top of the stack — in practice in `fc` (idx 60–61).

### D.2 "Body" vs "head" vs the scope knobs

| scope (`tap_scope`) | keeps trainable | tensors | scalars | % of weights |
|---|---|---|---|---|
| `head`  | `fc` only | last **2** | 51,300 | **0.46 %** |
| `block` | `layer4.1` + `fc` | last **8** | ~7.1 M | ~63 % |
| `block2`| `layer4` + tail of `layer3` + `fc` | last **20** | 9,035,364 | **80.5 %** |
| `full`  | everything | all **62** | 11,227,812 | 100 % |

**"Freezing the head/body" in code** = for each parameter tensor, set
`requires_grad = (i >= cut)`. Everything before the cut (the body) gets
`requires_grad=False`; everything from the cut up (the head) stays `True`.

### D.3 What training actually changes

One training step does three things, in order:

```
 1. FORWARD   run the WHOLE stack (body+head) on the batch → logits → loss L = L_cl + λ·L_wm
 2. BACKWARD  autograd computes ∂L/∂θ, but ONLY for tensors with requires_grad=True,
              and only propagates back far enough to reach them.
 3. STEP      optimizer does θ ← θ − lr·(∂L/∂θ) for tensors that received a gradient.
              Frozen tensors got no gradient ⇒ they are untouched ⇒ they stay EXACTLY
              at the global model's values.
```

So during a `block2` tap:

```
 body  (idx 0..41)  requires_grad=False   grad = None      θ unchanged (== global body)
 ───── cut ─────────────────────────────────────────────────────────────────────────
 head  (idx 42..61) requires_grad=True    grad computed    θ moves to re-lower BER
```

Two consequences worth stating in the paper:
- The forward pass is still full (you pay one full forward), but the backward pass is
  **truncated** at the cut — no gradients are computed for `layer1..layer3.1.conv1`.
  That makes a `block2` tap cheaper in **gpu_ms** than a `full` tap, but *not* cheaper
  in **samples** (see H below).
- Because the body is left at the global values, a tapped model = *(current-ish body)*
  + *(freshly re-embedded head)*. A **graft coast** takes this idea to the limit:
  don't even tap — just staple your saved head onto a verbatim copy of the current
  global body.

### D.4 Graft, drawn

```
 GLOBAL model this round W_g          FR's SAVED watermarked head θ_H★  (from last tap)
 ┌───────────── body θ_C^(t) ─────────────┐        ┌──── head (idx 42..61) ────┐
 │ conv1 … layer3.1.bn1  (idx 0..41)       │        │ layer3.1.conv2 … fc.*      │
 └─────────────────────────────────────────┘        └───────────────────────────┘
              │  copy verbatim (free)                          │  keep frozen (free)
              ▼                                                ▼
 SUBMITTED coast model  =  [ body = W_g body ]  ++  [ head = (1−γ)·θ_H★ + γ·W_g head ]
                            └ tracks the crowd ┘     └ carries the mark, γ=tap_graft_decay ┘
```

- γ = 0 → head is the frozen watermarked head verbatim.
- small γ > 0 → head bleeds slightly toward the current global head each round so a
  fully-frozen head doesn't desync from the slowly-moving body (this removes the
  late-run BER spikes seen in single-seed J4/K7).

## E. The self‑probe: the free‑rider running the server's verifier privately

The free‑rider cannot see the server's BER. So it **replicates the verifier on its own
held‑out trigger images**. During `_prepare` (once), it splits its shard:

```
 own shard ─▶  ├─ hold out up to H=16 trigger imgs  → PROBE SET Q   (never trained on)
               ├─ keep ≥8 trigger imgs              → embedded on during taps
               └─ d imgs per common class (cpc)     → reduced training loader  D̂
```

Its probe (`_probe_ber`) is line‑for‑line the server's check, only on Q instead of the
server's test bank:

```
 probe_BER(W) = BER( sign(mean_over_Q( project( f(softmax(W(Q))), key M ) )),  target bits B )
                └──────────────── identical to Eq. 15/16, same M, same B ──────────────┘
```

The **only** difference from the server is the *images*: Q comes from the attacker's
**training** shard; the server extracts on **test** triggers. That train/test gap is the
one quantity the attacker can't observe — and it is exactly what the safety **margin ρ**
absorbs (D.4 / G).

**Threshold self‑estimation.** During the last κ warm‑up rounds (the *calibration
window*), it records its probe BER each round into a list **C**. Once, at defection, it
freezes

```
 η̂  =  mean(C) + k·std(C)          # k = tap_eta_k = 3   → mirrors the server's μ+3σ rule
 ρ  =  max(margin, ρ_k·std(C))      # "derived" margin: widens when its own estimate is noisy
 τ  =  max(0, η̂ − ρ)                # the target BER it will steer to, kept safely below η
```

In the **oracle** ablation (J4/Y) it skips this and is *handed* the true η = 0.264 — an
upper bound on the defender's advantage — and still evades. `η̂` (self) is the honest,
no‑oracle version and is the flagship setting.

## F. The tap/coast control law (one round)

```
 every free‑ride round:
   M_coast ← CoastCandidate(W_g)                 # the model it WOULD submit if it coasts
   b       ← probe_BER(M_coast)                  # a forward pass on Q only → 0 training samples
   force   ← (coast_streak ≥ max_coast)          # safety: never coast forever
   tap     ← (b is None) or (b > τ)  or force    # "threshold" mode
   if tap:  freeze body(scope); honest‑wm update on D̂; unfreeze; save head; streak←0
   else:    submit M_coast; streak←1
```

Coast while the *probed* coasting BER stays under target τ; surface for a cheap tap the
moment it would drift above τ. Alternating produces the sawtooth ("submarine") BER.

## G. Why graft is optimal — and why not the cheap alternatives

The mark is an output‑space quantity, `head ∘ body`. Four "free" strategies exist; only
graft satisfies **both** (a) preserve the key‑specific softmax pattern *and* (b) track the
moving global body:

| strategy | preserves mark? | tracks global body? | result |
|---|---|---|---|
| **Gaussian noise** `W_g+N(0,σ²)` (H6) | ✗ isotropic noise randomises `sign(z_k)` → E[BER]→0.5 | n/a | **caught** |
| **Previous‑models** `2W_t−W_{t−1}` (H5) | ✗ global never trained on *this* key | n/a | **caught** |
| **Decay** = replay own last full tap | ✓ | ✗ whole body frozen → goes stale, desyncs, accuracy drifts | worse than graft |
| **Graft** = global body + frozen head | ✓ | ✓ | **evades, cheap** |

Mathematically, why it is *cheap*: the detector reads `m = n//10 = 10` sign bits from the
last‑layer softmax, so the watermark occupies a low‑dimensional readout. The gradient of
`L_wm` that fixes those signs concentrates at the head; re‑establishing the mark needs
`O(head)` work, while the honest client "wastes" `O(whole network)` work every round.
Graft operationalises this: take the `O(whole network)` part for free from the global,
pay only for the `O(head)` part, and pay even that only on the rare rounds the probe
demands. Gaussian/prev‑model destroy the readout; decay keeps it but forfeits the free
body improvement and drifts on any accuracy/anomaly axis.

## H. **[refines B]** Where the compute saving actually comes from — and the "few parameters" wording

A caution to bake into §2 and §6 so a reviewer can't turn it against us:

- **`block2` is not "a small chunk of parameters."** By tensor *count* it is the last 20
  of 62; by *weight* it is **80.5 %** of all scalars (the `layer4` convs are the widest in
  the net). The genuinely tiny part — the true "few parameters" of the §2 intuition — is
  the **output layer `fc`** (0.46 %), which is *where the mark is decoded*. State the
  intuition as **"the mark is read from, and re‑embedded through, the output layer, a
  <0.5 % slice"**, and treat `block2` as an *implementation scope for the tap*, not as the
  evidence for "few parameters."
- **The sample‑cost saving does not come from the parameter scope at all.** Per tap, the
  cost metric (samples seen) is `|D̂|` regardless of whether the tap is `block2` or `full`
  — both iterate the same reduced loader. The ~70 % sample saving comes from **(i) the
  reduced shard** (≈27 % of honest data per tap, cf. D1 / E2) and **(ii) coasting**
  (0‑sample rounds). Scope only affects **gpu_ms** (a `block2` backward is truncated) and,
  indirectly, **how often it taps**: `full`‑scope taps perturb the body, desync the
  self‑probe from the server BER, so the FR believes it must tap ~every round and stops
  coasting (that is the K5 result — taps 95–100 %, saves *less*). So the honest one‑liner
  is: **coasting + reduced shard buy the samples; localizing the tap to `block2` keeps the
  probe aligned so it can keep coasting.**
- **Why `block2` and not `head` (fc only)?** On hard classes, fc‑only re‑embedding is too
  weak to pull BER back under τ, so it would tap every round or fail; `full` over‑perturbs
  and desyncs. `block2` is the empirically found middle that re‑embeds strongly enough
  while leaving the probe usable. This *is* the K4(block2) vs K5(full) ablation, and a
  `head`‑scope run would complete the scope sweep on the cheap side.

## I. Reading the H / D / E / EA results

### I.1 H — positive controls (the detector is real)
`H5_prevmodel_c100` and `H6_gaussian_c100`: the free‑rider mean BER sits at **≈0.50–0.58
every round**, far above η_loose = 0.264, while honest sits at ≈0.05. So the detector
**flags naive zero‑work free‑riders on every round** — exactly what a working defence
should do. This is the control that makes the submarine's sub‑η evasion *mean* something:
the detector is not broken; it is specifically the *output‑layer* structure the submarine
exploits. Put H5/H6 in the paper immediately before the submarine result.

### I.2 D — data‑budget sweep (the attack's data floor is tiny)
`D1` sweeps common‑images‑per‑class ∈ {triggers‑only(0), +1, +2, +5, +10, full}:
- **Triggers‑only collapses**: BER → ≈0.44 (no common data ⇒ the softmax collapses and the
  mark can't be shaped) — *above* η_loose, i.e. this extreme is caught. You need *some*
  common data.
- **+1/class already passes**: converged BER ≈0.13, and **+2/+5/+10/full barely move it**
  (≈0.11–0.14). The curve is a cliff at 0, then flat.

Two paper points: (1) the marginal value of data above a tiny budget is ≈0, so the attack
is cheap by construction; (2) it directly **justifies cpc=5** (safely past the cliff) and
shows **cpc=10 (K7) buys essentially nothing over cpc=5 (K4)** — an argument *against*
spending the extra data. (Note D1's η_tight is 0.084 for classes 6,8, vs 0.064 for 3,6 —
another instance of class‑dependent thresholds.)

### I.3 E / EA — non‑IID amplification (and the honest fix)
- `E1` honest floors *balloon and spread* under Dirichlet α=0.5: **cls3 = 0.341**,
  cls1 = 0.175, cls6 = 0.174 — vs cls3 = 0.045 in IID (A1). The single‑threshold problem
  is far worse: no η separates honest‑hard from anything.
- `E2` reduced FR (classes 3,6, 27 % data): FR mean BER ≈0.25 — **below its own hard‑class
  honest floor (cls3 0.34)** and far below η_loose = 0.576. The attacker hides *inside* the
  honest cloud. `E3` (α=0.1 and α=1.0) shows the same across the non‑IID spectrum, with
  η_loose blowing up to 1.06 at α=0.1.
- `EA1` distribution‑aware assignment is the **honest‑side mitigation**: give each client a
  trigger class it actually holds, and most floors drop to **0.000** (a few remain, e.g.
  cls63 0.20, cls73 0.25). Report this honestly: *part* of the non‑IID floor inflation is
  an assignment artifact the defender can fix — **but** the fix repairs honest floors, it
  does not detect the attacker, because the submarine exploits the head/body asymmetry
  regardless of how trigger classes are assigned.

**Gap this exposes:** E2/E3 are the *reduced* (tap‑every‑round) attack under non‑IID, not
the adaptive submarine. The "non‑IID amplifies" story for *our* attack still needs at least
one **`adaptive_tap` run under Dirichlet** (a K‑family config with `PARTITION=dirichlet`).
Prioritise that over further IID tuning.

## J. Is K8 the best submarine config to run?

Short answer: **there is no single "best" config — K4 and K8 are two ends of a
cost↔stealth frontier, and which is "best" depends on the claim you headline.** Run K8,
but as a *complement* to K4, not a replacement.

- **K4 (derived margin, self‑η, dynamic warm‑up, block2, cpc5, max_coast6)** — the
  **flagship**. Its margin is *derived* (`ρ = max(0.03, σ_calib)`), so there is no
  hand‑tuned constant: it is the "fully self‑estimating, no oracle, no magic number"
  story, and it gives the best headline saving (≈70–76 %). Its weak spot: individual
  seeds can nick above η on the easy class.
- **K8 (fixed margin 0.12, block2, cpc5, max_coast8)** — the **stealth‑optimised** point.
  The wide 0.12 margin keeps every seed comfortably under η (kills the over‑threshold
  excursions visible in J4/K7), at the cost of a few more taps (slightly lower saving).
  Its weak spot for the *narrative*: 0.12 is a **hand‑picked constant** a reviewer will
  ask you to justify, which slightly undercuts the "fully dynamic/self‑estimating" pitch.

Recommendation:
1. Keep **K4 as the headline** (principled + best cost).
2. Run **K8** and present it as *the margin knob trading cost for stealth* — this directly
   answers "can you make it stealthier?" with "yes, one knob, here's the frontier."
3. Above both in priority for a solid paper: **one non‑IID submarine run** (§I.3 gap) and
   the **K5 full‑scope ablation** (§H). If compute is tight, do those before more IID
   margin variants.
4. Optional cheap add: a **`head`‑scope** run to complete the scope sweep on the small side.

**Config‑correctness note (from the code):** with `tap_margin_mode="derived"` the effective
margin is `max(tap_margin, margin_k·σ_calib)`, and with `warmup_mode="dynamic"` the calib
window (and thus η̂) is resolved *after convergence*, in the κ rounds just before defection
— so K4's η̂ and margin are both data‑driven, whereas K8 fixes the margin by hand. Say which
you used in each figure caption; the two are different threat models (self‑calibrated vs
partially hand‑tuned).