#!/usr/bin/env python
"""to_pgfplots -- turn result.json runs into pgfplots-ready .dat tables + .tex figures
=======================================================================================
default (3 seeds, std shown):
  fig1  timeline BER vs round, honest vs our free-rider (reduced + head2) -- FareMark,
        FedIPR, sign  -> fig1_faremark_timeline / fig1_fedipr_timeline / fig1_sign_timeline
  tab1  cost table (samples + GPU-time, honest vs FR) FareMark/FedIPR/sign     -> tab1_costs
  fig2  BER vs round: gaussian vs previous-models vs OUR attack -- FareMark,
        sign          -> fig2_attack_compare / fig2_sign_attack_compare
  fig3  FareMark class difficulty: per-class BER honest vs FR + entropy on side-> fig3_class_difficulty
  fig4  FedIPR-sign: final BER vs #watermarked layers (the fix vs our attack)  -> fig4_layers

Everything else (per-class band, overlap, iso, savings, non-IID, other datasets/models,
submarine tap views, ...) lives in APPENDIX_FIGURES and is emitted only with --appendix.

Each figure/table becomes:
  export/data/<name>.dat   whitespace table with a header row  (pgfplots `table`)
  export/fig/<name>.tex    a \\begin{figure}/\\begin{table} float that plots/prints it

Usage
  python scripts/to_pgfplots.py --res '.../results/*/result.json' --out export --tail 20
  python scripts/to_pgfplots.py --res '...'  --out export --appendix          # appendix set
  python scripts/to_pgfplots.py --res '...'  --out export --only fig4_layers   # one figure
"""
import argparse, glob, json, os, statistics as st
from collections import defaultdict

# ============================================================================
# PAPER FIGURES  (default) -- 3 seeds, std shown
# ============================================================================
FIGURES = [
    # ==========================================================================
    # FareMark (box-free) -- HEAD-ONLY free-rider (softmax fc only, last 2 tensors)
    # ==========================================================================
    dict(name="fig1_faremark_timeline_head", kind="timeline", fr="L6_graftblock_head_c36",
         eta_t=0.064, eta_l=0.264,
         caption="FareMark (CIFAR-100): honest clients vs.\\ our head-only free-rider "
                 "(softmax fc only, last 2 tensors), hard classes 3,6. Bands are $\\pm 1$ s.d.",
         takeaway="Restricting the free-rider to the softmax fc alone is still enough to re-embed "
                  "the FareMark mark and evade."),
    dict(name="fig1_faremark_timeline_head_c17", kind="timeline", fr="L7_graftblock_head_c17",
         eta_t=0.064, eta_l=0.264,
         caption="FareMark (CIFAR-100): honest clients vs.\\ our head-only free-rider "
                 "(softmax fc only), \\emph{easy} classes 1,7 -- companion to the hard-class panel. "
                 "Bands are $\\pm 1$ s.d.",
         takeaway="same as before but on the easy classes 1,7 to show the BER of a FR can be below the honest even"),

    # ==========================================================================
    # FedIPR white-box sign -- HEAD2 free-rider only (carrier lives in head2)
    # ==========================================================================
    dict(name="fig1_sign_timeline", kind="timeline", fr="G_L1_graftblock_head2_c36_ws",
         eta_t=0.20, eta_l=0.50,
         hon_emphasis=True, fr_mark=0.6, hon_label="honest",   
         caption="FedIPR white-box sign (CIFAR-100, 3 seeds, 1 output layer): watermark BER "
                 "vs.\\ round for honest clients and our head2 free-rider. Bands are $\\pm 1$ s.d.",
         takeaway="(white box fedipr style, resnet18, cifar100. use this for output layer white box scheme)"
                  "same story as faremark timeline, 0.5 BER is random and 0 is best. FR and honest match to avoid detection."
                  "white box version has much lower BER in general since sign watermark encodes each bit as the sign of a chosen weight + no sampling noise or softmax projection and class difficulty. \\jg{does someone know how to modify the figure such that the blue line is ABOVE the orange, please? } \\gz{i made the orange dots smaller idk if this is a bit better?}"),

    # ---- tab1: one combined cost table -- FareMark (head) + white-box sign (head2) ----
    dict(name="tab1_costs", kind="costtable",
         rows=[("FareMark (head)", ["L6_graftblock_head_c36", "L7_graftblock_head_c17"]),
               ("FedIPR-sign (head2)", ["G_L1_graftblock_head2_c36_ws"])],
         caption="Per-client training cost of an honest client vs.\\ our free-rider: FareMark with "
                 "a head-only (softmax fc) free-rider and the FedIPR white-box sign with a head2 "
                 "free-rider, CIFAR-100, mean $\\pm$ s.d.\\ over seeds. The FareMark row averages the "
                 "easy (1,7) and hard (3,6) free-riders, since the free-rider's cost is "
                 "class-independent (same scope, same reduced shard).",
         takeaway="The free-rider matches the honest watermark BER while spending only "
                  "$\\approx\\!0.3\\times$ the honest client's samples and GPU time -- evasion is cheap."),

    # ---- fig2: attack comparison (previous-models / gaussian / ours) ----
    dict(name="fig2_attack_compare_head", kind="attackcompare", honest="A1_honest_c100", ymax=0.8,
         attacks=[("previous models", "H5_prevmodel_c100"),
                  ("gaussian",        "H6_gaussian_c100"),
                  ("ours (head)",     "L6_graftblock_head_c36")],
         eta_t=0.064, eta_l=0.264,
         caption="FareMark (CIFAR-100): free-rider BER vs.\\ round for the two baselines and our "
                 "head-only attack (softmax fc only). Bands are $\\pm 1$ s.d.",
         takeaway="last layer only FR compared to the easy FR attacks from paper"),
    dict(name="fig2_sign_attack_compare", kind="attackcompare", honest="G_A1_honest_c100_ws", ymax=0.8,
         hon_on_top=True,   # honest floor coincides with ours (~0); draw it opaque, above the attack bands
         attacks=[("previous models", "G_H5_prevmodel_c100_ws"),
                  ("gaussian",        "G_H6_gaussian_c100_ws"),
                  ("ours (head2)",    "G_L1_graftblock_head2_c36_ws")],
         eta_t=0.20, eta_l=0.50,
         caption="FedIPR white-box sign (CIFAR-100, 3 seeds): free-rider BER vs.\\ round for the "
                 "two baseline attacks (previous-models, Gaussian) and ours. Bands are $\\pm 1$ s.d.",
         takeaway="Same picture for the white-box sign mark: the baselines are caught at chance, "
                  "our head2 free-rider evades at the honest floor. "),

    # ---- fig3: FareMark class difficulty -- (a) per-class BER bars (head), (b) entropy ----
    dict(name="fig3a_class_ber_head", kind="classbars",
         honest="A1_honest_c100", fr="L6_graftblock_head_c36",
         caption="FareMark (CIFAR-100): per trigger-class watermark BER, honest vs.\\ our "
                 "head-only free-rider (softmax fc only). Bars are mean $\\pm 1$ s.d.\\ over seeds.",
         takeaway="The fc-only free-rider still matches the honest per-class BER floor."),
    dict(name="fig3b_class_entropy", kind="classscatter", honest="A1_honest_c100",
         caption="FareMark (CIFAR-100, 3 seeds): watermark BER floor ($\\Delta$BER, mean over "
                 "seeds) vs.\\ softmax entropy on the trigger class, one labelled point per class. "
                 "Lower-entropy (more peaked, harder-to-embed) classes carry the higher BER floor.",
         takeaway="Harder, lower-entropy trigger classes carry a higher honest BER floor"),

    # ---- fig4: detection + cost on one panel vs #watermarked layers ----
    #   fixed head2 FR BER (climbs -> caught) + adaptive FR BER (~0, evades) + adaptive FR compute
    dict(name="fig4_detect_cost", kind="costdetect",
         fixed_fmt="G_L1_graftblock_head2_c36_ws_L{nl}",   # fixed head2 free-rider (detection)
         adapt_fmt="G_Ladapt_c36_ws_L{nl}",                # free-rider, reduced shard (cpc=5)
         full_fmt="G_Lfull_c36_ws_L{nl}",                  # free-rider, full data (cpc=-1)
         layers=[1, 6, 20],   # rn18/c100: head2=1, block2=6, full=20
         caption="FedIPR white-box sign (CIFAR-100) vs.\\ the number of layers $N$ the server embeds "
                 "the sign into. A \\emph{fixed-scope} free-rider that only ever retrains head2 (black) "
                 "loses the deeper bits as $N$ grows, so its watermark BER climbs and it is caught. An "
                 "\\emph{adaptive} free-rider that widens its retrained scope to cover the mark (orange) "
                 "keeps its watermark BER $\\approx 0$ on either data budget and always evades --- but "
                 "pays for it in compute: as a fraction of an honest client's cost it climbs with $N$, "
                 "reaching $\\approx\\!0.53$ on a reduced shard (green) and $\\approx\\!1.0$, the full "
                 "honest cost, on the full data budget (purple). Error bars are $\\pm 1$ s.d.\\ over seeds.",
         takeaway="the defender's dilemma --- a cheap fixed free-rider is caught by embedding deeper, "
                  "while a free-rider that stays hidden must widen its scope until, at full depth on full "
                  "data, it costs as much as an honest client."),
]

# ============================================================================
# APPENDIX FIGURES  (emitted only with --appendix)  -- TBD
#   non-IID (E/EA), other datasets (food101), other models, band/overlap/savings, etc.
# ============================================================================
APPENDIX_FIGURES = [
    # ---- FedIPR BACKDOOR (black-box) timelines -- MOVED here from the main paper set ----
    # dict(name="fig1_fedipr_timeline", kind="timeline", fr="F_L1_graftblock_head2_c36_fi",
    #      eta_t=0.20, eta_l=0.50,
    #      caption="FedIPR backdoor (CIFAR-100, 3 seeds): watermark BER (=$1-$trigger accuracy) "
    #              "vs.\\ round for honest clients and our head2 free-rider. Bands are $\\pm 1$ s.d.",
    #      takeaway="The black-box backdoor scheme behaves like FareMark: the head2 free-rider "
    #               "re-embeds the trigger set and evades at the honest floor."),
    # dict(name="fig1_fedipr_timeline_head", kind="timeline", fr="F_L6_graftblock_head_c36_fi",
    #      eta_t=0.20, eta_l=0.50,
    #      caption="FedIPR backdoor (CIFAR-100): honest vs.\\ our HEAD-ONLY free-rider (fc only). "
    #              "Bands are $\\pm 1$ s.d.",
    #      takeaway="Even fc-only, the backdoor free-rider re-memorises the trigger set and evades."),

    # dict(name="app_faremark_overlap", kind="overlap", honest="A1_honest_c100",
    #      fr=["L1_graftblock_head2_c36", "L5_graftblock_head2_c17",
    #          "K9_alldyn_head2_c36", "K9_alldyn_head2_c17"], eta_t=0.064, eta_l=0.264,
    #      caption="Honest per-class BER band vs.\\ free-rider operating points (FareMark). "
    #              "Free-riders land inside the band; no single threshold separates them.",
    #      takeaway="Every free-rider operating point falls inside the honest band, so no single "
    #               "BER threshold separates honest from free-rider."),
    # ---- (1) CLASS DIFFICULTY: ranked honest per-class BER, pooled over A1 + whichever T decades
    #   exist. x = RANK (not class id), so missing decades leave no gaps / bunching. One colour
    #   per decade (shows the spread is within every decade, not a decade artefact); the easiest /
    #   hardest classes are named. Caption numbers (@N@, @LO@, @HI@, @RATIO@) are filled from data.
    dict(name="app_faremark_class_band", kind="classrank",
         families=["A1_honest_c100",
                   "T4_honest_c100_cls1019", "T5_honest_c100_cls2029", "T8_honest_c100_cls3039",
                   "T1_honest_c100_cls4049", "T9_honest_c100_cls5059", "T6_honest_c100_cls6069",
                   "T7_honest_c100_cls7079", "T10_honest_c100_cls8089", "T2_honest_c100_cls9099"],
         n_label=3,   # name the 3 easiest + 3 hardest classes
         caption="FareMark (CIFAR-100): converged honest watermark BER of every trigger class run so "
                 "far (@N@ classes), sorted from easiest to hardest. Each dot is one class (mean "
                 "$\\pm 1$ s.d.\\ over seeds, last @TAIL@ rounds); colour marks the run (class "
                 "decade) it comes from. The honest floor ranges from @LO@ to @HI@ "
                 "(@RATIO@$\\times$) depending only on which class is the trigger.",
         takeaway="Class difficulty alone moves the honest BER floor by several-fold, so no single "
                  "threshold fits every client."),

    # ---- (2) NON-IID: (a) the alpha=0.5 timeline (unchanged content) + (b) converged BER vs alpha.
    #   (b) is one compact panel: honest mean + 10-90% client band, reduced FR (3,6) +-1 s.d. seeds,
    #   at alpha = 0.1 / 0.5 / 1.0 / IID. Missing settings are skipped.
    dict(name="app_niid_reduced_timeline", kind="niidalpha", wide=True,
         honest="E1_honest_niid_c100", ymax=0.6, eta_t=0.161, eta_l=0.576,
         attacks=[("reduced FR (random assign)",       "E2_reduced_niid_c36"),
                  ("reduced FR (distribution assign)", "EA2_reduced_niid_distrib_c36")],
         sweep=[("0.1", "E3_honest_niid_c100_a01", "E3_reduced_niid_c36_a01"),
                ("0.5", "E1_honest_niid_c100",     "E2_reduced_niid_c36"),
                ("1.0", "E3_honest_niid_c100_a10", "E3_reduced_niid_c36_a10"),
                ("IID", "A1_honest_c100",          "A3_reduced_c100_c36")],
         caption="Non-IID CIFAR-100. \\textbf{(a)} Dirichlet $\\alpha{=}0.5$: honest floor vs.\\ our "
                 "reduced free-rider (classes 3,6) under random (E2) and distribution-aware (EA2) "
                 "trigger assignment; bands are $\\pm 1$ s.d.\\ over seeds. \\textbf{(b)} Converged "
                 "BER (last @TAIL@ rounds) vs.\\ the Dirichlet concentration $\\alpha$ (smaller = more "
                 "skewed): honest mean with the 10--90\\,\\% band over honest clients, and the reduced "
                 "free-rider (random assignment) $\\pm 1$ s.d.\\ over seeds.",
         takeaway="Stronger skew raises and widens the honest floor, and the reduced free-rider stays "
                  "inside the honest band at every $\\alpha$."),

    # ---- (3) SUBMARINE (IID, head2): BER timeline / when it taps / what it costs, shared x axis.
    #   bottom panel: cumulative FR samples / honest samples -- solid = actual (taps on the reduced
    #   cpc shard), dashed = same tap schedule if every tap used full honest data. 
    dict(name="app_submarine_timeline_k9", kind="submarine",
         honest="A1_honest_c100", ymax=0.6, eta_t=0.064, eta_l=0.264,
         attacks=[("easy 1,7", "K9_alldyn_head2_c17"),
                  ("hard 3,6", "K9_alldyn_head2_c36")],
         caption="Submarine free-rider (FareMark, CIFAR-100, IID, head2 scope), 3 seeds. "
                 "\\textbf{Top:} watermark BER vs.\\ round, honest floor vs.\\ the submarine at easy "
                 "(1,7) and hard (3,6) classes ($\\pm 1$ s.d.). \\textbf{Middle:} what the free-riders "
                 "do each round (majority over free-riders $\\times$ seeds): honest warm-up, "
                 "\\emph{tap} (retrain on the reduced shard) or \\emph{coast} (no training). "
                 "\\textbf{Bottom:} cumulative training samples relative to an honest client; solid = "
                 "actual, dashed = the same tap schedule if every tap used full data.",
         takeaway="The submarine only taps when its BER drifts up, stays on the honest floor, and "
                  "ends at @COST@ of the honest cost (@COSTFULL@ if taps used full data)."),
    # ---- (4) ROC: no BER threshold separates honest clients from free-riders (the "money plot").
    #   negatives = honest clients across all trigger classes (A1 + every T decade + benign clients of
    #   the FR runs); positives = free-riders pooled over the head-only, reduced-shard and submarine
    dict(name="app_roc_faremark", kind="roc", fpr_budget=0.05,
         honest=["A1_honest_c100",
                 "T4_honest_c100_cls1019", "T5_honest_c100_cls2029", "T8_honest_c100_cls3039",
                 "T1_honest_c100_cls4049", "T9_honest_c100_cls5059", "T6_honest_c100_cls6069",
                 "T7_honest_c100_cls7079", "T10_honest_c100_cls8089", "T2_honest_c100_cls9099"],
         fr=["L6_graftblock_head_c36", "L7_graftblock_head_c17",       # head-only (our attack)
             "A2_reduced_c100_c17", "A3_reduced_c100_c36",             # reduced-shard
             "K9_alldyn_head2_c17", "K9_alldyn_head2_c36",             # submarine (head2)
             "K4_alldyn_block2_c17", "K4_alldyn_block2_c36"],          # submarine (block2)
         caption="FareMark (CIFAR-100): ROC of a per-client watermark-BER threshold detector. "
                 "Negatives are honest clients across every trigger class (@NNEG@ operating points); "
                 "positives are our free-riders (@NPOS@ points, pooled over the head-only, "
                 "reduced-shard and submarine variants). A client is flagged when its watermark BER "
                 "exceeds a threshold. The curve tracks the chance diagonal (AUC @AUC@), so no "
                 "threshold catches free-riders without flagging honest clients: at a @FPRB@\\% "
                 "false-positive budget only @TPRB@\\% of free-riders are detected. Tail-mean over "
                 "the last @TAIL@ rounds.",
         takeaway="The free-rider operating points lie inside the honest band, so a BER threshold "
                  "detects them only at chance (AUC @AUC@) --- watermark verification cannot "
                  "separate honest clients from free-riders."),
]

# ----------------------------------------------------------------------------
# Okabe-Ito, matching scripts/plots.py
COLORDEF = (r"\definecolor{chonest}{HTML}{0072B2}"  "\n"
            r"\definecolor{cfr}{HTML}{D55E00}"      "\n"
            r"\definecolor{cacc}{HTML}{009E73}"     "\n"
            r"\definecolor{cprev}{HTML}{000000}"    "\n"
            r"\definecolor{cfull}{HTML}{CC79A7}"    "\n"
            r"\definecolor{ctail}{HTML}{DDDDDD}"    "\n"
            r"\definecolor{cyel}{HTML}{E69F00}"     "\n"
            r"\definecolor{csky}{HTML}{56B4E9}"     "\n"
            r"\definecolor{cgrey}{HTML}{999999}"    "\n")
AXBASE = ("width=\\linewidth,height=4.2cm,cycle list={{chonest},{cfr},{black}},"
          "every axis plot/.append style={line width=1pt},"
          "legend style={font=\\scriptsize,draw=none,fill=white},"
          "tick label style={font=\\footnotesize}")


# ----------------------------------------------------------------------------
def _dataset_of(r):
    """The dataset a result was produced on (summary > config > manifest)."""
    return ((r.get("summary") or {}).get("dataset")
            or (r.get("config") or {}).get("dataset")
            or (r.get("manifest") or {}).get("dataset"))


def load(res_glob, dataset=None):
    """Group result.json by family"""
    runs = defaultdict(list)
    seen = defaultdict(set)
    for f in sorted(glob.glob(res_glob)):
        try:
            r = json.load(open(f))
        except Exception as e:
            print(f"  (skip {f}: {e})"); continue
        ds = _dataset_of(r)
        if dataset and ds and ds != dataset:
            continue
        fam = (r.get("manifest", {}) or {}).get("family")
        if fam:
            runs[fam].append(r); seen[fam].add(ds)
    if not dataset:
        mixed = sorted(fam for fam, dss in seen.items() if len({d for d in dss if d}) > 1)
        if mixed:
            print("  [WARN] families span >1 dataset and will be MERGED "
                  "(pass --dataset to separate): " + ", ".join(mixed))
    return runs

def _hist(r, tail=0):
    h = r.get("history", []) or []
    return h[-tail:] if tail else h

def per_class_honest(runs, tail):
    out = defaultdict(lambda: defaultdict(list))
    for r in runs:
        frs = set(r.get("free_rider_indices") or [])
        for h in _hist(r, tail):
            for p in (h.get("wm_per_client") or []):
                if p["cid"] in frs:
                    continue
                c = int(p["trigger_class"])
                for k in ("ber", "entropy", "dominance", "pmax"):
                    v = p.get(k)
                    if v is not None:
                        out[c][k].append(float(v))
    return out

def per_class_fr(runs, tail):
    """Per trigger-class BER of the FREE-RIDER clients (tail rounds)."""
    out = defaultdict(list)
    for r in runs:
        frs = set(r.get("free_rider_indices") or [])
        for h in _hist(r, tail):
            for p in (h.get("wm_per_client") or []):
                if p["cid"] in frs and p.get("ber") is not None:
                    out[int(p["trigger_class"])].append(float(p["ber"]))
    return out

def per_class_honest_by_seed(runs, tail):
    """Per trigger-class, per-SEED tail-mean of ber/entrop"""
    out = defaultdict(lambda: defaultdict(list))
    for r in runs:                                   # r = one seed/rep
        frs = set(r.get("free_rider_indices") or [])
        acc = defaultdict(lambda: defaultdict(list))
        for h in _hist(r, tail):
            for p in (h.get("wm_per_client") or []):
                if p["cid"] in frs:
                    continue
                c = int(p["trigger_class"])
                for k in ("ber", "entropy"):
                    v = p.get(k)
                    if v is not None:
                        acc[c][k].append(float(v))
        for c, d in acc.items():
            for k, vs in d.items():
                if vs:
                    out[c][k].append(st.mean(vs))    # this seed's tail-mean
    return out

def per_class_fr_by_seed(runs, tail):
    """Per trigger-class, per-SEED tail-mean BER of the FREE-RIDER clients."""
    out = defaultdict(list)
    for r in runs:                                   # r = one seed/rep
        frs = set(r.get("free_rider_indices") or [])
        acc = defaultdict(list)
        for h in _hist(r, tail):
            for p in (h.get("wm_per_client") or []):
                if p["cid"] in frs and p.get("ber") is not None:
                    acc[int(p["trigger_class"])].append(float(p["ber"]))
        for c, vs in acc.items():
            if vs:
                out[c].append(st.mean(vs))           # this seed's tail-mean
    return out

def per_class_test(runs):
    acc, loss = defaultdict(list), defaultdict(list)
    for r in runs:
        bc = ((r.get("per_class") or {}).get("by_class") or {})
        for c, d in bc.items():
            acc[int(c)].append(d["acc"]); loss[int(c)].append(d["loss"])
    return acc, loss

def pct(xs, q):
    xs = sorted(xs)
    if not xs: return float("nan")
    i = min(len(xs)-1, max(0, int(round(q*(len(xs)-1)))))
    return xs[i]

def _ms(v):
    return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)

def write_dat(path, header, rows):
    with open(path, "w") as f:
        f.write(" ".join(header) + "\n")
        for row in rows:
            f.write(" ".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in row) + "\n")

# short on-plot titles for the Overleaf figures 
TITLES = {
    "fig1_faremark_timeline":        "FareMark style (head2): honest vs.\\ free-rider BER timeline",
    "fig1_fedipr_timeline":          "FedIPR backdoor style (head2): honest vs.\\ free-rider BER timeline",
    "fig1_sign_timeline":            "FedIPR white-box style (head2, 1 layer): honest vs.\\ free-rider BER timeline",
    "fig2_attack_compare":           "FareMark style (head2): attack comparison",
    "fig2_sign_attack_compare":      "FedIPR white-box style (head2): attack comparison",
    "fig3a_class_ber":               "FareMark style (head2): per-class watermark BER",
    "fig3b_class_entropy":           "FareMark style (head2): BER floor vs.\\ entropy",
    # appendix
    "app_faremark_class_band":       "FareMark: honest BER floor per trigger class (ranked)",
    "app_niid_reduced_timeline":     "Non-IID (Dirichlet $\\alpha{=}0.5$): reduced free-rider under both trigger assignments",
    "app_submarine_timeline_k9":     "Submarine (head2, IID): honest floor vs.\\ tapping free-rider (1,7 and 3,6)",
    # head-only twins
    "fig1_faremark_timeline_head":   "FareMark style (head, hard 3,6): honest vs.\\ free-rider BER timeline",
    "fig1_faremark_timeline_head_c17": "FareMark style (head, easy 1,7): honest vs.\\ free-rider BER timeline",
    "fig1_fedipr_timeline_head":     "FedIPR backdoor style (head): honest vs.\\ free-rider BER timeline",
    "fig1_sign_timeline_head":       "FedIPR white-box style (head): honest vs.\\ free-rider BER timeline",
    "fig2_attack_compare_head":      "FareMark style (head): attack comparison",
    "fig2_sign_attack_compare_head": "FedIPR white-box style (head): attack comparison",
    "fig3a_class_ber_head":          "FareMark style (head): per-class BER",
}

def _env(fig):
    """figure* for wide (two-column-spanning) multi-panel appendix figures, else figure."""
    return "figure*" if fig.get("wide") else "figure"

def fig_open(fig, axopts):
    t = TITLES.get(fig.get("_base", fig["name"]))                      # short title above the axes (Overleaf)
    title_opt = f"title={{{t}}},title style={{font=\\small,yshift=-2pt}}," if t else ""
    return (f"\\begin{{{_env(fig)}}}[t]\\centering\n{COLORDEF}"
            f"\\begin{{tikzpicture}}\n\\begin{{axis}}[{AXBASE},{title_opt}{axopts}]\n")

def _cap(fig):
    """Caption + an optional bold one-sentence TAKEAWAY appended after it."""
    cap = fig["caption"]
    tk = fig.get("takeaway")
    if tk:
        cap += f" \\textbf{{Takeaway:}} {tk}"
    ds = fig.get("_dslabel")
    if ds:
        cap = cap.replace("CIFAR-100", ds).replace("cifar100", ds)
    return cap

def fig_close(fig):
    return (f"\\end{{axis}}\n\\end{{tikzpicture}}\n"
            f"\\caption{{{_cap(fig)}}}\\label{{fig:{fig['name']}}}\n\\end{{{_env(fig)}}}\n")

def fig_begin_multi(fig):
    """Open a float + tikzpicture WITHOUT an axis (multi-axis figures place their own axes)."""
    return f"\\begin{{{_env(fig)}}}[t]\\centering\n{COLORDEF}\\begin{{tikzpicture}}\n"

def fig_end_multi(fig):
    return (f"\\end{{tikzpicture}}\n"
            f"\\caption{{{_cap(fig)}}}\\label{{fig:{fig['name']}}}\n\\end{{{_env(fig)}}}\n")

# ============================================================================
# PAPER emitters
# ============================================================================
def emit_timeline(fig, runs, out, tail):
    """fig1: honest floor (benign clients in the FR run) vs FR mean BER per round, +-std over seeds."""
    fr = runs.get(fig["fr"])
    if not fr: return None
    by_round = defaultdict(lambda: {"fr": [], "hon": []})
    for r in fr:
        for h in _hist(r):
            rd = h["round"]
            if h.get("wm_fr_ber") is not None: by_round[rd]["fr"].append(h["wm_fr_ber"])
            if h.get("wm_benign_ber") is not None: by_round[rd]["hon"].append(h["wm_benign_ber"])
    rows, rmax = [], 0
    for rd in sorted(by_round):
        fm, fs = _ms(by_round[rd]["fr"]); hm, hs = _ms(by_round[rd]["hon"])
        rows.append((rd, fm, max(0.0, fm-fs), fm+fs, hm, max(0.0, hm-hs), hm+hs)); rmax = rd
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat),
              ["round","fr","fr_lo","fr_hi","hon","hon_lo","hon_hi"], rows)
    et, el = fig.get("eta_t"), fig.get("eta_l")
    tstart = max(1, rmax - tail + 1)
    eta = ""                                     # -- threshold lines disabled for now --
    # if et is not None:
    #     eta += (f"\\addplot[cprev,dashed,forget plot,domain=1:{rmax}]{{{et}}};\n"
    #             f"\\node[anchor=south west,font=\\scriptsize] at (axis cs:1,{et}) {{$\\eta_t$}};\n")
    # if el is not None:
    #     eta += (f"\\addplot[chonest,densely dashed,forget plot,domain=1:{rmax}]{{{el}}};\n"
    #             f"\\node[anchor=north west,font=\\scriptsize] at (axis cs:1,{el}) {{$\\eta_\\ell$}};\n")
    ymax = fig.get("ymax", 0.6)   # zoom the BER axis 
    frms = fig.get("fr_mark", 1.1)         
    hon_label = fig.get("hon_label", "honest floor")
    hon_emph = ("\\addplot[chonest!50,line width=3pt,mark=none,forget plot] "
                f"table[x=round,y=hon]{{{dat}}};\n") if fig.get("hon_emphasis") else ""
    tex = (fig_open(fig, f"xlabel={{communication round}},ylabel={{watermark BER}},ymin=0,ymax={ymax},"
                    "legend pos=north east,reverse legend") +   
           f"\\addplot[name path=flo,draw=none,forget plot] table[x=round,y=fr_lo]{{{dat}}};\n"
           f"\\addplot[name path=fhi,draw=none,forget plot] table[x=round,y=fr_hi]{{{dat}}};\n"
           f"\\addplot[cfr!15,forget plot] fill between[of=flo and fhi];\n"
           f"\\addplot[cfr,mark=*,mark size={frms}pt] table[x=round,y=fr]{{{dat}}};\\addlegendentry{{free-rider}}\n"
           f"\\addplot[name path=hlo,draw=none,forget plot] table[x=round,y=hon_lo]{{{dat}}};\n"
           f"\\addplot[name path=hhi,draw=none,forget plot] table[x=round,y=hon_hi]{{{dat}}};\n"
           f"\\addplot[chonest!22,forget plot] fill between[of=hlo and hhi];\n"   
           f"{hon_emph}"
           f"\\addplot[chonest,mark=none] table[x=round,y=hon]{{{dat}}};\\addlegendentry{{{hon_label}}}\n"
           f"{eta}" + fig_close(fig))
    return dat, tex

def _compute_cols(runs, fam):
    fams = fam if isinstance(fam, (list, tuple)) else [fam]
    rr = [r for f in fams for r in runs.get(f, [])]
    def col(key):
        vs = [(r.get("compute", {}).get("summary", {}) or {}).get(key) for r in rr]
        vs = [float(v) for v in vs if v is not None]
        return _ms(vs)
    return col, len(rr)

def emit_costtable(fig, runs, out, tail):
    """tab1: LaTeX table -- honest vs FR samples + GPU-time, mean+-std over seeds, both schemes."""
    rows, any_ok = [], False
    for label, fam in fig["rows"]:
        col, n = _compute_cols(runs, fam)
        if n == 0:
            continue
        any_ok = True
        hs_m, hs_s = col("honest_mean_samples"); fs_m, fs_s = col("fr_mean_samples")
        # GPU cost as within run free-rider/honest ratio (effort_ratio_gpu), averaged over seeds
        gr_m, gr_s = col("effort_ratio_gpu")
        ratio_s = (fs_m / hs_m) if hs_m else float("nan")
        rows.append((label, n, hs_m, hs_s, fs_m, fs_s, gr_m, gr_s, ratio_s))
    if not any_ok:
        return None
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat),
              ["scheme","seeds","hon_samp","hon_samp_sd","fr_samp","fr_samp_sd",
               "gpu_ratio","gpu_ratio_sd","ratio_samp"], rows)
    def pm(m, s, dp=0):
        if m != m: return "--"
        return f"${m:,.{dp}f} \\pm {s:,.{dp}f}$"
    body = ""
    for (label, n, hs_m, hs_s, fs_m, fs_s, gr_m, gr_s, rs) in rows:
        body += (f"\\multirow{{2}}{{*}}{{{label}}} & honest & {pm(hs_m,hs_s)} & $1.00$ & $1.00$ \\\\\n"
                 f" & FR (ours) & {pm(fs_m,fs_s)} & {pm(gr_m,gr_s,3)} & ${rs:.3f}$ \\\\\n"
                 f"\\midrule\n")
    if body.endswith("\\midrule\n"):
        body = body[:-len("\\midrule\n")]
    # -- "GPU / honest" = effort_ratio_gpu, "Samples / honest" = fr/honest sample ratio.
    tex = (f"\\begin{{table}}[t]\\centering\n"
           f"\\caption{{{_cap(fig)}}}\\label{{tab:{fig['name']}}}\n"
           f"\\setlength{{\\tabcolsep}}{{4pt}}\n"
           f"\\resizebox{{\\columnwidth}}{{!}}{{%\n"
           f"\\begin{{tabular}}{{llrrr}}\n\\toprule\n"
           f"Scheme & Client & Samples & GPU\\,/\\,honest & Samples\\,/\\,honest \\\\\n\\midrule\n"
           f"{body}"
           f"\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n")
    return dat, tex

def emit_attackcompare(fig, runs, out, tail):
    """fig2: honest floor + one BER-per-round line (+-std) per attack family."""
    parts = _attackcompare_parts(fig, runs, out, tail)
    if parts is None:
        return None
    dat, body = parts
    eta = ""                                     # -- threshold lines disabled for now  --
    ymax = fig.get("ymax", 0.6)   # zoom the BER axis
    tex = (fig_open(fig, f"xlabel={{communication round}},ylabel={{watermark BER}},ymin=0,ymax={ymax},"
                    "legend columns=2,legend style={at={(0.5,-0.32)},anchor=north,"
                    "/tikz/every even column/.append style={column sep=6pt}}")
           + body + eta + fig_close(fig))
    return dat, tex

def _attackcompare_parts(fig, runs, out, tail):
    """Shared by emit_attackcompare and the non-IID panel (a): writes the .dat, returns
    (dat, pgfplots body) or None."""
    atk = fig["attacks"]
    series = []  # (label, {round: [ber over seeds]} or None)
    hon_by_round = defaultdict(list)
    for label, fam in atk:
        rr = runs.get(fam, [])
        if not rr:
            series.append((label, None)); continue
        byr = defaultdict(list)
        for r in rr:
            for h in _hist(r):
                if h.get("wm_fr_ber") is not None:
                    byr[h["round"]].append(h["wm_fr_ber"])
                if h.get("wm_benign_ber") is not None:
                    hon_by_round[h["round"]].append(h["wm_benign_ber"])
        series.append((label, byr))
    hon_runs = runs.get(fig.get("honest"))
    if hon_runs:                                   # cleaner honest floor from the honest family
        hon_by_round = defaultdict(list)
        for r in hon_runs:
            for h in _hist(r):
                if h.get("wm_benign_ber") is not None:
                    hon_by_round[h["round"]].append(h["wm_benign_ber"])
    rounds = sorted(hon_by_round) or sorted({rd for _, b in series if b for rd in b})
    if not rounds:
        return None
    hdr = ["round", "hon", "hon_lo", "hon_hi"]
    cols = {"round": rounds}
    hm = [_ms(hon_by_round.get(rd, [])) for rd in rounds]
    cols["hon"] = [m for m, s in hm]; cols["hon_lo"] = [max(0.0, m-s) for m, s in hm]; cols["hon_hi"] = [m+s for m, s in hm]
    keys = []
    for i, (label, byr) in enumerate(series):
        k = f"a{i}"; keys.append((k, label, byr is not None))
        hdr += [k, f"{k}_lo", f"{k}_hi"]
        if byr is None:
            cols[k] = [float("nan")]*len(rounds); cols[f"{k}_lo"] = list(cols[k]); cols[f"{k}_hi"] = list(cols[k])
        else:
            ms = [_ms(byr.get(rd, [])) for rd in rounds]
            cols[k] = [m for m, s in ms]; cols[f"{k}_lo"] = [max(0.0, m-s) for m, s in ms]; cols[f"{k}_hi"] = [m+s for m, s in ms]
    rows = list(zip(*[cols[h] for h in hdr]))
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat), hdr, rows)
    rmax = rounds[-1]
    et, el = fig.get("eta_t"), fig.get("eta_l")
    palette = ["cprev", "cacc", "cfr", "chonest"]
    marks = ["square*", "triangle*", "*", "o"]
    hon_on_top = fig.get("hon_on_top", False)
    hon_fill = "chonest!50" if hon_on_top else "chonest!12"
    hon_band = (f"\\addplot[name path=hlo,draw=none,forget plot] table[x=round,y=hon_lo]{{{dat}}};\n"
                f"\\addplot[name path=hhi,draw=none,forget plot] table[x=round,y=hon_hi]{{{dat}}};\n"
                f"\\addplot[{hon_fill},forget plot] fill between[of=hlo and hhi];\n")
    hon_line = f"\\addplot[chonest,mark=none] table[x=round,y=hon]{{{dat}}};\\addlegendentry{{honest floor}}\n"
    atk_bands = ""; atk_lines = ""
    for i, (k, label, ok) in enumerate(keys):
        if not ok: continue
        col = palette[i % len(palette)]; mk = marks[i % len(marks)]
        msz = "0.6pt" if label.startswith("ours") else "1pt"  
        atk_bands += (f"\\addplot[name path={k}lo,draw=none,forget plot] table[x=round,y={k}_lo]{{{dat}}};\n"
                      f"\\addplot[name path={k}hi,draw=none,forget plot] table[x=round,y={k}_hi]{{{dat}}};\n"
                      f"\\addplot[{col}!12,forget plot] fill between[of={k}lo and {k}hi];\n")
        atk_lines += (f"\\addplot[{col},mark={mk},mark size={msz}] table[x=round,y={k}]{{{dat}}};"
                      f"\\addlegendentry{{{label}}}\n")
    body = (atk_bands + hon_band + hon_line + atk_lines) if hon_on_top \
           else (hon_band + hon_line + atk_bands + atk_lines)
    # if et is not None: eta += f"\\addplot[cprev,dashed,forget plot,domain=1:{rmax}]{{{et}}};\n"
    # if el is not None: eta += f"\\addplot[chonest,densely dashed,forget plot,domain=1:{rmax}]{{{el}}};\n"
    return dat, body

def emit_classbars(fig, runs, out, tail):
    """fig3a: per-class honest BER bars (all classes) + FR BER bars (its trigger classes)."""
    hon = runs.get(fig["honest"]); fr = runs.get(fig["fr"])
    if not hon: return None
    pch = per_class_honest_by_seed(hon, tail)          # tail-mean per seed -> clean over-seeds std
    pcf = per_class_fr_by_seed(fr, tail) if fr else {}
    hrows, frows = [], []
    for c in sorted(pch):
        hb = pch[c]["ber"]
        if not hb: continue
        hm, hs = _ms(hb); hrows.append((c, hm, hs))    # mean +/- s.d. over seeds
        fb = pcf.get(c, [])
        if fb:                                
            fm, fs = _ms(fb); frows.append((c, fm, fs))
    if not hrows: return None
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat), ["class","hon_ber","hon_sd"], hrows)
    fdat = f"{fig['name']}_fr.dat"
    write_dat(os.path.join(out, "data", fdat), ["class","fr_ber","fr_sd"], frows)
    fr_plot = (f"\\addplot[fill=cfr!55,draw=none,\n"
               f"  error bars/error bar style={{cfr,line width=0.8pt}},\n"
               f"  error bars/.cd,y dir=both,y explicit]\n"
               f"  table[x=class,y=fr_ber,y error=fr_sd]{{{fdat}}};\\addlegendentry{{free-rider}}\n"
               if frows else "")
    tex = (fig_open(fig, "ybar,bar width=5pt,xlabel={trigger class},ylabel={watermark BER},"
                    "xtick=data,ymin=0,enlarge x limits=0.08,legend pos=north west,legend cell align=left,"
                    "legend image code/.code={\\draw[#1,draw=none] (0cm,-0.09cm) rectangle (0.26cm,0.09cm);}") +
           f"\\addplot[fill=chonest!55,draw=none,\n"
           f"  error bars/error bar style={{chonest,line width=0.8pt}},\n"
           f"  error bars/.cd,y dir=both,y explicit]\n"
           f"  table[x=class,y=hon_ber,y error=hon_sd]{{{dat}}};\\addlegendentry{{honest}}\n"
           f"{fr_plot}" + fig_close(fig))
    return dat, tex

def emit_classscatter(fig, runs, out, tail):
    """fig3b: SCATTER (points, not bars) -- honest watermark BER floor (y) vs softmax
    entropy (x), ONE point per trigger class, mean +/- 1 s.d. over the 3 seeds on both axes"""
    hon = runs.get(fig["honest"])
    if not hon: return None
    pcs = per_class_honest_by_seed(hon, tail)          # per-seed tail-means -> clean std
    rows = []
    for c in sorted(pcs):
        en, be = pcs[c].get("entropy", []), pcs[c].get("ber", [])
        if not en or not be: continue
        em, es = _ms(en); bm, bs = _ms(be)
        rows.append((c, em, es, bm, bs))
    if not rows: return None
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat),
              ["class","entropy","entropy_sd","ber","ber_sd"], rows)
    # class-number label to the upper-right of each dot
    labels = "".join(
        f"\\node[font=\\footnotesize,anchor=west,inner sep=2pt] "
        f"at (axis cs:{e:.5f},{b:.5f}) {{{c}}};\n" for (c, e, es, b, bs) in rows)
    tex = (fig_open(fig, "height=4cm,xlabel={softmax entropy on trigger class},"
                    "ylabel={$\\Delta$ BER},ymin=0,ymajorgrids=true,"
                    "grid style={gray!25},enlarge x limits=0.16,enlarge y limits=0.14") +
           f"\\addplot[only marks,black,mark=*,mark size=2pt] "
           f"table[x=entropy,y=ber]{{{dat}}};\n"
           f"{labels}" + fig_close(fig))
    return dat, tex

def emit_layers(fig, runs, out, tail):
    """fig4: final BER (tail-mean per seed, then mean+-std over seeds) vs #watermarked layers."""
    def final_ber(fam, want_fr):
        rr = runs.get(fam, [])
        vals = []
        for r in rr:
            tailh = _hist(r, tail)
            if want_fr:
                v = [h["wm_fr_ber"] for h in tailh if h.get("wm_fr_ber") is not None]
            else:
                v = [h["wm_benign_ber"] for h in tailh if h.get("wm_benign_ber") is not None]
            if v: vals.append(st.mean(v))
        return _ms(vals), len(vals)
    rows = []
    for nl in fig["layers"]:
        (fm, fs), nf = final_ber(fig["fr_fmt"].format(nl=nl), True)
        (hm, hs), nh = final_ber(fig["honest_fmt"].format(nl=nl), False)
        if nf == 0 and nh == 0:
            continue
        rows.append((nl, fm, fs, hm, hs))
    if not rows: return None
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat), ["nl","fr","fr_sd","hon","hon_sd"], rows)
    et, el = fig.get("eta_t"), fig.get("eta_l")
    nlmin, nlmax = rows[0][0], rows[-1][0]
    eta = ""                                     # -- threshold lines disabled for now --
    # if et is not None:
    #     eta += (f"\\addplot[cprev,dashed,forget plot] coordinates {{({nlmin},{et}) ({nlmax},{et})}};\n"
    #             f"\\node[anchor=south east,font=\\scriptsize] at (axis cs:{nlmax},{et}) {{$\\eta_t$}};\n")
    # if el is not None:
    #     eta += f"\\addplot[chonest,densely dashed,forget plot] coordinates {{({nlmin},{el}) ({nlmax},{el})}};\n"
    tex = (fig_open(fig, "xlabel={number of watermarked layers $N$},ylabel={final watermark BER},"
                    "xtick=data,ymin=0,legend pos=north west,unbounded coords=discard") +
           f"\\addplot[cfr,mark=*,error bars/.cd,y dir=both,y explicit] "
           f"table[x=nl,y=fr,y error=fr_sd]{{{dat}}};\\addlegendentry{{free-rider}}\n"
           f"\\addplot[chonest,mark=square*,error bars/.cd,y dir=both,y explicit] "
           f"table[x=nl,y=hon,y error=hon_sd]{{{dat}}};\\addlegendentry{{honest}}\n"
           f"{eta}" + fig_close(fig))
    return dat, tex


def emit_costlayers(fig, runs, out, tail):
    """fig4 (cost framing): an adaptive free-rider vs #watermarked layers N"""
    def seed_vals(fam):
        rr = runs.get(fam, [])
        costs, bers = [], []
        for r in rr:
            c = (r.get("compute", {}).get("summary", {}) or {}).get("effort_ratio_gpu")
            if c is not None:
                costs.append(c)
            v = [h["wm_fr_ber"] for h in _hist(r, tail) if h.get("wm_fr_ber") is not None]
            if v:
                bers.append(st.mean(v))
        return costs, bers
    rows = []
    for nl in fig["layers"]:
        costs, bers = seed_vals(fig["fr_fmt"].format(nl=nl))
        if not costs and not bers:
            continue
        cm, cs = _ms(costs); bm, bs = _ms(bers)
        rows.append((nl, cm, cs, bm, bs))
    if not rows:
        return None
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat), ["nl", "cost", "cost_sd", "ber", "ber_sd"], rows)
    nlmin, nlmax = rows[0][0], rows[-1][0]
    tex = (fig_open(fig, "xlabel={number of watermarked layers $N$},"
                    "ylabel={fraction of honest cost / watermark BER},"
                    "xtick=data,ymin=0,ymax=1.08,legend pos=north west,unbounded coords=discard") +
           f"\\addplot[chonest,densely dashed,forget plot] coordinates {{({nlmin},1) ({nlmax},1)}};\n"
           f"\\node[anchor=south east,font=\\scriptsize] at (axis cs:{nlmax},1) {{honest cost}};\n"
           f"\\addplot[cfr,mark=*,error bars/.cd,y dir=both,y explicit] "
           f"table[x=nl,y=cost,y error=cost_sd]{{{dat}}};\\addlegendentry{{FR compute (rel.\\ honest)}}\n"
           f"\\addplot[cprev,mark=square*,error bars/.cd,y dir=both,y explicit] "
           f"table[x=nl,y=ber,y error=ber_sd]{{{dat}}};\\addlegendentry{{FR watermark BER}}\n"
           + fig_close(fig))
    return dat, tex

def emit_costdetect(fig, runs, out, tail):
    """MERGED fig4 (detection + cost, one axis) vs #watermarked layers N. Four series in [0,1]:
      (i)   FIXED-scope head2 free-rider watermark BER -- climbs as N grows -> CAUGHT.
      (ii)  ADAPTIVE free-rider watermark BER -- stays ~0 (either data budget) -> EVADES.
      (iii) ADAPTIVE free-rider compute on the REDUCED shard (effort_ratio_gpu) -- climbs with N.
      (iv)  ADAPTIVE free-rider compute on the FULL data budget -- climbs to ~1.0 (== honest)"""
    def final_ber(fam):
        vals = []
        for r in runs.get(fam, []):
            v = [h["wm_fr_ber"] for h in _hist(r, tail) if h.get("wm_fr_ber") is not None]
            if v: vals.append(st.mean(v))
        return _ms(vals), len(vals)
    def cost(fam):
        vals = [(r.get("compute", {}).get("summary", {}) or {}).get("effort_ratio_gpu")
                for r in runs.get(fam, [])]
        vals = [float(v) for v in vals if v is not None]
        return _ms(vals), len(vals)
    rows = []
    for nl in fig["layers"]:
        (dm, ds), nd = final_ber(fig["fixed_fmt"].format(nl=nl))     # fixed-FR BER (detection)
        (am, as_), na = final_ber(fig["adapt_fmt"].format(nl=nl))    # adaptive-FR BER (evasion)
        (cm, cs), nc = cost(fig["adapt_fmt"].format(nl=nl))          # adaptive-FR compute, reduced shard
        (fm, fs), nf = cost(fig["full_fmt"].format(nl=nl))           # adaptive-FR compute, full data (old fig4c)
        if nd == 0 and na == 0 and nc == 0 and nf == 0:
            continue
        rows.append((nl, dm, ds, am, as_, cm, cs, fm, fs))
    if not rows:
        return None
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat),
              ["nl", "fixber", "fixber_sd", "adber", "adber_sd",
               "adcost", "adcost_sd", "fullcost", "fullcost_sd"], rows)
    nlmin, nlmax = rows[0][0], rows[-1][0]
    tex = (fig_open(fig, "xlabel={number of watermarked layers $N$},"
                    "ylabel={BER / fraction of honest cost},"
                    "xtick=data,ymin=0,ymax=1.08,unbounded coords=discard,"
                    "legend columns=2,legend style={at={(0.5,-0.34)},anchor=north,"
                    "/tikz/every even column/.append style={column sep=6pt}}") +
           f"\\addplot[chonest,densely dashed,forget plot] coordinates {{({nlmin},1) ({nlmax},1)}};\n"
           f"\\node[anchor=south east,font=\\scriptsize] at (axis cs:{nlmax},1) {{honest cost}};\n"
           f"\\addplot[cprev,mark=square*,error bars/.cd,y dir=both,y explicit] "
           f"table[x=nl,y=fixber,y error=fixber_sd]{{{dat}}};\\addlegendentry{{fixed-scope FR: watermark BER (caught)}}\n"
           f"\\addplot[cfr,mark=*,error bars/.cd,y dir=both,y explicit] "
           f"table[x=nl,y=adber,y error=adber_sd]{{{dat}}};\\addlegendentry{{adaptive FR: watermark BER (evades)}}\n"
           f"\\addplot[cacc,mark=triangle*,error bars/.cd,y dir=both,y explicit] "
           f"table[x=nl,y=adcost,y error=adcost_sd]{{{dat}}};\\addlegendentry{{adaptive FR: compute, reduced data}}\n"
           f"\\addplot[cfull,mark=diamond*,error bars/.cd,y dir=both,y explicit] "
           f"table[x=nl,y=fullcost,y error=fullcost_sd]{{{dat}}};\\addlegendentry{{adaptive FR: compute, full data ($=$ honest)}}\n"
           + fig_close(fig))
    return dat, tex

# ============================================================================
# APPENDIX emitters (band / overlap / savings)
# ============================================================================
def emit_band(fig, runs, out, tail):
    # `families` pools several honest runs into one per-class band (e.g. A1 covers classes 0-9,
    # the T decades cover 10-99
    fams = fig.get("families") or ([fig["family"]] if fig.get("family") else [])
    r = [x for f in fams for x in runs.get(f, [])]
    if not r: return None
    pc = per_class_honest(r, tail); acc, loss = per_class_test(r)
    rows = []
    for c in sorted(pc):
        b = pc[c]["ber"]
        if not b: continue
        mean = st.mean(b)
        rows.append((c, mean, max(0.0, mean-pct(b,0.10)), max(0.0, pct(b,0.90)-mean),
                     st.mean(pc[c]["entropy"]) if pc[c]["entropy"] else 0.0,
                     st.mean(pc[c]["dominance"]) if pc[c]["dominance"] else 0.0,
                     st.mean(acc[c]) if acc[c] else 0.0, st.mean(loss[c]) if loss[c] else 0.0))
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat),
              ["class","ber","elo","ehi","entropy","dominance","acc","loss"], rows)
    tex = (fig_open(fig, "ybar,bar width=7pt,ylabel={watermark BER},xlabel={trigger class},"
                    "xtick=data,ymin=0,enlarge x limits=0.08") +
           "\\addplot+[chonest,fill=chonest!55,draw=chonest,error bars/.cd,y dir=both,y explicit]"
           f" table[x=class,y=ber,y error plus=ehi,y error minus=elo]{{{dat}}};\n" +
           fig_close(fig))
    return dat, tex

def emit_overlap(fig, runs, out, tail):
    hon = runs.get(fig["honest"])
    if not hon: return None
    pc = per_class_honest(hon, tail)
    hrows = [(c, st.mean(pc[c]["ber"]), pct(pc[c]["ber"],0.10), pct(pc[c]["ber"],0.90))
             for c in sorted(pc) if pc[c]["ber"]]
    if not hrows: return None
    hdat = f"{fig['name']}_honest.dat"
    write_dat(os.path.join(out, "data", hdat), ["class","ber","lo","hi"], hrows)
    frrows = []
    for fam in fig["fr"]:
        for r in runs.get(fam, []):
            frs = set(r.get("free_rider_indices") or [])
            per = defaultdict(list)
            for h in _hist(r, tail):
                for p in (h.get("wm_per_client") or []):
                    if p["cid"] in frs and p.get("ber") is not None:
                        per[int(p["trigger_class"])].append(p["ber"])
            for c, v in per.items():
                frrows.append((c, st.mean(v)))
    fdat = f"{fig['name']}_fr.dat"
    write_dat(os.path.join(out, "data", fdat), ["class","ber"], frrows)
    cmin = min(r[0] for r in hrows); cmax = max(r[0] for r in hrows)
    et = fig.get("eta_t")
    eta = ""                                     # -- threshold lines disabled for now --
    # eta = (f"\\addplot[cprev,dashed,forget plot] coordinates {{({cmin},{et}) ({cmax},{et})}};\n"
    #        f"\\node[anchor=south east,font=\\scriptsize] at (axis cs:{cmax},{et}) {{$\\eta_t$}};\n"
    #        if et is not None else "")
    tex = (fig_open(fig, "xlabel={trigger class},ylabel={watermark BER},ymin=0,xtick=data") +
           f"\\addplot[name path=lo,draw=none,forget plot] table[x=class,y=lo]{{{hdat}}};\n"
           f"\\addplot[name path=hi,draw=none,forget plot] table[x=class,y=hi]{{{hdat}}};\n"
           f"\\addplot[black!12,forget plot] fill between[of=lo and hi];\n"
           f"\\addlegendimage{{area legend,fill=black!12}}\\addlegendentry{{honest band}}\n"
           f"\\addplot[chonest,mark=*,mark size=1pt] table[x=class,y=ber]{{{hdat}}};\\addlegendentry{{honest mean}}\n"
           f"\\addplot[only marks,mark=x,cfr,mark size=3pt,line width=1pt] table[x=class,y=ber]{{{fdat}}};"
           f"\\addlegendentry{{free-riders}}\n{eta}" + fig_close(fig))
    return " + ".join([hdat, fdat]), tex

def emit_savings(fig, runs, out, tail):
    rows = []
    for i, fam in enumerate(fig["fr"]):
        rr = runs.get(fam, [])
        if not rr: continue
        s = [(r.get("compute",{}).get("summary",{}) or {}) for r in rr]
        fs = [x.get("effort_ratio_samples") for x in s if x.get("effort_ratio_samples") is not None]
        fg = [x.get("effort_ratio_gpu") for x in s if x.get("effort_ratio_gpu") is not None]
        rows.append((i, fam.replace("_","-"), st.mean(fs) if fs else 0.0, st.mean(fg) if fg else 0.0))
    if not rows: return None
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat), ["idx","family","frac_samples","frac_gpu"], rows)
    tex = (fig_open(fig, "ybar,bar width=8pt,ylabel={cost / honest client},"
                    f"symbolic x coords={{{','.join(r[1] for r in rows)}}},xtick=data,"
                    "x tick label style={rotate=25,anchor=east,font=\\scriptsize},ymin=0") +
           f"\\addplot[chonest,fill=chonest!55,draw=chonest] table[x=family,y=frac_samples]{{{dat}}};\\addlegendentry{{samples}}\n"
           f"\\addplot[cfr,fill=cfr!55,draw=cfr] table[x=family,y=frac_gpu]{{{dat}}};\\addlegendentry{{GPU time}}\n" +
           fig_close(fig))
    return dat, tex

# ============================================================================
# APPENDIX (v2) -- class-difficulty rank / non-IID + alpha sweep / submarine tap+cost
# ============================================================================
CIFAR100_NAMES = [
    "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn mower", "leopard", "lion", "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak tree", "orange", "orchid", "otter", "palm tree", "pear", "pickup truck", "pine tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow tree", "wolf", "woman", "worm"]
assert len(CIFAR100_NAMES) == 100

def class_name(c, runs_list=None):
    """CIFAR-100 fine-label name (torchvision index order)"""
    ds = None
    for r in (runs_list or []):
        ds = _dataset_of(r) or ds
    if (ds in (None, "cifar100")) and 0 <= int(c) < 100:
        return CIFAR100_NAMES[int(c)]
    return f"class {c}"

def fill_tokens(fig, **tok):
    f = dict(fig)
    for k in ("caption", "takeaway"):
        if f.get(k):
            for key, val in tok.items():
                f[k] = f[k].replace(f"@{key}@", str(val))
    return f

def classrank_stats(runs, families, tail):
    """Ranked honest per-class BER"""
    rows, groups, seen = [], [], set()
    for fam in families:
        rr = runs.get(fam) or []
        if not rr:
            continue
        pcs = per_class_honest_by_seed(rr, tail)
        cls = [c for c in sorted(pcs) if pcs[c].get("ber") and c not in seen]
        if not cls:
            continue
        gi = len(groups); groups.append((fam, min(cls), max(cls)))
        for c in cls:
            m, s = _ms(pcs[c]["ber"])
            rows.append(dict(cls=c, mean=m, sd=s, n=len(pcs[c]["ber"]), g=gi)); seen.add(c)
    rows.sort(key=lambda d: d["mean"])
    for i, d in enumerate(rows):
        d["rank"] = i + 1
    return rows, groups

def _tail_client_means(rr, tail, want_fr):
    """Per (seed, cid) tail-mean BER of honest (want_fr=False) or free-rider clients"""
    out = []
    for r in rr:
        frs = set(r.get("free_rider_indices") or [])
        acc = defaultdict(list)
        for h in _hist(r, tail):
            for p in (h.get("wm_per_client") or []):
                if ((p["cid"] in frs) == want_fr) and p.get("ber") is not None:
                    acc[p["cid"]].append(float(p["ber"]))
        out += [st.mean(v) for v in acc.values() if v]
    return out

def alpha_sweep_stats(runs, sweep, tail):
    """One row per alpha setting present: honest mean + p10/p90 over (seed, honest client)
    tail means; FR tail-mean wm_fr_ber per seed -> mean/sd over seeds."""
    rows = []
    for lbl, hf, ff in sweep:
        hon, fr = runs.get(hf) or [], runs.get(ff) or []
        if not hon and not fr:
            continue
        hv = _tail_client_means(hon or fr, tail, want_fr=False)
        fv = []
        for r in fr:
            v = [h["wm_fr_ber"] for h in _hist(r, tail) if h.get("wm_fr_ber") is not None]
            if v:
                fv.append(st.mean(v))
        hm = st.mean(hv) if hv else float("nan")
        fm, fs = _ms(fv)
        rows.append(dict(label=lbl, hmean=hm, hp10=pct(hv, 0.10), hp90=pct(hv, 0.90),
                         fmean=fm, fsd=fs, nh=len(hon), nf=len(fr)))
    return rows

def honest_round_stats(runs, honest_fam, fallback_fams=()):
    """{round: (mean, sd)} of wm_benign_ber over seeds (honest family, else benign clients
    of the given attack families)."""
    src = runs.get(honest_fam) or [r for f in fallback_fams for r in runs.get(f, [])]
    by = defaultdict(list)
    for r in src:
        for h in _hist(r):
            if h.get("wm_benign_ber") is not None:
                by[h["round"]].append(float(h["wm_benign_ber"]))
    return {rd: _ms(v) for rd, v in sorted(by.items())}

def submarine_stats(runs, fam):
    """Per round, for one submarine family (all FR cids x seeds):
         ber[rd]    = (mean, sd over seeds) of wm_fr_ber
         state[rd]  = majority action: 'tap' | 'coast' | 'warm' (trained, no tap/coast yet)
         tapfrac[rd]= share of FR x seeds that tapped (among tap/coast)
         cost[rd]   = mean cumulative FR samples / cumulative honest samples  (actual)
         costcf[rd] = same, if every round the FR trained had cost an honest round
                      (= the tap schedule WITHOUT the data reduction)"""
    rr = runs.get(fam) or []
    if not rr:
        return None
    ber_by, cnt = defaultdict(list), defaultdict(lambda: defaultdict(int))
    cost_by, cf_by = defaultdict(list), defaultdict(list)
    for r in rr:
        hist = _hist(r)
        rounds = [h["round"] for h in hist]
        for h in hist:
            if h.get("wm_fr_ber") is not None:
                ber_by[h["round"]].append(float(h["wm_fr_ber"]))
        comp = (r.get("compute", {}) or {}).get("per_client", {}) or {}
        frs = {int(c) for c in (r.get("free_rider_indices") or [])} or \
              {int(k) for k, c in comp.items() if c.get("is_free_rider")}
        hon = [k for k in comp if int(k) not in frs]
        def samples(k, rd):
            pr = (comp[k].get("per_round") or {})
            cell = pr.get(str(rd)) or pr.get(rd)
            return None if cell is None else float(cell.get("samples", 0.0))
        for k in comp:
            if int(k) not in frs:
                continue
            act = {t.get("round"): t.get("action") for t in (comp[k].get("trace") or [])
                   if t.get("action") in ("tap", "coast")}
            ch = ca = ccf = 0.0
            for rd in rounds:
                hs = [samples(j, rd) for j in hon]
                hs = [v for v in hs if v is not None]
                h_rd = st.mean(hs) if hs else 0.0
                fs = samples(k, rd)
                a = act.get(rd)
                trained = (a == "tap") or (a is None and (fs or 0.0) > 0)
                state = a or ("warm" if trained else None)
                if state:
                    cnt[rd][state] += 1
                ch += h_rd; ca += (fs or 0.0); ccf += (h_rd if trained else 0.0)
                if ch > 0:
                    cost_by[rd].append(ca / ch); cf_by[rd].append(ccf / ch)
    rounds = sorted(set(ber_by) | set(cnt) | set(cost_by))
    state, tapfrac = {}, {}
    for rd in rounds:
        c = cnt.get(rd, {})
        if c:
            state[rd] = max(("tap", "coast", "warm"), key=lambda s: (c.get(s, 0), s == "tap"))
            n = c.get("tap", 0) + c.get("coast", 0)
            tapfrac[rd] = c.get("tap", 0) / n if n else float("nan")
    cost = {rd: st.mean(v) for rd, v in cost_by.items()}
    costcf = {rd: st.mean(v) for rd, v in cf_by.items()}
    last = max(cost) if cost else None
    return dict(rounds=rounds, ber={rd: _ms(v) for rd, v in ber_by.items()}, state=state,
                tapfrac=tapfrac, cost=cost, costcf=costcf, n=len(rr),
                final=cost.get(last, float("nan")) if last else float("nan"),
                finalcf=costcf.get(last, float("nan")) if last else float("nan"))

def roc_stats(runs, honest_fams, fr_fams, tail, fpr_budget=0.05):
    """Per-client tail-mean watermark BER -> ROC of a BER-threshold free-rider detector"""
    def per_client(fam):
        out = []
        for r in runs.get(fam, []):
            acc = {}
            for h in _hist(r, tail):
                for p in (h.get("wm_per_client") or []):
                    if p.get("ber") is None:
                        continue
                    d = acc.setdefault(p.get("cid"), {"ber": [], "fr": bool(p.get("is_free_rider"))})
                    d["ber"].append(float(p["ber"]))
            for d in acc.values():
                if d["ber"]:
                    out.append((st.mean(d["ber"]), d["fr"]))
        return out
    neg, pos = [], []
    for fam in list(honest_fams) + list(fr_fams):
        for b, isfr in per_client(fam):
            (pos if isfr else neg).append(b)
    if not neg or not pos:
        return None
    # AUC via Mann-Whitney (P(pos score > neg score), ties 0.5) for the flag-if-BER>tau detector
    gt = eq = 0
    for a in pos:
        for b in neg:
            if a > b: gt += 1
            elif a == b: eq += 1
    auc = (gt + 0.5 * eq) / (len(pos) * len(neg))
    # ROC: sweep tau over all observed scores (flag if BER > tau)
    scores = sorted(set(neg + pos))
    roc = sorted({(sum(1 for x in neg if x > tau) / len(neg),
                   sum(1 for x in pos if x > tau) / len(pos))
                  for tau in [scores[0] - 1e-9] + scores})
    within = [(f, t) for f, t in roc if f <= fpr_budget]
    op = max(within, key=lambda t: t[1]) if within else (0.0, 0.0)   # best TPR within the FPR budget
    return dict(neg=neg, pos=pos, roc=roc, auc=auc, n_neg=len(neg), n_pos=len(pos),
                fpr_budget=fpr_budget, op=op)


GROUP_COLORS = ["chonest", "cfr", "cacc", "cfull", "cyel", "csky", "cprev", "cgrey"]
GROUP_MARKS  = ["*", "square*", "triangle*", "diamond*", "pentagon*", "o", "square", "triangle"]
SUB_COLORS   = ["cfr", "cfull", "cacc", "cyel"]
NAN = float("nan")

def emit_classrank(fig, runs, out, tail):
    """(1) ranked honest per-class BER (x = rank, so partial decade coverage never bunches)."""
    rows, groups = classrank_stats(runs, fig["families"], tail)
    if not rows:
        return None
    allruns = [r for f in fig["families"] for r in runs.get(f, [])]
    name = fig["name"]; N = len(rows)
    write_dat(os.path.join(out, "data", f"{name}.dat"), ["rank", "class", "ber", "sd", "group"],
              [(d["rank"], d["cls"], d["mean"], d["sd"], d["g"]) for d in rows])
    plots = ""
    for gi, (fam, lo, hi) in enumerate(groups):
        gd = f"{name}_g{gi}.dat"
        write_dat(os.path.join(out, "data", gd), ["rank", "class", "ber", "sd"],
                  [(d["rank"], d["cls"], d["mean"], d["sd"]) for d in rows if d["g"] == gi])
        col = GROUP_COLORS[gi % len(GROUP_COLORS)]; mk = GROUP_MARKS[gi % len(GROUP_MARKS)]
        plots += (f"\\addplot[only marks,{col},mark={mk},mark size=1.6pt,"
                  f"error bars/error bar style={{{col}!70,line width=0.5pt}},"
                  f"error bars/.cd,y dir=both,y explicit] "
                  f"table[x=rank,y=ber,y error=sd]{{{gd}}};\\addlegendentry{{classes {lo}--{hi}}}\n")
    k = min(fig.get("n_label", 3), N // 2)
    lab = rows[:k] + rows[N - k:] if k else []
    nodes = "".join(f"\\node[rotate=90,anchor=west,font=\\tiny,inner sep=1.5pt] at "
                    f"(axis cs:{d['rank']},{d['mean'] + d['sd']:.5f}) "
                    f"{{{class_name(d['cls'], allruns)} ({d['cls']})}};\n" for d in lab)
    top = max(d["mean"] + d["sd"] for d in rows)
    lo, hi = rows[0]["mean"], rows[-1]["mean"]
    ratio = f"{hi / lo:.0f}" if lo > 1e-3 else "$\\gg$10"
    f2 = fill_tokens(fig, N=N, LO=f"{lo:.3f}", HI=f"{hi:.3f}", RATIO=ratio, TAIL=tail)
    ncol = min(5, len(groups))
    tex = (fig_open(f2, f"xlabel={{trigger classes, ranked easiest $\\to$ hardest}},"
                        f"ylabel={{honest watermark BER}},ymin=0,ymax={top * 1.55:.4f},"
                        f"xmin=0,xmax={N + 1},clip=false,ymajorgrids=true,grid style={{gray!20}},"
                        f"legend columns={ncol},legend style={{at={{(0.5,-0.30)}},anchor=north,"
                        "/tikz/every even column/.append style={column sep=5pt}}")
           + plots + nodes + fig_close(f2))
    return f"{name}.dat", tex

def emit_niidalpha(fig, runs, out, tail):
    """(2) non-IID: (a) alpha=0.5 timeline (same as before) | (b) converged BER vs alpha."""
    parts = _attackcompare_parts(fig, runs, out, tail)
    rows = alpha_sweep_stats(runs, fig.get("sweep", []), tail)
    if parts is None and not rows:
        return None
    name = fig["name"]; ymax = fig.get("ymax", 0.6)
    W = "0.43\\textwidth"; H = "4.6cm"
    legend = "legend columns=1,legend cell align=left,legend style={at={(0.5,-0.30)},anchor=north}"
    tex = fig_begin_multi(fig)
    at = ""
    if parts is not None:
        dat, body = parts
        tex += (f"\\begin{{axis}}[{AXBASE},name=pa,width={W},height={H},"
                f"title={{(a) Dirichlet $\\alpha{{=}}0.5$ timeline}},title style={{font=\\small}},"
                f"xlabel={{communication round}},ylabel={{watermark BER}},ymin=0,ymax={ymax},{legend}]\n"
                f"{body}\\end{{axis}}\n")
        at = "at={(pa.south east)},anchor=south west,xshift=1.5cm,"
    if rows:
        sd = f"{name}_alpha.dat"
        write_dat(os.path.join(out, "data", sd),
                  ["x", "alpha", "hon", "hon_p10", "hon_p90", "fr", "fr_sd"],
                  [(i, r["label"], r["hmean"], r["hp10"], r["hp90"], r["fmean"], r["fsd"])
                   for i, r in enumerate(rows)])
        n = len(rows)
        xt = ",".join(str(i) for i in range(n))
        xl = ",".join(("IID" if r["label"].upper() == "IID" else f"${r['label']}$") for r in rows)
        tex += (f"\\begin{{axis}}[{AXBASE},{at}width={W},height={H},"
                f"title={{(b) converged BER vs.\\ skew}},title style={{font=\\small}},"
                f"xlabel={{Dirichlet $\\alpha$ (more skewed $\\leftarrow$)}},ylabel={{converged BER}},"
                f"xtick={{{xt}}},xticklabels={{{xl}}},xmin=-0.4,xmax={n - 0.6},ymin=0,ymax={ymax},"
                f"unbounded coords=jump,{legend}]\n"
                f"\\addplot[name path=blo,draw=none,forget plot] table[x=x,y=hon_p10]{{{sd}}};\n"
                f"\\addplot[name path=bhi,draw=none,forget plot] table[x=x,y=hon_p90]{{{sd}}};\n"
                f"\\addplot[chonest!18,forget plot] fill between[of=blo and bhi];\n"
                f"\\addplot[chonest,mark=square*,mark size=1.6pt] table[x=x,y=hon]{{{sd}}};"
                f"\\addlegendentry{{honest mean}}\n"
                f"\\addlegendimage{{area legend,fill=chonest!18,draw=chonest!18}}"
                f"\\addlegendentry{{honest 10--90\\,\\% clients}}\n"
                f"\\addplot[cfr,mark=*,mark size=1.6pt,error bars/.cd,y dir=both,y explicit] "
                f"table[x=x,y=fr,y error=fr_sd]{{{sd}}};\\addlegendentry{{reduced FR (random)}}\n"
                f"\\end{{axis}}\n")
    f2 = fill_tokens(fig, TAIL=tail)
    return name + ".dat", tex + fig_end_multi(f2)

def emit_submarine(fig, runs, out, tail):
    """(3) submarine: BER timeline / tap-coast raster / cumulative cost, one shared x axis."""
    stats = [(lbl, fam, submarine_stats(runs, fam)) for lbl, fam in fig["attacks"]]
    stats = [(l, f, s) for l, f, s in stats if s]
    if not stats:
        return None
    name = fig["name"]
    hon = honest_round_stats(runs, fig.get("honest"), [f for _, f, _ in stats])
    rounds = sorted(set(hon) | {rd for _, _, s in stats for rd in s["rounds"]})
    xmin, xmax = rounds[0] - 0.5, rounds[-1] + 0.5
    # --- data files
    hdr = ["round", "hon", "hon_lo", "hon_hi"]; cols = []
    for rd in rounds:
        m, s = hon.get(rd, (NAN, 0.0)); cols.append([rd, m, max(0.0, m - s), m + s])
    for i, (_, _, s) in enumerate(stats):
        hdr += [f"a{i}", f"a{i}_lo", f"a{i}_hi", f"a{i}_cost", f"a{i}_costcf",
                f"a{i}_tap", f"a{i}_coast", f"a{i}_warm"]
        y = len(stats) - i
        for row, rd in zip(cols, rounds):
            m, sd = s["ber"].get(rd, (NAN, 0.0)); stt = s["state"].get(rd)
            row += [m, max(0.0, m - sd), m + sd, s["cost"].get(rd, NAN), s["costcf"].get(rd, NAN),
                    y if stt == "tap" else NAN, y if stt == "coast" else NAN,
                    y if stt == "warm" else NAN]
    dat = f"{name}.dat"
    write_dat(os.path.join(out, "data", dat), hdr, [tuple(r) for r in cols])
    # scale only axis: all three panels share the exact same plot width, so rounds line up.
    X = f"xmin={xmin},xmax={xmax},scale only axis,width=0.80\\linewidth"
    # --- top: BER
    top = (f"\\begin{{axis}}[{AXBASE},name=s1,{X},height=3.0cm,ymin=0,ymax={fig.get('ymax', 0.6)},"
           f"ylabel={{watermark BER}},xticklabels={{}},legend pos=north east,legend cell align=left,"
           f"unbounded coords=jump]\n"
           f"\\addplot[name path=hlo,draw=none,forget plot] table[x=round,y=hon_lo]{{{dat}}};\n"
           f"\\addplot[name path=hhi,draw=none,forget plot] table[x=round,y=hon_hi]{{{dat}}};\n"
           f"\\addplot[chonest!18,forget plot] fill between[of=hlo and hhi];\n"
           f"\\addplot[chonest,mark=none,line width=1.2pt] table[x=round,y=hon]{{{dat}}};"
           f"\\addlegendentry{{honest floor}}\n")
    for i, (lbl, _, _) in enumerate(stats):
        c = SUB_COLORS[i % len(SUB_COLORS)]
        top += (f"\\addplot[name path=a{i}lo,draw=none,forget plot] table[x=round,y=a{i}_lo]{{{dat}}};\n"
                f"\\addplot[name path=a{i}hi,draw=none,forget plot] table[x=round,y=a{i}_hi]{{{dat}}};\n"
                f"\\addplot[{c}!12,forget plot] fill between[of=a{i}lo and a{i}hi];\n"
                f"\\addplot[{c},mark=none] table[x=round,y=a{i}]{{{dat}}};"
                f"\\addlegendentry{{submarine {lbl}}}\n")
    top += "\\end{axis}\n"
    # --- middle: what it does each round
    k = len(stats)
    yt = ",".join(str(k - i) for i in range(k))
    yl = ",".join("{" + lbl + "}" for lbl, _, _ in stats)   # braces: labels contain commas
    mid = (f"\\begin{{axis}}[{AXBASE},name=s2,at={{(s1.south west)}},anchor=north west,yshift=-0.15cm,"
           f"{X},height={0.42 * k:.2f}cm,ymin=0.4,ymax={k + 0.6},ytick={{{yt}}},"
           f"yticklabel style={{font=\\scriptsize}},"
           f"yticklabels={{{yl}}},xticklabels={{}},ymajorgrids=false,xmajorgrids=true,"
           f"grid style={{gray!15}},unbounded coords=discard]\n")
    for i in range(k):
        c = SUB_COLORS[i % len(SUB_COLORS)]
        mid += (f"\\addplot[only marks,mark=|,cgrey,mark size=2pt] table[x=round,y=a{i}_warm]{{{dat}}};\n"
                f"\\addplot[only marks,mark=o,{c},mark size=1.3pt] table[x=round,y=a{i}_coast]{{{dat}}};\n"
                f"\\addplot[only marks,mark=*,{c},mark size=1.6pt] table[x=round,y=a{i}_tap]{{{dat}}};\n")
    mid += "\\end{axis}\n"
    # --- bottom: cumulative cost
    bot = (f"\\begin{{axis}}[{AXBASE},name=s3,at={{(s2.south west)}},anchor=north west,yshift=-0.2cm,"
           f"{X},height=2.2cm,ymin=0,ymax=1.12,ylabel={{cost / honest}},"
           f"xlabel={{communication round}},unbounded coords=jump,legend columns=3,legend cell align=left,"
           f"legend style={{at={{(0.5,-0.42)}},anchor=north,"
           f"/tikz/every even column/.append style={{column sep=5pt}}}}]\n"
           f"\\addplot[chonest,densely dashed,forget plot,domain={xmin}:{xmax}] {{1}};\n"
           f"\\node[anchor=north east,font=\\scriptsize,chonest] at (axis cs:{xmax},1) {{honest}};\n")
    for i in range(k):
        c = SUB_COLORS[i % len(SUB_COLORS)]
        bot += (f"\\addplot[{c},forget plot] table[x=round,y=a{i}_cost]{{{dat}}};\n"
                f"\\addplot[{c},dashed,forget plot] table[x=round,y=a{i}_costcf]{{{dat}}};\n")
    bot += ("\\addlegendimage{only marks,mark=*,black,mark size=1.6pt}\\addlegendentry{tap}\n"
            "\\addlegendimage{only marks,mark=o,black,mark size=1.3pt}\\addlegendentry{coast}\n"
            "\\addlegendimage{only marks,mark=|,cgrey,mark size=2pt}\\addlegendentry{warm-up}\n"
            "\\addlegendimage{black}\\addlegendentry{actual cost}\n"
            "\\addlegendimage{black,dashed}\\addlegendentry{cost w/ full-data taps}\n"
            "\\end{axis}\n")
    cost = ", ".join(f"${s['final']:.2f}\\times$ ({l})" for l, _, s in stats)
    costf = ", ".join(f"${s['finalcf']:.2f}\\times$" for _, _, s in stats)
    f2 = fill_tokens(fig, COST=cost, COSTFULL=costf, TAIL=tail)
    return dat, fig_begin_multi(fig) + top + mid + bot + fig_end_multi(f2)

def emit_roc(fig, runs, out, tail):
    """(4) ROC of a per-client watermark-BER threshold detector"""
    s = roc_stats(runs, fig.get("honest", []), fig.get("fr", []), tail, fig.get("fpr_budget", 0.05))
    if s is None:
        return None
    name = fig["name"]; dat = f"{name}.dat"
    write_dat(os.path.join(out, "data", dat), ["fpr", "tpr"], s["roc"])
    b = s["fpr_budget"]; bpct = f"{b * 100:.0f}"; opf, opt = s["op"]; tprb = f"{opt * 100:.0f}"
    shade = (f"\\addplot[name path=rbz,draw=none,forget plot] coordinates {{(0,0) ({b},0)}};\n"
             f"\\addplot[name path=rbo,draw=none,forget plot] coordinates {{(0,1) ({b},1)}};\n"
             f"\\addplot[cprev!7,forget plot] fill between[of=rbz and rbo];\n"
             f"\\node[anchor=south west,font=\\scriptsize,cprev!70] at (axis cs:{b},0.02) "
             f"{{FPR $\\le$ {bpct}\\%}};\n")
    op = (f"\\addplot[cprev,only marks,mark=o,mark size=2.2pt,forget plot] coordinates {{({opf},{opt})}};\n"
          f"\\node[anchor=north west,font=\\scriptsize] at (axis cs:{opf},{opt}) {{{tprb}\\% TPR}};\n")
    f2 = fill_tokens(fig, AUC=f"{s['auc']:.2f}", NNEG=s["n_neg"], NPOS=s["n_pos"],
                     FPRB=bpct, TPRB=tprb, TAIL=tail)
    tex = (fig_open(f2, "width=6cm,height=6cm,xlabel={false-positive rate (honest clients flagged)},"
                    "ylabel={true-positive rate (free-riders detected)},"
                    "xmin=0,xmax=1,ymin=0,ymax=1,legend pos=south east,legend cell align=left")
           + shade
           + "\\addplot[cgrey,densely dashed] coordinates {(0,0) (1,1)};\\addlegendentry{chance (AUC 0.5)}\n"
           + f"\\addplot[cfr,mark=*,mark size=0.9pt,thick] table[x=fpr,y=tpr]{{{dat}}};"
             f"\\addlegendentry{{BER threshold (AUC {s['auc']:.2f})}}\n"
           + op + fig_close(f2))
    return dat, tex

EMIT = {"timeline": emit_timeline, "costtable": emit_costtable,
        "attackcompare": emit_attackcompare, "classbars": emit_classbars,
        "classscatter": emit_classscatter, "layers": emit_layers,
        "costlayers": emit_costlayers, "costdetect": emit_costdetect,
        "band": emit_band, "overlap": emit_overlap, "savings": emit_savings,
        "classrank": emit_classrank, "niidalpha": emit_niidalpha, "submarine": emit_submarine,
        "roc": emit_roc}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", default="export")
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--only", default=None)
    ap.add_argument("--dataset", default=None,
                    help="keep only runs from this dataset (e.g. cifar100 | food101)")
    ap.add_argument("--appendix", action="store_true",
                    help="emit the APPENDIX_FIGURES set instead of the paper set.")
    ap.add_argument("--name-prefix", default="", dest="name_prefix",
                    help="prepend to every figure/data filename AND its \\label, so a second dataset "
                         "(e.g. Food-101) does not collide with the CIFAR-100 files on the pgfplots "
                         "search path. E.g. --name-prefix food101_")
    ap.add_argument("--dataset-label", default=None, dest="dataset_label",
                    help="rewrite the hardcoded 'CIFAR-100' in captions to this (e.g. 'Food-101').")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "fig"), exist_ok=True)
    runs = load(a.res, a.dataset)
    print(f"loaded families: {sorted(runs)}")
    figset = APPENDIX_FIGURES if a.appendix else FIGURES
    only = set(a.only.split(",")) if a.only else None
    made = []
    for fig in figset:
        if only and fig["name"] not in only:
            continue
        f = fig
        if a.name_prefix or a.dataset_label:
            f = dict(fig, name=a.name_prefix + fig["name"], _base=fig["name"],
                     _dslabel=a.dataset_label)
        res = EMIT[f["kind"]](f, runs, a.out, a.tail)
        if res is None:
            print(f"  skip {f['name']} (family/data missing)"); continue
        dat, tex = res
        open(os.path.join(a.out, "fig", f["name"] + ".tex"), "w").write(tex)
        made.append(f["name"]); print(f"  wrote fig/{f['name']}.tex  <- {dat}")
    menu = "all_figures_appendix.tex" if a.appendix else "all_figures.tex"
    with open(os.path.join(a.out, menu), "w") as f:
        f.write("% \\input this, or copy individual \\input lines where you want each float.\n")
        for n in made:
            f.write(f"\\input{{plots/export/fig/{n}.tex}}\n")
    # open(os.path.join(a.out, "preamble_snippet.tex"), "w").write(PREAMBLE)
    # open(os.path.join(a.out, "README_OVERLEAF.md"), "w").write(README)
    kind = "appendix" if a.appendix else "paper"
    print(f"\n{len(made)} {kind} figures -> {a.out}/  (menu: {a.out}/{menu})")

if __name__ == "__main__":
    main()