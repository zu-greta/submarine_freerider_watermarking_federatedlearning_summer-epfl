#!/usr/bin/env python
"""to_pgfplots -- turn result.json runs into pgfplots-ready .dat tables + .tex figures
=======================================================================================
default (3 seeds, std shown):
  fig1  timeline BER vs round, honest vs OUR free-rider (reduced + head2) -- FareMark,
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
  python scripts/to_pgfplots.py --res '/mnt/nfs/home/zu/results/*/result.json' --out export --tail 20
  python scripts/to_pgfplots.py --res '...'  --out export --appendix          # appendix set
  python scripts/to_pgfplots.py --res '...'  --out export --only fig4_layers   # one figure
"""
import argparse, glob, json, os, statistics as st
from collections import defaultdict

# ============================================================================
# PAPER FIGURES  (the only ones emitted by default) -- 3 seeds, std shown
# ============================================================================
FIGURES = [
    # ---- fig1: honest vs OUR free-rider (reduced data + head2), timeline, cifar-100 ----
    dict(name="fig1_faremark_timeline", kind="timeline", fr="L1_graftblock_head2_c36",
         eta_t=0.064, eta_l=0.264,
         caption="FareMark (CIFAR-100, 3 seeds): watermark BER vs.\\ communication round for "
                 "honest clients and our reduced-data head-only free-rider. The free-rider "
                 "re-embeds the mark and its BER stays inside the honest band, below the "
                 "detection threshold. Bands are $\\pm 1$ s.d.\\ over seeds."),
    dict(name="fig1_fedipr_timeline", kind="timeline", fr="F_L1_graftblock_head2_c36_fi",
         eta_t=0.20, eta_l=0.50,
         caption="FedIPR backdoor (CIFAR-100, 3 seeds): watermark BER (=$1-$trigger accuracy) "
                 "vs.\\ round for honest clients and our head-only free-rider. Same outcome as "
                 "FareMark: the free-rider evades. Bands are $\\pm 1$ s.d."),
    dict(name="fig1_sign_timeline", kind="timeline", fr="G_L1_graftblock_head2_c36_ws",
         eta_t=0.20, eta_l=0.50,
         caption="FedIPR white-box sign (CIFAR-100, 3 seeds, 1 output layer): watermark BER "
                 "(=Hamming$/N$) vs.\\ round for honest clients and our head2 free-rider. The "
                 "free-rider re-embeds the single-layer sign mark and evades, same as the "
                 "black-box schemes. Bands are $\\pm 1$ s.d."),

    # ---- tab1: cost of honest vs OUR free-rider, all three schemes ----
    dict(name="tab1_costs", kind="costtable",
         rows=[("FareMark", "L1_graftblock_head2_c36"),
               ("FedIPR",   "F_L1_graftblock_head2_c36_fi"),
               ("FedIPR-sign", "G_L1_graftblock_head2_c36_ws")],
         caption="Per-client training cost of an honest client vs.\\ our free-rider "
                 "(reduced data + head2), CIFAR-100, mean $\\pm$ s.d.\\ over seeds. Samples are "
                 "device-independent; GPU cost is the within-run free-rider/honest GPU-time "
                 "ratio (honest and free-rider are timed in the same run under identical GPU "
                 "load, so the ratio is unaffected by how many jobs shared the pool). "
                 "Both columns show the free-rider needs $\\approx\\!0.3\\times$ an honest client."),

    # ---- fig2: attack comparison (gaussian / previous-models / ours), FareMark ----
    dict(name="fig2_attack_compare", kind="attackcompare", honest="A1_honest_c100",
         attacks=[("previous models", "H5_prevmodel_c100"),
                  ("gaussian",        "H6_gaussian_c100"),
                  ("ours (head2)",    "L1_graftblock_head2_c36")],
         eta_t=0.064, eta_l=0.264,
         caption="FareMark (CIFAR-100, 3 seeds): free-rider BER vs.\\ round for the two "
                 "baseline attacks (previous-models, Gaussian) and ours. The baselines sit "
                 "near chance (caught); ours stays in the honest band (evades). "
                 "Bands are $\\pm 1$ s.d."),
    dict(name="fig2_sign_attack_compare", kind="attackcompare", honest="G_A1_honest_c100_ws",
         attacks=[("previous models", "G_H5_prevmodel_c100_ws"),
                  ("gaussian",        "G_H6_gaussian_c100_ws"),
                  ("ours (head2)",    "G_L1_graftblock_head2_c36_ws")],
         eta_t=0.20, eta_l=0.50,
         caption="FedIPR white-box sign (CIFAR-100, 3 seeds): free-rider BER vs.\\ round for the "
                 "two baseline attacks (previous-models, Gaussian) and ours. The baselines climb "
                 "to chance ($0.5$, caught); ours stays at the honest floor (evades). "
                 "Bands are $\\pm 1$ s.d."),

    # ---- fig3: FareMark class difficulty -- TWO plots: (a) BER bars, (b) entropy ----
    dict(name="fig3a_class_ber", kind="classbars",
         honest="A1_honest_c100", fr="L1_graftblock_head2_c36",
         caption="FareMark (CIFAR-100, 3 seeds): per trigger-class watermark BER for honest "
                 "clients vs.\\ our free-rider. Harder classes have a higher honest floor; the "
                 "free-rider sits at or below it. Bars are mean $\\pm 1$ s.d.\\ over seeds."),
    dict(name="fig3b_class_entropy", kind="classscatter", honest="A1_honest_c100",
         caption="FareMark (CIFAR-100, 3 seeds): watermark BER floor ($\\Delta$BER, mean over "
                 "seeds) vs.\\ softmax entropy on the trigger class, one labelled point per class. "
                 "Lower-entropy (more peaked, harder-to-embed) classes carry the higher BER floor "
                 "-- the classes the free-rider hides behind in Fig.~\\ref{fig:fig3a_class_ber}."),

    # ---- fig4a: DETECTION -- FIXED head2 free-rider, final BER vs #watermarked layers ----
    #   Claim A: hold the attacker at its cheap head2 scope; as the mark spreads past head2
    #   the free-rider cannot maintain the deep bits -> BER rises above eta -> caught.
    dict(name="fig4a_detection_layers", kind="layers",
         honest_fmt="G_A1_honest_c100_ws_L{nl}", fr_fmt="G_L1_graftblock_head2_c36_ws_L{nl}",
         layers=[1, 6, 20], eta_t=0.20, eta_l=0.50,
         caption="FedIPR white-box sign (CIFAR-100): final watermark BER vs.\\ the number of "
                 "layers the server embeds the sign into, for honest clients and a \\emph{fixed} "
                 "head2 free-rider (does not adapt its scope). At one output layer the free-rider "
                 "re-embeds and evades; embedding deeper -- past its head2 reach -- pushes its BER "
                 "above the threshold: deeper embedding \\emph{catches} the naive free-rider. "
                 "Error bars are $\\pm 1$ s.d.\\ over seeds."),

    # ---- fig4b: COST -- ADAPTIVE free-rider cost vs #watermarked layers ----
    dict(name="fig4b_cost_layers", kind="costlayers",
         fr_fmt="G_Ladapt_c36_ws_L{nl}", layers=[1, 6, 20],   # rn18/c100: head2=1, block2=6, full=20
         caption="FedIPR white-box sign watermark (CIFAR-100): an \\emph{adaptive} free-rider "
                 "widens the scope it retrains to cover the sign wherever it is embedded, so its "
                 "watermark BER stays near zero -- it always evades. But its compute, as a "
                 "fraction of an honest client's, climbs from $\\approx0.3$ at one (output) layer "
                 "toward $1.0$ (honest) once the mark spans the full depth: deep embedding does "
                 "not stop evasion, it removes the free-rider's compute saving. Error bars are "
                 "$\\pm 1$ s.d.\\ over seeds."),

    # ======================================================================
    # HEAD figures (TAP_SCOPE=head = softmax fc only,last 2 tensors)
    # ======================================================================
    dict(name="fig1_faremark_timeline_head", kind="timeline", fr="L6_graftblock_head_c36",
         eta_t=0.064, eta_l=0.264,
         caption="FareMark (CIFAR-100): honest clients vs.\\ our HEAD-ONLY free-rider "
                 "(softmax fc only, last 2 tensors). The output-space mark is re-embeddable by "
                 "the fc alone, so the free-rider still evades. Bands are $\\pm 1$ s.d."),
    dict(name="fig1_fedipr_timeline_head", kind="timeline", fr="F_L6_graftblock_head_c36_fi",
         eta_t=0.20, eta_l=0.50,
         caption="FedIPR backdoor (CIFAR-100): honest vs.\\ our HEAD-ONLY free-rider (fc only). "
                 "The fc alone re-memorises the trigger set, so the free-rider evades. "
                 "Bands are $\\pm 1$ s.d."),
    dict(name="fig1_sign_timeline_head", kind="timeline", fr="G_L6_graftblock_head_c36_ws",
         eta_t=0.20, eta_l=0.50,
         caption="FedIPR white-box sign (CIFAR-100): honest vs.\\ our HEAD-ONLY free-rider "
                 "(fc only). The sign carrier is the last BN scale, which is NOT in the fc, so a "
                 "head-only free-rider cannot maintain its bits and is CAUGHT -- unlike the "
                 "output-space schemes above. Bands are $\\pm 1$ s.d."),

    dict(name="tab1_costs_head", kind="costtable",
         rows=[("FareMark", "L6_graftblock_head_c36"),
               ("FedIPR",   "F_L6_graftblock_head_c36_fi"),
               ("FedIPR-sign", "G_L6_graftblock_head_c36_ws")],
         caption="Per-client training cost of an honest client vs.\\ our HEAD-ONLY free-rider "
                 "(reduced data + softmax fc only), CIFAR-100, mean $\\pm$ s.d.\\ over seeds. "
                 "GPU cost is the within-run free-rider/honest GPU-time ratio."),

    dict(name="fig2_attack_compare_head", kind="attackcompare", honest="A1_honest_c100",
         attacks=[("previous models", "H5_prevmodel_c100"),
                  ("gaussian",        "H6_gaussian_c100"),
                  ("ours (head)",     "L6_graftblock_head_c36")],
         eta_t=0.064, eta_l=0.264,
         caption="FareMark (CIFAR-100): free-rider BER vs.\\ round for the two baselines and our "
                 "HEAD-ONLY attack (softmax fc only). The baselines are caught; ours evades. "
                 "Bands are $\\pm 1$ s.d."),
    dict(name="fig2_sign_attack_compare_head", kind="attackcompare", honest="G_A1_honest_c100_ws",
         attacks=[("previous models", "G_H5_prevmodel_c100_ws"),
                  ("gaussian",        "G_H6_gaussian_c100_ws"),
                  ("ours (head)",     "G_L6_graftblock_head_c36_ws")],
         eta_t=0.20, eta_l=0.50,
         caption="FedIPR white-box sign (CIFAR-100): free-rider BER vs.\\ round for the two "
                 "baselines and our HEAD-ONLY attack (fc only). Here our head-only attack is "
                 "caught alongside the baselines, because the sign carrier sits below the fc. "
                 "Bands are $\\pm 1$ s.d."),

    dict(name="fig3a_class_ber_head", kind="classbars",
         honest="A1_honest_c100", fr="L6_graftblock_head_c36",
         caption="FareMark (CIFAR-100): per trigger-class watermark BER, honest vs.\\ our "
                 "HEAD-ONLY free-rider (softmax fc only). Bars are mean $\\pm 1$ s.d.\\ over seeds."),
]

# ============================================================================
# APPENDIX FIGURES  (emitted only with --appendix)  -- TBD
#   non-IID (E/EA), other datasets (food101), other models, band/overlap/savings, etc.
# ============================================================================
APPENDIX_FIGURES = [
    dict(name="app_faremark_overlap", kind="overlap", honest="A1_honest_c100",
         fr=["L1_graftblock_head2_c36", "L5_graftblock_head2_c17",
             "K9_alldyn_head2_c36", "K9_alldyn_head2_c17"], eta_t=0.064, eta_l=0.264,
         caption="Honest per-class BER band vs.\\ free-rider operating points (FareMark). "
                 "Free-riders land inside the band; no single threshold separates them."),
    dict(name="app_faremark_class_band", kind="band", family="A1_honest_c100",
         caption="Per-trigger-class honest watermark BER (FareMark, CIFAR-100)."),
    dict(name="app_faremark_savings", kind="savings",
         fr=["L1_graftblock_head2_c36", "L5_graftblock_head2_c17",
             "K9_alldyn_head2_c36", "K4_alldyn_block2_c36"],
         caption="Free-rider cost as a fraction of an honest client (samples and GPU-time)."),
    dict(name="app_fedipr_overlap", kind="overlap", honest="F_A1_honest_c100_fi",
         fr=["F_L1_graftblock_head2_c36_fi", "F_L5_graftblock_head2_c17_fi",
             "F_K9_alldyn_head2_c36_fi", "F_K9_alldyn_head2_c17_fi"], eta_t=0.20, eta_l=0.50,
         caption="Honest band vs.\\ free-rider points (FedIPR backdoor)."),
    dict(name="app_fedipr_sign_overlap", kind="overlap", honest="G_A1_honest_c100_ws",
         fr=["G_L1_graftblock_head2_c36_ws", "G_L5_graftblock_head2_c17_ws",
             "G_K9_alldyn_head2_c36_ws", "G_K9_alldyn_head2_c17_ws"], eta_t=0.20, eta_l=0.50,
         caption="Honest band vs.\\ free-rider points (FedIPR white-box sign, 1 layer)."),
    dict(name="app_niid_timeline_c36", kind="timeline", fr="E2_reduced_niid_c36",
         eta_t=0.161, eta_l=0.576,
         caption="Non-IID (Dirichlet $\\alpha{=}0.5$): honest vs.\\ reduced free-rider BER."),
    # TODO(appendix): other datasets (food101 -> point --res at results/food101), other models,
    #   submarine tap views, alpha sweep, distribution-aware assignment (EA), accuracy panels.
]

# ----------------------------------------------------------------------------
# Okabe-Ito, matching scripts/plots.py. Emitted into every figure so figures are
# self-contained and independent of your Set1 cycle list.
COLORDEF = (r"\definecolor{chonest}{HTML}{0072B2}"  "\n"
            r"\definecolor{cfr}{HTML}{D55E00}"      "\n"
            r"\definecolor{cacc}{HTML}{009E73}"     "\n"
            r"\definecolor{cprev}{HTML}{000000}"    "\n"
            r"\definecolor{ctail}{HTML}{DDDDDD}"    "\n")
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

def fig_open(fig, axopts):
    return (f"\\begin{{figure}}[t]\\centering\n{COLORDEF}"
            f"\\begin{{tikzpicture}}\n\\begin{{axis}}[{AXBASE},{axopts}]\n")

def fig_close(fig):
    return (f"\\end{{axis}}\n\\end{{tikzpicture}}\n"
            f"\\caption{{{fig['caption']}}}\\label{{fig:{fig['name']}}}\n\\end{{figure}}\n")

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
    eta = ""
    if et is not None:
        eta += (f"\\addplot[cprev,dashed,forget plot,domain=1:{rmax}]{{{et}}};\n"
                f"\\node[anchor=south west,font=\\scriptsize] at (axis cs:1,{et}) {{$\\eta_t$}};\n")
    if el is not None:
        eta += (f"\\addplot[chonest,densely dashed,forget plot,domain=1:{rmax}]{{{el}}};\n"
                f"\\node[anchor=north west,font=\\scriptsize] at (axis cs:1,{el}) {{$\\eta_\\ell$}};\n")
    tex = (fig_open(fig, "xlabel={communication round},ylabel={watermark BER},ymin=0,"
                    "legend pos=north east") +
           f"\\fill[ctail,opacity=0.5] (axis cs:{tstart},0) rectangle (rel axis cs:1,1);\n"
           f"\\addplot[name path=hlo,draw=none,forget plot] table[x=round,y=hon_lo]{{{dat}}};\n"
           f"\\addplot[name path=hhi,draw=none,forget plot] table[x=round,y=hon_hi]{{{dat}}};\n"
           f"\\addplot[chonest!12,forget plot] fill between[of=hlo and hhi];\n"
           f"\\addplot[chonest,mark=none] table[x=round,y=hon]{{{dat}}};\\addlegendentry{{honest floor}}\n"
           f"\\addplot[name path=flo,draw=none,forget plot] table[x=round,y=fr_lo]{{{dat}}};\n"
           f"\\addplot[name path=fhi,draw=none,forget plot] table[x=round,y=fr_hi]{{{dat}}};\n"
           f"\\addplot[cfr!15,forget plot] fill between[of=flo and fhi];\n"
           f"\\addplot[cfr,mark=*,mark size=1.1pt] table[x=round,y=fr]{{{dat}}};\\addlegendentry{{free-rider (ours)}}\n"
           f"{eta}" + fig_close(fig))
    return dat, tex

def _compute_cols(runs, fam):
    """mean,std over seeds for honest/FR samples and gpu_ms from compute.summary."""
    rr = runs.get(fam, [])
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
        # GPU cost as the WITHIN-RUN free-rider/honest ratio (effort_ratio_gpu), averaged
        # over seeds. honest & FR are timed in the same run under the same GPU contention,
        # so this ratio is concurrency-robust -- unlike absolute gpu_ms, it does NOT depend
        # on how many jobs shared the GPU. (samples ratio is contention-free by construction.)
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
    tex = (f"\\begin{{table}}[t]\\centering\n"
           f"\\caption{{{fig['caption']}}}\\label{{tab:{fig['name']}}}\n"
           f"\\begin{{tabular}}{{llrrr}}\n\\toprule\n"
           f"Scheme & Client & Samples (run) & GPU (\\(\\times\\) honest) & Cost / honest \\\\\n\\midrule\n"
           f"{body}"
           f"\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
    return dat, tex

def emit_attackcompare(fig, runs, out, tail):
    """fig2: honest floor + one BER-per-round line (+-std) per attack family."""
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
    body = (f"\\addplot[name path=hlo,draw=none,forget plot] table[x=round,y=hon_lo]{{{dat}}};\n"
            f"\\addplot[name path=hhi,draw=none,forget plot] table[x=round,y=hon_hi]{{{dat}}};\n"
            f"\\addplot[chonest!12,forget plot] fill between[of=hlo and hhi];\n"
            f"\\addplot[chonest,mark=none] table[x=round,y=hon]{{{dat}}};\\addlegendentry{{honest floor}}\n")
    for i, (k, label, ok) in enumerate(keys):
        if not ok: continue
        col = palette[i % len(palette)]; mk = marks[i % len(marks)]
        body += (f"\\addplot[name path={k}lo,draw=none,forget plot] table[x=round,y={k}_lo]{{{dat}}};\n"
                 f"\\addplot[name path={k}hi,draw=none,forget plot] table[x=round,y={k}_hi]{{{dat}}};\n"
                 f"\\addplot[{col}!12,forget plot] fill between[of={k}lo and {k}hi];\n"
                 f"\\addplot[{col},mark={mk},mark size=1pt] table[x=round,y={k}]{{{dat}}};\\addlegendentry{{{label}}}\n")
    eta = ""
    if et is not None: eta += f"\\addplot[cprev,dashed,forget plot,domain=1:{rmax}]{{{et}}};\n"
    if el is not None: eta += f"\\addplot[chonest,densely dashed,forget plot,domain=1:{rmax}]{{{el}}};\n"
    # legend INSIDE the axes (top strip, 2 cols) so it never overflows the column,
    # matching the timelines/fig4. Data plateaus <= ~0.5 so the upper half is free.
    tex = (fig_open(fig, "xlabel={communication round},ylabel={watermark BER},ymin=0,ymax=1,"
                    "legend pos=north east,legend columns=2,"
                    "legend style={/tikz/every even column/.append style={column sep=6pt}}")
           + body + eta + fig_close(fig))
    return dat, tex

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
        hm, hs = _ms(hb); hrows.append((c, hm, hs))    # mean +/- s.d. OVER SEEDS
        fb = pcf.get(c, [])
        if fb:                                  # FR only occupies its trigger class(es)
            fm, fs = _ms(fb); frows.append((c, fm, fs))
    if not hrows: return None
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat), ["class","hon_ber","hon_sd"], hrows)
    fdat = f"{fig['name']}_fr.dat"
    write_dat(os.path.join(out, "data", fdat), ["class","fr_ber","fr_sd"], frows)
    fr_plot = (f"\\addplot[cfr,fill=cfr!55,draw=cfr,error bars/.cd,y dir=both,y explicit] "
               f"table[x=class,y=fr_ber,y error=fr_sd]{{{fdat}}};\\addlegendentry{{free-rider (ours)}}\n"
               if frows else "")
    tex = (fig_open(fig, "ybar,bar width=5pt,xlabel={trigger class},ylabel={watermark BER},"
                    "xtick=data,ymin=0,enlarge x limits=0.08,legend pos=north west") +
           f"\\addplot[chonest,fill=chonest!55,draw=chonest,error bars/.cd,y dir=both,y explicit] "
           f"table[x=class,y=hon_ber,y error=hon_sd]{{{dat}}};\\addlegendentry{{honest}}\n"
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
    # class-number label to the upper-right of each dot (so a point traces back to fig3a).
    labels = "".join(
        f"\\node[font=\\scriptsize,anchor=west,inner sep=2pt] "
        f"at (axis cs:{e:.5f},{b:.5f}) {{{c}}};\n" for (c, e, es, b, bs) in rows)
    # CLEAN scatter (no error bars) + taller box + gridlines -> spread out, matches the
    # matplotlib "before" look. Override AXBASE height (last key wins) so points breathe.
    tex = (fig_open(fig, "height=6cm,xlabel={softmax entropy on trigger class},"
                    "ylabel={$\\Delta$ BER},ymin=0,ymajorgrids=true,"
                    "grid style={gray!25},enlarge x limits=0.16,enlarge y limits=0.14") +
           f"\\addplot[only marks,cfr,mark=*,mark size=2pt] "
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
    eta = ""
    if et is not None:
        eta += (f"\\addplot[cprev,dashed,forget plot] coordinates {{({nlmin},{et}) ({nlmax},{et})}};\n"
                f"\\node[anchor=south east,font=\\scriptsize] at (axis cs:{nlmax},{et}) {{$\\eta_t$}};\n")
    if el is not None:
        eta += f"\\addplot[chonest,densely dashed,forget plot] coordinates {{({nlmin},{el}) ({nlmax},{el})}};\n"
    tex = (fig_open(fig, "xlabel={number of watermarked layers $N$},ylabel={final watermark BER},"
                    "xtick=data,ymin=0,legend pos=north west,unbounded coords=discard") +
           f"\\addplot[cfr,mark=*,error bars/.cd,y dir=both,y explicit] "
           f"table[x=nl,y=fr,y error=fr_sd]{{{dat}}};\\addlegendentry{{free-rider (ours)}}\n"
           f"\\addplot[chonest,mark=square*,error bars/.cd,y dir=both,y explicit] "
           f"table[x=nl,y=hon,y error=hon_sd]{{{dat}}};\\addlegendentry{{honest}}\n"
           f"{eta}" + fig_close(fig))
    return dat, tex


def emit_costlayers(fig, runs, out, tail):
    """fig4 (cost framing): an adaptive free-rider vs #watermarked layers N.
    Two series on one [0,1] axis: (i) its compute as a fraction of an honest client's
    (effort_ratio_gpu, within-run FR/honest) -- rises toward 1.0 as the mark deepens;
    (ii) its watermark BER (tail-mean wm_fr_ber) -- stays ~0, it always evades."""
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

# ============================================================================
# APPENDIX emitters (band / overlap / savings) -- kept from the original exporter
# ============================================================================
def emit_band(fig, runs, out, tail):
    r = runs.get(fig["family"])
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
    eta = (f"\\addplot[cprev,dashed,forget plot] coordinates {{({cmin},{et}) ({cmax},{et})}};\n"
           f"\\node[anchor=south east,font=\\scriptsize] at (axis cs:{cmax},{et}) {{$\\eta_t$}};\n"
           if et is not None else "")
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

EMIT = {"timeline": emit_timeline, "costtable": emit_costtable,
        "attackcompare": emit_attackcompare, "classbars": emit_classbars,
        "classscatter": emit_classscatter, "layers": emit_layers,
        "costlayers": emit_costlayers,
        "band": emit_band, "overlap": emit_overlap, "savings": emit_savings}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", default="export")
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--only", default=None)
    ap.add_argument("--dataset", default=None,
                    help="keep only runs from this dataset (e.g. cifar100 | food101); "
                         "Food-101 reuses CIFAR-100 family names, so set this when a "
                         "results folder mixes datasets.")
    ap.add_argument("--appendix", action="store_true",
                    help="emit the APPENDIX_FIGURES set instead of the paper set.")
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
        res = EMIT[fig["kind"]](fig, runs, a.out, a.tail)
        if res is None:
            print(f"  skip {fig['name']} (family/data missing)"); continue
        dat, tex = res
        open(os.path.join(a.out, "fig", fig["name"] + ".tex"), "w").write(tex)
        made.append(fig["name"]); print(f"  wrote fig/{fig['name']}.tex  <- {dat}")
    menu = "all_figures_appendix.tex" if a.appendix else "all_figures.tex"
    with open(os.path.join(a.out, menu), "w") as f:
        f.write("% \\input this, or copy individual \\input lines where you want each float.\n")
        for n in made:
            f.write(f"\\input{{plots/export/fig/{n}.tex}}\n")
    open(os.path.join(a.out, "preamble_snippet.tex"), "w").write(PREAMBLE)
    open(os.path.join(a.out, "README_OVERLEAF.md"), "w").write(README)
    kind = "appendix" if a.appendix else "paper"
    print(f"\n{len(made)} {kind} figures -> {a.out}/  (menu: {a.out}/{menu})")

PREAMBLE = r"""% --- paste into your main.tex preamble (once) ---
\usepackage{pgfplots}
\usepgfplotslibrary{fillbetween}     % for the +-std / honest bands
\usepackage{multirow}                % for the cost table (tab1)
\usepackage{booktabs}                % \toprule \midrule \bottomrule
\pgfplotsset{compat=1.17}
% pgfplots resolves table{...} relative to MAIN.tex; point it at the data folder:
\pgfplotsset{table/search path={plots/export/data}}   % <-- set to where your .dat live
% The exported figures set their own colours + per-axis `cycle list`, so they do NOT
% depend on colorbrewer/Set1.
"""

README = r"""# Overleaf: paper figures + table (vector pgfplots, matplotlib-matched)

`.dat` = raw numbers, `.tex` = the pgfplots/tabular that draws them. Paper set:
  fig1_faremark_timeline, fig1_fedipr_timeline, fig1_sign_timeline   (fig 1)
  tab1_costs                                      (table 1: FareMark/FedIPR/sign)
  fig2_attack_compare, fig2_sign_attack_compare   (fig 2)
  fig3a_class_ber, fig3b_class_entropy            (fig 3, two panels)
  fig4_layers                                     (fig 4)
Appendix set (regenerate with `--appendix`): app_* figures.

## Preamble (once)
See preamble_snippet.tex. Needs: pgfplots (+fillbetween), booktabs, multirow.
Set the data search path:  \pgfplotsset{table/search path={plots/export/data}}

## Place & reference
Each fig/<name>.tex is a full float with \caption+\label. \input it where you want:
    \input{plots/export/fig/fig1_faremark_timeline.tex}
    \input{plots/export/fig/tab1_costs.tex}
Span both IEEE columns: change \begin{figure} -> \begin{figure*} in that file.
Reference with \ref{fig:<name>} or \ref{tab:tab1_costs}.
Do NOT route through an externalization/\inputplot macro.

## Regenerate (automated from the runbook)
    ./runbook.sh paper       # paper set   -> export/
    ./runbook.sh appendix    # appendix set
or directly:
    python scripts/to_pgfplots.py --res '<results>/*/result.json' --out export --tail 20
    python scripts/to_pgfplots.py --res '<results>/*/result.json' --out export --appendix
"""

if __name__ == "__main__":
    main()