#!/usr/bin/env python
"""to_pgfplots -- turn result.json runs into pgfplots-ready .dat tables + .tex figures
=======================================================================================
default:
  fig1  timeline BER vs round, honest vs OUR free-rider (reduced + head2) -- one FareMark,
        one FedIPR                                   -> fig1_faremark_timeline / fig1_fedipr_timeline
  tab1  cost table (samples + GPU-time, honest vs FR) for FareMark & FedIPR   -> tab1_costs
  fig2  BER vs round: gaussian vs previous-models vs OUR attack (FareMark)     -> fig2_attack_compare
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

# TODO add the fedipr white box runs

# ============================================================================
# PAPER FIGURES  (default) -- 3 seeds, std shown 
# - TODO change captions when they have been decided
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

    # ---- tab1: cost of honest vs OUR free-rider, both schemes ----
    dict(name="tab1_costs", kind="costtable",
         rows=[("FareMark", "L1_graftblock_head2_c36"),
               ("FedIPR",   "F_L1_graftblock_head2_c36_fi")],
         caption="Per-client training cost of an honest client vs.\\ our free-rider "
                 "(reduced data + head2), CIFAR-100, mean $\\pm$ s.d.\\ over 3 seeds. "
                 "Samples are device-independent; GPU-time is wall-time on the shared pool "
                 "(absolute values indicative, the honest/FR \\emph{ratio} is the reliable "
                 "quantity)."),

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

    # ---- fig3: FareMark class difficulty -- TWO plots: (a) BER bars, (b) entropy ----
    dict(name="fig3a_class_ber", kind="classbars",
         honest="A1_honest_c100", fr="L1_graftblock_head2_c36",
         caption="FareMark (CIFAR-100, 3 seeds): per trigger-class watermark BER for honest "
                 "clients vs.\\ our free-rider. Harder classes have a higher honest floor; the "
                 "free-rider sits at or below it. Bars are mean $\\pm 1$ s.d.\\ over seeds."),
    dict(name="fig3b_class_entropy", kind="classentropy", honest="A1_honest_c100",
         caption="FareMark (CIFAR-100, 3 seeds): mean softmax entropy on each trigger class. "
                 "Lower-entropy (peaked) classes are the harder-to-embed ones -- the same "
                 "classes with the higher BER floor in Fig.~\\ref{fig:fig3a_class_ber}."),

    # ---- fig4: FedIPR-sign final BER vs #watermarked layers (the fix) ----
    dict(name="fig4_layers", kind="layers",
         honest_fmt="G_A1_honest_c100_ws_L{nl}", fr_fmt="G_L1_graftblock_head2_c36_ws_L{nl}",
         layers=[1, 2, 4], eta_t=0.20, eta_l=0.50,
         caption="FedIPR white-box sign watermark (CIFAR-100, 3 seeds): final watermark BER "
                 "vs.\\ the number of normalization layers the mark is embedded into, for "
                 "honest clients and our head-only free-rider. At one (output) layer the "
                 "free-rider evades; embedding into deeper layers it cannot retrain pushes its "
                 "BER above the threshold -- the defence against our attack. Error bars are "
                 "$\\pm 1$ s.d.\\ over seeds."),
]

# ============================================================================
# APPENDIX FIGURES  (emitted only with --appendix)  -- TBD, placeholders for now
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
def load(res_glob):
    runs = defaultdict(list)
    for f in sorted(glob.glob(res_glob)):
        try:
            r = json.load(open(f))
        except Exception as e:
            print(f"  (skip {f}: {e})"); continue
        fam = (r.get("manifest", {}) or {}).get("family")
        if fam:
            runs[fam].append(r)
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
        hg_m, hg_s = col("honest_mean_gpu_ms");  fg_m, fg_s = col("fr_mean_gpu_ms")
        ratio_s = (fs_m / hs_m) if hs_m else float("nan")
        rows.append((label, n, hs_m, hs_s, fs_m, fs_s, hg_m/1e3, hg_s/1e3, fg_m/1e3, fg_s/1e3,
                     ratio_s))
    if not any_ok:
        return None
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat),
              ["scheme","seeds","hon_samp","hon_samp_sd","fr_samp","fr_samp_sd",
               "hon_gpu_s","hon_gpu_s_sd","fr_gpu_s","fr_gpu_s_sd","ratio_samp"], rows)
    def pm(m, s, dp=0):
        if m != m: return "--"
        return f"${m:,.{dp}f} \\pm {s:,.{dp}f}$"
    body = ""
    for (label, n, hs_m, hs_s, fs_m, fs_s, hg_m, hg_s, fg_m, fg_s, rs) in rows:
        body += (f"\\multirow{{2}}{{*}}{{{label}}} & honest & {pm(hs_m,hs_s)} & {pm(hg_m,hg_s,1)} & $1.00$ \\\\\n"
                 f" & FR (ours) & {pm(fs_m,fs_s)} & {pm(fg_m,fg_s,1)} & ${rs:.3f}$ \\\\\n"
                 f"\\midrule\n")
    if body.endswith("\\midrule\n"):
        body = body[:-len("\\midrule\n")]
    tex = (f"\\begin{{table}}[t]\\centering\n"
           f"\\caption{{{fig['caption']}}}\\label{{tab:{fig['name']}}}\n"
           f"\\begin{{tabular}}{{llrrr}}\n\\toprule\n"
           f"Scheme & Client & Samples (run) & GPU-time (s) & Cost / honest \\\\\n\\midrule\n"
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
    tex = (fig_open(fig, "xlabel={communication round},ylabel={watermark BER},ymin=0,ymax=1,"
                    "legend pos=outer north east") + body + eta + fig_close(fig))
    return dat, tex

def emit_classbars(fig, runs, out, tail):
    """fig3a: per-class honest BER bars (all classes) + FR BER bars (its trigger classes)."""
    hon = runs.get(fig["honest"]); fr = runs.get(fig["fr"])
    if not hon: return None
    pch = per_class_honest(hon, tail)
    pcf = per_class_fr(fr, tail) if fr else {}
    hrows, frows = [], []
    for c in sorted(pch):
        hb = pch[c]["ber"]
        if not hb: continue
        hm, hs = _ms(hb); hrows.append((c, hm, hs))
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

def emit_classentropy(fig, runs, out, tail):
    """fig3b: per-class mean softmax entropy (bars), +-1 s.d. over the pooled rounds/seeds."""
    hon = runs.get(fig["honest"])
    if not hon: return None
    pch = per_class_honest(hon, tail)
    rows = []
    for c in sorted(pch):
        en = pch[c]["entropy"]
        if not en: continue
        em, es = _ms(en); rows.append((c, em, es))
    if not rows: return None
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat), ["class","entropy","entropy_sd"], rows)
    tex = (fig_open(fig, "ybar,bar width=6pt,xlabel={trigger class},ylabel={softmax entropy},"
                    "xtick=data,ymin=0,enlarge x limits=0.08") +
           f"\\addplot[cacc,fill=cacc!55,draw=cacc,error bars/.cd,y dir=both,y explicit] "
           f"table[x=class,y=entropy,y error=entropy_sd]{{{dat}}};\n" +
           fig_close(fig))
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
                    "xtick=data,ymin=0,legend pos=north west") +
           f"\\addplot[cfr,mark=*,error bars/.cd,y dir=both,y explicit] "
           f"table[x=nl,y=fr,y error=fr_sd]{{{dat}}};\\addlegendentry{{free-rider (ours)}}\n"
           f"\\addplot[chonest,mark=square*,error bars/.cd,y dir=both,y explicit] "
           f"table[x=nl,y=hon,y error=hon_sd]{{{dat}}};\\addlegendentry{{honest}}\n"
           f"{eta}" + fig_close(fig))
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
        "classentropy": emit_classentropy, "layers": emit_layers,
        "band": emit_band, "overlap": emit_overlap, "savings": emit_savings}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", default="export")
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--only", default=None)
    ap.add_argument("--appendix", action="store_true",
                    help="emit the APPENDIX_FIGURES set instead of the paper set.")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "fig"), exist_ok=True)
    runs = load(a.res)
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

PREAMBLE = r"""% --- paste into your main.tex preamble ---
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
  fig1_faremark_timeline, fig1_fedipr_timeline   (fig 1)
  tab1_costs                                      (table 1)
  fig2_attack_compare                             (fig 2)
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