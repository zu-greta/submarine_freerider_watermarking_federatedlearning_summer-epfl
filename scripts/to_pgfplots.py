#!/usr/bin/env python
"""to_pgfplots -- turn result.json runs into pgfplots-ready .dat tables + .tex figures
=======================================================================================
No images. Each MAIN figure becomes:
  export/data/<name>.dat   whitespace table with a header row  (pgfplots `table`)
  export/fig/<name>.tex    a \\begin{figure} float that \\addplot table{...}s it

Figures are styled to MATCH the matplotlib output of scripts/plots.py (Okabe-Ito:
honest=#0072B2, free-rider=#D55E00, eta_t black dashed, eta_l blue dashed, +-std
bands, converged-tail shading) and every \\addplot has EXPLICIT colour/mark, plus a
per-axis `cycle list`, so they do NOT depend on the `Set1` cycle list or on library
load order in your preamble.

Usage
  python scripts/to_pgfplots.py --res '/mnt/nfs/home/zu/results/*/result.json' \
         --out export --tail 20
  # then upload export/ to Overleaf (see export/README_OVERLEAF.md)

Pick & choose which figures via the FIGURES list below, or --only fig_a,fig_b.
"""
import argparse, glob, json, os, statistics as st
from collections import defaultdict

# ----------------------------------------------------------------------------
FIGURES = [
    dict(name="fig_class_band",        kind="band",       family="A1_honest_c100",
         caption="Per-trigger-class watermark BER for all-honest clients (CIFAR-100, "
                 "10 classes). Some classes are structurally harder to embed."),
    dict(name="fig_ber_entropy",       kind="berentropy", family="A1_honest_c100",
         caption="Watermark BER vs.\\ softmax entropy on the trigger class: the mark "
                 "rides the output tail, so peaked (low-entropy) classes have high BER."),
    dict(name="fig_ber_timeline_L1",   kind="timeline",   fr="L1_graftblock_head2_c36",
         honest="A1_honest_c100", eta_t=0.064, eta_l=0.264,
         caption="Final-block free-rider (head2) vs.\\ honest floor, hard classes 3,6. "
                 "The free-rider re-embeds and stays under the threshold."),
    dict(name="fig_ber_timeline_L5",   kind="timeline",   fr="L5_graftblock_head2_c17",
         honest="A1_honest_c100", eta_t=0.064, eta_l=0.264,
         caption="Final-block free-rider (head2) vs.\\ honest floor, easy classes 1,7."),
    dict(name="fig_overlap",           kind="overlap",    honest="A1_honest_c100",
         fr=["L1_graftblock_head2_c36", "L5_graftblock_head2_c17",
             "K9_alldyn_head2_c36", "K9_alldyn_head2_c17"], eta_t=0.064, eta_l=0.264,
         caption="Honest per-class BER band vs.\\ free-rider operating points. Free-riders "
                 "land inside the honest band, so no single threshold separates them."),
    dict(name="fig_savings",           kind="savings",
         fr=["L1_graftblock_head2_c36", "L5_graftblock_head2_c17",
             "K9_alldyn_head2_c36", "K4_alldyn_block2_c36"],
         caption="Free-rider training cost as a fraction of an honest client "
                 "(samples and GPU-time)."),
    dict(name="fig_tap_perfr_L1",      kind="tap_perfr",  fr="L1_graftblock_head2_c36",
         honest="A1_honest_c100", eta_t=0.064, eta_l=0.264,
         caption="Free-rider BER across seeds (band = min--max over seeds/free-riders) "
                 "vs.\\ the honest floor. The calibration window is shaded; the dashed "
                 "vertical marks where free-riding begins."),
    dict(name="fig_iso_c6",            kind="iso",        honest="A1_honest_c100",
         fr="L1_graftblock_head2_c36", cls=6, eta_t=0.064, eta_l=0.264,
         caption="Same trigger class (6): honest client vs.\\ free-rider BER per round -- "
                 "the free-rider tracks the honest client, so they are indistinguishable."),
    dict(name="fig_iso_c3",            kind="iso",        honest="A1_honest_c100",
         fr="L1_graftblock_head2_c36", cls=3, eta_t=0.064, eta_l=0.264,
         caption="Same trigger class (3): honest vs.\\ free-rider BER per round."),

    # --- FedIPR (uncomment once the F_* runs land; identical plots, 2nd scheme) ---
    # dict(name="fig_F_class_band",  kind="band",     family="F_A1_honest_c100_fi", caption="FedIPR: honest per-class band."),
    # dict(name="fig_F_timeline_L1", kind="timeline", fr="F_L1_graftblock_head2_c36_fi", honest="F_A1_honest_c100_fi", eta_t=0.20, eta_l=0.50, caption="FedIPR: final-block FR vs honest floor."),
    # dict(name="fig_F_overlap",     kind="overlap",  honest="F_A1_honest_c100_fi",
    #      fr=["F_L1_graftblock_head2_c36_fi","F_L5_graftblock_head2_c17_fi","F_K9_alldyn_head2_c36_fi","F_K9_alldyn_head2_c17_fi"], eta_t=0.20, eta_l=0.50, caption="FedIPR: honest band vs FR points."),
]

# Okabe-Ito, matching scripts/plots.py (C_HONEST/C_FR/etc.). Emitted into every figure
# so figures are self-contained and independent of your Set1 cycle list.
COLORDEF = (r"\definecolor{chonest}{HTML}{0072B2}"  "\n"
            r"\definecolor{cfr}{HTML}{D55E00}"      "\n"
            r"\definecolor{cacc}{HTML}{009E73}"     "\n"
            r"\definecolor{ctail}{HTML}{DDDDDD}"    "\n")
# per-axis cycle list so pgfplots never looks up `Set1` (kills the load-order bug)
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

# ---- figure emitters -------------------------------------------------------
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

def emit_berentropy(fig, runs, out, tail):
    r = runs.get(fig["family"])
    if not r: return None
    pc = per_class_honest(r, tail)
    rows = [(c, st.mean(pc[c]["ber"]), st.mean(pc[c]["entropy"]),
             st.mean(pc[c]["dominance"]), st.mean(pc[c]["pmax"]))
            for c in sorted(pc) if pc[c]["ber"]]
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat), ["class","ber","entropy","dominance","pmax"], rows)
    tex = (fig_open(fig, "xlabel={softmax entropy on trigger class},ylabel={watermark BER},"
                    "ymin=0,nodes near coords,point meta=explicit symbolic,"
                    "visualization depends on={value \\thisrow{class}\\as\\lbl},"
                    "every node near coord/.append style={font=\\scriptsize,color=black,anchor=west}") +
           f"\\addplot[only marks,mark=*,cfr,mark size=1.6pt] table[x=entropy,y=ber,meta=class]{{{dat}}};\n" +
           fig_close(fig))
    return dat, tex

def emit_timeline(fig, runs, out, tail):
    fr = runs.get(fig["fr"])
    if not fr: return None
    by_round = defaultdict(lambda: {"fr": [], "hon": []})
    for r in fr:
        for h in _hist(r):
            rd = h["round"]
            if h.get("wm_fr_ber") is not None: by_round[rd]["fr"].append(h["wm_fr_ber"])
            if h.get("wm_benign_ber") is not None: by_round[rd]["hon"].append(h["wm_benign_ber"])
    def ms(v):
        return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)
    rows, rmax = [], 0
    for rd in sorted(by_round):
        fm, fs = ms(by_round[rd]["fr"]); hm, hs = ms(by_round[rd]["hon"])
        rows.append((rd, fm, max(0.0, fm-fs), fm+fs, hm, max(0.0, hm-hs), hm+hs)); rmax = rd
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat),
              ["round","fr","fr_lo","fr_hi","hon","hon_lo","hon_hi"], rows)
    et, el = fig.get("eta_t"), fig.get("eta_l")
    tstart = max(1, rmax - tail + 1)
    eta = ""
    if et is not None:
        eta += (f"\\addplot[black,dashed,forget plot,domain=1:{rmax}]{{{et}}};\n"
                f"\\node[anchor=south west,font=\\scriptsize] at (axis cs:1,{et}) {{$\\eta_t$}};\n")
    if el is not None:
        eta += (f"\\addplot[chonest,densely dashed,forget plot,domain=1:{rmax}]{{{el}}};\n"
                f"\\node[anchor=north west,font=\\scriptsize] at (axis cs:1,{el}) {{$\\eta_\\ell$}};\n")
    tex = (fig_open(fig, "xlabel={communication round},ylabel={watermark BER},ymin=0,"
                    "legend pos=north east") +
           # converged-region shading (matches plots.py axvspan of the tail)
           f"\\fill[ctail,opacity=0.5] (axis cs:{tstart},0) rectangle (rel axis cs:1,1);\n"
           # honest +-std band + mean line
           f"\\addplot[name path=hlo,draw=none,forget plot] table[x=round,y=hon_lo]{{{dat}}};\n"
           f"\\addplot[name path=hhi,draw=none,forget plot] table[x=round,y=hon_hi]{{{dat}}};\n"
           f"\\addplot[chonest!12,forget plot] fill between[of=hlo and hhi];\n"
           f"\\addplot[chonest,mark=none] table[x=round,y=hon]{{{dat}}};\\addlegendentry{{honest floor}}\n"
           # free-rider +-std band + mean line with markers
           f"\\addplot[name path=flo,draw=none,forget plot] table[x=round,y=fr_lo]{{{dat}}};\n"
           f"\\addplot[name path=fhi,draw=none,forget plot] table[x=round,y=fr_hi]{{{dat}}};\n"
           f"\\addplot[cfr!15,forget plot] fill between[of=flo and fhi];\n"
           f"\\addplot[cfr,mark=*,mark size=1.1pt] table[x=round,y=fr]{{{dat}}};\\addlegendentry{{free-rider}}\n"
           f"{eta}" + fig_close(fig))
    return dat, tex

def emit_overlap(fig, runs, out, tail):
    hon = runs.get(fig["honest"])
    if not hon: return None
    pc = per_class_honest(hon, tail)
    hrows = [(c, st.mean(pc[c]["ber"]), pct(pc[c]["ber"],0.10), pct(pc[c]["ber"],0.90))
             for c in sorted(pc) if pc[c]["ber"]]
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
    eta = (f"\\addplot[black,dashed,forget plot] coordinates {{({cmin},{et}) ({cmax},{et})}};\n"
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

def emit_tap_perfr(fig, runs, out, tail):
    fr = runs.get(fig["fr"])
    if not fr: return None
    frr, honr = defaultdict(list), defaultdict(list)
    for r in fr:                                    # FR band = pooled FR per-client bers
        frset = set(r.get("free_rider_indices") or [])
        for h in _hist(r):
            for p in (h.get("wm_per_client") or []):
                if p.get("ber") is None: continue
                (frr if p["cid"] in frset else honr)[h["round"]].append(p["ber"])
    hon = runs.get(fig.get("honest"))
    if hon:                                         # cleaner honest floor from a honest family
        honr = defaultdict(list)
        for r in hon:
            for h in _hist(r):
                for p in (h.get("wm_per_client") or []):
                    if p.get("ber") is not None: honr[h["round"]].append(p["ber"])
    rows, rmax = [], 0
    for rd in sorted(frr):
        fv, hv = frr[rd], honr.get(rd, [])
        rows.append((rd, st.mean(fv), min(fv), max(fv),
                     st.mean(hv) if hv else float("nan"),
                     pct(hv, 0.10) if hv else float("nan"),
                     pct(hv, 0.90) if hv else float("nan"))); rmax = rd
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat),
              ["round","fr","fr_lo","fr_hi","hon","hon_lo","hon_hi"], rows)
    c0 = fr[0].get("config", {}) or {}
    W = int(c0.get("autop_honest_until", 12) or 12); K = int(c0.get("autop_calib_rounds", 4) or 4)
    clo, chi = max(1, W-K), max(1, W-1)
    et, el = fig.get("eta_t"), fig.get("eta_l")
    ann = f"\\fill[cacc,opacity=0.18] (axis cs:{clo},0) rectangle (axis cs:{chi},0.6);\n" \
          f"\\draw[gray,dashed] (axis cs:{W},0) -- (axis cs:{W},0.6);\n" \
          f"\\node[anchor=south east,font=\\scriptsize,gray,rotate=90] at (axis cs:{W},0.05) {{free-ride}};\n"
    if et is not None: ann += f"\\addplot[black,dashed,forget plot,domain=1:{rmax}]{{{et}}};\n"
    if el is not None: ann += f"\\addplot[chonest,densely dashed,forget plot,domain=1:{rmax}]{{{el}}};\n"
    tex = (fig_open(fig, "xlabel={communication round},ylabel={watermark BER},ymin=0,ymax=0.6,"
                    "legend pos=north east") +
           f"\\addplot[name path=hlo,draw=none,forget plot] table[x=round,y=hon_lo]{{{dat}}};\n"
           f"\\addplot[name path=hhi,draw=none,forget plot] table[x=round,y=hon_hi]{{{dat}}};\n"
           f"\\addplot[chonest!12,forget plot] fill between[of=hlo and hhi];\n"
           f"\\addplot[chonest] table[x=round,y=hon]{{{dat}}};\\addlegendentry{{honest floor}}\n"
           f"\\addplot[name path=flo,draw=none,forget plot] table[x=round,y=fr_lo]{{{dat}}};\n"
           f"\\addplot[name path=fhi,draw=none,forget plot] table[x=round,y=fr_hi]{{{dat}}};\n"
           f"\\addplot[cfr!18,forget plot] fill between[of=flo and fhi];\n"
           f"\\addplot[cfr,mark=*,mark size=1pt] table[x=round,y=fr]{{{dat}}};\\addlegendentry{{free-rider}}\n"
           f"{ann}" + fig_close(fig))
    return dat, tex

def emit_iso(fig, runs, out, tail):
    hon, fr, c = runs.get(fig["honest"]), runs.get(fig["fr"]), int(fig["cls"])
    if not hon or not fr: return None
    def series(rr, want_fr):
        by = defaultdict(list)
        for r in rr:
            frset = set(r.get("free_rider_indices") or [])
            for h in _hist(r):
                for p in (h.get("wm_per_client") or []):
                    if int(p.get("trigger_class", -1)) != c: continue
                    if (p["cid"] in frset) != want_fr: continue
                    if p.get("ber") is not None: by[h["round"]].append(p["ber"])
        return by
    hs, fs = series(hon, False), series(fr, True)
    rounds = sorted(set(hs) | set(fs)); rmax = rounds[-1] if rounds else 50
    rows = [(rd, st.mean(hs[rd]) if hs.get(rd) else float("nan"),
                 st.mean(fs[rd]) if fs.get(rd) else float("nan")) for rd in rounds]
    dat = f"{fig['name']}.dat"
    write_dat(os.path.join(out, "data", dat), ["round","honest","fr"], rows)
    et, el = fig.get("eta_t"), fig.get("eta_l")
    eta = ""
    if et is not None: eta += f"\\addplot[black,dashed,forget plot,domain=1:{rmax}]{{{et}}};\\node[anchor=south west,font=\\scriptsize] at (axis cs:1,{et}) {{$\\eta_t$}};\n"
    if el is not None: eta += f"\\addplot[chonest,densely dashed,forget plot,domain=1:{rmax}]{{{el}}};\n"
    tex = (fig_open(fig, "xlabel={communication round},ylabel={watermark BER "
                    f"(class {c})}},ymin=0,legend pos=north east") +
           f"\\addplot[chonest,mark=none] table[x=round,y=honest]{{{dat}}};\\addlegendentry{{honest (cls {c})}}\n"
           f"\\addplot[cfr,mark=*,mark size=1pt] table[x=round,y=fr]{{{dat}}};\\addlegendentry{{free-rider (cls {c})}}\n"
           f"{eta}" + fig_close(fig))
    return dat, tex

EMIT = {"band": emit_band, "berentropy": emit_berentropy, "timeline": emit_timeline,
        "overlap": emit_overlap, "savings": emit_savings,
        "tap_perfr": emit_tap_perfr, "iso": emit_iso}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", default="export")
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "fig"), exist_ok=True)
    runs = load(a.res)
    print(f"loaded families: {sorted(runs)}")
    only = set(a.only.split(",")) if a.only else None
    made = []
    for fig in FIGURES:
        if only and fig["name"] not in only:
            continue
        res = EMIT[fig["kind"]](fig, runs, a.out, a.tail)
        if res is None:
            print(f"  skip {fig['name']} (family missing)"); continue
        dat, tex = res
        open(os.path.join(a.out, "fig", fig["name"] + ".tex"), "w").write(tex)
        made.append(fig["name"]); print(f"  wrote fig/{fig['name']}.tex  <- {dat}")
    with open(os.path.join(a.out, "all_figures.tex"), "w") as f:
        f.write("% \\input this, or copy individual \\input lines where you want each float.\n")
        for n in made:
            f.write(f"\\input{{plots/export/fig/{n}.tex}}\n")
    open(os.path.join(a.out, "preamble_snippet.tex"), "w").write(PREAMBLE)
    open(os.path.join(a.out, "README_OVERLEAF.md"), "w").write(README)
    print(f"\n{len(made)} figures -> {a.out}/  (menu: {a.out}/all_figures.tex)")

PREAMBLE = r"""% --- paste into your main.tex preamble (once) ---
\usepackage{pgfplots}
\usepgfplotslibrary{colorbrewer}     % MUST be loaded BEFORE \input{...pgfplots-config}
\usepgfplotslibrary{fillbetween}     % for the +-std / honest bands
\usepackage{siunitx}
\usepackage{sansmath}
\input{plots/pgfplots-config}        % your config (adjust path/name to yours)
% pgfplots resolves table{...} relative to MAIN.tex; point it at the data folder:
\pgfplotsset{table/search path={plots/export/data}}   % <-- set to where your .dat live

% The exported figures set their own colours + per-axis `cycle list`, so they do NOT
% need Set1 -- but your OTHER figures do, so keep the colorbrewer-before-config order.
"""

README = r"""# Overleaf: raw data + pgfplots figures (vector, matplotlib-matched)

`.dat` = the raw numbers, `.tex` = pgfplots that draws them. Figures use the Okabe-Ito
palette from scripts/plots.py (honest #0072B2, free-rider #D55E00), +-std bands, eta
lines and converged-region shading, and set explicit colours + a per-axis `cycle list`
so they render regardless of your Set1 load order.

## Fixing the "no data / no such cycle list Set1" error
Two things caused blank plots:
1. TABLE PATH -- pgfplots looks relative to main.tex. Set (in preamble):
       \pgfplotsset{table/search path={plots/export/data}}
2. CYCLE LIST ORDER -- your pgfutils.tex \input's the config (which names cycle
   list=Set1) BEFORE loading colorbrewer (which defines Set1). Move the colorbrewer
   load ABOVE the \input:
       \usepgfplotslibrary{colorbrewer}
       \input{plots/pgfplots-config}
   (These exported figures don't rely on Set1, but your Python-exported ones do.)

Also: \input these figures directly -- do NOT route them through your \inputplot /
externalization macro (that swaps in a pre-built main-figureN.pdf that doesn't exist
for these). Plain:
       \input{plots/export/fig/fig_overlap.tex}

## Layout (root / plots / sections)
    main.tex
    plots/pgfplots-config.tex
    plots/export/{data/*.dat, fig/*.tex, all_figures.tex, preamble_snippet.tex}
    sections/*.tex

## Place & reference
Each fig/<name>.tex is a full \begin{figure} float with \caption + \label. Put it where
you want it: \input{plots/export/fig/fig_ber_timeline_L1.tex}. Span both IEEE columns by
changing \begin{figure} -> \begin{figure*} in that file. Cite with \cref{fig:<name>}
(you load cleveref) or \ref{fig:<name>}.

## Pick & choose / re-run
Edit the FIGURES list in scripts/to_pgfplots.py (comment rows), or
  python scripts/to_pgfplots.py --res '<results>/*/result.json' --out export --only fig_overlap,fig_class_band
Re-run when FedIPR F_* results land (uncomment the fig_F_* rows) -- \input/\ref lines
don't change.
"""

if __name__ == "__main__":
    main()