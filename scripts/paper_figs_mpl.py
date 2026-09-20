#!/usr/bin/env python3
# =============================================================================
# paper_figs_mpl.py -- matplotlib twins of the PAPER figures (the same set
# to_pgfplots.py emits for Overleaf), rendered as PNG+PDF.
#
#   fig1_*_timeline   honest floor vs OUR free-rider, BER vs round (+-s.d. seeds)
#                       variants: faremark / fedipr / sign(white-box)
#   fig2_*_attack     honest floor + prev-model / gaussian / ours, BER vs round
#                       variants: faremark / sign(white-box)
#   fig3              class difficulty: grouped honest/FR bars + DeltaBER-vs-entropy
#                       scatter (the "before" look)
#   fig4_layers       white-box: final BER vs # watermarked layers, honest vs FR
#   tab1_costs        honest vs FR samples + GPU-time, both schemes -> .csv + .png
#
# Same result.json tree as to_pgfplots.py; families kept in lock-step with it.
# Missing families are skipped (so it runs on a partial results dir).
# All stats are over SEEDS (tail-mean per seed, then mean/std over seeds).
#
#   python paper_figs_mpl.py --res 'results/*/result.json' --out figs --tail 20
#   python paper_figs_mpl.py --res '...' --out figs --only fig1_sign_timeline
# =============================================================================
import argparse, glob, json, os, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Okabe-Ito palette (identical hues to to_pgfplots COLORDEF) -------------
C_HONEST = "#0072B2"   # blue   -> honest floor
C_FR     = "#D55E00"   # orange -> our free-rider
C_ACC    = "#009E73"   # green  -> gaussian
C_PREV   = "#000000"   # black  -> previous-model
C_GOLD   = "#F0C24B"   # gold   -> honest bars (fig3 left)
GRID     = "#CFCFCF"
plt.rcParams.update({
    "font.size": 12, "axes.linewidth": 0.9, "axes.edgecolor": "#333333",
    "figure.dpi": 150, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": False,
})

# ============================================================================
# loading / aggregation (over seeds)
# ============================================================================
def _dataset_of(r):
    return ((r.get("summary") or {}).get("dataset")
            or (r.get("config") or {}).get("dataset")
            or (r.get("manifest") or {}).get("dataset"))


def load(res_glob, dataset=None):
    """Group by family; if `dataset` is set, keep only that dataset's runs. Food-101
    reuses the CIFAR-100 family names, so without this a mixed folder merges datasets."""
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
            r["__src"] = f                         # remember the source file (for the per-fig seed report)
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

def _ms(v):
    return (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0) if v else (float("nan"), 0.0)

def per_round(runs, key):
    """{round: [value over seeds]} for a top-level history key (e.g. wm_fr_ber)."""
    by = defaultdict(list)
    for r in runs:
        for h in _hist(r):
            v = h.get(key)
            if v is not None:
                by[h["round"]].append(float(v))
    return by

def honest_by_seed(runs, tail):
    out = defaultdict(lambda: defaultdict(list))
    for r in runs:
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
                    out[c][k].append(st.mean(vs))
    return out

def fr_by_seed(run_lists, tail):
    out = defaultdict(list)
    for runs in run_lists:
        for r in runs:
            frs = set(r.get("free_rider_indices") or [])
            acc = defaultdict(list)
            for h in _hist(r, tail):
                for p in (h.get("wm_per_client") or []):
                    if p["cid"] in frs and p.get("ber") is not None:
                        acc[int(p["trigger_class"])].append(float(p["ber"]))
            for c, vs in acc.items():
                if vs:
                    out[c].append(st.mean(vs))
    return out

def compute_col(runs, key):
    vs = [(r.get("compute", {}).get("summary", {}) or {}).get(key) for r in runs]
    vs = [float(v) for v in vs if v is not None]
    return _ms(vs)

def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png",):            # PNG only -- PDF output disabled ("pdf" removed)
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png")
    return True                     # signal "produced" so main() prints the seed report

def _band(ax, rounds, mean, sd, color, label, marker=None):
    lo = [max(0.0, m - s) for m, s in zip(mean, sd)]
    hi = [m + s for m, s in zip(mean, sd)]
    ax.fill_between(rounds, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(rounds, mean, color=color, lw=1.8, marker=marker, ms=3, label=label)

# ============================================================================
# fig1: honest floor vs OUR free-rider timeline
# ============================================================================
def fig_timeline(spec, runs, out, tail):
    fr = runs.get(spec["fr"])
    if not fr:
        print(f"  skip {spec['name']} (family '{spec['fr']}' missing)"); return
    frR = per_round(fr, "wm_fr_ber"); benR = per_round(fr, "wm_benign_ber")
    rounds = sorted(set(frR) | set(benR))
    if not rounds:
        print(f"  skip {spec['name']} (no history)"); return
    fm = [_ms(frR.get(rd, [])) for rd in rounds]; bm = [_ms(benR.get(rd, [])) for rd in rounds]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    # free-rider drawn FIRST (underneath), honest band+line LAST (on top) so honest is never
    # hidden behind the FR fill -- matches the Overleaf timeline. Grey tail-highlight removed.
    _band(ax, rounds, [m for m, s in fm], [s for m, s in fm], C_FR, "free-rider", marker="o")
    _band(ax, rounds, [m for m, s in bm], [s for m, s in bm], C_HONEST, "honest floor")
    ax.set_xlabel("communication round"); ax.set_ylabel("watermark BER")
    ax.set_ylim(0, spec.get("ymax", 0.6)); ax.grid(axis="x", visible=False)   # zoom BER axis (matches overleaf)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: 0 if "honest" in l[i] else 1)   # honest floor listed first
    ax.legend([h[i] for i in order], [l[i] for i in order], fontsize=10, loc="upper right")
    ax.set_title(spec.get("title", ""), fontsize=11)
    return _save(fig, out, spec["name"])

# ============================================================================
# fig2: attack comparison (prev-model / gaussian / ours)
# ============================================================================
def fig_attack(spec, runs, out, tail):
    hon = runs.get(spec["honest"])
    present = [(lbl, fam) for lbl, fam in spec["attacks"] if runs.get(fam)]
    if not present:
        print(f"  skip {spec['name']} (no attack families present)"); return
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    # honest floor: prefer the dedicated honest family, else benign clients of first attack
    if hon:
        benR = per_round(hon, "wm_benign_ber")
    else:
        benR = per_round(runs.get(present[0][1]), "wm_benign_ber")
    rounds = sorted(benR)
    if rounds:
        bm = [_ms(benR[rd]) for rd in rounds]
        _band(ax, rounds, [m for m, s in bm], [s for m, s in bm], C_HONEST, "honest floor")
    style = {"previous models": (C_PREV, "s"), "gaussian": (C_ACC, "^"),
             "ours (head2)": (C_FR, "o"), "ours (head)": (C_FR, "o")}
    for lbl, fam in present:
        fR = per_round(runs.get(fam), "wm_fr_ber")
        rr = sorted(fR)
        if not rr:
            continue
        col, mk = style.get(lbl, (C_FR, "o"))
        ms = [_ms(fR[rd]) for rd in rr]
        _band(ax, rr, [m for m, s in ms], [s for m, s in ms], col, lbl, marker=mk)
    # -- threshold lines disabled for now (commented out) --
    # if spec.get("eta_t") is not None:
    #     ax.axhline(spec["eta_t"], color=C_PREV, ls="--", lw=1, zorder=1)
    # if spec.get("eta_l") is not None:
    #     ax.axhline(spec["eta_l"], color=C_HONEST, ls=(0, (5, 2)), lw=1, zorder=1)
    ax.set_xlabel("communication round"); ax.set_ylabel("watermark BER")
    ax.set_ylim(0, spec.get("ymax", 0.8)); ax.grid(axis="x", visible=False)   # zoom BER axis (matches overleaf; baselines ~0.5-0.75 stay in frame)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="center right"); ax.set_title(spec.get("title", ""), fontsize=11)
    return _save(fig, out, spec["name"])

# ============================================================================
# fig3: class difficulty -- grouped bars + DeltaBER-vs-entropy scatter
# ============================================================================
def fig_classdiff(spec, runs, out, tail):
    hon = runs.get(spec["honest"])
    if not hon:
        print(f"  skip {spec['name']} (honest '{spec['honest']}' missing)"); return
    H = honest_by_seed(hon, tail)
    F = fr_by_seed([runs.get(f, []) for f in spec["fr"]], tail)
    classes = sorted(H)
    hb = {c: _ms(H[c].get("ber", [])) for c in classes}
    he = {c: _ms(H[c].get("entropy", [])) for c in classes}
    fb = {c: _ms(F.get(c, [])) for c in classes}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.0),
                                   gridspec_kw=dict(width_ratios=[1.15, 1.0], wspace=0.28))
    w = 0.38; xs = list(range(len(classes)))
    axL.bar([x - w/2 for x in xs], [hb[c][0] for c in classes], width=w,
            yerr=[hb[c][1] for c in classes], capsize=2.5, color=C_GOLD,
            edgecolor="#B8901F", linewidth=0.8, error_kw=dict(ecolor="#8a6d1a", elinewidth=0.9),
            label="honest", zorder=3)
    fx = [x + w/2 for x, c in zip(xs, classes) if fb[c][0] == fb[c][0]]
    axL.bar(fx, [fb[c][0] for c in classes if fb[c][0] == fb[c][0]], width=w,
            yerr=[fb[c][1] for c in classes if fb[c][0] == fb[c][0]], capsize=2.5,
            color=C_FR, edgecolor="#8a3d00", linewidth=0.8,
            error_kw=dict(ecolor="#5f2a00", elinewidth=0.9), label="free-rider", zorder=3)
    axL.set_xticks(xs); axL.set_xticklabels([str(c) for c in classes])
    axL.set_xlabel("trigger class"); axL.set_ylabel("watermark BER"); axL.set_ylim(bottom=0)
    axL.grid(axis="x", visible=False); axL.legend(fontsize=10, loc="upper right")
    ex = [he[c][0] for c in classes]; ey = [hb[c][0] for c in classes]
    axR.scatter(ex, ey, s=42, color=C_FR, zorder=3)
    for c, x, y in zip(classes, ex, ey):
        axR.annotate(str(c), (x, y), textcoords="offset points", xytext=(6, 1), fontsize=10)
    axR.set_xlabel("softmax entropy on trigger class"); axR.set_ylabel(r"$\Delta$ BER")
    axR.set_ylim(bottom=0); axR.grid(axis="x", visible=False)
    if ex:
        dx = (max(ex) - min(ex)) or 1.0
        axR.set_xlim(min(ex) - 0.08*dx, max(ex) + 0.12*dx); axR.margins(y=0.12)
    for ax in (axL, axR):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return _save(fig, out, spec["name"])

# ============================================================================
# fig4: white-box final BER vs # watermarked layers
# ============================================================================
def fig_layers(spec, runs, out, tail):
    def final_ber(fam, key):
        vals = []
        for r in runs.get(fam, []):
            v = [h[key] for h in _hist(r, tail) if h.get(key) is not None]
            if v: vals.append(st.mean(v))
        return _ms(vals), len(vals)
    rows = []
    for nl in spec["layers"]:
        (fm, fs), nf = final_ber(spec["fr_fmt"].format(nl=nl), "wm_fr_ber")
        (hm, hs), nh = final_ber(spec["honest_fmt"].format(nl=nl), "wm_benign_ber")
        if nf == 0 and nh == 0:
            continue
        rows.append((nl, fm, fs, hm, hs))
    if not rows:
        print(f"  skip {spec['name']} (no layer-sweep families present)"); return
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    nls = [r[0] for r in rows]
    ax.errorbar(nls, [r[1] for r in rows], yerr=[r[2] for r in rows], color=C_FR,
                marker="o", ms=5, lw=1.8, capsize=3, label="free-rider (ours)")
    ax.errorbar(nls, [r[3] for r in rows], yerr=[r[4] for r in rows], color=C_HONEST,
                marker="s", ms=5, lw=1.8, capsize=3, label="honest")
    # -- threshold lines disabled for now (commented out) --
    # if spec.get("eta_t") is not None:
    #     ax.axhline(spec["eta_t"], color=C_PREV, ls="--", lw=1); ax.text(nls[-1], spec["eta_t"], r" $\eta_t$", fontsize=10, va="bottom", ha="right")
    # if spec.get("eta_l") is not None:
    #     ax.axhline(spec["eta_l"], color=C_HONEST, ls=(0, (5, 2)), lw=1)
    ax.set_xticks(nls); ax.set_xlabel("number of watermarked layers $N$")
    ax.set_ylabel("final watermark BER"); ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="center right"); ax.set_title(spec.get("title", ""), fontsize=11)
    return _save(fig, out, spec["name"])


def fig_costlayers(spec, runs, out, tail):
    """Cost framing: an adaptive free-rider vs # watermarked layers N"""
    def seed_vals(fam):
        costs, bers = [], []
        for r in runs.get(fam, []):
            c = (r.get("compute", {}).get("summary", {}) or {}).get("effort_ratio_gpu")
            if c is not None: costs.append(c)
            v = [h["wm_fr_ber"] for h in _hist(r, tail) if h.get("wm_fr_ber") is not None]
            if v: bers.append(st.mean(v))
        return costs, bers
    rows = []
    for nl in spec["layers"]:
        costs, bers = seed_vals(spec["fr_fmt"].format(nl=nl))
        if not costs and not bers:
            continue
        cm, cs = _ms(costs); bm, bs = _ms(bers)
        rows.append((nl, cm, cs, bm, bs))
    if not rows:
        print(f"  skip {spec['name']} (no adaptive-sweep families present)"); return
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    nls = [r[0] for r in rows]
    ax.axhline(1.0, color=C_HONEST, ls="--", lw=1)
    ax.text(nls[-1], 1.0, " honest cost", fontsize=9, va="bottom", ha="right")
    ax.errorbar(nls, [r[1] for r in rows], yerr=[r[2] for r in rows], color=C_FR,
                marker="o", ms=5, lw=1.8, capsize=3, label="FR compute (rel. honest)")
    ax.errorbar(nls, [r[3] for r in rows], yerr=[r[4] for r in rows], color=C_PREV,
                marker="s", ms=5, lw=1.8, capsize=3, label="FR watermark BER")
    ax.set_xticks(nls); ax.set_xlabel("number of watermarked layers $N$")
    ax.set_ylabel("fraction of honest cost / watermark BER"); ax.set_ylim(0, 1.08)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="center left"); ax.set_title(spec.get("title", ""), fontsize=11)
    return _save(fig, out, spec["name"])

# ============================================================================
# tab1: cost table -> CSV + rendered PNG
# ============================================================================
def tab_costs(spec, runs, out, tail):
    os.makedirs(out, exist_ok=True)
    header = ["scheme", "client", "samples", "samples_sd", "gpu_s", "gpu_s_sd", "cost_over_honest"]
    csv_rows, tbl_rows, any_ok = [], [], False
    for label, fam in spec["rows"]:
        fams = fam if isinstance(fam, (list, tuple)) else [fam]   # a list pools+averages its seeds (e.g. FareMark easy+hard)
        rr = [r for f in fams for r in runs.get(f, [])]
        if not rr:
            continue
        any_ok = True
        hs_m, hs_s = compute_col(rr, "honest_mean_samples"); fs_m, fs_s = compute_col(rr, "fr_mean_samples")
        hg_m, hg_s = compute_col(rr, "honest_mean_gpu_ms");  fg_m, fg_s = compute_col(rr, "fr_mean_gpu_ms")
        ratio = (fs_m / hs_m) if hs_m else float("nan")
        csv_rows.append([label, "honest", f"{hs_m:.0f}", f"{hs_s:.0f}", f"{hg_m/1e3:.1f}", f"{hg_s/1e3:.1f}", "1.00"])
        csv_rows.append([label, "FR(ours)", f"{fs_m:.0f}", f"{fs_s:.0f}", f"{fg_m/1e3:.1f}", f"{fg_s/1e3:.1f}", f"{ratio:.3f}"])
        tbl_rows.append([label, "honest", f"{hs_m:,.0f}", f"{hg_m/1e3:,.1f}", "1.00"])
        tbl_rows.append(["", "FR (ours)", f"{fs_m:,.0f}", f"{fg_m/1e3:,.1f}", f"{ratio:.3f}"])
    if not any_ok:
        print(f"  skip {spec['name']} (no cost families present)"); return
    with open(os.path.join(out, f"{spec['name']}.csv"), "w") as f:
        f.write(",".join(header) + "\n")
        for row in csv_rows:
            f.write(",".join(row) + "\n")
    # rendered PNG table
    fig, ax = plt.subplots(figsize=(7.2, 0.5 + 0.4*len(tbl_rows)))
    ax.axis("off")
    col = ["Scheme", "Client", "Samples", "GPU-time (s)", "Samples / honest"]  # last col = fr/honest sample ratio
    t = ax.table(cellText=tbl_rows, colLabels=col, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.4)
    for (r_i, c_i), cell in t.get_celld().items():
        if r_i == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#EEEEEE")
        cell.set_edgecolor("#CCCCCC")
    return _save(fig, out, spec["name"])

# ============================================================================
# figure registry -- families kept in lock-step with to_pgfplots.py
# ============================================================================
# NOTE: kept in lock-step with to_pgfplots.py 
FIGS = [
    # ---- FareMark (box-free) -- HEAD-only free-rider ----
    dict(name="fig1_faremark_timeline_head", kind="timeline", fr="L6_graftblock_head_c36",
         eta_t=0.064, eta_l=0.264, title="FareMark (head, hard 3,6): honest vs free-rider"),
    dict(name="fig1_faremark_timeline_head_c17", kind="timeline", fr="L7_graftblock_head_c17",
         eta_t=0.064, eta_l=0.264, title="FareMark (head, easy 1,7): honest vs free-rider"),
    # ---- FedIPR white-box sign -- HEAD2 free-rider ----
    dict(name="fig1_sign_timeline", kind="timeline", fr="G_L1_graftblock_head2_c36_ws",
         eta_t=0.20, eta_l=0.50, title="FedIPR white-box sign (head2): honest vs free-rider"),
    # ---- ONE merged cost table (FareMark head avg L6+L7, sign head2 G_L1) ----
    dict(name="tab1_costs", kind="costtable",
         rows=[("FareMark (head)", ["L6_graftblock_head_c36", "L7_graftblock_head_c17"]),
               ("FedIPR-sign (head2)", ["G_L1_graftblock_head2_c36_ws"])]),
    # ---- attack comparisons ----
    dict(name="fig2_attack_compare_head", kind="attack", honest="A1_honest_c100", ymax=0.8,
         attacks=[("previous models", "H5_prevmodel_c100"), ("gaussian", "H6_gaussian_c100"),
                  ("ours (head)", "L6_graftblock_head_c36")],
         eta_t=0.064, eta_l=0.264, title="FareMark (head): attack comparison"),
    dict(name="fig2_sign_attack_compare", kind="attack", honest="G_A1_honest_c100_ws", ymax=0.8,
         attacks=[("previous models", "G_H5_prevmodel_c100_ws"), ("gaussian", "G_H6_gaussian_c100_ws"),
                  ("ours (head2)", "G_L1_graftblock_head2_c36_ws")],
         eta_t=0.20, eta_l=0.50, title="FedIPR white-box sign (head2): attack comparison"),
    # ---- FareMark class difficulty (combined bars + entropy scatter twin), head families ----
    dict(name="fig3_class_difficulty", kind="classdiff",
         honest="A1_honest_c100", fr=["L6_graftblock_head_c36", "L7_graftblock_head_c17"]),
    # ---- white-box layer sweeps: detection (fig4a) + cost (fig4b) + full-data cost (fig4c) ----
    dict(name="fig4a_detection_layers", kind="layers",
         honest_fmt="G_A1_honest_c100_ws_L{nl}", fr_fmt="G_L1_graftblock_head2_c36_ws_L{nl}",
         layers=[1, 6, 20], eta_t=0.20, eta_l=0.50,
         title="White-box: deeper embedding catches the fixed free-rider"),
    dict(name="fig4b_cost_layers", kind="costlayers",
         fr_fmt="G_Ladapt_c36_ws_L{nl}", layers=[1, 6, 20],   # rn18/c100: head2=1, block2=6, full=20
         title="White-box: adaptive free-rider pays honest cost as the mark deepens"),
    dict(name="fig4c_cost_layers_full", kind="costlayers",
         fr_fmt="G_Ladaptfull_c36_ws_L{nl}", layers=[1, 6, 20],   # cpc=-1 full-data variant of fig4b
         title="White-box: adaptive free-rider, full data (cpc=-1)"),
]
EMIT = {"timeline": fig_timeline, "attack": fig_attack, "classdiff": fig_classdiff,
        "layers": fig_layers, "costlayers": fig_costlayers, "costtable": tab_costs}


# ============================================================================
# per-figure seed / source-file report (so you can confirm ALL seeds were read)
# ============================================================================
def _spec_families(spec):
    """The result families a figure pulls from (used only for the report)."""
    k = spec["kind"]
    if k == "timeline":
        return [spec["fr"]]
    if k == "attack":
        return ([spec["honest"]] if spec.get("honest") else []) + [fam for _, fam in spec["attacks"]]
    if k == "classdiff":
        return [spec["honest"]] + list(spec["fr"])
    if k == "layers":
        return ([spec["honest_fmt"].format(nl=nl) for nl in spec["layers"]]
                + [spec["fr_fmt"].format(nl=nl) for nl in spec["layers"]])
    if k == "costlayers":
        return [spec["fr_fmt"].format(nl=nl) for nl in spec["layers"]]
    if k == "costtable":
        out = []
        for _, fam in spec["rows"]:
            out += (list(fam) if isinstance(fam, (list, tuple)) else [fam])
        return out
    return []


def _runtag(r):
    """'.../<FAMILY>_rep<seed>/result.json' -> '<FAMILY>_rep<seed>' (shows the seed)."""
    src = r.get("__src", "")
    return os.path.basename(os.path.dirname(src)) or os.path.basename(src) or "?"


def _report(spec, runs):
    """Print how many seeds + which files fed the figure that was just produced."""
    fams = _spec_families(spec)
    total = set()
    lines = []
    for fam in fams:
        rr = runs.get(fam, [])
        if rr:
            tags = sorted(_runtag(r) for r in rr)
            lines.append(f"        {fam}: {len(rr)} seed(s)  [{', '.join(tags)}]")
            total.update(r.get("__src") for r in rr)
        else:
            lines.append(f"        {fam}: MISSING (0 seeds)")
    print(f"    -> {spec['name']}: read {len(total)} result.json over {len(fams)} family(ies)")
    for ln in lines:
        print(ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", default="figs")
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--only", default=None, help="comma-separated figure names")
    ap.add_argument("--dataset", default=None,
                    help="keep only runs from this dataset (cifar100 | food101); set it "
                         "when a results folder mixes datasets (families are shared).")
    a = ap.parse_args()
    runs = load(a.res, a.dataset)
    print(f"loaded families: {sorted(runs)}")
    only = set(a.only.split(",")) if a.only else None
    print(f">>> PAPER FIGURES (matplotlib) -> {a.out}")
    for spec in FIGS:
        if only and spec["name"] not in only:
            continue
        produced = EMIT[spec["kind"]](spec, runs, a.out, a.tail)   # emitters now return True when a png was written
        if produced:                                               # only report for figures that were actually made
            _report(spec, runs)                                    # print seed count + which result.json files fed it

if __name__ == "__main__":
    main()