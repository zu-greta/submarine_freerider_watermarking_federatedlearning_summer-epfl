#!/usr/bin/env python
"""preflight_fedipr -- fast gate for the FedIPR honest baseline.

A FedIPR honest run MUST embed the backdoor: honest detection BER = 1 - trigger_acc
should be LOW (paper eta_T ~ 0.95 -> ber ~ 0.05). If the trigger source is OOD
(svhn/noise) on a BatchNorm model, the mark collapses at eval time and honest ber
stays ~0.8-1.0 -- a broken baseline. This checks a short honest run and fails loudly
so you find out in minutes, not after a full sweep.

    python scripts/preflight_fedipr.py --in '/mnt/nfs/.../fedipr_probe_rep0/result.json' --max 0.2

Exit 0 = PASS (honest ber < max), exit 1 = FAIL (with guidance).
"""
import argparse, glob, json, statistics as st, sys


def honest_ber(run, tail):
    frset = set(run.get("free_rider_indices") or [])
    hist = (run.get("history") or [])[-tail:]
    vals = [p["ber"] for h in hist for p in (h.get("wm_per_client") or [])
            if p.get("ber") is not None and p["cid"] not in frset]
    return (st.mean(vals), len(vals)) if vals else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--max", type=float, default=0.2, help="max honest ber to pass")
    ap.add_argument("--tail", type=int, default=3, help="rounds from the end to average")
    a = ap.parse_args()
    files = [f for g in a.inp for f in glob.glob(g)]
    if not files:
        print(f"FEDIPR PREFLIGHT: no result.json matched {a.inp}  (did the probe run finish?)")
        sys.exit(1)
    run = json.load(open(sorted(files)[0]))
    scheme = (run.get("config", {}) or {}).get("wm_scheme")
    src = (run.get("config", {}) or {}).get("fedipr_trigger_source")
    ber, n = honest_ber(run, a.tail)
    print(f"FEDIPR PREFLIGHT  scheme={scheme} trigger_source={src}  "
          f"honest ber(last {a.tail}r, n={n}) = {ber}")
    if ber is None:
        print("  FAIL: no honest BER recorded (is this a watermarked honest run?)"); sys.exit(1)
    if ber < a.max:
        print(f"  PASS: honest mark embeds (ber {ber:.3f} < {a.max}). Trigger acc ~ {1-ber:.3f}. "
              f"Proceed: calibrate FEDIPR_ETA = mu+3sigma of honest ber, then run H/L/K.")
        sys.exit(0)
    print(f"  FAIL: honest ber {ber:.3f} >= {a.max}  -> the honest watermark is NOT embedding.")
    print("        Most likely: OOD triggers (svhn/noise) + BatchNorm collapse the mark at eval.")
    print("        Fix: FEDIPR_TRIGGER_SOURCE=indist (real task images; BN-robust). "
          "Also check trigger acc during training in run.log (should be high).")
    sys.exit(1)


if __name__ == "__main__":
    main()
