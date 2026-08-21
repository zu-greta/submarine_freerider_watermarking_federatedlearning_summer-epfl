#!/usr/bin/env bash
# =============================================================================
# runbook.sh -- entry point 
#
#   ./runbook.sh help          print the phase order
#   ./runbook.sh probe         0. embedding sanity check (gates paper rows)
#   ./runbook.sh manifest      1. build jobs.tsv for the next batch (run_now.sh)
#   ./runbook.sh submit        2. run the pool (PODS x WORKERS, submit_pool.sh)
#   ./runbook.sh monitor       3. watch progress
#   ./runbook.sh plot          4. ALL figures (plots.py, organised by group)
#   ./runbook.sh grade         5. paper_check tables (optional, paper-repro rows)
#
# Knobs (env): BATCH, PODS, WORKERS, RES, OUT, FAST_DATA(1), DETERMINISM(0)
#
#   BATCH=<> ./runbook.sh manifest 
#   WORKERS=<> PODS=<> BATCH=<> ./runbook.sh submit      
#   RES=~/local/results ./runbook.sh plot            
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"

# ---- batch selection --------------------------------------------------------
# Default = A
BATCH="${BATCH:-A}"   # WHOLE tokens, space/comma separated: "H T EA" 
PAPER_OK="${PAPER_OK:-0}"          # 1 = also build probe-gated paper rows (grade phase)
FAST_DATA="${FAST_DATA:-1}"        # 1 = GPU-resident loaders (kills DataLoader fork storms)
DETERMINISM="${DETERMINISM:-0}"    # 0 = cuDNN autotuner on (~1.3-2x; stat. identical over seeds)
PODS="${PODS:-2}"; WORKERS="${WORKERS:-6}"
MPS="${MPS:-1}"

RES="${RES:-/mnt/nfs/home/zu/results}"   # cluster results (submit) OR local dir (plot)
OUT="${OUT:-$RES/figs}"
ALL="$RES/*/result.json"

# ---- frozen references. Change here, applied to every plot ----
HON=A1_honest_c100                 # honest calibration family (IID, c100, 10 clients)
HONCLASS="${HONCLASS:-A1_honest_c100}"   # all-honest family for the class-acc bar chart
ETA_T="0.064"; ETA_L="0.264"       # IID  eta tight / loose
ETA_T_NIID="0.161"; ETA_L_NIID="0.576"   # non-IID eta tight / loose

PL="python ../scripts/plots.py"   
PC="python ../scripts/paper_check.py"
run(){ echo "== $*"; eval "$*" || echo "   (skipped -- family may not exist yet)"; }

# ---------------------------------------------------------------------------
phase_probe(){
  echo ">>> PROBE: embedding sanity (watch run.log: ber_h should drop, pmax not nan)"
  NUM_WORKERS=0 FAMILY=probe_fix WM_BITS=2 WM_NUM_TRIGGERS=50 ROUNDS=25 \
      ./submit_experiment.sh 11 0
}

phase_manifest(){
  echo ">>> MANIFEST: groups=[$BATCH] PAPER_OK=$PAPER_OK FAST_DATA=$FAST_DATA DETERMINISM=$DETERMINISM"
  rm -f jobs.tsv
  FAST_DATA="$FAST_DATA" DETERMINISM="$DETERMINISM" PAPER_OK="$PAPER_OK" ./run_now.sh "$BATCH"
  echo "   -> jobs.tsv built. Review the per-family counts above."
}

phase_submit(){
  echo ">>> SUBMIT: $PODS pod(s) x $WORKERS workers, shared queue over jobs.tsv"
  [ -s jobs.tsv ] || { echo "!! no jobs.tsv -- run ./runbook.sh manifest first"; return 1; }
  [ "$MPS" = "1" ] && echo "   (MPS=1: start nvidia-cuda-mps-control -d in each pod before the workers)"
  unset DRYRUN
  WORKERS="$WORKERS" PODS="$PODS" ./submit_pool.sh
  echo "   monitor with:  ./runbook.sh monitor"
}

phase_monitor(){
  echo ">>> MONITOR"
  run "runai list jobs"
  echo "--- quick digest of whatever has landed ---"
  run "python ../scripts/resultio.py digest --in '$ALL'"
  echo "--- speed: seconds/round (last col) ---"
  run "for d in $RES/*/run.log; do echo \"\$d:\"; awk '\$2==\"R\" && \$3 ~ /^[0-9]/ {print \$3, \$NF}' \"\$d\" | tail -3; done"
}

# ---------------------------------------------------------------------------
# 4. ALL FIGURES. eta is FROZEN.
#    Organised strictly by experiment group; every call maps to a merged plots.py
#    subcommand. Families that are not present yet are skipped by run{} with a note.
# ---------------------------------------------------------------------------
phase_plot(){
  mkdir -p "$OUT"; echo ">>> PLOT -> $OUT"

  # ===================== GROUP A -- honest baseline =========================
  run "$PL honest_lines     --in '$ALL' --family $HON --tail 20 --out $OUT/A1_class_floors"
  run "$PL honest_per_round --in '$ALL' --family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/A1_honest_per_round"
  run "$PL class_acc        --in '$ALL' --family $HONCLASS --out $OUT/A0_class_acc"
  run "$PL timeline --in '$ALL' --family A2_reduced_c100_c17 --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/A2_easy_timeline"
  run "$PL timeline --in '$ALL' --family A3_reduced_c100_c36 --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/A3_hard_timeline"
  run "$PL iso_pair --honest_in '$ALL' --fr_in '$ALL' --family A3_reduced_c100_c36 --class 3 --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/iso_A3_c3"
  run "$PL iso_pair --honest_in '$ALL' --fr_in '$ALL' --family A3_reduced_c100_c36 --class 6 --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/iso_A3_c6"
  # isolated ACCURACY: hard class 6 (FR from K4), easy-medium class 7 (FR from A2)
  run "$PL iso_acc --honest_in '$ALL' --fr_in '$ALL' --family K4_alldyn_block2_c36 --class 6 --out $OUT/iso_acc_c6"
  run "$PL iso_acc --honest_in '$ALL' --fr_in '$ALL' --family A2_reduced_c100_c17 --class 7 --out $OUT/iso_acc_c7"

  # ===================== GROUP T -- honest-band generality (classes 40-49, 90-99) ==
  # Same honest-floor view as A1 but on other CIFAR-100 decades
  # Every decade is its own 10-client / 10-class honest run; together they cover 0-99.
  TDECADES="T4_honest_c100_cls1019 T5_honest_c100_cls2029 T8_honest_c100_cls3039 T1_honest_c100_cls4049 T9_honest_c100_cls5059 T6_honest_c100_cls6069 T7_honest_c100_cls7079 T10_honest_c100_cls8089 T2_honest_c100_cls9099"
  for fam in $TDECADES; do
    run "$PL honest_lines     --in '$ALL' --family $fam --tail 20 --out $OUT/${fam}_class_floors"
    run "$PL honest_per_round --in '$ALL' --family $fam --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/${fam}_per_round"
  done
  # MERGED honest-floor timeline across A1 + every decade run (scales to 100 classes).
  ATFAMS="$HON $TDECADES"
  run "$PL honest_floors_all --in '$ALL' --families $ATFAMS --eta_tight $ETA_T --eta_loose $ETA_L --tail 20 --out $OUT/honest_floors_all"
  # Ranked per-class band (same data, sorted). Not framed as a paper reproduction.
  run "$PL pooled_band --in '$ALL' --families $ATFAMS --tail 20 --out $OUT/pooled_band_AT"

  # ===================== GROUP H -- baseline free-riders (must be CAUGHT) ====
  # Positive controls: base free-riders sit near BER 0.5 
  run "$PL timeline --in '$ALL' --family H5_prevmodel_c100 --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/H5_prevmodel_timeline"
  run "$PL timeline --in '$ALL' --family H6_gaussian_c100 --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/H6_gaussian_timeline"

  # ===================== GROUP D -- reduced data-budget spectrum ============
  run "$PL sweep       --in '$ALL' --family D1_reduced_c100_c36 --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/D1_spectrum"
  run "$PL gpu_savings --in '$ALL' --family D1_reduced_c100_c36_n5 --out $OUT/gpu_savings_D1_reduced_c100_c36_n5"

  # ===================== GROUP E -- starved non-IID =========================
  run "$PL honest_lines     --in '$ALL' --family E1_honest_niid_c100 --tail 20 --out $OUT/E1_class_floors"
  run "$PL honest_per_round --in '$ALL' --family E1_honest_niid_c100 --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/E1_honest_per_round"
  run "$PL timeline --in '$ALL' --family E2_reduced_niid_c36 --honest_in '$RES/E1_honest_niid_c100_rep*/result.json' --honest_family E1_honest_niid_c100 --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/E2_niid_timeline"
  for at in a01 a10; do
    run "$PL timeline --in '$ALL' --family E3_reduced_niid_c36_${at} --honest_in '$ALL' --honest_family E3_honest_niid_c100_${at} --eta_tight $ETA_T_NIID --out $OUT/E3_${at}_timeline"
  done
  run "$PL iso_pair    --honest_in '$RES/E1_honest_niid_c100_rep*/result.json' --fr_in '$ALL' --family E2_reduced_niid_c36 --class 6 --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/iso_E2_c6"
  run "$PL gpu_savings --in '$ALL' --family E2_reduced_niid_c36 --out $OUT/gpu_savings_E2_reduced_niid_c36"
  run "$PL dirichlet_dist --out $OUT/dirichlet_dist"

  # ===================== GROUP EA -- fair non-IID (distribution assignment) ==
  run "$PL honest_per_round --in '$ALL' --family EA1_honest_niid_distrib_c100 --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/EA1_honest_per_round"
  run "$PL iso_pair    --honest_in '$RES/EA1_honest_niid_distrib_c100_rep*/result.json' --fr_in '$ALL' --family EA2_reduced_niid_distrib_c36 --class 6 --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/iso_EA2_c6"
  run "$PL gpu_savings --in '$ALL' --family EA2_reduced_niid_distrib_c36 --out $OUT/gpu_savings_EA2_reduced_niid_distrib_c36"

  # ===== NON-IID SUBMARINE: full suite + timelines (same views as IID K group) =====
  # E* = random trigger assignment (honest floor from E1) ; EA* = fair (honest floor from EA1)
  NIID_SUB="E4_submarine_niid_c36 E4_submarine_niid_c17 E4_submarine_niid_c36_a01 E4_submarine_niid_c36_a10 EA3_submarine_niid_distrib_c36 EA3_submarine_niid_distrib_c17"
  for fam in $NIID_SUB; do
    case $fam in
      EA*) HF=EA1_honest_niid_distrib_c100 ;;
      *)   HF=E1_honest_niid_c100 ;;
    esac
    HG="$RES/${HF}_rep*/result.json"
    run "$PL tap_perfr   --in '$ALL' --family $fam --honest_in '$HG' --honest_family $HF --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/tap_perfr_${fam}"
    run "$PL tap_perseed --in '$ALL' --family $fam --honest_in '$HG' --honest_family $HF --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/tap_perseed_${fam}"
    run "$PL tap_effort  --in '$ALL' --family $fam --honest_in '$HG' --honest_family $HF --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/tap_effort_${fam}"
    run "$PL timeline    --in '$ALL' --family $fam --honest_in '$HG' --honest_family $HF --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/timeline_${fam}"
    run "$PL gpu_savings --in '$ALL' --family $fam --out $OUT/gpu_savings_${fam}"
    run "$PL accuracy    --in '$ALL' --family $fam --honest_in '$HG' --honest_family $HF --out $OUT/accuracy_${fam}"
  done
  # isolated same-class twin for the random-assign submarine (fixed classes 3,6 vs E1 honest)
  for cls in 3 6; do
    run "$PL iso_pair --honest_in '$RES/E1_honest_niid_c100_rep*/result.json' --fr_in '$RES/E4_submarine_niid_c36_rep*/result.json' --class $cls --eta_tight $ETA_T_NIID --eta_loose $ETA_L_NIID --out $OUT/iso_E4_c${cls}"
  done

  # ===== AMPLIFICATION PANEL: ROC + IID-vs-nonIID side-by-side + starvation + savings =====
  # alpha axis: IID (K4) + non-IID random a=1.0/0.5/0.1 (+ fair EA3). One curve/bar per setting.
  AMP_FAMS="K4_alldyn_block2_c36 E4_submarine_niid_c36_a10 E4_submarine_niid_c36 E4_submarine_niid_c36_a01 EA3_submarine_niid_distrib_c36"
  # ROC / threshold-dilemma (honest pop = each family's own honest clients)
  run "$PL roc --in '$ALL' --families $AMP_FAMS --tail 20 --out $OUT/roc_iid_vs_niid"
  # IID vs non-IID(random) isolated same-class, side by side, at cls 3 and 6
  for cls in 3 6; do
    run "$PL iso_compare --class $cls \
       --honest_in '$RES/${HON}_rep*/result.json' '$RES/E1_honest_niid_c100_rep*/result.json' \
       --fr_in '$RES/K4_alldyn_block2_c36_rep*/result.json' '$RES/E4_submarine_niid_c36_rep*/result.json' \
       --families 'IID' 'non-IID random' --out $OUT/iso_compare_c${cls}"
  done
  # starvation (embedding health) + savings/tap-fraction across the alpha sweep
  run "$PL starvation       --in '$ALL' --families $AMP_FAMS --tail 20 --out $OUT/starvation_vs_alpha"
  run "$PL savings_vs_alpha --in '$ALL' --families $AMP_FAMS --out $OUT/savings_vs_alpha"

  # ===================== GROUP K -- the submarine ===========================
  # all submarine families. c36 => free-riders on classes 3,6 ; c17 => classes 1,7.
  #   tap_perfr = seed-band ; tap_perseed = per-seed panels ; tap_effort = BER+samples ;
  #   accuracy = main-task ; gpu_savings/timeline = cost & mean. FR cids auto-detected.
  K_C36="K4_alldyn_block2_c36 K5_alldyn_full_c36 K7_costopt_block2_cpc10_c36 K8_opt_block2_c36"
  K_C17="K4_alldyn_block2_c17 K8_opt_block2_c17"
  for fam in $K_C36 $K_C17; do
    run "$PL tap_perfr   --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_perfr_${fam}"
    run "$PL tap_perseed --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_perseed_${fam}"
    run "$PL tap_effort  --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_effort_${fam}"
    run "$PL accuracy    --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --out $OUT/accuracy_${fam}"
    run "$PL gpu_savings --in '$ALL' --family $fam --out $OUT/gpu_savings_${fam}"
    run "$PL timeline    --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/timeline_${fam}"
  done
  # isolated same-class twin (honest from A1): c36 families at cls 3,6 ; c17 families at cls 1,7
  for fam in $K_C36; do
    for cls in 3 6; do
      run "$PL iso_pair --honest_in '$RES/${HON}_rep*/result.json' --fr_in '$RES/${fam}_rep*/result.json' --class $cls --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/iso_${fam}_c${cls}"
    done
  done
  for fam in $K_C17; do
    for cls in 1 7; do
      run "$PL iso_pair --honest_in '$RES/${HON}_rep*/result.json' --fr_in '$RES/${fam}_rep*/result.json' --class $cls --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/iso_${fam}_c${cls}"
    done
  done
  # honest per-class band vs ALL submarine operating points on one axis.
  KFR=""
  for fam in $K_C36 $K_C17 J4_scope_graft_block2_c36 J4_scope_graft_block2_c17; do
    KFR="$KFR '$RES/${fam}_rep*/result.json'"
  done
  run "$PL overlap --in '$ALL' --families $ATFAMS --fr_in $KFR --eta_tight $ETA_T --eta_loose $ETA_L --tail 20 --out $OUT/overlap_band_vs_fr"

  # ===================== GROUP Z -- NO-WATERMARK control ====================
  # A0_nowm = all-honest, lambda=0 (embedding OFF) 
  run "$PL honest_per_round --in '$ALL' --family A0_nowm_honest_c100 --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/A0_nowm_per_round"
  run "$PL class_acc        --in '$ALL' --family A0_nowm_honest_c100 --out $OUT/A0_nowm_class_acc"

  # ===================== GROUP Y -- ORACLE-THRESHOLD ablation (J4) ==========
  # J4 = the submarine handed the TRUE eta (0.264) instead of self-estimating it.
  # c36 => classes 3,6 ; c17 => classes 1,7. Distinct out-prefixes (no overwrite).
  for fam in J4_scope_graft_block2_c36 J4_scope_graft_block2_c17; do
    run "$PL tap_perfr   --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_perfr_${fam}"
    run "$PL tap_perseed --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_perseed_${fam}"
    run "$PL tap_effort  --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_effort_${fam}"
    run "$PL timeline    --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/timeline_${fam}"
  done

  # ===================== GROUP L -- block-graft free-rider =================
  # reduced + last-layers-only (+ optional graft)
  #   head2  scope = softmax fc + the conv layer just before it (last 5 tensors)
  #   block2 scope = last 20 tensors 
  L_FAMS="L1_graftblock_head2_c36 L2_graftblock_block2_c36 L3_graftblock_head2_graft_c36 L4_graftblock_block2_graft_c36 L5_graftblock_head2_c17"
  for fam in $L_FAMS; do
    run "$PL tap_perfr   --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_perfr_${fam}"
    run "$PL tap_perseed --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_perseed_${fam}"
    run "$PL tap_effort  --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/tap_effort_${fam}"
    run "$PL accuracy    --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --out $OUT/accuracy_${fam}"
    run "$PL gpu_savings --in '$ALL' --family $fam --out $OUT/gpu_savings_${fam}"
    run "$PL timeline    --in '$ALL' --family $fam --honest_in '$ALL' --honest_family $HON --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/timeline_${fam}"
    for cls in 3 6; do
      run "$PL iso_pair --honest_in '$RES/${HON}_rep*/result.json' --fr_in '$RES/${fam}_rep*/result.json' --class $cls --eta_tight $ETA_T --eta_loose $ETA_L --out $OUT/iso_${fam}_c${cls}"
    done
  done
  # scope x graft ablation on one axis (savings + tap-fraction across the 4 L variants)
  run "$PL savings_vs_alpha --in '$ALL' --families $L_FAMS --out $OUT/savings_graftblock"

  echo "   done -> $OUT  (A honest / D reduced / E starved-niid / EA fair-niid / K+Y submarine / Z no-wm control)"
}

phase_grade(){
  # OPTIONAL paper-reproduction tables (needs the probe-gated paper rows, PAPER_OK=1).
  echo ">>> GRADE vs the FareMark paper (optional; needs paper-repro families)"
  run "$PC --row t9 --in '$ALL' --family F3_tableIX_c10_nc50 --heldout-family F3_tableIX_c10_nc50_heldout"
  run "$PC --row c10 --in '$ALL' --family H1_honest_c10"
}

case "${1:-help}" in
  probe)     phase_probe ;;
  manifest)  phase_manifest ;;
  submit)    phase_submit ;;
  monitor)   phase_monitor ;;
  plot)      phase_plot ;;
  grade)     phase_grade ;;
  all-submit) phase_probe; phase_manifest; phase_submit ;;   # 0 -> 1 -> 2
  *)
    cat <<USAGE
runbook.sh -- run phases 

  ON THE CLUSTER (has submit_experiment.sh + .env):
    ./runbook.sh probe       0. embedding sanity
    ./runbook.sh manifest    1. build jobs.tsv     (BATCH=$BATCH)
    ./runbook.sh submit      2. run the pool       (PODS=$PODS WORKERS=$WORKERS)
    ./runbook.sh monitor     3. progress
       ... wait for jobs to finish ...

  LOCALLY (set RES=~/local/results):
    RES=~/local/results ./runbook.sh plot     4. ALL figures
    RES=~/local/results ./runbook.sh grade    5. paper tables (optional)

  batch tokens (whole, space/comma separated): A T D E EA H K Y Z .
USAGE
    ;;
esac