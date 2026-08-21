#!/usr/bin/env bash
# =============================================================================
# run_now.sh -- builds jobs.tsv for a chosen set of groups
# Called by runbook.sh manifest. Submits nothing; submit_pool.sh runs it.
#
#   ./run_now.sh              # default groups (A)
#   ./run_now.sh "E EA K"     # E + EA + K together   (whole tokens, space/comma separated)
#   ./run_now.sh A            # just group A
#
# GROUPS (tokens are matched whole: "H T EA" -> H, T, EA; not E or A)
#   A  proven IID baseline (A1 honest, A2 reduced easy c17, A3 reduced hard c36)
#   T  honest band generality: trigger classes beyond 0-9 (check the mapping)
#   D  reduced free-rider +N data-budget spectrum (c36)
#   E  non-IID starved (E1 honest, E2 reduced, E3 alpha sweep)
#   EA non-IID distribution-aware assignment (EA1 honest, EA2 reduced)
#   H  baseline free-riders (positive controls that MUST be caught): H5 previous-models
#      on c100 (our setting) + H6 gaussian-noise on c100.  Paper-setting CIFAR-10 rows
#      are the optional `grade` phase (paper_check), not here.
#   K  the submarine: K4 (block2 scope) + K9 (head2 scope) - 1,7 and 3,6
#   L  graftblock free-rider (last-block-only) - 1,7 and 3,6
#   Z  no-watermark control (all-honest, lambda=0) for the trig_acc check
#   Y  oracle threshold (J4): the submarine K4 run handed the true eta (1,7 and 3,6)
#
# =============================================================================
set -uo pipefail
export DRYRUN=1 JOBS_FILE="${JOBS_FILE:-jobs.tsv}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
WANT="${1:-A}"
PAPER_OK="${PAPER_OK:-0}"
rm -f "$JOBS_FILE"
WANT=" ${WANT//,/ } "
echo "== building $JOBS_FILE  groups=[${WANT# }]  PAPER_OK=$PAPER_OK  NUM_WORKERS=$NUM_WORKERS =="
has(){ [[ "$WANT" == *" $1 "* ]]; }

# ---------------------------------------------------------------------------
# GROUP A -- proven IID baseline (cifar100, 10 clients). 
# ---------------------------------------------------------------------------
if has A; then
  echo "   (group A -- uncomment loops below to regenerate)"
  for s in 0 1 2 3 4 5; do
    env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 \
        FAMILY="A1_honest_c100" NOTE="A1 honest baseline" ./submit_experiment.sh 14 "$s"
  done
  for s in 0 1 2; do
    env ATTACK=reduced FREE_RIDER_IDS=1,7 AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 \
        AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
        FAMILY="A2_reduced_c100_c17" NOTE="A2 reduced +5 easy classes 1,7" ./submit_experiment.sh 14 "$s"
  done
  for s in 0 1 2; do
    env ATTACK=reduced FREE_RIDER_IDS=3,6 AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 \
        AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
        FAMILY="A3_reduced_c100_c36" NOTE="A3 reduced +5 hard classes 3,6" ./submit_experiment.sh 14 "$s"
  done
fi

# ---------------------------------------------------------------------------
# GROUP T -- honest-band generality: show the per-class BER band is not an artifact
#   of only ever using classes 0-9 (cid % num_classes)
# ---------------------------------------------------------------------------
if has T; then
  SEEDS_T="${SEEDS_T:-0 1 2}"
  MAP1019="0:10,1:11,2:12,3:13,4:14,5:15,6:16,7:17,8:18,9:19"
  MAP2029="0:20,1:21,2:22,3:23,4:24,5:25,6:26,7:27,8:28,9:29"
  MAP3039="0:30,1:31,2:32,3:33,4:34,5:35,6:36,7:37,8:38,9:39"
  MAP4049="0:40,1:41,2:42,3:43,4:44,5:45,6:46,7:47,8:48,9:49"
  MAP5059="0:50,1:51,2:52,3:53,4:54,5:55,6:56,7:57,8:58,9:59"
  MAP6069="0:60,1:61,2:62,3:63,4:64,5:65,6:66,7:67,8:68,9:69"
  MAP7079="0:70,1:71,2:72,3:73,4:74,5:75,6:76,7:77,8:78,9:79"
  MAP8089="0:80,1:81,2:82,3:83,4:84,5:85,6:86,7:87,8:88,9:89"
  MAP9099="0:90,1:91,2:92,3:93,4:94,5:95,6:96,7:97,8:98,9:99"
  for s in $SEEDS_T; do
    env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 TRIGGER_CLASS_MAP="$MAP4049" \
        FAMILY="T1_honest_c100_cls4049" NOTE="T1 honest band, trigger classes 40-49" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in $SEEDS_T; do
    env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 TRIGGER_CLASS_MAP="$MAP9099" \
        FAMILY="T2_honest_c100_cls9099" NOTE="T2 honest band, trigger classes 90-99" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in $SEEDS_T; do
    env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 TRIGGER_CLASS_MAP="$MAP1019" \
        FAMILY="T4_honest_c100_cls1019" NOTE="T4 honest band, trigger classes 10-19" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in $SEEDS_T; do
    env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 TRIGGER_CLASS_MAP="$MAP2029" \
        FAMILY="T5_honest_c100_cls2029" NOTE="T5 honest band, trigger classes 20-29" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in $SEEDS_T; do
    env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 TRIGGER_CLASS_MAP="$MAP6069" \
        FAMILY="T6_honest_c100_cls6069" NOTE="T6 honest band, trigger classes 60-69" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in $SEEDS_T; do
    env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 TRIGGER_CLASS_MAP="$MAP7079" \
        FAMILY="T7_honest_c100_cls7079" NOTE="T7 honest band, trigger classes 70-79" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in $SEEDS_T; do
    env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 TRIGGER_CLASS_MAP="$MAP3039" \
        FAMILY="T8_honest_c100_cls3039" NOTE="T8 honest band, trigger classes 30-39" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in $SEEDS_T; do
    env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 TRIGGER_CLASS_MAP="$MAP5059" \
        FAMILY="T9_honest_c100_cls5059" NOTE="T9 honest band, trigger classes 50-59" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in $SEEDS_T; do
    env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 TRIGGER_CLASS_MAP="$MAP8089" \
        FAMILY="T10_honest_c100_cls8089" NOTE="T10 honest band, trigger classes 80-89" \
        ./submit_experiment.sh 14 "$s"
  done
fi

# ---------------------------------------------------------------------------
# GROUP D -- +N reduced spectrum at the hard classes (3,6). 
# ---------------------------------------------------------------------------
if has D; then
  echo "   (group D -- uncomment loop below to regenerate)"
  for N in -1 0 1 2 5 10; do
    for s in 0 1 2; do
      env ATTACK=reduced FREE_RIDER_IDS=6,8 AUTOP_COMMON_PER_CLASS=$N \
          AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 WM_ETA_FIXED=0.064 ROUNDS=50 \
          FAMILY="D1_reduced_c100_c68_n${N}" NOTE="D1 +N spectrum N=$N" ./submit_experiment.sh 14 "$s"
    done
  done
fi

# ---------------------------------------------------------------------------
# GROUP E -- non-IID (Dirichlet). E1 honest a=0.5, E2 reduced a=0.5, E3 alpha sweep.
# ---------------------------------------------------------------------------
if has E; then
  SEEDS_E="${SEEDS_E:-0 1 2}"
  for s in $SEEDS_E; do
    env ATTACK=none NUM_FREE_RIDERS=0 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 ROUNDS=50 \
        FAMILY="E1_honest_niid_c100" NOTE="E1 non-iid honest a=0.5" ./submit_experiment.sh 14 "$s"
  done
  for s in $SEEDS_E; do
    env ATTACK=reduced FREE_RIDER_IDS=3,6 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 \
        AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
        WM_ETA_FIXED=0.161 ROUNDS=50 \
        FAMILY="E2_reduced_niid_c36" NOTE="E2 non-iid reduced hard a=0.5" ./submit_experiment.sh 14 "$s"
  done
  for A in 0.1 1.0; do
    ATAG="a$(printf '%s' "$A" | tr -d '.')"
    for s in $SEEDS_E; do
      env ATTACK=none NUM_FREE_RIDERS=0 PARTITION=dirichlet DIRICHLET_ALPHA=$A ROUNDS=50 \
          FAMILY="E3_honest_niid_c100_${ATAG}" NOTE="E3 non-iid honest alpha=$A" ./submit_experiment.sh 14 "$s"
    done
    for s in $SEEDS_E; do
      env ATTACK=reduced FREE_RIDER_IDS=3,6 PARTITION=dirichlet DIRICHLET_ALPHA=$A \
          AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
          WM_ETA_FIXED=0.161 ROUNDS=50 \
          FAMILY="E3_reduced_niid_c36_${ATAG}" NOTE="E3 non-iid reduced hard alpha=$A" ./submit_experiment.sh 14 "$s"
    done
  done
  # submarine attack on E non-iid
  for s in $SEEDS_E; do
    env ATTACK=adaptive_tap FREE_RIDER_IDS=3,6 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 \
        AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 AUTOP_ORACLE_ETA=0.264 \
        WM_ETA_FIXED=0.064 TAP_DATA_CPC=5 TAP_COAST_MODE=graft TAP_WHEN=threshold \
        TAP_PROBE_HOLDOUT=16 TAP_MARGIN=0.03 TAP_MAX_COAST=6 TAP_GRAFT_DECAY=0.25 ROUNDS=50 FAST_DATA=1 \
        TAP_ETA_SOURCE=self TAP_ETA_K=3.0 TAP_MARGIN_MODE=derived TAP_MARGIN_K=1.0 \
        TAP_WARMUP_MODE=dynamic TAP_CONV_EPS=0.03 TAP_CONV_PATIENCE=2 \
        TAP_HONEST_MIN=6 TAP_WARMUP_CAP=15 TAP_SCOPE=block2 \
        FAMILY="E4_submarine_niid_c36" \
        NOTE="E4 hard non-iid submarine (self-eta, derived margin, dynamic warmup) on distribution-aware assignment" ./submit_experiment.sh 14 "$s"
  done
  for s in $SEEDS_E; do
    env ATTACK=adaptive_tap FREE_RIDER_IDS=1,7 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 \
        AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 AUTOP_ORACLE_ETA=0.264 \
        WM_ETA_FIXED=0.064 TAP_DATA_CPC=5 TAP_COAST_MODE=graft TAP_WHEN=threshold \
        TAP_PROBE_HOLDOUT=16 TAP_MARGIN=0.03 TAP_MAX_COAST=6 TAP_GRAFT_DECAY=0.25 ROUNDS=50 FAST_DATA=1 \
        TAP_ETA_SOURCE=self TAP_ETA_K=3.0 TAP_MARGIN_MODE=derived TAP_MARGIN_K=1.0 \
        TAP_WARMUP_MODE=dynamic TAP_CONV_EPS=0.03 TAP_CONV_PATIENCE=2 \
        TAP_HONEST_MIN=6 TAP_WARMUP_CAP=15 TAP_SCOPE=block2 \
        FAMILY="E4_submarine_niid_c17" \
        NOTE="E4 easy non-iid submarine (self-eta, derived margin, dynamic warmup) on distribution-aware assignment" ./submit_experiment.sh 14 "$s"
  done
  # --- submarine non-iid ALPHA SWEEP (hard c36) ---
  for A in 0.1 1.0; do
    ATAG=$(echo "a$A" | tr -d '.')
    for s in $SEEDS_E; do
      env ATTACK=adaptive_tap FREE_RIDER_IDS=3,6 PARTITION=dirichlet DIRICHLET_ALPHA=$A \
          AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 AUTOP_ORACLE_ETA=0.264 \
          WM_ETA_FIXED=0.064 TAP_DATA_CPC=5 TAP_COAST_MODE=graft TAP_WHEN=threshold \
          TAP_PROBE_HOLDOUT=16 TAP_MARGIN=0.03 TAP_MAX_COAST=6 TAP_GRAFT_DECAY=0.25 ROUNDS=50 FAST_DATA=1 \
          TAP_ETA_SOURCE=self TAP_ETA_K=3.0 TAP_MARGIN_MODE=derived TAP_MARGIN_K=1.0 \
          TAP_WARMUP_MODE=dynamic TAP_CONV_EPS=0.03 TAP_CONV_PATIENCE=2 \
          TAP_HONEST_MIN=6 TAP_WARMUP_CAP=15 TAP_SCOPE=block2 \
          FAMILY="E4_submarine_niid_c36_${ATAG}" \
          NOTE="E4 hard non-iid submarine alpha=$A (amplification sweep)" ./submit_experiment.sh 14 "$s"
    done
  done
fi

# ---------------------------------------------------------------------------
# GROUP EA -- distribution-aware trigger assignment (non-IID fairness fix).
#   Server assigns each client a class it holds a lot of (instead of blind cid%n).
# ---------------------------------------------------------------------------
if has EA; then
  SEEDS_EA="${SEEDS_EA:-0 1 2}"
  for s in $SEEDS_EA; do
    env ATTACK=none NUM_FREE_RIDERS=0 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 \
        WM_TRIGGER_ASSIGN=distribution ROUNDS=50 \
        FAMILY="EA1_honest_niid_distrib_c100" \
        NOTE="EA1 non-iid honest a=0.5, DISTRIBUTION trigger assignment" ./submit_experiment.sh 14 "$s"
  done
  for s in $SEEDS_EA; do
    env ATTACK=reduced FREE_RIDER_IDS=3,6 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 \
        WM_TRIGGER_ASSIGN=distribution \
        AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
        WM_ETA_FIXED=0.161 ROUNDS=50 \
        FAMILY="EA2_reduced_niid_distrib_c36" \
        NOTE="EA2 non-iid reduced a=0.5, DISTRIBUTION assignment for all incl. free-riders" ./submit_experiment.sh 14 "$s"
  done
  # submarine attack on EA distribution aware non-iid
  for s in $SEEDS_EA; do
    env ATTACK=adaptive_tap FREE_RIDER_IDS=3,6 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 \
        WM_TRIGGER_ASSIGN=distribution \
        AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 AUTOP_ORACLE_ETA=0.264 \
        WM_ETA_FIXED=0.064 TAP_DATA_CPC=5 TAP_COAST_MODE=graft TAP_WHEN=threshold \
        TAP_PROBE_HOLDOUT=16 TAP_MARGIN=0.03 TAP_MAX_COAST=6 TAP_GRAFT_DECAY=0.25 ROUNDS=50 FAST_DATA=1 \
        TAP_ETA_SOURCE=self TAP_ETA_K=3.0 TAP_MARGIN_MODE=derived TAP_MARGIN_K=1.0 \
        TAP_WARMUP_MODE=dynamic TAP_CONV_EPS=0.03 TAP_CONV_PATIENCE=2 \
        TAP_HONEST_MIN=6 TAP_WARMUP_CAP=15 TAP_SCOPE=block2 \
        FAMILY="EA3_submarine_niid_distrib_c36" \
        NOTE="EA3 hard non-iid submarine (self-eta, derived margin, dynamic warmup) on distribution-aware assignment" ./submit_experiment.sh 14 "$s"
  done
  for s in $SEEDS_EA; do
    env ATTACK=adaptive_tap FREE_RIDER_IDS=1,7 PARTITION=dirichlet DIRICHLET_ALPHA=0.5 \
        WM_TRIGGER_ASSIGN=distribution \
        AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 AUTOP_ORACLE_ETA=0.264 \
        WM_ETA_FIXED=0.064 TAP_DATA_CPC=5 TAP_COAST_MODE=graft TAP_WHEN=threshold \
        TAP_PROBE_HOLDOUT=16 TAP_MARGIN=0.03 TAP_MAX_COAST=6 TAP_GRAFT_DECAY=0.25 ROUNDS=50 FAST_DATA=1 \
        TAP_ETA_SOURCE=self TAP_ETA_K=3.0 TAP_MARGIN_MODE=derived TAP_MARGIN_K=1.0 \
        TAP_WARMUP_MODE=dynamic TAP_CONV_EPS=0.03 TAP_CONV_PATIENCE=2 \
        TAP_HONEST_MIN=6 TAP_WARMUP_CAP=15 TAP_SCOPE=block2 \
        FAMILY="EA3_submarine_niid_distrib_c17" \
        NOTE="EA3 easy non-iid submarine (self-eta, derived margin, dynamic warmup) on distribution-aware assignment" ./submit_experiment.sh 14 "$s"
  done
fi


# ---------------------------------------------------------------------------
# GROUP H -- H5 base previous-models free-rider on c100 (positive control).
#   Must be CAUGHT (BER ~0.5-0.8). cids 3,6 to match K4/D1.
# ---------------------------------------------------------------------------
if has H; then
  for s in 0 1 2; do
    env ATTACK=previous_models NUM_FREE_RIDERS=2 FREE_RIDER_IDS=3,6 WM_ETA_FIXED=0.064 ROUNDS=50 \
        FAMILY="H5_prevmodel_c100" NOTE="H5 baseline FR previous-models on c100 (positive control)" \
        ./submit_experiment.sh 14 "$s"
  done
  for s in 0 1 2; do
    env ATTACK=gaussian NUM_FREE_RIDERS=2 FREE_RIDER_IDS=3,6 NOISE_SIGMA=0.1 WM_ETA_FIXED=0.064 ROUNDS=50 \
        FAMILY="H6_gaussian_c100" NOTE="H6 baseline FR gaussian-noise on c100 (positive control)" \
        ./submit_experiment.sh 14 "$s"
  done
fi

# ---------------------------------------------------------------------------
# GROUP K -- SUBMARINE (self-estimated eta, derived margin, dynamic warmup).
#   K4 = block2 scope (1,7 and 3,6 - 3 seeds). K9 = head2 scope (1,7 and 3,6 - 3 seeds)
# ---------------------------------------------------------------------------
if has K; then
  SEEDS_K="${SEEDS_K:-0 1 2}"

  # kbase = ALL-DYNAMIC (self-eta, DERIVED margin, dynamic warmup). Headline K4 + ablation K5.
  kbase="ATTACK=adaptive_tap FREE_RIDER_IDS=3,6 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
         AUTOP_ORACLE_ETA=0.264 WM_ETA_FIXED=0.064 TAP_DATA_CPC=5 \
         TAP_COAST_MODE=graft TAP_WHEN=threshold TAP_PROBE_HOLDOUT=16 \
         TAP_MARGIN=0.03 TAP_MAX_COAST=6 TAP_GRAFT_DECAY=0.25 ROUNDS=50 FAST_DATA=1 \
         TAP_ETA_SOURCE=self TAP_ETA_K=3.0 TAP_MARGIN_MODE=derived TAP_MARGIN_K=1.0 \
         TAP_WARMUP_MODE=dynamic TAP_CONV_EPS=0.03 TAP_CONV_PATIENCE=2 \
         TAP_HONEST_MIN=6 TAP_WARMUP_CAP=15"

  # K4 -- block2 scope, hard/medium classes 3,6 
  for s in $SEEDS_K; do
    env $kbase TAP_SCOPE=block2 FAMILY="K4_alldyn_block2_c36" \
        NOTE="K4 hard classes all-dynamic + block2 (self-eta, derived margin, dynamic warmup)" \
        ./submit_experiment.sh 14 "$s"
  done
  # K4 -- block2 scope, EASY classes 1,7  
  for s in $SEEDS_K; do
    env $kbase TAP_SCOPE=block2 FREE_RIDER_IDS=1,7 FAMILY="K4_alldyn_block2_c17" \
        NOTE="K4 easy classes all-dynamic + block2 (self-eta, derived margin, dynamic warmup)" \
        ./submit_experiment.sh 14 "$s"
  done
  # # K5 -- full scope - classes 3,6
  # for s in $SEEDS_K; do
  #   env $kbase TAP_SCOPE=full FAMILY="K5_alldyn_full_c36" \
  #       NOTE="K5 all-dynamic + full-scope taps (scope ablation vs K4/block2)" \
  #       ./submit_experiment.sh 14 "$s"
  # done

  # K9 -- HEAD2 scope submarine (all-dynamic), matches LastBlock's head2 scope 
  #       K9 hard/medium classes 3,6
  for s in $SEEDS_K; do
    env $kbase TAP_SCOPE=head2 FAMILY="K9_alldyn_head2_c36" \
        NOTE="K9 hard classes all-dynamic + head2 (self-eta, derived margin, dynamic warmup)" \
        ./submit_experiment.sh 14 "$s"
  done
  # K9 -- head2 scope, EASY classes 1,7
  for s in $SEEDS_K; do
    env $kbase TAP_SCOPE=head2 FREE_RIDER_IDS=1,7 FAMILY="K9_alldyn_head2_c17" \
        NOTE="K9 easy classes all-dynamic + head2 (self-eta, derived margin, dynamic warmup)" \
        ./submit_experiment.sh 14 "$s"
  done

  # # kfix = FIXED-margin cost-optimised base (self-eta, FIXED margin, dynamic warmup, block2).
  # #   K7 and K8 differ only in margin / max_coast / cpc (set per-run below).
  # kfix="ATTACK=adaptive_tap FREE_RIDER_IDS=3,6 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
  #        AUTOP_ORACLE_ETA=0.264 WM_ETA_FIXED=0.064 \
  #        TAP_COAST_MODE=graft TAP_WHEN=threshold TAP_PROBE_HOLDOUT=16 \
  #        TAP_GRAFT_DECAY=0.10 TAP_MARGIN_MODE=fixed ROUNDS=50 FAST_DATA=1 \
  #        TAP_ETA_SOURCE=self TAP_ETA_K=3.0 \
  #        TAP_WARMUP_MODE=dynamic TAP_CONV_EPS=0.03 TAP_CONV_PATIENCE=2 \
  #        TAP_HONEST_MIN=6 TAP_WARMUP_CAP=15 TAP_SCOPE=block2"

  # # K7 -- hard-class data bump: fixed margin 0.03, longer coast (12), cpc 10, classes 3,6
  # for s in $SEEDS_K; do
  #   env $kfix TAP_MARGIN=0.03 TAP_MAX_COAST=12 TAP_DATA_CPC=10 \
  #       FAMILY="K7_costopt_block2_cpc10_c36" \
  #       NOTE="K7 fixed-margin block2, cpc=10 (more embed signal on the hard class)" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # # K8 -- margin optimized submarine: fixed margin 0.12 (stops over-threshold spikes), coast 8, cpc 5.
  # for s in $SEEDS_K; do
  #   env $kfix TAP_MARGIN=0.12 TAP_MAX_COAST=8 TAP_DATA_CPC=5 \
  #       FAMILY="K8_opt_block2_c36" \
  #       NOTE="K8 optimised block2 (fixed margin 0.12) -- hard classes 3,6" \
  #       ./submit_experiment.sh 14 "$s"
  # done
  # for s in $SEEDS_K; do
  #   env $kfix TAP_MARGIN=0.12 TAP_MAX_COAST=8 TAP_DATA_CPC=5 FREE_RIDER_IDS=1,7 \
  #       FAMILY="K8_opt_block2_c17" \
  #       NOTE="K8 optimised block2 (fixed margin 0.12) -- easy classes 1,7" \
  #       ./submit_experiment.sh 14 "$s"
  # done
fi

# ---------------------------------------------------------------------------
# GROUP Z -- NO-WATERMARK sanity check (trigger class accuracy) 
#   All-honest run with the watermark embedding disabled (WM_LAMBDA=0) 
# ---------------------------------------------------------------------------
if has Z; then
  SEEDS_Z="${SEEDS_Z:-0}"                      # single seed
  for s in $SEEDS_Z; do
    env ATTACK=none NUM_FREE_RIDERS=0 ROUNDS=50 WM_LAMBDA=0 \
        FAMILY="A0_nowm_honest_c100" \
        NOTE="Z no-watermark control: lambda=0 (embedding OFF) but verifier ON, so trig_acc is logged; confirms A1 trig_acc~0 is caused by the watermark, not the class" \
        ./submit_experiment.sh 14 "$s"
  done
fi

# ---------------------------------------------------------------------------
# GROUP Y -- J4 oracle threshold - K4 submarine match 
#   J4 = the oracle-eta version of K4 - (AUTOP_ORACLE_ETA=0.264, target 0.234)
# ---------------------------------------------------------------------------
if has Y; then
  SEEDS_Y="${SEEDS_Y:-0 1 2}"                      
  jbase="ATTACK=adaptive_tap AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
         AUTOP_ORACLE_ETA=0.264 WM_ETA_FIXED=0.064 TAP_DATA_CPC=5 TAP_ETA_SOURCE=oracle \
         TAP_PROBE_HOLDOUT=16 TAP_SCOPE=block2 TAP_COAST_MODE=graft TAP_WHEN=threshold \
         TAP_MARGIN=0.03 TAP_MAX_COAST=12 ROUNDS=50 FAST_DATA=1"
  # J4 at hard/medium classes 3,6
  for s in $SEEDS_Y; do
    env $jbase FREE_RIDER_IDS=3,6 \
        FAMILY="J4_scope_graft_block2_c36" \
        NOTE="J4 oracle-eta block2+graft submarine, classes 3,6 (single seed)" \
        ./submit_experiment.sh 14 "$s"
  done
  # J4 at easy classes 1,7
  for s in $SEEDS_Y; do
    env $jbase FREE_RIDER_IDS=1,7 \
        FAMILY="J4_scope_graft_block2_c17" \
        NOTE="J4 oracle-eta block2+graft submarine, classes 1,7 (single seed)" \
        ./submit_experiment.sh 14 "$s"
  done
fi

if has L; then
  # ---------------------------------------------------------------------------
  # GROUP L -- BLOCK-GRAFT free-rider (attack="graftblock").
  #   Warm up honest, then free-ride train only the last layers on a reduced shard (cpc=5)
  #     head2  = softmax fc + the conv layer just before it   (last 5 tensors)
  # ---------------------------------------------------------------------------
  SEEDS_L="${SEEDS_L:-0 1 2}"                        
  lbase="ATTACK=graftblock PARTITION=iid ROUNDS=50 FAST_DATA=1 \
         AUTOP_COMMON_PER_CLASS=5 AUTOP_HONEST_UNTIL=12 AUTOP_CALIB_ROUNDS=4 \
         WM_ETA_FIXED=0.064 FREE_RIDER_IDS=3,6"
  for s in $SEEDS_L; do
    # L1 -- head2 scope (softmax + previous layer)
    env $lbase TAP_SCOPE=head2  TAP_COAST_MODE=decay \
        FAMILY="L1_graftblock_head2_c36" \
        NOTE="L1 graftblock reduced cpc5 + head2 (softmax+prev layer) no graft (classes 3,6)" \
        ./submit_experiment.sh 14 "$s"
    # L2 -- block2 (20 tensors) scope, no graft
    # env $lbase TAP_SCOPE=block2 TAP_COAST_MODE=decay \
    #     FAMILY="L2_graftblock_block2_c36" \
    #     NOTE="L2 graftblock reduced cpc5 + block2 no graft (classes 3,6)" \
    #     ./submit_experiment.sh 14 "$s"
    # L3 -- head2 scope + graft trained scope onto global each round
    # env $lbase TAP_SCOPE=head2  TAP_COAST_MODE=graft \
    #     FAMILY="L3_graftblock_head2_graft_c36" \
    #     NOTE="L3 graftblock reduced cpc5 + head2 + graft onto global each round (classes 3,6)" \
    #     ./submit_experiment.sh 14 "$s"
    # L4 -- block2 (20 tensors) scope + graft trained scope onto global each round
    # env $lbase TAP_SCOPE=block2 TAP_COAST_MODE=graft \
    #     FAMILY="L4_graftblock_block2_graft_c36" \
    #     NOTE="L4 graftblock reduced cpc5 + block2 + graft onto global each round (classes 3,6)" \
    #     ./submit_experiment.sh 14 "$s"
    # L5 -- L1 but on the easy classes 1,7
    env $lbase TAP_SCOPE=head2  TAP_COAST_MODE=decay \
        FREE_RIDER_IDS=1,7 \
        FAMILY="L5_graftblock_head2_c17" \
        NOTE="L5 graftblock reduced cpc5 + head2 (softmax+prev layer) no graft (classes 1,7)" \
        ./submit_experiment.sh 14 "$s"
  done
fi

N=$(grep -c . "$JOBS_FILE" 2>/dev/null || echo 0)
echo
echo "== $N runs queued  (groups: $WANT) =="
cut -f1 "$JOBS_FILE" 2>/dev/null | sed 's/_rep[0-9]*$//' | sort | uniq -c | sed 's/^/   /'
cat <<NEXT

Next:
    unset DRYRUN
    WORKERS=6 PODS=2 ./submit_pool.sh
    runai list jobs
NEXT