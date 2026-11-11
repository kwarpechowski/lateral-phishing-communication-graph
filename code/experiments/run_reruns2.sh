#!/bin/bash
# FALA 3 re-runow po: (a) event-causal COMPA (exp_cascade_sweep._compa_features +
# cascadebench/features.compa), (b) HopperNoPrecursor w panelu cb, (c) per-seed zrzuty.
# 5 rownoleglych lane'ow x OMP=4 (22 rdzenie -> ~20 watkow + OS).
cd "$(dirname "$0")/.."
PY="C:/Users/kamilwa_admin/Desktop/doktorat/venv/Scripts/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs
run() { local name=$1; shift; echo "=== $name start $(date +%H:%M:%S)" >> logs/$name.log; "$PY" -u "$@" >> logs/$name.log 2>&1 && echo "=== $name OK $(date +%H:%M:%S)" >> logs/$name.log || echo "=== $name FAIL $(date +%H:%M:%S)" >> logs/$name.log; }
runc() { local name=$1; local code=$2; echo "=== $name start $(date +%H:%M:%S)" >> logs/$name.log; "$PY" -u -c "$code" >> logs/$name.log 2>&1 && echo "=== $name OK $(date +%H:%M:%S)" >> logs/$name.log || echo "=== $name FAIL $(date +%H:%M:%S)" >> logs/$name.log; }

case "$1" in
lane1)
  # panel cb z HopperNoPrecursor (event-causal COMPA) + per-seed
  run cb_stealth experiments/exp_cb_stealth.py
  run cb_surgical experiments/exp_cb_stealth.py surgical
  run cb_labelnoise experiments/exp_cb_stealth.py labelnoise
  ;;
lane2)
  # framework-1: glowny + odroznienia + sweep + arch + tgn
  run cascade experiments/exp_cascade.py
  run cascade_org experiments/exp_cascade.py org
  run cascade_org240 experiments/exp_cascade.py org scaleN=240
  run cascade_org1600 experiments/exp_cascade.py org scaleN=1600
  run sweep experiments/exp_cascade_sweep.py
  run arch experiments/exp_cascade_arch.py
  run tgn experiments/exp_cascade_tgn.py
  ;;
lane3)
  # real-data: Enron (split + kaskady z rezymem stealthy) + stealth sweep A
  run enron_split experiments/exp_enron_temporal_split.py
  run enron_casc experiments/exp_cascade_enron.py
  run stealth experiments/exp_cascade_stealth.py
  ;;
lane4)
  # rigor: headline_ci (+perseed) + mechanism + fpr + realgraphs + compromised
  runc headline "import sys; sys.path.insert(0,'.'); from cascadebench.experiments import headline_ci; headline_ci(seeds=20)"
  runc mechanism "import sys; sys.path.insert(0,'.'); from cascadebench.experiments import mechanism_ablation; mechanism_ablation()"
  run fpr experiments/exp_cascade_fpr.py
  run realgraphs experiments/exp_cascade_realgraphs.py
  run compromised experiments/exp_compromised.py
  ;;
lane5)
  # NOWE (po recenzji zewnetrznej): trudne negatywy, tc-split, CI osint, fab100
  run legal_cascade experiments/exp_legal_cascade.py
  run tc_split experiments/exp_tc_split.py
  run osint_shuffle experiments/exp_osint_shuffle.py
  run multiplex experiments/exp_multiplex.py
  ;;
esac
echo "=== lane $1 KONIEC $(date +%H:%M:%S)"
