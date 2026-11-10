#!/bin/bash
# Re-run wszystkich eksperymentow po fixach leak-aware (momentum, node features, liczniki).
# 4 rownolegle sciezki; OMP_NUM_THREADS=6 dla torch.
cd "$(dirname "$0")/.."   # code/
PY="C:/Users/kamilwa_admin/Desktop/doktorat/venv/Scripts/python"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
mkdir -p logs
run() { local name=$1; shift; echo "=== $name start $(date +%H:%M:%S)" >> logs/$name.log; "$PY" -u "$@" >> logs/$name.log 2>&1 && echo "=== $name OK $(date +%H:%M:%S)" >> logs/$name.log || echo "=== $name FAIL $(date +%H:%M:%S)" >> logs/$name.log; }

case "$1" in
lane1)
  run cascade experiments/exp_cascade.py
  run cascade_org experiments/exp_cascade.py org
  run cascade_org240 experiments/exp_cascade.py org scaleN=240
  run cascade_org1600 experiments/exp_cascade.py org scaleN=1600
  run sweep experiments/exp_cascade_sweep.py
  run arch experiments/exp_cascade_arch.py
  ;;
lane2)
  run stealth experiments/exp_cascade_stealth.py
  run cb_stealth experiments/exp_cb_stealth.py
  run cb_labelnoise experiments/exp_cb_stealth.py labelnoise
  ;;
lane3)
  run osint_shuffle experiments/exp_osint_shuffle.py
  run enron_split experiments/exp_enron_temporal_split.py
  run compromised experiments/exp_compromised.py
  run fpr experiments/exp_cascade_fpr.py
  run enron experiments/exp_cascade_enron.py
  run realgraphs experiments/exp_cascade_realgraphs.py
  ;;
lane4)
  run headline -c "import sys; sys.path.insert(0,'.'); from cascadebench.experiments import headline_ci; headline_ci(seeds=20)"
  run mechanism -c "import sys; sys.path.insert(0,'.'); from cascadebench.experiments import mechanism_ablation; mechanism_ablation()"
  ;;
esac
