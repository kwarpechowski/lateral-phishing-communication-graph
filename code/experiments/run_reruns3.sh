#!/bin/bash
# FALA 4 (recenzja wewnetrzna r2): przyczynowosc zdarzeniowa WSZYSTKICH detektorow (A1),
# volume na Enronie (A5), kontrasty w protokole off-hours (A7), trudne negatywy (A4),
# sonda iniekcji (A2). 6 lane'ow x OMP=4.
cd "$(dirname "$0")/.."
PY="C:/Users/kamilwa_admin/Desktop/doktorat/venv/Scripts/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs
run() { local name=$1; shift; echo "=== $name start $(date +%H:%M:%S)" >> logs/$name.log; "$PY" -u "$@" >> logs/$name.log 2>&1 && echo "=== $name OK $(date +%H:%M:%S)" >> logs/$name.log || echo "=== $name FAIL $(date +%H:%M:%S)" >> logs/$name.log; }
runc() { local name=$1; local code=$2; echo "=== $name start $(date +%H:%M:%S)" >> logs/$name.log; "$PY" -u -c "$code" >> logs/$name.log 2>&1 && echo "=== $name OK $(date +%H:%M:%S)" >> logs/$name.log || echo "=== $name FAIL $(date +%H:%M:%S)" >> logs/$name.log; }

case "$1" in
lane1)
  run cb_stealth experiments/exp_cb_stealth.py
  run cb_surgical experiments/exp_cb_stealth.py surgical
  run cb_labelnoise experiments/exp_cb_stealth.py labelnoise
  ;;
lane2)
  run cascade experiments/exp_cascade.py
  run cascade_org experiments/exp_cascade.py org
  run cascade_org240 experiments/exp_cascade.py org scaleN=240
  run cascade_org1600 experiments/exp_cascade.py org scaleN=1600
  run arch experiments/exp_cascade_arch.py
  run tgn experiments/exp_cascade_tgn.py
  ;;
lane3)
  run enron_split experiments/exp_enron_temporal_split.py
  run enron_casc experiments/exp_cascade_enron.py
  ;;
lane4)
  runc headline "import sys; sys.path.insert(0,'.'); from cascadebench.experiments import headline_ci; headline_ci(seeds=20)"
  runc headline_oh "import sys; sys.path.insert(0,'.'); from cascadebench.experiments import headline_ci; headline_ci(seeds=20, off_hours=0.15, tag='_oh')"
  runc mechanism "import sys; sys.path.insert(0,'.'); from cascadebench.experiments import mechanism_ablation; mechanism_ablation()"
  run compromised experiments/exp_compromised.py
  ;;
lane5)
  run sweep experiments/exp_cascade_sweep.py
  run stealth experiments/exp_cascade_stealth.py
  run fpr experiments/exp_cascade_fpr.py
  ;;
lane6)
  run legal_cascade experiments/exp_legal_cascade.py
  run injection_probe experiments/exp_injection_probe.py
  run realgraphs experiments/exp_cascade_realgraphs.py
  ;;
esac
echo "=== lane $1 KONIEC $(date +%H:%M:%S)"
