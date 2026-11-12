#!/usr/bin/env bash
# Fala R4: kontrola iniekcji (A1/A2) + B1/B2/B3 + analiza resztki/mechanizmu.
# 8 rownoleglych lane'ow, OMP=4 (TGN), LGBM n_jobs=1. Logi: logs/r4_*.log
cd "$(dirname "$0")"
mkdir -p logs
PY="C:/Users/kamilwa_admin/Desktop/doktorat/venv/Scripts/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# L1 (A1, framework-1): stealth sweep det+emp, 20 seedow, KS+sonda
nohup bash -c "
  echo '=== r4_stealth start '\$(date +%H:%M:%S)
  '$PY' -u experiments/exp_cascade_stealth.py && echo '=== r4_stealth OK '\$(date +%H:%M:%S) || echo '=== r4_stealth FAIL '\$(date +%H:%M:%S)
" > logs/r4_stealth.log 2>&1 &

# L2 (A1 framework-2 + B3): cb stealth det+emp; potem ablacja tc z NaN (R4-B3)
nohup bash -c "
  echo '=== r4_cb start '\$(date +%H:%M:%S)
  '$PY' -u experiments/exp_cb_stealth.py stealth && '$PY' -u experiments/exp_tc_ablation.py && echo '=== r4_cb OK '\$(date +%H:%M:%S) || echo '=== r4_cb FAIL '\$(date +%H:%M:%S)
" > logs/r4_cb.log 2>&1 &

# L3 (A1/A7): kontrasty Rigor — off-hours (FIX: off_hours trafia teraz do build_scenario) + timing=emp
nohup bash -c "
  echo '=== r4_headline start '\$(date +%H:%M:%S)
  '$PY' -u -c \"import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'experiments'); from cascadebench.experiments import headline_ci; headline_ci(seeds=20, off_hours=0.15, tag='_oh')\" && echo '=== r4_headline_oh OK '\$(date +%H:%M:%S) || echo '=== r4_headline_oh FAIL '\$(date +%H:%M:%S)
  '$PY' -u -c \"import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'experiments'); from cascadebench.experiments import headline_ci; headline_ci(seeds=20, timing='emp', tag='_emp')\" && echo '=== r4_headline_emp OK '\$(date +%H:%M:%S) || echo '=== r4_headline_emp FAIL '\$(date +%H:%M:%S)
" > logs/r4_headline.log 2>&1 &

# L4 (A2): Enron blast (8 seedow, timing det, KS+sonda in-run)
nohup bash -c "
  echo '=== r4_enron_blast start '\$(date +%H:%M:%S)
  '$PY' -u experiments/exp_cascade_enron.py --rezym blast && echo '=== r4_enron_blast OK '\$(date +%H:%M:%S) || echo '=== r4_enron_blast FAIL '\$(date +%H:%M:%S)
" > logs/r4_enron_blast.log 2>&1 &

# L5 (A2): Enron stealthy det+emp (8 seedow)
nohup bash -c "
  echo '=== r4_enron_stealth start '\$(date +%H:%M:%S)
  '$PY' -u experiments/exp_cascade_enron.py --rezym stealthy && echo '=== r4_enron_stealth OK '\$(date +%H:%M:%S) || echo '=== r4_enron_stealth FAIL '\$(date +%H:%M:%S)
" > logs/r4_enron_stealth.log 2>&1 &

# L6 (A2): SNAP email-Eu-core (10 seedow, blast det + stealthy det/emp)
nohup bash -c "
  echo '=== r4_snap_eu start '\$(date +%H:%M:%S)
  '$PY' -u experiments/exp_cascade_realgraphs.py --only email-Eu-core && echo '=== r4_snap_eu OK '\$(date +%H:%M:%S) || echo '=== r4_snap_eu FAIL '\$(date +%H:%M:%S)
" > logs/r4_snap_eu.log 2>&1 &

# L7 (A2): SNAP CollegeMsg (10 seedow)
nohup bash -c "
  echo '=== r4_snap_college start '\$(date +%H:%M:%S)
  '$PY' -u experiments/exp_cascade_realgraphs.py --only CollegeMsg && echo '=== r4_snap_college OK '\$(date +%H:%M:%S) || echo '=== r4_snap_college FAIL '\$(date +%H:%M:%S)
" > logs/r4_snap_college.log 2>&1 &

# L8 (B1/B2/B5 + A2-reszka + A3): multiplex, audyt, probe_residual, inversion, susceptibility
nohup bash -c "
  echo '=== r4_b start '\$(date +%H:%M:%S)
  '$PY' -u experiments/exp_multiplex.py && echo '=== r4_multiplex OK '\$(date +%H:%M:%S) || echo '=== r4_multiplex FAIL '\$(date +%H:%M:%S)
  '$PY' -u experiments/exp_enron_leak_audit.py && echo '=== r4_audit OK '\$(date +%H:%M:%S) || echo '=== r4_audit FAIL '\$(date +%H:%M:%S)
  '$PY' -u experiments/exp_susceptibility.py && echo '=== r4_susc OK '\$(date +%H:%M:%S) || echo '=== r4_susc FAIL '\$(date +%H:%M:%S)
  '$PY' -u experiments/exp_probe_residual.py && echo '=== r4_residual OK '\$(date +%H:%M:%S) || echo '=== r4_residual FAIL '\$(date +%H:%M:%S)
  '$PY' -u experiments/exp_probe_residual.py inversion && echo '=== r4_inversion OK '\$(date +%H:%M:%S) || echo '=== r4_inversion FAIL '\$(date +%H:%M:%S)
" > logs/r4_b.log 2>&1 &

echo "8 lane'ow R4 wystartowalo; logi: logs/r4_*.log"
