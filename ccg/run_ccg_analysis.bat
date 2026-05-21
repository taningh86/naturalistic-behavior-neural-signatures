@echo off
cd /d "H:\NPX ANALYSIS REPO"
call C:\Users\Gregg\anaconda3\condabin\conda.bat activate si_env
python coor1_ccg_and_significant_pairs.py
python coor1_spike_sync_kernel_signed_pairs.py
