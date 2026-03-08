# Run all NWF experiments (from nwf-research root)
$env:PYTHONPATH = (Get-Location).Path
Write-Host "Training VAE..."
python train_vae.py --epochs 15
Write-Host "Experiment 00: Convergence Kalman vs GD"
python experiments/00_convergence/run.py --n_samples 30 --n_iter 5 --plot
Write-Host "Experiment 01: Compression / Precision@10"
python experiments/01_compression/run.py --plot
Write-Host "Experiment 02: Noise robustness"
python experiments/02_noise/run.py --plot
Write-Host "Experiment 03: Incremental learning"
python experiments/03_incremental/run.py --plot
Write-Host "Experiment 04: OOD detection"
python experiments/04_ood/run.py
Write-Host "Experiment 05: Calibration (ECE)"
python experiments/05_calibration/run.py
Write-Host "Experiment 06: Online update (drift)"
python experiments/06_online_update/run.py --n_sequences 30 --n_frames 8
Write-Host "Experiment 07: Semantic interpolation"
python experiments/07_interpolation/run.py
Write-Host "Experiment 08: Landscape (PCA)"
python experiments/08_landscape/run.py --method pca --max_samples 1500
Write-Host "Experiment 09: Encoding speed"
python experiments/09_encoding_speed/run.py --n_samples 300
Write-Host "Done. Results in results/"
