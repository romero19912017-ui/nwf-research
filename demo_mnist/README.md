# Демонстрация NWF на MNIST (упрощённая реализация)

Proof-of-concept: эмбеддинги CNN + фиксированная sigma. Результаты 160x и 2.7x.

```powershell
conda activate nwf_torch
cd demo_mnist

python experiments/run_incremental_aggressive.py --k 10 --use-tfds --cpu --finetune-epochs 50
python experiments/run_noise_aggressive.py --k 10 --noise-levels "0,0.3,0.5,0.7,1.0" --use-tfds --cpu
python experiments/plot_results.py --aggressive
```
