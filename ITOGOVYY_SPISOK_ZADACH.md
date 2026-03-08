# Итоговый список задач NWF Research

## 1. Теоретическая доработка

- [ ] **Аксиома А5 (Динамического обновления)** — включить в статью, связь со свободной энергией
- [ ] **Теорема о сходимости Kalman-NWF** — при линейной динамике и гауссовых шумах
- [ ] **Нелинейные обобщения** — EKF, Sigma-point фильтры

## 2. Реализация (выполнено)

- [x] **KalmanEncoder** — encode_static, update, batch
- [x] **Симметричная метрика Махаланобиса** — (Sigma_q + Sigma_i)^{-1}
- [x] **Метрики уверенности** — potential_at_query, min_mahalanobis, trace_sigma, agreement_ratio

## 3. Эксперименты

### 3.1 Сжатие (01)
- [x] Сравнение NWF vs FAISS vs HDC
- [x] Симметричная метрика (--metric symmetric)
- [x] Kalman-кодирование (--use_kalman)
- [ ] Квантование z, Sigma
- [ ] Разные latent_dim: 16, 32, 64

### 3.2 Шум (02)
- [x] NWF_Mahalanobis, NWF_Symmetric, NWF_Euclidean, FAISS, HDC
- [ ] Kalman при зашумленных запросах

### 3.3 Инкрементальность (03)
- [x] NWF vs EWC vs Fine-tuning
- [x] Матрица забывания (forgetting matrix)
- [ ] iCaRL (буфер примеров + distillation)

### 3.4 OOD (04)
- [x] Mahalanobis distance
- [x] Potential Phi(z) как метрика OOD

### 3.5 Калибровка (05)
- [x] ECE для NWF и MLP
- [ ] Несколько метрик уверенности, reliability diagram
- [ ] Platt scaling, temperature scaling

### 3.6 Онлайн-обновление (06)
- [x] Дрейф концепций (вращающаяся цифра)

### 3.7 Семантическая интерполяция (07)
- [x] z_alpha = (1-a)*z1 + a*z2, декодирование

### 3.8 Ландшафт (08)
- [x] t-SNE/PCA + контуры потенциала Phi(r)

### 3.9 Скорость кодирования (09)
- [x] Kalman vs VAE vs HDC

## 4. Подготовка публикации

- [ ] Структурировать статью (введение, теория, методология, эксперименты, обсуждение)
- [ ] Итоговые графики
- [ ] README с инструкциями по воспроизведению

## 5. Запуск экспериментов

```powershell
cd c:\nwf\nwf-research
$env:PYTHONPATH = "c:\nwf\nwf-research"
python train_vae.py --epochs 15
python experiments/00_convergence/run.py --plot
python experiments/01_compression/run.py --metric symmetric
python experiments/02_noise/run.py --plot
python experiments/03_incremental/run.py --plot
python experiments/04_ood/run.py
python experiments/05_calibration/run.py
python experiments/06_online_update/run.py
python experiments/07_interpolation/run.py
python experiments/08_landscape/run.py --method pca
python experiments/09_encoding_speed/run.py
```
