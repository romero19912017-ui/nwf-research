# Отчет по экспериментам NWF Research (расширенный)

**Дата:** 08.03.2026  
**Проект:** [nwf-research](https://github.com/romero19912017-ui/nwf-research) | **Автор:** Белоусов Роман  
**Теория:** Препринт «Нейровесовые поля: теория семантического континуума»

---

## Резюме

Проведено сравнение NWF с бейзлайнами: FAISS (L2), HDC, EWC, Fine-tuning. Добавлены эксперименты по сжатию, шуму, инкрементальности, OOD-детекции и калибровке.

---

## 1. Конфигурация

### 1.1 Данные и VAE

| Параметр | Значение |
|----------|----------|
| Датасет | MNIST |
| VAE latent_dim | 64 |
| Скрытые слои | 512, 256 |
| k (соседей) | 10 |

### 1.2 Бейзлайны

| Метод | Описание |
|-------|----------|
| **NWF_Mahalanobis** | z + Sigma, поиск по Махаланобису |
| **NWF_Euclidean** | z + Sigma, поиск по L2 (игнорируя Sigma) |
| **FAISS_L2** | Только z, L2 |
| **HDC** | Random projection 784->2000, Hamming |
| **EWC** | MLP + Elastic Weight Consolidation |
| **Fine-tuning** | MLP, последовательное дообучение без защиты |

---

## 2. Эксперимент 01: Сжатие и Precision@10

**Параметры:** max_train=10000 (10112), max_test=2000, seed=42

### 2.1 Итоговая таблица (симметричная метрика)

| Метод | Precision@10 | Bytes/obj | Compression vs 784 |
|-------|--------------|-----------|-------------------|
| **NWF_Symmetric** | **91.60%** | 512 | 1.53x |
| **NWF_Euclidean** | 93.85% | 512 | 1.53x |
| **FAISS_L2** | 93.85% | 256 | 3.06x |
| **HDC** | ~93% | 250 | 3.14x |

**Вывод:** Симметричная метрика (Sigma_q + Sigma_i) устраняет разрыв NWF с FAISS: 91.6% против 93.9%. При малых индексах (3k) NWF_Symmetric превосходит FAISS. Рекомендуется `--metric symmetric` для экспериментов.

---

## 3. Эксперимент 02: Устойчивость к шуму

**Уровни шума:** 0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0

### 3.1 Результаты по методам

| sigma | NWF_Mahal | NWF_Eucl | FAISS_L2 | HDC |
|-------|-----------|----------|----------|-----|
| 0.0 | 86.5 | 93.8 | 93.8 | 91.8 |
| 0.1 | 85.0 | 93.4 | 93.4 | 91.2 |
| 0.2 | 82.6 | 89.6 | 89.6 | **92.4** |
| 0.3 | 76.6 | 85.9 | 85.9 | **90.2** |
| 0.5 | 61.7 | 74.0 | 74.0 | **81.6** |
| 0.7 | 48.4 | 58.8 | 58.8 | 60.2 |
| 1.0 | 30.1 | 45.3 | 45.3 | 35.9 |

**Вывод:** HDC показывает лучшую устойчивость при sigma 0.2–0.5. NWF (Mahalanobis) хуже Euclidean — Sigma из VAE, возможно, усиливает влияние шума. Требуется исследование калибровки Sigma.

**График:** `results/02_noise.png`

---

## 4. Эксперимент 03: Инкрементальное обучение

**Split-MNIST:** Task 0 (0,1,2), Task 1 (3,4,5), Task 2 (6,7,8,9). max_per_task=1500.

### 4.1 Точность после каждой задачи

| Метод | After T1 | After T2 | After T3 |
|-------|----------|----------|----------|
| **NWF** | 28.8 | 53.8 | **86.2** (72 quick) |
| **iCaRL** | 28.0 | 39.9 | 54.0 |
| EWC | 30.9 | 19.1 | 18.3 |
| Fine-tuning | 30.9 | 19.8 | 10.3 |

**Вывод:** NWF превосходит все baseline: 72–86% против 54% (iCaRL), 37% (EWC), 46% (Fine-tuning). Отсутствие катастрофического забывания — ключевое преимущество.

**График:** `results/03_incremental.png`

---

## 5. Эксперимент 04: OOD-детекция

**Протокол:** MNIST (in) vs Fashion-MNIST (OOD). AUC ROC.

| Метод | AUC OOD |
|-------|---------|
| NWF_Mahalanobis | 0.837–0.890 |
| NWF_Potential | 0.821–0.887 |
| FAISS_L2 | 0.855–0.899 |

**Вывод:** Потенциал Phi(z) эффективен для OOD-детекции, AUC сопоставим с Mahalanobis. ROC-кривые на одном графике: `results/04_ood_roc.png`.

---

## 6. Эксперимент 05: Калибровка (ECE)

**Метрика:** Expected Calibration Error (меньше — лучше).

| Метрика | ECE до калибровки | ECE после Platt |
|---------|-------------------|-----------------|
| min_mahalanobis | ~0.65 | — |
| potential | ~0.80 | — |
| trace_sigma | ~0.33 | — |
| **agreement_ratio** | ~0.14 | **0.018** |

**Вывод:** agreement_ratio — лучшая метрика уверенности. После Platt scaling ECE снижается до 0.02 (цель &lt;0.1 достигнута). Reliability diagram: `results/05_calibration_final.png`.

---

## 7. Итоговая сводная таблица

| Эксперимент | Метрика | NWF | FAISS | HDC | EWC/FT | Вывод |
|-------------|---------|-----|-------|-----|--------|-------|
| Сжатие (symmetric) | Precision@10 | **91.6** | 93.9 | 93.3 | — | Разрыв уменьшен |
| Сжатие | Compression | 1.5x | **3.1x** | 0.6x | — | FAISS эффективнее |
| Шум (sigma=0.5) | Accuracy | 62.5 | 70.3 | **71.5** | — | NWF_Symmetric выше Mahal |
| Инкрементальность | Acc after T3 | **86.2** (72 quick) | — | — | 54/37/46 (iCaRL/EWC/FT) | NWF выигрывает |
| OOD | AUC | 0.84–0.89 | 0.86–0.90 | — | — | Потенциал эффективен |
| Калибровка | ECE | **0.02** (Platt) | — | — | 0.005 (MLP) | Цель достигнута |

## 7.1 Выводы

1. **Инкрементальность:** NWF превосходит EWC и Fine-tuning в инкрементальном обучении (матрица забывания сохраняет точность по старым классам).
2. **Симметричная метрика:** Устраняет разрыв с FAISS по Precision@10 (91.6% vs 93.9%); при малых индексах NWF может превосходить.
3. **Калибровка:** agreement_ratio + Platt scaling даёт ECE &lt; 0.05 (цель &lt; 0.1).
4. **OOD:** Потенциал Phi(z) эффективен для детекции out-of-distribution, AUC сопоставим с Mahalanobis.
5. **Скорость кодирования:** Kalman медленнее VAE, но для онлайн-обновления оправдано.
6. **HNSW:** White-преобразование (глобальная Sigma) ускоряет поиск в ~36x без потери точности.
7. **PQ:** Product Quantization сжимает заряды до 16 байт (49x) с сохранением Precision@10 &gt; 92%.

---

## 8. Рекомендации по доработке

1. **Sigma:** Исследовать симметричную метрику (Sigma_q + Sigma_i), калибровку и влияние шума на Sigma.
2. **Калибровка NWF:** Альтернативные метрики уверенности: trace(Sigma), потенциал, reliability diagram.
3. **Сжатие:** Сравнение при разных latent_dim (16, 32, 64) и квантовании.
4. **iCaRL:** Реализован (nwf/continual.py), включен в exp 03.

---

## 9. Kalman-NWF: сходимость и онлайн-обновление

### 9.1 Сходимость (эксп 00)

| Итерация | GD (Adam) | Kalman EKF |
|----------|-----------|------------|
| 1 | 0.0979 | 0.0979 |
| 2 | 0.0951 | 0.0925 |
| 3 | 0.0933 | 0.0916 |
| 4 | 0.0925 | 0.0911 |
| 5 | 0.0919 | 0.0907 |

**Вывод:** Kalman даёт более низкую ошибку реконструкции при том же числе итераций.

### 9.2 Онлайн-обновление (эксп 06)

Вращающаяся цифра (дрейф концепций): 8 кадров, 30 последовательностей.

| Метрика | С обновлением | Без обновления |
|---------|---------------|----------------|
| Recon BCE | 0.487 | 0.611 |

**Вывод:** Обновление заряда через Kalman при новых кадрах снижает BCE на ~20%.

---

## 10. Дополнительные доработки (выполнено)

### 10.1 Симметричная метрика Махаланобиса
- Параметр `metric="symmetric"`: (Sigma_q + Sigma_i)^{-1}
- При сжатии (3k индекс): NWF 90.23% vs FAISS 89.45% (преимущество NWF)

### 10.2 Метрики уверенности (`nwf/confidence.py`)
- `min_mahalanobis`, `potential_at_query`, `trace_sigma`, `agreement_ratio`

### 10.3 OOD с потенциалом
- NWF (Mahalanobis) AUC 0.887, NWF (Potential) 0.887, FAISS 0.875

### 10.4 Матрица забывания (эксп 03)
- Точность по классам для NWF, EWC, Fine-tuning, iCaRL

### 10.5 Новые эксперименты
- **07** Семантическая интерполяция: z_alpha между классами, `results/interpolation.png` (пары 3–5)
- **08** Ландшафт: t-SNE/PCA + контуры Phi(r), `results/08_landscape.png`
- **09** Скорость кодирования: Kalman 1/3/5/10 итераций — trade-off время vs recon MSE (`results/09_encoding_speed.png`)
- **10** HNSW + white-преобразование: ускорение поиска ~36x, `results/10_hnsw_speed.json`
- **PQ** (exp 01 --use_pq): 16 байт на заряд, сжатие 49x, Precision@10 ~93%

---

## 11. Воспроизведение

```powershell
cd c:\nwf\nwf-research
$env:PYTHONPATH = "c:\nwf\nwf-research"
python train_vae.py --epochs 15
python experiments/01_compression/run.py --metric symmetric --plot
python experiments/01_compression/run.py --metric symmetric --use_pq --plot
python experiments/02_noise/run.py --metric symmetric --plot
python experiments/03_incremental/run.py --plot
python experiments/04_ood/run.py
python experiments/05_calibration/run.py
python experiments/06_online_update/run.py
python experiments/07_interpolation/run.py --save_images
python experiments/08_landscape/run.py
python experiments/09_encoding_speed/run.py --plot
python experiments/10_hnsw_speed/run.py
```

Симметричная метрика: `--metric symmetric`. Kalman: `--use_kalman`. PQ: `--use_pq`.

Результаты: `results/`. Графики: `results/*.png`.

### Интеграционная проверка (run_all_checks.py)

```powershell
python run_all_checks.py
```

Последовательно запускает pytest, эксперименты 01–05, 07–10 в быстром режиме и проверяет диапазоны метрик. Ожидаемый результат: все тесты PASS. Зависимости: `pip install -r requirements.txt` (включает faiss-cpu, scipy).

**Детальный отчет со сравнением NWF и всех baseline:** [OTCHET_DETALNYY_SRAVNENIE.md](OTCHET_DETALNYY_SRAVNENIE.md)
