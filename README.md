# NWF Research

Научно-корректная реализация Нейровесовых Полей (NWF) по теории препринта.

**Теория:** [Препринт](https://doi.org/10.24108/preprints-3113697) «Нейровесовые поля: теория семантического континуума»

**План исследований:** [PLAN_ISSLEDOVANIY.md](PLAN_ISSLEDOVANIY.md)

## Гипотезы

1. **H1:** NWF дают более высокий коэффициент семантического сжатия vs статические эмбеддинги
2. **H2:** Учёт ковариации Σ повышает устойчивость поиска к шуму
3. **H3:** Инкрементальное обновление без катастрофического забывания

## Структура

```
nwf/           - ядро: Charge, VAE, Махаланобис, поиск
experiments/   - эксперименты по гипотезам H1, H2, H3
configs/       - конфигурации
```

## Установка

```powershell
conda create -n nwf_research python=3.11 -y
conda activate nwf_research
pip install torch torchvision numpy scikit-learn tensorflow-datasets
```

## Статус

- [ ] Фаза 1: VAE, core, storage
- [ ] Фаза 2: Эксперименты H1, H2, H3
- [ ] Фаза 3: Расширение (CIFAR, IMDB)

**Лицензия:** CC BY-NC 4.0
