# Домашнее задание 1

**Козлов Кирилл Андреевич.** [Ноутбук](experiments.ipynb) · [Условие](<ressys_Домашнее задание 1.pdf>).

Основной эксперимент — **Amazon Reviews 2018 Video Games 5-core**.
`v2` — исправленная версия признаков FM, а не выпуск датасета.
[Отдельная внешняя оценка](external_2019/README.md) использует Amazon Reviews 2023 и настройки v2.

## Данные

[Источник](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) ·
[Отзывы](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz) ·
[Карточки для EDA](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz).

Рекомендуются игры, консоли и аксессуары. Событие — отзыв с оценкой 1–5 и временем публикации.
Из 497 577 отзывов оставлены самые ранние user–item пары, товары с непустым названием и восстановлен 5-core.
Итог: **452 989 событий, 50 546 пользователей, 16 859 товаров**.
В [проверенных семинарах 26f](https://github.com/girafe-ai/recsys/tree/c514701f9275fb6020c6382f0029ea92ab509a72/seminars) этот датасет не использовался.

## Протокол

| Часть | Период UTC | Событий |
|---|---|---:|
| Train | до 2016-01-01 | 330 951 |
| Validation | 2016 год | 72 730 |
| Test | с 2017-01-01 | 49 308 |
| Финальный fit | train + validation | 403 681 |

- Релевантность: rating ≥ 4. Основная метрика — macro NDCG@10, дополнительная — macro Recall@10.
- Запрос на начало окна, история фиксирована. Кандидаты — весь fit-каталог минус все известные fit-товары, включая оценки 1–3. Без sampling; ties по ASIN.
- Общая когорта: известные пользователи с ≥1 доступным релевантным товаром и ≥10 кандидатами. При недоступной персонализации применяется общий popularity fallback.
- Подбор только на validation: CF — 12, iALS — 12, EASE — 7, FM — 8, timeSVD++ — 8 конфигураций. Затем все пять моделей переобучены на train + validation.
- Test-когорта: **9 521 пользователь, 23 454 релевантных события**; охват положительных событий — 57,95%.

FM получает ID и признаки из fit-событий: год/месяц первого отзыва, log-count, возраст истории и давность.
Все исходные timestamps предшествуют запросу; карточки и тексты отзывов в признаки не входят.
Репрезентация фиксируется на дату запроса для обучения и рекомендации.

## Результаты

Все модели реализованы с нуля в `models/`. Формулы, выбранные параметры, графики и ablation — в [ноутбуке](experiments.ipynb).

| Модель | Test NDCG@10 | Test Recall@10 |
|---|---:|---:|
| Popularity | 0.009712 | 0.012274 |
| ItemKNN | 0.021241 | 0.032328 |
| iALS | 0.022217 | 0.034842 |
| EASE | 0.022048 | 0.033390 |
| FM | 0.006611 | 0.010603 |
| timeSVD++ | 0.000579 | 0.001113 |

iALS — точечный лидер. Для iALS − EASE по NDCG 95% paired CI [−0,001465; 0,001961] включает ноль.
Метапризнаки FM повысили validation Recall, но не NDCG относительно ID-only при тех же настройках.
TimeSVD++ не улучшила ranking относительно статической ablation.

## Запуск и воспроизведение

[Google Colab](https://colab.research.google.com/github/noobmaster-ru/recsys_cmc_msu_2026_course/blob/769803fd8897a09bbb416ef0938f90c2fe921876/homework_1/experiments.ipynb):
«Среда выполнения → Выполнить всё». CPU; репозиторий и данные загружаются автоматически.
Пересчитывается EDA, модельные результаты читаются из отчётов. Полное обучение — отдельной командой ниже.

Локально (Python 3.13), из корня репозитория:

```sh
cd homework_1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Пересборка модельных отчётов без данных и обучения:

```sh
python scripts/execute_notebook.py --section reports
```

EDA и все ячейки с сохранёнными результатами моделей:

```sh
python scripts/prepare_data.py --download
python scripts/execute_notebook.py --section all
```

Полное обучение и оценка в новой папке вне репозитория:

```sh
python scripts/reproduce.py --output ../../recsys-reproduction
```

Ориентир: 16 GB RAM и 10 GB диска; одна финальная матрица EASE занимает 2,08 GiB.
Seed=42, negative sampling=137, shuffle=43. Данные и веса исключены из Git.

## Ограничения

Test уже использовался в v1; v2 — корректирующая повторная оценка того же holdout.
Отбор 5-core и наличие названия учитывают полный период. Оценка ограничена известными пользователями и товарами;
отзывы не равны показам или покупкам. Один seed; bootstrap не учитывает вариацию обучения и множественные сравнения.

[Внешняя оценка за 2019 год](external_2019/README.md). Старые эксперименты доступны в истории Git.
