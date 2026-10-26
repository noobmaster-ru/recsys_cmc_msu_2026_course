# Рекомендательные системы — ВМК МГУ, 2026

**Козлов Кирилл Андреевич.**

## Домашнее задание 1

ItemKNN, iALS, EASE, FM и timeSVD++ с нуля.

| Эксперимент | Материалы |
|---|---|
| Amazon Reviews 2018 Video Games — основной, v2 | [README](homework_1/README.md) · [Ноутбук](homework_1/experiments.ipynb) · [Google Colab](https://colab.research.google.com/github/noobmaster-ru/recsys_cmc_msu_2026_course/blob/main/homework_1/experiments.ipynb) |
| Amazon Reviews 2023 Video Games — внешний test за 2019 год | [README](homework_1/external_2019/README.md) · [Ноутбук](homework_1/external_2019/experiments.ipynb) |

v1/v2 — версии эксперимента на данных 2018 года. Внешняя оценка использует настройки v2.
Test v2 оценивался ранее. Внешний test охватывает 4 477 пользователей и 2,36% положительных событий категории.

## Запуск в Colab

Открыть ссылку **Google Colab** выше → **Среда выполнения → Выполнить всё**. Достаточно CPU.
Первая ячейка загрузит репозиторий, зависимости и данные.
Выполняются EDA и вывод сохранённых результатов моделей. Полное обучение — по [инструкции](homework_1/README.md#запуск-и-воспроизведение).

## Структура

```text
homework_1/
├── README.md, experiments.ipynb   # Основная работа
├── models/, evaluation.py        # Модели и общий оценщик
├── configs/, scripts/, tests/    # Параметры, запуск и проверки
├── reports/, requirements.txt    # Отчёты и зависимости
├── archive/                      # Предыдущие эксперименты
└── external_2019/                # Внешняя оценка
```

[Условие](<homework_1/ressys_Домашнее задание 1.pdf>) ·
[Источник 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) ·
[Источник 2023](https://amazon-reviews-2023.github.io/main.html).
Данные, веса и персональные выдачи исключены из Git.
