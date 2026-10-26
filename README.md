# Рекомендательные системы — ВМК МГУ, 2026

**Козлов Кирилл Андреевич.** Домашние работы по курсу.

## Домашнее задание 1

Пять моделей с нуля: ItemKNN, iALS, EASE, FM и timeSVD++.

| Эксперимент | Материалы |
|---|---|
| **Amazon Reviews 2018 Video Games** — основной эксперимент, исправление FM v2 | [README](homework_1/README.md) · [Ноутбук](homework_1/experiments.ipynb) |
| **Amazon Reviews 2023 Video Games** — внешний test за 2019 год, настройки v2 | [README](homework_1/external_2019/README.md) · [Ноутбук](homework_1/external_2019/experiments.ipynb) |

[Условие задания](<homework_1/ressys_Домашнее задание 1.pdf>) ·
[Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) ·
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html).

**v1 и v2 — версии эксперимента на датасете 2018 года, не названия выпусков данных.**
Test v2 использовался ранее; её результаты — корректирующая повторная оценка.
Внешняя оценка выполнена отдельно с зафиксированными гиперпараметрами v2.
Она охватывает 4 477 пользователей и 2,36% положительных событий категории за 2019 год.

## Структура

```text
homework_1/
├── README.md             # Описание и запуск v2
├── experiments.ipynb     # EDA, подбор и результаты v2
├── models/               # Реализации пяти моделей
├── data_utils.py         # Подготовка fit-данных
├── evaluation.py         # Общий оценщик
├── configs/              # Сетки и выбранные параметры
├── scripts/              # Загрузка, обучение и воспроизведение
├── tests/                # Проверки
├── reports/              # Отчёты v2
├── requirements.txt      # Зависимости
├── archive/              # Предыдущие эксперименты
└── external_2019/        # Внешняя оценка и отдельный ноутбук
```

Инструкции запуска — в README каждого эксперимента.
Данные, веса и персональные выдачи исключены из Git.
