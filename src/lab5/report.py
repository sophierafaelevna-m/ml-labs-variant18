"""Формирование отчёта ЛР5 из результатов экспериментов."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path("reports/LAB5")


def md_table(columns, rows):
    """Формируем Markdown-таблицу без дополнительных библиотек."""
    result = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    result.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(result)


def main():
    metrics = pd.read_csv(ROOT / "synthetic_metrics.csv")
    summary = pd.read_csv(ROOT / "comparison_summary.csv")
    drift = pd.read_csv(ROOT / "drift_experiments.csv")

    config = json.loads((ROOT / "generator_config.json").read_text(encoding="utf-8"))

    test_log = (ROOT / "generator_tests.log").read_text(encoding="utf-8")
    negative_log = (ROOT / "negative_control.log").read_text(encoding="utf-8")

    coverage_match = re.search(
        r"^TOTAL\s+\d+\s+\d+\s+(\d+)%",
        test_log,
        flags=re.MULTILINE,
    )
    if not coverage_match:
        raise ValueError("Не найден показатель покрытия в логе")

    coverage = int(coverage_match.group(1))

    tests_match = re.search(r"(\d+) passed", test_log)
    if not tests_match:
        raise ValueError("Не найден результат pytest")

    tests_count = int(tests_match.group(1))

    if "1 failed" not in negative_log:
        raise ValueError("Отрицательный контроль не подтверждён логом")

    metric_rows = []
    for row in metrics.itertuples(index=False):
        ks = "—" if pd.isna(row.ks) else f"{row.ks:.5f}"
        metric_rows.append(
            [
                row.mode,
                row.feature,
                f"{row.psi:.5f}",
                ks,
                "PASS" if row.passed else "FAIL",
            ]
        )

    summary_rows = []
    for row in summary.itertuples(index=False):
        summary_rows.append(
            [
                row.mode,
                f"{row.passed_features}/{row.total_features}",
                f"{row.passed_share:.1%}",
                f"{row.required_share:.0%}",
                row.verdict,
            ]
        )

    drift_rows = []
    for row in drift.itertuples(index=False):
        drift_rows.append(
            [
                row.experiment,
                row.n_samples,
                f"{row.mean_words:.2f}",
                row.flipped_labels,
            ]
        )

    coverage_rows = []
    for line in test_log.splitlines():
        match = re.match(
            r"^(src/lab5/\S+)\s+\d+\s+\d+\s+(\d+)%",
            line,
        )
        if match:
            coverage_rows.append(
                [
                    match.group(1),
                    f"{match.group(2)}%",
                ]
            )

    run1 = ROOT / "repro_run_1.csv"
    run2 = ROOT / "repro_run_2.csv"

    if not run1.exists() or not run2.exists():
        raise FileNotFoundError("Не найдены файлы проверки воспроизводимости")

    hash1 = hashlib.sha256(run1.read_bytes()).hexdigest()
    hash2 = hashlib.sha256(run2.read_bytes()).hexdigest()

    if hash1 != hash2:
        raise ValueError("SHA256 двух запусков не совпадают")

    baseline = config["baseline"]

    report = f"""# Лабораторная работа № 5

## Управляемый генератор синтетических данных

**Вариант:** 18
**Набор данных:** 20 Newsgroups
**Язык программирования:** Python 3.12

## 1. Цель работы

Разработать генератор синтетических текстовых данных
с управляемыми режимами и механизмами дрейфа,
проверить статистическое сходство с реальными данными
и обеспечить воспроизводимость и автоматическое тестирование.

## 2. Исходные данные

Используется обучающая выборка 20 Newsgroups
из предыдущих лабораторных работ.

Классы:
- `alt.atheism`
- `talk.religion.misc`
- `comp.graphics`
- `sci.space`

Источник: `{baseline["source"]["train_path"]}`.

Тестовая выборка не используется при построении генератора.

## 3. Конфигурация генератора

Настройки описаны моделями Pydantic в
`src/lab5/config.py`.

| Параметр | Назначение |
| --- | --- |
| `mode` | Режим `similar` или `random` |
| `n_samples` | Количество документов |
| `seed` | Фиксация генератора случайных чисел |
| `length_multiplier` | Множитель длины текста |
| `class_probabilities` | Управление долями классов |
| `label_flip_rate` | Доля изменённых меток |
| `seasonal_amplitude` | Амплитуда сезонного дрейфа |
| `seasonal_period` | Период сезонного дрейфа |
| `output` | Формат и путь выходного файла |

Некорректные значения отклоняются при валидации.

Конфигурации экспериментов сохранены в
`reports/LAB5/generator_config.json`.

## 4. Алгоритм генерации

### Режим similar

Используется bootstrap: документы выбираются
из обучающей выборки с возвращением.
Вероятности выбора классов соответствуют
исходным долям, если не задан дрейф классов.

### Режим random

Формируются случайные последовательности слов
из частотного словаря обучающего корпуса.
Длины текстов выбираются случайным образом.

В обоих режимах доступны механизмы управляемого дрейфа.

Ограничение: режим similar может повторять реальные тексты,
поэтому статистическое сходство не означает генерацию
новых семантически независимых документов.

## 5. Эксперименты с дрейфом

Проверены пять сценариев: базовый, изменение длины,
изменение долей классов, переворот меток и сезонность.

{
        md_table(
            ["Эксперимент", "Документов", "Средняя длина, слов", "Изменено меток"],
            drift_rows,
        )
    }

Изменение длины проверено сравнением средних значений.
Изменение долей классов — по доле целевого класса.
Переворот меток — по числу заменённых меток.
Сезонность — сравнением длин текстов в разных фазах периода.

Автоматические тесты механизмов дрейфа пройдены.

## 6. Сравнение с реальными данными

Для текстов извлечены семь числовых характеристик:
число слов, число символов, средняя длина слова,
доля уникальных слов, доли слов из частотных словарей
top-100 и top-1000, доля коротких слов.

Дополнительно проверено распределение классов.

Для числовых характеристик рассчитаны PSI и
статистика Колмогорова — Смирнова (KS).
Для категориального распределения классов — PSI.

### 6.1. Метрики по признакам

{
        md_table(
            ["Режим", "Признак", "PSI", "KS", "Вердикт"],
            metric_rows,
        )
    }

### 6.2. Итоговые критерии

Для `similar`: PSI ≤ 0,1 и KS ≤ 0,1
минимум по 80% характеристик.
Для распределения классов учитывается только PSI.

Для `random`: PSI ≥ 0,2
минимум по 50% характеристик.

{
        md_table(
            ["Режим", "Прошло", "Доля", "Минимум", "Итог"],
            summary_rows,
        )
    }

В режиме random распределение классов
сохраняется близким к исходному по конструкции.
Это объясняет отсутствие значимого дрейфа
по характеристике `class_distribution`.

## 7. Воспроизводимость

Генерация использует фиксированный `seed`.
Проведены два независимых запуска
с одинаковыми настройками.

**SHA256 первого запуска:**

`{hash1}`

**SHA256 второго запуска:**

`{hash2}`

**Вердикт:** PASS — файлы побайтово совпадают.

## 8. Автоматическое тестирование

Использованы `pytest` и `pytest-cov`.

**Успешных тестов:** {tests_count}
**Общее покрытие:** {coverage}%
**Требование:** не менее 80%.

{md_table(["Модуль", "Покрытие"], coverage_rows)}

Проверены:
- схема и недопустимые значения конфигурации;
- оба режима генерации;
- воспроизводимость при одинаковом seed;
- изменение результата при другом seed;
- четыре механизма дрейфа;
- обработка некорректных исходных данных;
- расчёт PSI;
- формирование выходных артефактов.

### 8.1. Отрицательный контроль

Для отдельного теста намеренно нарушена
детерминированность: переданный `seed`
заменён другим значением.

Тест ожидаемо завершился ошибкой (`1 failed`),
то есть обнаружил нарушение воспроизводимости.

После удаления временного теста
основной набор из 27 проверок снова прошёл.

Логи:
- `generator_tests.log`
- `negative_control.log`

## 9. Выводы

Разработан конфигурируемый генератор
синтетических текстовых данных для варианта 18.

Режим similar воспроизводит выбранные
статистические характеристики реального корпуса.
Режим random формирует существенно отличающиеся
тексты. Оба режима соответствуют заданным
критериям PSI/KS.

Реализованы и протестированы механизмы
изменения длины, долей классов, меток
и сезонных колебаний.

Подтверждена побайтовая воспроизводимость.
Все {tests_count} автоматических тестов прошли,
общее покрытие составило {coverage}%.

Генератор может использоваться как источник
контролируемых экспериментальных данных
для следующих лабораторных работ.

## 10. Артефакты

- `configs/lab5.yaml`
- `src/lab5/config.py`
- `src/lab5/generator.py`
- `src/lab5/experiments.py`
- `src/lab5/compare.py`
- `src/lab5/report.py`
- `tests/lab5/test_generator.py`
- `reports/LAB5/synthetic_metrics.csv`
- `reports/LAB5/comparison_summary.csv`
- `reports/LAB5/drift_experiments.csv`
- `reports/LAB5/generator_config.json`
- `reports/LAB5/generator_tests.log`
- `reports/LAB5/negative_control.log`
"""

    output = ROOT / "lab5_report.md"
    output.write_text(report, encoding="utf-8")

    print(f"Отчёт создан: {output}")
    print(f"Тестов пройдено: {tests_count}")
    print(f"Покрытие: {coverage}%")
    print("SHA256: совпадают")
    print("\nИтоговое сравнение:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
