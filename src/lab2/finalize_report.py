import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "LAB2"

METRICS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "rare_class_recall": "Rare-class Recall",
}


def read_json(filename):
    return json.loads((OUT / filename).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def git_head():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main():
    full = read_json("full_results.json")
    chunk = read_json("chunk_results.json")
    leakage = read_json("leakage_test.json")
    reproducibility = read_json("training_reproducibility.json")
    scaling = read_json("chunk_scaling.json")

    config = yaml.safe_load((ROOT / "configs/lab2.yaml").read_text(encoding="utf-8"))
    negative_log = (OUT / "config_negative_test.log").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert scaling["status"] == "PASS"
    assert reproducibility["full_independent_training_predictions_identical"]
    assert reproducibility["chunk_independent_training_predictions_identical"]
    assert negative_log.count("PASS:") == 3

    assert config["pipeline"]["vectorizer"] == "hashing"
    assert config["pipeline"]["model"] == "sgd_classifier"

    for result in (full, chunk):
        model_path = ROOT / result["model_path"]
        assert model_path.exists()
        assert sha256(model_path) == result["model_sha256"]

    with (OUT / "ml_metrics.csv").open(encoding="utf-8", newline="") as stream:
        metric_rows = list(csv.DictReader(stream))

    with (OUT / "memory_compare.csv").open(encoding="utf-8", newline="") as stream:
        memory_rows = list(csv.DictReader(stream))

    assert len(metric_rows) == 10
    assert len(memory_rows) == 2

    run_ids = {
        "full": full["mlflow_run_id"],
        "chunk": chunk["mlflow_run_id"],
    }

    for row in metric_rows:
        assert row["mlflow_run_id"] == run_ids[row["mode"]]

    for row in memory_rows:
        assert row["mlflow_run_id"] == run_ids[row["mode"]]

    metric_table = [
        "| Метрика | Full [95% CI] | Chunk [95% CI] |",
        "|---|---:|---:|",
    ]

    for key, title in METRICS.items():

        def metric_cell(result):
            values = result["metrics"]
            return (
                f"{fmt(values[key])} "
                f"[{fmt(values[key + '_ci_lower'])}; "
                f"{fmt(values[key + '_ci_upper'])}]"
            )

        metric_table.append(f"| {title} | {metric_cell(full)} | {metric_cell(chunk)} |")

    full_memory = float(full["peak_memory_mb"])
    chunk_memory = float(chunk["peak_memory_mb"])
    memory_saved = full_memory - chunk_memory
    memory_saved_percent = 100 * memory_saved / full_memory

    full_time = float(full["training_time_sec"])
    chunk_time = float(chunk["training_time_sec"])

    scaling_rows = [
        "| Объём | Документы | Чанки | Пиковая RSS, МБ |",
        "|---|---:|---:|---:|",
    ]

    for item in scaling["measurements"]:
        scaling_rows.append(
            f"| {item['data_multiplier']}× | "
            f"{item['rows']} | {item['chunks']} | "
            f"{fmt(item['peak_memory_mb'], 1)} |"
        )

    train_path = ROOT / config["data"]["train_path"]
    test_path = ROOT / config["data"]["test_path"]

    today = datetime.now().astimezone().strftime("%d.%m.%Y")
    head = git_head()

    report = f"""# Лабораторная работа № 2

**Тема:** Воспроизводимый ML-пайплайн и сравнение полного и пакетного обучения

**Вариант:** 18
**Автор:** София Валиева
**Дата формирования отчёта:** {today}
**Git HEAD на момент формирования:** `{head}`

> Git HEAD фиксирует последнюю сохранённую ревизию репозитория.
> Изменения, ещё не включённые в коммит, необходимо сохранить
> перед сдачей.

## 1. Цель работы

Реализовать воспроизводимый конвейер машинного обучения для
классификации текстов, настроить проверяемую конфигурацию,
сравнить обучение на полном наборе и обучение чанками,
оценить качество, время и потребление памяти, сохранить
результаты экспериментов в MLflow.

## 2. Исходные данные

Используется набор **20 Newsgroups**, вариант 18.

Категории:

- `alt.atheism`
- `talk.religion.misc`
- `comp.graphics`
- `sci.space`

Обучающий набор: **2 034 документа**.
Тестовый набор: **1 353 документа**.

Файлы и контрольные суммы SHA-256:

- Train: `{config["data"]["train_path"]}`
- Train SHA-256: `{sha256(train_path)}`
- Test: `{config["data"]["test_path"]}`
- Test SHA-256: `{sha256(test_path)}`

Данные и их первичная проверка подготовлены в лабораторной № 1.

## 3. Конфигурация и алгоритм

Конфигурация: `configs/lab2.yaml`.

- Векторизатор: `HashingVectorizer`.
- Число признаков: **65 536**.
- Классификатор: `SGDClassifier(loss="hinge")`.
- Фиксированный seed: **{config["random_state"]}**.
- Размер чанка: **{config["chunking"]["chunk_size"]}**.
- Число проходов `max_iter` для Full: **1 000**.
- Эксперимент MLflow: `{config["mlflow"]["experiment_name"]}`.

Параметр C используется для вычисления
`alpha = 1 / (C × 1000)`.

Конфигурация проверяется через Pydantic.
Невалидные значения и неизвестные поля отклоняются.

## 4. Методика эксперимента

Обучающие данные разделяются на train и validation
в пропорции 80/20 со стратификацией.

Гиперпараметр выбирается по Macro F1 на validation
среди C = 0.1, 1.0, 10.0.

Лучший C: **{fmt(full["best_C"], 1)}**.
Validation Macro F1: **{fmt(full["best_validation_macro_f1"])}**.

После подбора выполняется обучение на всех
2 034 обучающих документах.

**Full:** модель обучается через `fit` на полном наборе.

**Chunk:** документы считываются из JSONL порциями,
векторизуются и последовательно передаются в `partial_fit`.
Полная матрица признаков не создаётся.

Оба режима используют одинаковые типы векторизатора
и классификатора. При этом процедуры оптимизации
`fit` и последовательный `partial_fit` различаются,
поэтому итоговые модели могут давать разные предсказания.

Test используется для итоговой оценки, но не для
подбора гиперпараметров.

## 5. Качество на тестовом наборе

{chr(10).join(metric_table)}

95%-е доверительные интервалы рассчитаны методом
percentile bootstrap с 1 000 повторных выборок.

**Наивный baseline:** предсказание наиболее частого
класса в обучающем наборе.

| Метрика baseline | Full | Chunk |
|---|---:|---:|
| Accuracy | {fmt(full["baseline"]["accuracy"])} | {fmt(chunk["baseline"]["accuracy"])} |
| Macro F1 | {fmt(full["baseline"]["macro_f1"])} | {fmt(chunk["baseline"]["macro_f1"])} |

## 6. Время и память

| Показатель | Full | Chunk |
|---|---:|---:|
| Время обучения, с | {fmt(full_time, 3)} | {fmt(chunk_time, 3)} |
| Пиковая RSS всего процесса, МБ | {fmt(full_memory, 1)} | {fmt(chunk_memory, 1)} |

Разница в пиковом потреблении памяти:
**{fmt(memory_saved, 1)} МБ**
(**{fmt(memory_saved_percent, 1)}%** относительно Full).

Пиковая RSS измерялась внешним наблюдателем за
полным временем работы каждого Python-процесса.
Время обучения измерялось внутри программы.
Поэтому это разные интервалы измерения.

Результаты зависят от оборудования, окружения
и условий конкретного запуска.

## 7. Проверка масштабирования Chunk

{chr(10).join(scaling_rows)}

На проверенных объёмах 1×, 2× и 4× пиковая RSS
оставалась приблизительно постоянной.

Эта проверка подтверждает ограниченный рост памяти
**в пределах проведённого эксперимента**, но не является
математическим доказательством для произвольных объёмов.

При масштабировании использовалось повторение
исходных обучающих записей. Эксперимент проверяет
потребление памяти, а не качество на новых данных.

## 8. Негативные проверки

**Конфигурация:** три некорректных случая отклонены:

1. `chunk_size = 0`.
2. `C = -1`.
3. Неизвестное поле `unexpected_option`.

Результат: **PASS (3 из 3)**.

**Проверка утечки:** при подборе гиперпараметра
обучение выполнялось только на 1 627 документах
train-части; финальное обучение — на 2 034 документах.
Специальный маркер из validation не попадал в `fit`.

Результат: **{leakage["status"]}**.

Проверка также подтвердила, что baseline использует
частотность классов из train, а не из test.

## 9. Воспроизводимость

Два независимых обучения с seed
**{reproducibility["seed"]}** дали идентичные
предсказания на {reproducibility["test_rows"]} тестовых документах.

- Full: **PASS**.
- Chunk: **PASS**.

Сравнивались предсказания после повторного обучения,
а не только повторные вызовы сохранённой модели.

## 10. MLflow и сохранённые модели

**Full Run ID:** `{full["mlflow_run_id"]}`

**Chunk Run ID:** `{chunk["mlflow_run_id"]}`

Модель Full:

- Файл: `{full["model_path"]}`
- SHA-256: `{full["model_sha256"]}`

Модель Chunk:

- Файл: `{chunk["model_path"]}`
- SHA-256: `{chunk["model_sha256"]}`

Контрольные суммы сохранённых файлов сверены
при генерации отчёта.

## 11. Артефакты

- `lab2_report.md` — итоговый отчёт.
- `ml_metrics.csv` — качество и доверительные интервалы.
- `memory_compare.csv` — время и память.
- `full_results.json`, `chunk_results.json` — результаты запусков.
- `ml_model.joblib`, `chunk_model.joblib` — обученные модели.
- `config_negative_test.log` — негативные тесты конфигурации.
- `leakage_test.json` — проверка утечки.
- `training_reproducibility.json` — проверка воспроизводимости.
- `chunk_scaling.json` — масштабирование и память.
- `full_run.log`, `chunk_run.log` — журналы запусков.

## 12. Выводы

1. На тестовом наборе из **1 353 документов**
   Full достиг Macro F1 **{fmt(full["metrics"]["macro_f1"])}**,
   Chunk — **{fmt(chunk["metrics"]["macro_f1"])}**.

2. Пиковая память полного процесса составила
   **{fmt(full_memory, 1)} МБ** для Full и
   **{fmt(chunk_memory, 1)} МБ** для Chunk.
   Разница — **{fmt(memory_saved_percent, 1)}%**
   в пользу Chunk в данном запуске.

3. При увеличении набора с **2 034** до **8 136**
   документов память потокового обучения
   не продемонстрировала устойчивого роста.

4. Все **3** негативных теста конфигурации прошли
   успешно. Проверка изоляции validation также пройдена.

5. Повторное независимое обучение с seed **42**
   дало одинаковые предсказания в обоих режимах.

6. Результаты экспериментов связаны с MLflow Run ID,
   сохранёнными моделями и их SHA-256.
"""

    report_path = OUT / "lab2_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"REPORT CREATED: {report_path}")
    print(f"Full Macro F1: {fmt(full['metrics']['macro_f1'])}")
    print(f"Chunk Macro F1: {fmt(chunk['metrics']['macro_f1'])}")
    print(f"Memory saving: {fmt(memory_saved_percent, 1)}%")
    print("REPORT CHECKS: PASS")


if __name__ == "__main__":
    main()
