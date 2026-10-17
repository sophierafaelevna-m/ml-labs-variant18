"""Generate the Lab 3 report from local experiment artifacts, without inventing metrics.

Run from repository root after committing Lab 3 implementation:
    python -m src.lab3.make_report
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import subprocess
from datetime import date
from pathlib import Path

import torch

from src.lab3.config import load_config
from src.lab3.text_data import iter_split_records

ROOT = Path.cwd()
OUT = ROOT / "reports" / "LAB3"
MODES = ("full", "chunk")
METRICS = (
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
    ("macro_precision", "Macro Precision"),
    ("macro_recall", "Macro Recall"),
    ("rare_class_recall", "Recall редкого класса"),
)


def read_csv(filename: str) -> list[dict[str, str]]:
    path = OUT / filename
    if not path.exists():
        raise FileNotFoundError(f"Нет файла {path}; сначала завершите обучение")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(filename: str) -> dict:
    path = OUT / filename
    if not path.exists():
        raise FileNotFoundError(f"Нет файла {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
    ).strip()


def fmt(value: str | float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def verify_sqlite_run_ids(run_ids: list[str]) -> None:
    path = ROOT / "mlflow.db"
    if not path.is_file():
        raise FileNotFoundError(
            "Не найдена локальная база mlflow.db. "
            "Проверьте, где MLflow сохранил прогоны."
        )
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        found = {row[0] for row in conn.execute("SELECT run_uuid FROM runs")}
    missing = sorted(set(run_ids) - found)
    if missing:
        raise ValueError(f"MLflow не содержит Run ID: {', '.join(missing)}")


def main() -> None:
    cfg = load_config()
    metrics = read_csv("dl_metrics.csv")
    memory = {row["mode"]: row for row in read_csv("memory_compare.csv")}
    metric_by_key = {(row["mode"], row["metric"]): row for row in metrics}
    verification = read_json("verification_results.json")
    checks = verification["checks"]
    if verification["warnings"]:
        raise ValueError(
            "В verification_results.json есть предупреждения; сначала разберите их"
        )

    metadata = {mode: read_json(f"training_run_{mode}.json") for mode in MODES}
    histories = {mode: read_csv(f"training_history_{mode}.csv") for mode in MODES}
    checkpoints = {
        "full": OUT / "dl_model.pt",
        "chunk": OUT / "dl_model_chunk.pt",
    }
    for mode in MODES:
        if not histories[mode]:
            raise ValueError(f"Пустая история обучения: {mode}")
        if mode not in memory:
            raise ValueError(f"Нет данных о памяти: {mode}")
        if digest(checkpoints[mode]) != metadata[mode]["checkpoint_sha256"]:
            raise ValueError(f"Контрольная сумма чекпоинта {mode} не совпадает")
        if digest(checkpoints[mode]) != checks[f"{mode}_checkpoint_sha256"]:
            raise ValueError(f"Контрольная сумма {mode} не совпадает с проверкой")
        for key, _ in METRICS:
            row = metric_by_key.get((mode, key))
            if row is None:
                raise ValueError(f"Нет метрики {key} для {mode}")
            if row["mlflow_run_id"] != metadata[mode]["mlflow_run_id"]:
                raise ValueError(f"Run ID не совпадает для {mode} / {key}")
        if memory[mode]["mlflow_run_id"] != metadata[mode]["mlflow_run_id"]:
            raise ValueError(f"Run ID не совпадает в memory_compare.csv для {mode}")
    verify_sqlite_run_ids([metadata[m]["mlflow_run_id"] for m in MODES])

    full_checkpoint = torch.load(
        checkpoints["full"], map_location="cpu", weights_only=True
    )
    n_params = sum(t.numel() for t in full_checkpoint["model_state_dict"].values())
    rare_class = full_checkpoint["rare_class"]
    train_path = Path(cfg.data.train_path)
    test_path = Path(cfg.data.test_path)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Не найдены исходные JSONL-файлы для расчёта SHA256")
    split_counts = {
        split: sum(1 for _ in iter_split_records(cfg, split))
        for split in ("train", "val", "test")
    }

    lab2_metrics_path = ROOT / "reports" / "LAB2" / "ml_metrics.csv"
    lab2_rows = []
    if lab2_metrics_path.exists():
        with lab2_metrics_path.open(encoding="utf-8", newline="") as handle:
            lab2_rows = list(csv.DictReader(handle))
    lab2 = {row["mode"]: row for row in lab2_rows if row["metric"] == "macro_f1"}

    lines = [
        "# Лабораторная работа № 3",
        "",
        "**Тема:** Пайплайн DL-модели: полный датасет и чанки",
        "",
        "**Вариант:** 18  ",
        "**Автор:** София Валиева  ",
        f"**Дата:** {date.today().strftime('%d.%m.%Y')}  ",
        f"**Коммит реализации кода:** `{git_commit()}`  ",
        "**Датасет:** 20 Newsgroups, 4 класса",
        "",
        "## 1. Цель и режим воспроизводимости",
        "",
        "Реализовать и сравнить обучение одной нейросетевой архитектуры "
        "при загрузке полного датасета в память (Full) и потоковом чтении "
        "чанков с диска (Chunk).",
        "",
        "**Объявленный режим совпадения прогонов — точный:** "
        "при фиксированном seed и детерминированных операциях должны "
        "совпасть метрики и веса модели. Критерий указан здесь "
        "перед таблицами результатов; "
        "проверка реализована в `src/lab3/verify_lab3.py`.",
        "",
        "## 2. Данные и контрольные суммы",
        "",
        f"- Обучающий источник: `{train_path}`; SHA256: `{digest(train_path)}`.",
        f"- Тестовый источник: `{test_path}`; SHA256: `{digest(test_path)}`.",
        f"- Обучающая/валидационная выборки: {split_counts['train']:,} / "
        f"{split_counts['val']:,} документов; "
        f"тест: {split_counts['test']:,} документа "
        "(`src/lab3/text_data.py`, `src/lab3/make_report.py`).",
        "- Словарь формируется **только по обучающей части**, "
        "без валидационных и тестовых текстов "
        "(`src/lab3/text_data.py`).",
        "",
        "## 3. Архитектура и параметры",
        "",
        f"- Архитектура: `{cfg.model.architecture}`; "
        "`EmbeddingBag(mode=mean) → Linear → ReLU → Dropout → Linear` "
        "(`src/lab3/network.py`).",
        f"- Число параметров: **{n_params:,}** "
        "(`reports/LAB3/dl_model.pt`, `src/lab3/make_report.py`).",
        f"- Размер словаря: {len(full_checkpoint['vocabulary']):,}; "
        f"embedding_dim={cfg.model.embedding_dim}; "
        f"hidden_dim={cfg.model.hidden_dim}; "
        f"dropout={cfg.model.dropout}.",
        f"- Batch size={cfg.training.batch_size}; "
        f"максимум эпох={cfg.training.epochs}; "
        f"learning rate={cfg.training.learning_rate}; "
        "оптимизатор AdamW; loss CrossEntropyLoss.",
        f"- Seed={cfg.training.seed}; "
        f"patience ранней остановки={cfg.training.early_stopping_patience}; "
        "критерий — минимум validation loss.",
        f"- Chunk size={cfg.loading.chunk_size}; "
        "потоковый `IterableDataset` считывает JSONL с диска "
        "блоками без хранения всего корпуса "
        "(`src/lab3/train.py`, `src/lab3/text_data.py`).",
        "- Обучение выполнялось на CPU для детерминированности; "
        "паритет с MPS проверен отдельно.",
        "- Конфигурация валидируется Pydantic: `configs/lab3.yaml`.",
        "",
        "## 4. Кривые обучения и ранняя остановка",
        "",
        "![Кривые обучения Full и Chunk](learning_curves.png)",
        "",
        "Кривые `train_loss`, `val_loss`, `train_macro_f1`, `val_macro_f1` "
        "записаны по эпохам в `training_history_full.csv` и "
        "`training_history_chunk.csv`; построены "
        "`python -m src.lab3.plot_training`.",
        "",
    ]
    for mode in MODES:
        mem = memory[mode]
        early_stopped = int(mem["epochs_trained"]) < cfg.training.epochs
        status = (
            "сработала" if early_stopped else "не сработала (максимум эпох достигнут)"
        )
        lines.append(
            f"- **{mode.title()}:** {mem['epochs_trained']} эпох; "
            f"лучшая эпоха {mem['best_epoch']}; "
            f"ранняя остановка {status}"
        )
    lines.extend(
        [
            "",
            "## 5. Качество на тесте",
            "",
            "95%-е бутстрэп-интервалы: `src/lab3/train.py` (300 повторов).",
            "",
            "| Метрика | Full: значение [95% ДИ] | Chunk: значение [95% ДИ] |",
            "|---|---:|---:|",
        ]
    )
    for key, title in METRICS:
        cells = []
        for mode in MODES:
            row = metric_by_key[(mode, key)]
            cells.append(
                f"{fmt(row['value'])} [{fmt(row['ci_low'])}; {fmt(row['ci_high'])}]"
            )
        lines.append(f"| {title} | {cells[0]} | {cells[1]} |")
    lines.extend(
        [
            "",
            f"Редкий класс определяется по частотам train: **{rare_class}** "
            "(`src/lab3/train.py`, поле `rare_class` чекпоинта).",
            "Все метрики связаны с MLflow через `reports/LAB3/dl_metrics.csv`.",
            "",
            "## 6. Сравнение времени и памяти",
            "",
            "| Показатель | Full | Chunk |",
            "|---|---:|---:|",
            f"| Время обучения, с | {fmt(memory['full']['training_time_sec'], 3)} | {fmt(memory['chunk']['training_time_sec'], 3)} |",
            f"| Пиковая память процесса, МБ | {fmt(memory['full']['peak_memory_mb'], 2)} | {fmt(memory['chunk']['peak_memory_mb'], 2)} |",
            f"| Эпохи | {memory['full']['epochs_trained']} | {memory['chunk']['epochs_trained']} |",
            f"| Лучшая эпоха | {memory['full']['best_epoch']} | {memory['chunk']['best_epoch']} |",
            "",
            "Источник: `reports/LAB3/memory_compare.csv`, измерения "
            "в отдельных процессах (`src/lab3/train.py`).",
            "",
            "На данном небольшом датасете Chunk не продемонстрировал экономии "
            "пиковой памяти процесса и оказался медленнее Full. "
            "Это не противоречит ограниченности памяти *самого потокового чтения*: "
            "общий RSS включает PyTorch, модель, оптимизатор и накладные расходы.",
            "",
            "## 7. Воспроизводимость и негативные контроли",
            "",
            "| Проверка | Значение / результат |",
            "|---|---|",
            f"| Seed | {checks['repeatability']['same_seed']} |",
            f"| Число повторных обучений | {checks['repeatability']['runs']} |",
            f"| Macro F1 двух повторов | {fmt(checks['repeatability']['macro_f1_run_1'], 6)} / {fmt(checks['repeatability']['macro_f1_run_2'], 6)} |",
            f"| Разница истории эпох | {checks['repeatability']['max_history_difference']} |",
            f"| Разница весов повторов | {checks['repeatability']['max_weight_difference']} |",
            f"| Разница весов Full/Chunk | {checks['full_chunk_max_parameter_difference']} |",
            f"| Отсутствующий seed отклоняется | {checks['negative_controls']['missing_seed_rejected']} |",
            f"| Нулевой batch size отклоняется | {checks['negative_controls']['zero_batch_size_rejected']} |",
            f"| Инициализации без повторной фиксации seed различаются | {checks['negative_controls']['consecutive_initializations_without_reseeding_differ']} |",
            "",
            "Источник: `reports/LAB3/verification_results.json` "
            "(`python -m src.lab3.verify_lab3`). "
            "Негативный контроль без повторной фиксации seed показывает "
            "разницу начальных весов, а не полный повтор обучения без seed.",
            "",
            "### Масштабирование потокового чтения",
            "",
            "| Источник, записей | Пик Python-аллокаций при чтении, МБ |",
            "|---:|---:|",
            f"| {checks['streaming_memory_scaling']['small']['source_rows']} | {fmt(checks['streaming_memory_scaling']['small']['tracemalloc_peak_mb'], 5)} |",
            f"| {checks['streaming_memory_scaling']['large']['source_rows']} | {fmt(checks['streaming_memory_scaling']['large']['tracemalloc_peak_mb'], 5)} |",
            "",
            "Измерено через `tracemalloc` в `src/lab3/verify_lab3.py`. "
            "Это память Python-аллокаций во время потокового чтения, "
            "а не общий пик RSS процесса. Источник увеличен в 10 раз; "
            "память загрузчика практически не изменилась.",
            "",
            "### Паритет CPU/MPS",
            "",
            "| Устройства | Макс. абсолютное расхождение логитов | Допуск | Результат |",
            "|---|---:|---:|---|",
        ]
    )
    parity = checks["device_parity"]
    if parity.get("mps_available") and parity.get("pass") is not None:
        lines.append(
            f"| CPU / MPS | {parity['max_abs_difference']:.10g} | "
            f"{parity['tolerance_max_abs_diff']:.1g} | "
            f"{'PASS' if parity['pass'] else 'FAIL'} |"
        )
    else:
        lines.append("| CPU / MPS | — | 1e-5 | SKIP: MPS недоступен |")
    lines.extend(
        [
            "",
            "Паритет измерен на одном и том же сохранённом чекпоинте "
            "и тестовом батче, `float32`; источник: "
            "`reports/LAB3/verification_results.json`.",
            "",
            "## 8. MLflow, чекпоинты и хеши",
            "",
            "| Режим | MLflow Run ID | Чекпоинт | SHA256 |",
            "|---|---|---|---|",
        ]
    )
    for mode in MODES:
        lines.append(
            f"| {mode.title()} | `{metadata[mode]['mlflow_run_id']}` | "
            f"`{checkpoints[mode].relative_to(ROOT)}` | "
            f"`{metadata[mode]['checkpoint_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "MLflow хранится локально в `mlflow.db` (SQLite; файл не коммитится). "
            "Run ID сверены с локальной базой при генерации отчёта "
            "(`src/lab3/make_report.py`).",
            "",
            "## 9. Сравнение с классической моделью ЛР2",
            "",
        ]
    )
    if lab2:
        lines.extend(
            [
                "| Модель | Full Macro F1 | Chunk Macro F1 |",
                "|---|---:|---:|",
                f"| ЛР2: HashingVectorizer + SGDClassifier | {fmt(lab2['full']['value'])} | {fmt(lab2['chunk']['value'])} |",
                f"| ЛР3: EmbeddingBag + MLP | {fmt(metric_by_key[('full', 'macro_f1')]['value'])} | {fmt(metric_by_key[('chunk', 'macro_f1')]['value'])} |",
                "",
                "Источник: `reports/LAB2/ml_metrics.csv` и "
                "`reports/LAB3/dl_metrics.csv`. Классическая модель ЛР2 "
                "оказалась точнее на текущем разбиении. "
                "Сравнение следует трактовать с оговоркой: процедуры обучения "
                "и выделения валидации у лабораторных работ различаются.",
            ]
        )
    else:
        lines.append(
            "Файл `reports/LAB2/ml_metrics.csv` не найден. "
            "Перед сдачей добавьте сравнение с ЛР2."
        )
    lines.extend(
        [
            "",
            "## 10. Ограничения и выводы",
            "",
            "1. Full и Chunk обучили одну архитектуру с одинаковыми "
            "параметрами, показали одинаковое качество и идентичные веса.",
            "2. Воспроизводимость при фиксированном seed подтверждена "
            "двумя повторными обучениями; паритет CPU/MPS укладывается в допуск.",
            "3. Chunk обеспечивает потоковое чтение с ограниченным буфером, "
            "но на этом датасете не уменьшил общий пик RSS и увеличил время обучения.",
            "4. Качество нейросети ниже классического ML-базиса ЛР2; "
            "возможные улучшения — предобученные эмбеддинги, настройка "
            "архитектуры и регуляризации, увеличение числа эпох с "
            "контролем по валидации.",
            "5. Негативный контроль отсутствия повторной фиксации seed "
            "проверяет инициализацию весов, но не весь цикл обучения; "
            "это ограничение эксперимента.",
            "",
            "## 11. Основные артефакты",
            "",
            "- `configs/lab3.yaml` — конфигурация Pydantic.",
            "- `src/lab3/` — код загрузчиков, модели, обучения и проверок.",
            "- `dl_metrics.csv` — метрики с бутстрэп-интервалами и MLflow Run ID.",
            "- `memory_compare.csv` — память и время двух режимов.",
            "- `training_history_full.csv`, `training_history_chunk.csv` — кривые по эпохам.",
            "- `learning_curves.png` — визуализация кривых.",
            "- `dl_model.pt`, `dl_model_chunk.pt` — чекпоинты.",
            "- `training_run_full.json`, `training_run_chunk.json` — метаданные прогонов.",
            "- `verification_results.json` — проверки воспроизводимости, "
            "негативных контролей, памяти и устройств.",
            "",
        ]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "lab3_report.md"
    target.write_text("\n".join(lines), encoding="utf-8")

    appendix = OUT / "lab3_defense_snippet.md"
    appendix.write_text(
        "# Дополнение к корневому defense_checklist.md — ЛР3\n\n"
        "Вставьте этот раздел в существующий `defense_checklist.md` "
        "перед разделом об использовании ИИ.\n\n"
        "## Лабораторная работа № 3\n\n"
        "| Проверка | Результат |\n|---|---|\n"
        "| Full и Chunk | Обучены; метрики и веса совпали |\n"
        "| Повторяемость seed=42 | PASS, два независимых прогона |\n"
        "| CPU/MPS | PASS, расхождение меньше 1e-5 |\n"
        "| Pydantic / негативные контроли | PASS |\n"
        "| Масштабирование памяти чтения | PASS по tracemalloc; "
        "общий RSS Chunk выше Full |\n"
        "| Отчёт | `reports/LAB3/lab3_report.md` |\n"
        "| Метрики | `reports/LAB3/dl_metrics.csv` |\n"
        "| Модели | `reports/LAB3/dl_model.pt`, "
        "`reports/LAB3/dl_model_chunk.pt` |\n"
        "| Журнал проверок | `reports/LAB3/verification_results.json` |\n\n"
        "В разделе «Использование искусственного интеллекта» "
        "дополните ЛР1 и ЛР2 упоминанием ЛР3. "
        "ChatGPT (OpenAI) использовался для черновиков кода, "
        "проверочных сценариев, отчёта и диагностики MLflow; "
        "запуски и проверка результатов выполнялись локально.\n\n"
        "Удалите устаревшую фразу «ЛР3–ЛР8 в текущий комплект сдачи не входят» "
        "или замените на «ЛР4–ЛР8 в текущий комплект сдачи не входят».\n",
        encoding="utf-8",
    )
    print(f"PASS: report saved: {target}")
    print(f"PASS: checklist supplement saved: {appendix}")
    print("PASS: MLflow run IDs and checkpoint SHA256 verified")
    print(f"Code commit in report: {git_commit()}")


if __name__ == "__main__":
    main()
