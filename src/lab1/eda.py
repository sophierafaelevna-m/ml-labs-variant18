from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports" / "LAB1"
FIGURES_DIR = REPORT_DIR / "figures"

CATEGORIES = [
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
]


def load_split(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    return pd.read_json(path, lines=True)


def text_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["text_chars"] = result["text"].str.len()
    result["text_words"] = result["text"].str.split().str.len()
    return result


def add_stat(stats: list[dict], metric: str, value: float, unit: str) -> None:
    stats.append(
        {
            "metric": metric,
            "value": value,
            "unit": unit,
        }
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train.jsonl")
    test = load_split("test.jsonl")

    train["split"] = "train"
    test["split"] = "test"

    df = pd.concat([train, test], ignore_index=True)

    df = text_length_stats(df)

    stats: list[dict] = []

    # Общая структура
    add_stat(stats, "documents_total", len(df), "documents")
    add_stat(stats, "documents_train", len(train), "documents")
    add_stat(stats, "documents_test", len(test), "documents")
    add_stat(stats, "classes", df["target_name"].nunique(), "classes")

    # Пропуски и пустые документы
    add_stat(
        stats,
        "missing_text",
        int(df["text"].isna().sum()),
        "documents",
    )
    add_stat(
        stats,
        "empty_text",
        int((df["text"].fillna("").str.strip() == "").sum()),
        "documents",
    )
    add_stat(
        stats,
        "missing_target",
        int(df["target"].isna().sum()),
        "documents",
    )

    # Длины текстов
    add_stat(
        stats,
        "text_chars_mean",
        df["text_chars"].mean(),
        "characters",
    )
    add_stat(
        stats,
        "text_chars_median",
        df["text_chars"].median(),
        "characters",
    )
    add_stat(
        stats,
        "text_chars_min",
        df["text_chars"].min(),
        "characters",
    )
    add_stat(
        stats,
        "text_chars_max",
        df["text_chars"].max(),
        "characters",
    )
    add_stat(
        stats,
        "text_words_mean",
        df["text_words"].mean(),
        "words",
    )
    add_stat(
        stats,
        "text_words_median",
        df["text_words"].median(),
        "words",
    )
    add_stat(
        stats,
        "text_words_min",
        df["text_words"].min(),
        "words",
    )
    add_stat(
        stats,
        "text_words_max",
        df["text_words"].max(),
        "words",
    )

    # Распределение классов
    class_counts = df["target_name"].value_counts().sort_index()

    for class_name, count in class_counts.items():
        share = count / len(df) * 100

        safe_name = class_name.replace(".", "_").replace(" ", "_").replace("-", "_")

        add_stat(
            stats,
            f"class_count_{safe_name}",
            int(count),
            "documents",
        )
        add_stat(
            stats,
            f"class_share_pct_{safe_name}",
            share,
            "percent",
        )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_df=0.5,
        min_df=5,
        stop_words="english",
    )

    tfidf = vectorizer.fit_transform(df["text"])

    n_documents, n_features = tfidf.shape
    nonzero = tfidf.nnz
    total_elements = n_documents * n_features

    density = nonzero / total_elements
    sparsity = 1 - density

    add_stat(stats, "tfidf_documents", n_documents, "documents")
    add_stat(stats, "tfidf_features", n_features, "features")
    add_stat(stats, "tfidf_nonzero", nonzero, "values")
    add_stat(stats, "tfidf_total_elements", total_elements, "values")
    add_stat(stats, "tfidf_density_pct", density * 100, "percent")
    add_stat(stats, "tfidf_sparsity_pct", sparsity * 100, "percent")

    # Частоты терминов в корпусе
    count_vectorizer = CountVectorizer(
        stop_words="english",
        min_df=1,
    )

    count_matrix = count_vectorizer.fit_transform(df["text"])
    term_frequency = count_matrix.sum(axis=0).A1

    frequency_bins = pd.cut(
        term_frequency,
        bins=[0, 1, 2, 5, 10, 50, float("inf")],
        labels=[
            "1",
            "2",
            "3-5",
            "6-10",
            "11-50",
            "51+",
        ],
        include_lowest=True,
    )

    frequency_counts = pd.Series(frequency_bins).value_counts().sort_index()

    for bin_name, count in frequency_counts.items():
        add_stat(
            stats,
            f"term_frequency_bin_{bin_name}",
            int(count),
            "terms",
        )

    # Сохраняем статистику
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(
        REPORT_DIR / "eda_stats.csv",
        index=False,
    )

    # График 1. Распределение классов
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
    )
    plt.title("Распределение документов по классам")
    plt.xlabel("Класс")
    plt.ylabel("Количество документов")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "class_distribution.png",
        dpi=200,
    )
    plt.close()

    # График 2. Длины текстов
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x="text_words",
        bins=50,
    )
    plt.title(
        f"Распределение длины текстов (медиана = {df['text_words'].median():.0f} слов)"
    )
    plt.xlabel("Количество слов")
    plt.ylabel("Количество документов")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "text_length_distribution.png",
        dpi=200,
    )
    plt.close()

    # График 3. Частотные бины
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=frequency_counts.index.astype(str),
        y=frequency_counts.values,
    )
    plt.title("Распределение терминов по частотным бинам")
    plt.xlabel("Частотный бин")
    plt.ylabel("Количество терминов")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "term_frequency_buckets.png",
        dpi=200,
    )
    plt.close()

    # График 4. Разреженность TF-IDF — основной график проблемы
    labels = ["Ненулевые значения", "Нулевые значения"]
    values = [density * 100, sparsity * 100]

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        x=labels,
        y=values,
    )

    for i, value in enumerate(values):
        ax.text(
            i,
            value + 0.5,
            f"{value:.2f}%",
            ha="center",
            fontweight="bold",
        )

    plt.title(f"Разреженность TF-IDF: {sparsity * 100:.2f}% элементов равны нулю")
    plt.xlabel("Тип значения")
    plt.ylabel("Доля матрицы, %")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "tfidf_sparsity.png",
        dpi=200,
    )
    plt.close()

    print("EDA completed successfully.")
    print(f"Documents: {len(df)}")
    print(f"Classes: {df['target_name'].nunique()}")
    print(f"TF-IDF shape: {tfidf.shape}")
    print(f"TF-IDF density: {density * 100:.2f}%")
    print(f"TF-IDF sparsity: {sparsity * 100:.2f}%")
    print(f"Statistics: {REPORT_DIR / 'eda_stats.csv'}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
