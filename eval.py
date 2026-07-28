"""
Zentrale Auswertung & Vergleich aller drei Sentiment-Modelle
(VADER, RoBERTa, Own-NN).

Funktionen:
1. Jedes Modell wird auf denselben 1000 Testbeispielen einzeln bewertet
   (Gesamt-Genauigkeit, Genauigkeit getrennt nach positiven/negativen Texten).
2. Die Modelle werden untereinander verglichen (Ranking, Vergleichsplots,
   gemeinsame Problemfälle).
3. Man kann eigene Texte eingeben, die von allen drei Modellen live
   ausgewertet werden.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import vader
import roberta
import own_nn

# ==========================================
# 1. KONFIGURATION
# ==========================================
NUM_EVAL_SAMPLES = 1000
COMPARISON_PLOT_PATH = "model_comparison.png"
CONFUSION_PLOT_PATH = "model_confusion_matrices.png"

MODELS = {
    "VADER": {"module": vader, "results_path": vader.RESULTS_PATH},
    "RoBERTa": {"module": roberta, "results_path": roberta.RESULTS_PATH},
    "Own-NN": {"module": own_nn, "results_path": own_nn.RESULTS_PATH},
}


# ==========================================
# 2. ERGEBNISSE LADEN ODER GENERIEREN
# ==========================================
def load_or_generate_results():
    """Lädt für jedes Modell die gespeicherten Ergebnisse. Falls noch keine
    Ergebnis-Datei existiert, wird die Auswertung des jeweiligen Modells
    automatisch einmal ausgeführt (inkl. Training, falls nötig)."""
    all_results = {}
    for name, info in MODELS.items():
        path = info["results_path"]
        if os.path.exists(path):
            print(f"Lade vorhandene Ergebnisse für {name} aus '{path}'...")
            df = pd.read_csv(path)
        else:
            print(f"Keine Ergebnisse für {name} gefunden - führe Auswertung jetzt aus...")
            df = info["module"].run_evaluation(num_samples=NUM_EVAL_SAMPLES)
        all_results[name] = df
    return all_results


# ==========================================
# 3. METRIKEN PRO MODELL
# ==========================================
def compute_metrics(df, model_name):
    y_true = df["label"].values
    y_pred = df["prediction"].values

    overall_acc = accuracy_score(y_true, y_pred)

    neg_mask = y_true == 0
    pos_mask = y_true == 1
    neg_acc = accuracy_score(y_true[neg_mask], y_pred[neg_mask]) if neg_mask.sum() > 0 else float("nan")
    pos_acc = accuracy_score(y_true[pos_mask], y_pred[pos_mask]) if pos_mask.sum() > 0 else float("nan")

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 55}")
    print(f" {model_name} - Auswertung auf {len(df)} Beispielen")
    print(f"{'=' * 55}")
    print(f"Gesamt-Genauigkeit:              {overall_acc * 100:.2f}%")
    print(f"Genauigkeit bei Negativ-Texten:  {neg_acc * 100:.2f}%  (n={neg_mask.sum()})")
    print(f"Genauigkeit bei Positiv-Texten:  {pos_acc * 100:.2f}%  (n={pos_mask.sum()})")

    if neg_acc < pos_acc:
        print(f"-> {model_name} hat größere Schwierigkeiten bei NEGATIVEN Texten.")
    elif pos_acc < neg_acc:
        print(f"-> {model_name} hat größere Schwierigkeiten bei POSITIVEN Texten.")
    else:
        print(f"-> {model_name} performt bei positiven und negativen Texten etwa gleich gut.")

    print("\n" + classification_report(y_true, y_pred, target_names=["Negativ (0)", "Positiv (1)"]))

    return {
        "model": model_name,
        "overall_accuracy": overall_acc,
        "negative_accuracy": neg_acc,
        "positive_accuracy": pos_acc,
        "confusion_matrix": cm,
    }


# ==========================================
# 4. MODELLE UNTEREINANDER VERGLEICHEN
# ==========================================
def compare_models(all_results, metrics_list):
    print(f"\n{'=' * 55}")
    print(" MODELLVERGLEICH")
    print(f"{'=' * 55}")

    ranking = sorted(metrics_list, key=lambda m: m["overall_accuracy"], reverse=True)
    for rank, m in enumerate(ranking, start=1):
        print(f"{rank}. {m['model']}: {m['overall_accuracy'] * 100:.2f}% Gesamt-Genauigkeit")

    best = ranking[0]
    print(f"\nBestes Modell auf diesem Datensatz: {best['model']} "
          f"({best['overall_accuracy'] * 100:.2f}%)")

    # Beispiele, bei denen ALLE Modelle falsch lagen (besonders schwierige Fälle)
    dfs = list(all_results.values())
    n = min(len(df) for df in dfs)
    all_wrong_mask = np.ones(n, dtype=bool)
    for df in dfs:
        df_trunc = df.iloc[:n]
        all_wrong_mask &= (df_trunc["label"].values != df_trunc["prediction"].values)

    n_all_wrong = int(all_wrong_mask.sum())
    print(f"\nBeispiele, bei denen ALLE Modelle falsch lagen: {n_all_wrong} von {n}")
    if n_all_wrong > 0:
        example_idx = int(np.where(all_wrong_mask)[0][0])
        example_text = dfs[0].iloc[example_idx]["text"]
        print(f"Beispieltext (gekürzt): {str(example_text)[:200]}...")

    plot_comparison(metrics_list)


def plot_comparison(metrics_list):
    model_names = [m["model"] for m in metrics_list]
    overall = [m["overall_accuracy"] * 100 for m in metrics_list]
    neg = [m["negative_accuracy"] * 100 for m in metrics_list]
    pos = [m["positive_accuracy"] * 100 for m in metrics_list]

    x = np.arange(len(model_names))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, neg, width, label="Negativ-Texte", color="#e74c3c")
    plt.bar(x, overall, width, label="Gesamt", color="#3498db")
    plt.bar(x + width, pos, width, label="Positiv-Texte", color="#2ecc71")
    plt.xticks(x, model_names)
    plt.ylabel("Genauigkeit (%)")
    plt.ylim(0, 100)
    plt.title("Modellvergleich: Genauigkeit (gesamt / negativ / positiv)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT_PATH, dpi=200)
    print(f"\nVergleichsplot gespeichert als '{COMPARISON_PLOT_PATH}'.")
    plt.show()

    n_models = len(metrics_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics_list):
        sns.heatmap(
            m["confusion_matrix"], annot=True, fmt="d", cmap="Purples", ax=ax,
            xticklabels=["Vorhergesagt Neg.", "Vorhergesagt Pos."],
            yticklabels=["Echt Neg.", "Echt Pos."],
        )
        ax.set_title(m["model"])
    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT_PATH, dpi=200)
    print(f"Confusion-Matrizen gespeichert als '{CONFUSION_PLOT_PATH}'.")
    plt.show()


# ==========================================
# 5. EIGENEN TEXT AUSWERTEN LASSEN
# ==========================================
def load_all_inference_resources():
    print("\nLade Modelle für die Live-Auswertung eigener Texte...")
    resources = {
        "VADER": vader.load_resources(),
        "RoBERTa": roberta.load_resources(),
        "Own-NN": own_nn.load_resources(),
    }
    return resources


def evaluate_custom_text(text, resources):
    print(f"\nText: \"{text}\"")
    print("-" * 60)
    for name, module in (("VADER", vader), ("RoBERTa", roberta), ("Own-NN", own_nn)):
        label, prob = module.predict_single_text(text, resources[name])
        sentiment = "POSITIV" if label == 1 else "NEGATIV"
        print(f"{name:10s} -> {sentiment:8s} (Wahrscheinlichkeit positiv: {prob * 100:.1f}%)")


def interactive_mode(resources):
    print("\n" + "=" * 60)
    print(" EIGENEN TEXT AUSWERTEN LASSEN")
    print(" (leere Eingabe oder 'exit' zum Beenden)")
    print("=" * 60)
    while True:
        text = input("\nGib einen englischen Filmreview-Text ein: ").strip()
        if text == "" or text.lower() in ("exit", "quit", "q"):
            print("Beende die interaktive Auswertung.")
            break
        evaluate_custom_text(text, resources)


# ==========================================
# 6. MAIN
# ==========================================
if __name__ == "__main__":
    all_results = load_or_generate_results()
    metrics_list = [compute_metrics(df, name) for name, df in all_results.items()]
    compare_models(all_results, metrics_list)

    inference_resources = load_all_inference_resources()
    interactive_mode(inference_resources)