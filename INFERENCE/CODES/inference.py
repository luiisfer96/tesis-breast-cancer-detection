import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

# ================================
# CONFIGURACIÓN GENERAL
# ================================
PYTHON_EXEC = "python"
RUTA_SCRIPT = r"INFERENCE"

# Lista de modelos
MODELOS = [
    {
        "ruta": r"ddsm_vgg16_s10_512x1.h5",
        "tipo": "rgb"
    }
    # ,
    # {
    #     "ruta": r"ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5",
    #     "tipo": "rgb"
    # },
    # {
    #     "ruta": r"inbreast_vgg16_[512-512-1024]x2_hybrid.h5",
    #     "tipo": "rgb"
    # },
    # {
    #     "ruta": r"inbreast_vgg16_512x1.h5",
    #     "tipo": "rgb"
    # },
    # {
    #     "ruta": r"ddsm_YaroslavNet_s10.h5",
    #     "tipo": "gray"
    # }
]

# Dataset de evaluación
DATASETS = [
    {
        "pkl": r"Metadata.pkl",
        "img_folder": r"Images",
        "name": "Inference"
    }
]

# Carpeta de resultados
RESULTS_DIR = os.path.join(RUTA_SCRIPT, "RESULTS_TEST")
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary_metrics.csv")

# ================================
# FUNCIONES
# ================================
def run_predictions(model_path, model_type, pkl_path, preprocessed_folder, name_tag):
    """
    Ejecuta la predicción usando generate_predictions.py
    """
    prediction_file = os.path.join(RESULTS_DIR, f"predictions_{os.path.basename(model_path)}_{name_tag}.csv")
    try:
        subprocess.run([
            PYTHON_EXEC, os.path.join(RUTA_SCRIPT, "generate_predictions.py"),
            "--exam-list-path", pkl_path,
            "--input-data-folder", preprocessed_folder,
            "--prediction-file", prediction_file,
            "--model", model_path,
            "--rescale-factor", "0.003891",
            "--mean-pixel-intensity", "44.4"
        ], check=True, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en generate_predictions.py con modelo {model_path}:", e.stderr)
        return None

    return pd.read_csv(prediction_file)


def compute_metrics(df):
    """
    Calcula Accuracy y AUC de un DataFrame de predicciones.
    """
    labels = df["malignant_label"].astype(int)
    preds_bin = (df["malignant_pred"] > 0.5).astype(int)
    preds_prob = df["malignant_pred"].astype(float)

    acc = accuracy_score(labels, preds_bin)
    try:
        auc = roc_auc_score(labels, preds_prob)
    except ValueError:
        auc = float("nan")  # No se puede calcular AUC si solo hay una clase
    return acc, auc


# ================================
# MAIN
# ================================
def main():
    summary_records = []

    for modelo in MODELOS:
        model_name = os.path.basename(modelo["ruta"])
        print(f"\n🚀 Probando modelo: {model_name} [{modelo['tipo']}]")

        try:
            all_dfs = []
            for data in DATASETS:
                print(f"🔄 Procesando {data['name']}...")

                df = run_predictions(modelo["ruta"], modelo["tipo"], data["pkl"], data["img_folder"], data["name"])
                if df is None:
                    raise RuntimeError("Error en predicciones")

                acc, auc = compute_metrics(df)
                print(f"✅ {data['name']} -> Accuracy: {acc:.4f} | AUC: {auc:.4f}")

                all_dfs.append(df)

                # Guardar métricas dataset
                summary_records.append({
                    "Model": model_name,
                    "Dataset": data["name"],
                    "Accuracy": round(acc, 4),
                    "AUC": round(auc, 4)
                })

            # Métricas globales
            combined = pd.concat(all_dfs, ignore_index=True)
            acc_global, auc_global = compute_metrics(combined)
            print(f"🎯 Global -> Accuracy: {acc_global:.4f} | AUC: {auc_global:.4f}")

            summary_records.append({
                "Model": model_name,
                "Dataset": "GLOBAL",
                "Accuracy": round(acc_global, 4),
                "AUC": round(auc_global, 4)
            })

        except Exception as e:
            print(f"⚠ Modelo {model_name} falló: {e}")
            continue

    # Guardar CSV resumen
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\n📂 Resumen de métricas guardado en: {SUMMARY_CSV}")

    # ================================
    # GRAFICAR AUC COMPARATIVO
    # ================================
    try:
        df_auc = df_summary[df_summary["Dataset"] == "GLOBAL"]
        plt.figure(figsize=(10, 6))
        plt.bar(df_auc["Model"], df_auc["AUC"], color="skyblue")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("AUC")
        plt.title("Comparación de AUC (GLOBAL) por Modelo")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "auc_comparison.png"))
        plt.show()
        print(f"📊 Gráfica de AUC guardada en: {os.path.join(RESULTS_DIR, 'auc_comparison.png')}")
    except Exception as e:
        print(f"⚠ Error generando gráfica: {e}")


if __name__ == "__main__":
    main()
