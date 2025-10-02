import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt

# =========================
# 1. Cargar datasets
# =========================
train_df = pd.read_csv("PREDICCIONES/LABELS_TRAIN.csv")
val_df   = pd.read_csv("PREDICCIONES/LABELS_VAL.csv")
test_df  = pd.read_csv("PREDICCIONES/LABELS_TEST.csv")

target = "true_label"
drop_cols = ["group", "patient_uid_count", "id", target]
features = [c for c in train_df.columns if c not in drop_cols]

X_train, y_train = train_df[features], train_df[target]
X_val,   y_val   = val_df[features], val_df[target]
X_test,  y_test  = test_df[features], test_df[target]

# =========================
# 2. Definir mejores modelos con hiperparámetros encontrados
# =========================
best_models = {
    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=20, random_state=42)),
        ("clf", GradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=20,
            n_estimators=200,
            subsample=0.8,
            random_state=42
        ))
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=20, random_state=42)),
        ("clf", RandomForestClassifier(
            max_depth=None,
            max_features=0.3,
            min_samples_leaf=20,
            n_estimators=1000,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# =========================
# 3. Función bootstrap para IC95% del AUC
# =========================
def bootstrap_auc(y_true, y_prob, n_bootstrap=1000, alpha=0.95, seed=42):
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    for _ in range(n_bootstrap):
        indices = rng.randint(0, len(y_prob_np), len(y_prob_np))
        if len(np.unique(y_true_np[indices])) < 2:
            continue
        score = roc_auc_score(y_true_np[indices], y_prob_np[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.sort(bootstrapped_scores)
    lower = np.percentile(sorted_scores, ((1 - alpha) / 2) * 100)
    upper = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
    return roc_auc_score(y_true_np, y_prob_np), lower, upper

# =========================
# 4. Función para calcular threshold óptimo usando G-Mean 
# =========================
def optimal_threshold(y_true, y_prob):
    thresholds = np.linspace(0,1,1000)
    g_means = []
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    for t in thresholds:
        y_pred = (y_prob_np >= t).astype(int)
        tn = sum((y_true_np==0) & (y_pred==0))
        fp = sum((y_true_np==0) & (y_pred==1))
        fn = sum((y_true_np==1) & (y_pred==0))
        tp = sum((y_true_np==1) & (y_pred==1))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        g_means.append(np.sqrt(tpr*tnr))
    idx = np.argmax(g_means)
    return thresholds[idx]

# =========================
# 5. Entrenar y evaluar
# =========================
results = []

for name, pipe in best_models.items():
    print(f"\n>>> Entrenando {name}...")
    pipe.fit(X_train, y_train)

    # Probabilidades
    y_train_prob = pipe.predict_proba(X_train)[:,1]
    y_val_prob   = pipe.predict_proba(X_val)[:,1]
    y_test_prob  = pipe.predict_proba(X_test)[:,1]

    # Threshold óptimo en test
    thresh = optimal_threshold(y_test, y_test_prob)
    print(f"Threshold óptimo ({name}) en TEST: {thresh:.3f}")

    # Predicciones con threshold
    y_train_pred = (y_train_prob >= thresh).astype(int)
    y_val_pred   = (y_val_prob   >= thresh).astype(int)
    y_test_pred  = (y_test_prob  >= thresh).astype(int)

    # AUC con IC95%
    auc_train, l_train, u_train = bootstrap_auc(y_train, y_train_prob)
    auc_val,   l_val,   u_val   = bootstrap_auc(y_val, y_val_prob)
    auc_test,  l_test,  u_test  = bootstrap_auc(y_test, y_test_prob)

    # Confusion matrix (solo para test)
    tn = sum((y_test==0) & (y_test_pred==0))
    fp = sum((y_test==0) & (y_test_pred==1))
    fn = sum((y_test==1) & (y_test_pred==0))
    tp = sum((y_test==1) & (y_test_pred==1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # Formato AUC [IC95]
    auc_train_str = f"{auc_train:.3f} [{l_train:.3f}–{u_train:.3f}]"
    auc_val_str   = f"{auc_val:.3f} [{l_val:.3f}–{u_val:.3f}]"
    auc_test_str  = f"{auc_test:.3f} [{l_test:.3f}–{u_test:.3f}]"

    results.append({
        "Modelo": name,
        "AUC_Train [IC95]": auc_train_str,
        "AUC_Val [IC95]": auc_val_str,
        "AUC_Test [IC95]": auc_test_str,
        "Acc_Train": accuracy_score(y_train, y_train_pred),
        "Acc_Val": accuracy_score(y_val, y_val_pred),
        "Acc_Test": accuracy_score(y_test, y_test_pred),
        "Sensitivity_Test": round(sensitivity,4),
        "Specificity_Test": round(specificity,4),
        "Threshold": round(thresh,3)
    })

    # Curva ROC test con threshold marcado
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC={auc_test:.3f}")
    plt.scatter(fpr[np.argmax(tpr-np.array(fpr))], tpr[np.argmax(tpr-np.array(fpr))], 
                color='red', label=f"Threshold={thresh:.3f}")
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name} (Test Set)")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()

# =========================
# 6. Resultados finales
# =========================
df_results = pd.DataFrame(results)
print("\n===== Resultados finales con AUC [IC95], Accuracy, Sensitivity, Specificity y Threshold óptimo =====")
print(df_results)

df_results.to_csv("resultados_finales_auc_acc_sens_spec_threshold.csv", index=False)
