## ML + REG + PCA + GridSearchCV
## RandomForestClassifier + GradientBoostingClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================
# 1. Cargar datasets
# ============================
train_df = pd.read_csv("PREDICCIONES/LABELS_TRAIN.csv")
val_df   = pd.read_csv("PREDICCIONES/LABELS_VAL.csv")
test_df  = pd.read_csv("PREDICCIONES/LABELS_TEST.csv")

# ============================
# 2. Definir features y target
# ============================
target = "true_label"
drop_cols = ["group", "patient_uid_count", "id", target]

features = [c for c in train_df.columns if c not in drop_cols]

X_train, y_train = train_df[features], train_df[target]
X_val,   y_val   = val_df[features], val_df[target]
X_test,  y_test  = test_df[features], test_df[target]

# ============================
# 3. Pipelines
# ============================
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=1))
])

pipe_gb = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', GradientBoostingClassifier(random_state=42))
])

# ============================
# 4. Espacios de búsqueda
# ============================
param_grid_rf = {
    'pca__n_components': [10, 15, 20, 30],
    'clf__n_estimators': [200, 500, 1000],
    'clf__max_depth': [None, 5, 10, 20],
    'clf__min_samples_leaf': [1, 5, 10, 20],
    'clf__max_features': ['sqrt', 'log2', 0.3, 0.5]
}

param_grid_gb = {
    'pca__n_components': [10, 15, 20, 30],
    'clf__n_estimators': [200, 500, 1000],
    'clf__learning_rate': [0.05, 0.1, 0.2],
    'clf__max_depth': [2, 3, 4],
    'clf__min_samples_leaf': [1, 5, 10, 20],
    'clf__subsample': [0.6, 0.8, 1.0]
}

# ============================
# 5. GridSearchCV
# ============================
def grid_search_model(pipe, param_grid, model_name):
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        verbose=2
    )
    grid.fit(X_train, y_train)

    # Mejor modelo
    best_model = grid.best_estimator_
    best_params = grid.best_params_

    # Predicciones (probabilidades)
    y_train_pred = best_model.predict_proba(X_train)[:, 1]
    y_val_pred   = best_model.predict_proba(X_val)[:, 1]
    y_test_pred  = best_model.predict_proba(X_test)[:, 1]

    results = {
        "Final_Model": model_name,
        "Selector": "PCA",
        "k": best_params["pca__n_components"],
        "Best_Params": best_params,
        "AUC_Train": roc_auc_score(y_train, y_train_pred),
        "AUC_Val": roc_auc_score(y_val, y_val_pred),
        "AUC_Test": roc_auc_score(y_test, y_test_pred)
    }
    return results

results = []
results.append(grid_search_model(pipe_rf, param_grid_rf, "RandomForest"))
results.append(grid_search_model(pipe_gb, param_grid_gb, "GradientBoosting"))

# ============================
# 6. Resultados finales
# ============================
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="AUC_Val", ascending=False).reset_index(drop=True)

print("\n===== TOP 10 Modelos por AUC de Validación =====\n")
print(df_results.head(10)[["Final_Model", "Selector", "k", "AUC_Train", "AUC_Val", "AUC_Test", "Best_Params"]])

# Guardar CSV
df_results.to_csv("gridsearch_rf_gb_pca_results.csv", index=False)




##### METRICAS GUARDADAS EN GRIDSEARCH_RESULTS.CSV #####