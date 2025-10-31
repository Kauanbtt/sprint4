# -*- coding: utf-8 -*-
"""
Streamlit ML App — robust imports + sklearn compatibility fixes

What changed vs. your original file:
1) Robust dependency handling: clear error if scikit-learn isn't installed
   (and optional auto-install if running outside Streamlit Cloud).
2) Backward‑compatible OneHotEncoder: uses `sparse_output=False` on newer
   sklearn, and gracefully falls back to `sparse=False` on older versions.
3) Proper Streamlit UI (file uploader + buttons + on-screen metrics).
4) Safer CSV handling and messages.
"""
import os
import sys
import io
import importlib
import subprocess
from typing import List

# --- Optional: try to ensure dependencies when running locally ---------
# On Streamlit Cloud you should prefer requirements.txt (provided).
def _ensure_pkg(package: str, import_name: str = None, version: str = None):
    """
    Try to import a package, and if not available, attempt a pip install.
    Returns the imported module. If install fails on Streamlit Cloud,
    the UI will show a clear message.
    """
    name = import_name or package
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        spec = f"{package}=={version}" if version else package
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
        except Exception as e:
            # On Streamlit Cloud, this may fail if network is restricted.
            raise ModuleNotFoundError(
                f"Missing dependency '{package}'. Add it to requirements.txt. "
                f"(Tried to install '{spec}' and failed: {e})"
            )
        return importlib.import_module(name)

# Core libs
st = _ensure_pkg("streamlit", "streamlit")
np = _ensure_pkg("numpy", "numpy")
pd = _ensure_pkg("pandas", "pandas")
# Use a modern sklearn (>=1.2 for OneHotEncoder.sparse_output). 1.4.x is stable.
_ = _ensure_pkg("scikit-learn", "sklearn", version="1.4.2")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Classificador de Suspeita", page_icon="🧪", layout="wide")
st.title("🧪 Classificador de Suspeita (ML Pipeline)")

st.markdown(
    """
    Envie um CSV **separado por `;`** com a coluna alvo `alerta_suspeita` e colunas de features.
    Se você não enviar, o app tentará usar `dados_enriquecidos_com_alertas.csv` no diretório.
    """
)

# ---------- Helpers -----------------------------------------------------
def create_ohe():
    """Create OneHotEncoder compatible with old/new sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older sklearn (<1.2)
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def load_dataframe(uploaded_file):
    sep = ";"
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, sep=sep)
    default_path = "dados_enriquecidos_com_alertas.csv"
    if os.path.exists(default_path):
        st.info(f"Usando arquivo local **{default_path}**.")
        return pd.read_csv(default_path, sep=sep)
    st.warning("Nenhum CSV enviado e arquivo padrão não encontrado.")
    return None

# ---------- UI: CSV upload ----------------------------------------------
uploaded = st.file_uploader("📂 Envie seu CSV", type=["csv"])
df_modelo = load_dataframe(uploaded)

with st.expander("ℹ️ Instruções de colunas esperadas", expanded=False):
    st.write(
        {
            "Numericas (opcionais)": [
                "preco", "quantidade_vendida", "avaliacao_nota", "avaliacao_numero",
                "reviews_1_estrelas_pct", "reviews_5_estrelas_pct", "rendimento_paginas",
                "custo_por_pagina", "capacidade_num"
            ],
            "Categoricas (opcionais)": [
                "status_vendedor", "reputacao_cor", "categoria_produto", "modelo_cartucho"
            ],
            "Alvo (obrigatória)": "alerta_suspeita",
        }
    )

if df_modelo is None:
    st.stop()

# ---------- Basic validation & feature engineering ----------------------
if "capacidade" in df_modelo.columns:
    df_modelo["capacidade_num"] = (
        df_modelo["capacidade"].astype(str).str.extract(r"(\\d+\\.?\\d*)").astype(float)
    )
else:
    # Garantir a coluna para o pipeline (imputer cuidará dos NaNs)
    df_modelo["capacidade_num"] = np.nan

if "alerta_suspeita" not in df_modelo.columns:
    st.error("A coluna alvo **alerta_suspeita** não foi encontrada no CSV.")
    st.stop()

st.write("### Contagem do alvo `alerta_suspeita`")
st.dataframe(df_modelo["alerta_suspeita"].value_counts().to_frame("contagem"))

features_numericas_esperadas: List[str] = [
    "preco", "quantidade_vendida", "avaliacao_nota", "avaliacao_numero",
    "reviews_1_estrelas_pct", "reviews_5_estrelas_pct", "rendimento_paginas",
    "custo_por_pagina", "capacidade_num"
]
features_categoricas_esperadas: List[str] = [
    "status_vendedor", "reputacao_cor", "categoria_produto", "modelo_cartucho"
]

features_numericas = [c for c in features_numericas_esperadas if c in df_modelo.columns]
features_categoricas = [c for c in features_categoricas_esperadas if c in df_modelo.columns]

st.write("#### Features numéricas utilizadas", features_numericas)
st.write("#### Features categóricas utilizadas", features_categoricas)

X = df_modelo[features_numericas + features_categoricas]
y = df_modelo["alerta_suspeita"]

# ---------- Split -------------------------------------------------------
stratify = y if y.nunique() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=stratify
)
st.success(f"Dados divididos em treino ({len(X_train)}) e teste ({len(X_test)}).")

# ---------- Preprocessing pipelines ------------------------------------
preprocessor_numerico = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
preprocessor_categorico = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", create_ohe()),
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", preprocessor_numerico, features_numericas),
        ("cat", preprocessor_categorico, features_categoricas),
    ],
    remainder="passthrough",
)

st.info("Pipeline de pré-processamento criado.")

# ---------- Models ------------------------------------------------------
modelos = {
    "Regressão Logística": LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

if st.button("🚀 Treinar Modelos"):
    reports = {}
    for nome, modelo in modelos.items():
        with st.spinner(f"Treinando {nome} ..."):
            pipeline_modelo = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", modelo)])
            pipeline_modelo.fit(X_train, y_train)
            y_pred = pipeline_modelo.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cr = classification_report(y_test, y_pred, target_names=["Não Suspeito (0)", "Suspeito (1)"])
            cm = confusion_matrix(y_test, y_pred)

            st.subheader(f"Resultados — {nome}")
            st.metric("Acurácia", f"{acc:.4f}")
            st.text("Relatório de Classificação:")
            st.text(cr)
            st.write("Matriz de Confusão:")
            st.dataframe(pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]))
            st.divider()
            reports[nome] = {"accuracy": acc, "report": cr, "cm": cm.tolist()}

# ---------- Grid Search on RandomForest --------------------------------
st.markdown("### 🔧 Tuning (GridSearchCV) — Random Forest")
if st.button("🔎 Rodar GridSearch (recall)"):
    with st.spinner("Executando validação cruzada..."):
        pipeline_rf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced"))
        ])
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20, None],
            "classifier__min_samples_leaf": [1, 3],
        }
        grid_search = GridSearchCV(pipeline_rf, param_grid, cv=3, n_jobs=-1, verbose=1, scoring="recall")
        grid_search.fit(X_train, y_train)

        st.success("GridSearch finalizado.")
        st.write("**Melhores parâmetros (max recall):**", grid_search.best_params_)
        st.write("**Melhor score (Recall) em CV:**", f"{grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_
        y_pred_best = best_model.predict(X_test)

        acc_best = accuracy_score(y_test, y_pred_best)
        cr_best = classification_report(y_test, y_pred_best, target_names=["Não Suspeito (0)", "Suspeito (1)"])
        cm_best = confusion_matrix(y_test, y_pred_best)

        st.subheader("Avaliação no conjunto de teste — Modelo Otimizado")
        st.metric("Acurácia", f"{acc_best:.4f}")
        st.text("Relatório de Classificação:")
        st.text(cr_best)
        st.write("Matriz de Confusão:")
        st.dataframe(pd.DataFrame(cm_best, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]))
        st.divider()

st.caption("© 2025 — Pipeline de classificação com Streamlit + scikit-learn")