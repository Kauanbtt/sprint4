# -*- coding: utf-8 -*-
# Streamlined Streamlit ML App — fast boot & lazy imports
# - No runtime pip installs (use requirements.txt)
# - Lazy import scikit-learn only when needed
# - Minimal top-level work; caching for preprocessors
import os
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Classificador de Suspeita", page_icon="🧪", layout="wide")
st.title("🧪 Classificador de Suspeita (versão otimizada)")

st.markdown(
    "Envie um CSV **`;`-separado** com a coluna alvo `alerta_suspeita`. "
    "Se ausente, o app tenta `dados_enriquecidos_com_alertas.csv`."
)

# -------------------- Helpers --------------------
def _create_ohe():
    # Defer sklearn import
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

@st.cache_data(show_spinner=False)
def load_dataframe(file):
    if file is not None:
        return pd.read_csv(file, sep=";")
    path = "dados_enriquecidos_com_alertas.csv"
    if os.path.exists(path):
        return pd.read_csv(path, sep=";")
    return None

def ensure_capacidade_num(df):
    if "capacidade" in df.columns:
        # extrai números (ex.: "300 ml", "12.5")
        df["capacidade_num"] = (
            df["capacidade"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
        )
    elif "capacidade_num" not in df.columns:
        df["capacidade_num"] = np.nan
    return df

@st.cache_resource(show_spinner=False)
def build_preprocessor(num_cols, cat_cols):
    # Lazy imports
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    pre_num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    pre_cat = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", _create_ohe())
    ])
    return ColumnTransformer([
        ("num", pre_num, num_cols),
        ("cat", pre_cat, cat_cols)
    ], remainder="passthrough")

def train_and_eval(X_train, X_test, y_train, y_test, preprocessor):
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    modelos = {
        "Regressão Logística": LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    results = {}
    for nome, modelo in modelos.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", modelo)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        report = classification_report(y_test, pred, target_names=["Não Suspeito (0)", "Suspeito (1)"])
        cm = confusion_matrix(y_test, pred).tolist()
        results[nome] = {"acc": float(acc), "report": report, "cm": cm}
    return results

def run_grid_search(X_train, X_test, y_train, y_test, preprocessor):
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    pipe = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))])
    grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [10, 20, None],
        "clf__min_samples_leaf": [1, 3],
    }
    gs = GridSearchCV(pipe, grid, cv=3, n_jobs=-1, scoring="recall", verbose=0)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    pred = best.predict(X_test)
    return {
        "best_params": gs.best_params_,
        "best_cv_recall": float(gs.best_score_),
        "acc": float(accuracy_score(y_test, pred)),
        "report": classification_report(y_test, pred, target_names=["Não Suspeito (0)", "Suspeito (1)"]),
        "cm": confusion_matrix(y_test, pred).tolist(),
    }

# -------------------- UI --------------------
up = st.file_uploader("📂 Envie seu CSV", type=["csv"])
df = load_dataframe(up)

if df is None:
    st.warning("Envie um CSV ou inclua `dados_enriquecidos_com_alertas.csv` no diretório do app.")
    st.stop()

df = ensure_capacidade_num(df)

if "alerta_suspeita" not in df.columns:
    st.error("Coluna alvo `alerta_suspeita` não encontrada.")
    st.stop()

st.write("### Distribuição do alvo")
st.dataframe(df["alerta_suspeita"].value_counts().to_frame("contagem"))

NUM_COLS = [c for c in [
    "preco","quantidade_vendida","avaliacao_nota","avaliacao_numero",
    "reviews_1_estrelas_pct","reviews_5_estrelas_pct","rendimento_paginas",
    "custo_por_pagina","capacidade_num"
] if c in df.columns]
CAT_COLS = [c for c in ["status_vendedor","reputacao_cor","categoria_produto","modelo_cartucho"] if c in df.columns]

st.write("**Numéricas usadas:**", NUM_COLS)
st.write("**Categóricas usadas:**", CAT_COLS)

X = df[NUM_COLS + CAT_COLS]
y = df["alerta_suspeita"]

# Split com lazy import
from sklearn.model_selection import train_test_split
strat = y if y.nunique() > 1 else None
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)

pre = build_preprocessor(NUM_COLS, CAT_COLS)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Treinar modelos"):
        with st.spinner("Treinando..."):
            out = train_and_eval(Xtr, Xte, ytr, yte, pre)
        for nome, res in out.items():
            st.subheader(nome)
            st.metric("Acurácia", f"{res['acc']:.4f}")
            st.text(res["report"])
            st.dataframe(pd.DataFrame(res["cm"], index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"]))
            st.divider()

with col2:
    if st.button("🔎 GridSearch (recall)"):
        with st.spinner("Buscando melhores hiperparâmetros..."):
            tuned = run_grid_search(Xtr, Xte, ytr, yte, pre)
        st.subheader("Random Forest Otimizado")
        st.write("**Melhores parâmetros:**", tuned["best_params"])
        st.write("**Melhor Recall (CV):**", f"{tuned['best_cv_recall']:.4f}")
        st.metric("Acurácia (teste)", f"{tuned['acc']:.4f}")
        st.text(tuned["report"])
        st.dataframe(pd.DataFrame(tuned["cm"], index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"]))

st.caption("© 2025 — Versão otimizada para start rápido (imports tardios & cache).")