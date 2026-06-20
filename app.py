import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

RANDOM_STATE = 42
NUMERIC_FEATURES = ["age", "bmi", "children", "charges"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

st.set_page_config(
    page_title="Grupo 3 | Aprendizaje No Supervisado",
    page_icon="📊",
    layout="wide"
)

st.title("UTH-POSGRADOS-Grupo 3: Aprendizaje No Supervisado")
st.caption("Clustering con K-Means, visualización PCA, DBSCAN")


def make_demo_data(n=350, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, n)
    sex = rng.choice(["male", "female"], n)
    bmi = np.clip(rng.normal(30, 6, n), 16, 55).round(1)
    children = rng.integers(0, 5, n)
    smoker = rng.choice(["yes", "no"], n, p=[0.22, 0.78])
    region = rng.choice(["northeast", "northwest", "southeast", "southwest"], n)
    base = 2000 + age * 220 + bmi * 130 + children * 600
    smoke_effect = np.where(smoker == "yes", rng.normal(23000, 5000, n), rng.normal(0, 2500, n))
    charges = np.maximum(1000, base + smoke_effect + rng.normal(0, 2500, n)).round(2)
    return pd.DataFrame({
        "age": age, "sex": sex, "bmi": bmi, "children": children,
        "smoker": smoker, "region": region, "charges": charges
    })


@st.cache_data
def clean_data(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")
    df = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip().str.lower()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df


@st.cache_resource
def train_models(df, k_final=3, dbscan_eps=1.7, dbscan_min_samples=8):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    X = df[FEATURES].copy()
    prepared = preprocessor.fit_transform(X)
    prepared_dense = prepared.toarray() if hasattr(prepared, "toarray") else prepared

    k_rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(prepared)
        k_rows.append({
            "k": k,
            "inercia": km.inertia_,
            "silhouette": silhouette_score(prepared, labels),
        })
    k_results = pd.DataFrame(k_rows)

    kmeans = KMeans(n_clusters=k_final, random_state=RANDOM_STATE, n_init=10)
    pipeline = Pipeline([("preprocessor", preprocessor), ("kmeans", kmeans)])
    clusters = pipeline.fit_predict(X)

    result = df.copy()
    result["cluster"] = clusters
    resumen = result.groupby("cluster").agg(
        cantidad_clientes=("cluster", "count"),
        edad_promedio=("age", "mean"),
        bmi_promedio=("bmi", "mean"),
        hijos_promedio=("children", "mean"),
        cargos_promedio=("charges", "mean"),
        cargos_mediana=("charges", "median"),
        cargos_maximos=("charges", "max"),
        porcentaje_fumadores=("smoker", lambda s: (s == "yes").mean() * 100),
    ).round(2)

    orden = resumen.sort_values("cargos_promedio").index.tolist()
    risk_map = {orden[0]: "Bajo", orden[1]: "Medio", orden[2]: "Alto"}
    result["riesgo_actuarial"] = result["cluster"].map(risk_map)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_xy = pca.fit_transform(prepared_dense)
    pca_df = pd.DataFrame(pca_xy, columns=["PC1", "PC2"])
    pca_df["cluster"] = result["cluster"].astype(str)
    pca_df["riesgo_actuarial"] = result["riesgo_actuarial"]
    pca_df["smoker"] = result["smoker"]
    pca_df["charges"] = result["charges"]

    # SVM didáctico: aprende a separar las etiquetas generadas por K-Means en el plano PCA.
    X_svm = pca_df[["PC1", "PC2"]]
    y_svm = result["riesgo_actuarial"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_svm, y_svm, test_size=0.30, random_state=RANDOM_STATE, stratify=y_svm
    )
    kernels = {
        "linear": {"kernel": "linear", "C": 1},
        "poly": {"kernel": "poly", "C": 1, "degree": 3, "gamma": "scale"},
        "rbf": {"kernel": "rbf", "C": 1, "gamma": "scale"},
        "sigmoid": {"kernel": "sigmoid", "C": 1, "gamma": "scale"},
    }
    svm_models, svm_results = {}, []
    for name, params in kernels.items():
        svm = SVC(**params, random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        svm_models[name] = svm
        svm_results.append({
            "kernel": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        })
    svm_results = pd.DataFrame(svm_results).round(3)

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    db_labels = dbscan.fit_predict(prepared_dense)
    dbscan_df = pca_df.copy()
    dbscan_df["dbscan_cluster"] = db_labels.astype(str)

    final_score = silhouette_score(pipeline.named_steps["preprocessor"].transform(X), clusters)

    return {
        "pipeline": pipeline,
        "prepared_dense": prepared_dense,
        "k_results": k_results,
        "result": result,
        "summary": resumen,
        "risk_map": risk_map,
        "pca_df": pca_df,
        "pca_model": pca,
        "pca_variance": pca.explained_variance_ratio_,
        "svm_results": svm_results,
        "svm_models": svm_models,
        "dbscan_df": dbscan_df,
        "silhouette": final_score,
    }


def load_dataset():
    st.sidebar.header("Datos")
    uploaded = st.sidebar.file_uploader("Suba insurance.csv", type=["csv"])
    local_path = Path("insurance.csv")

    if uploaded is not None:
        return pd.read_csv(uploaded), "Archivo CSV cargado por el usuario"
    if local_path.exists():
        return pd.read_csv(local_path), "Archivo local insurance.csv"

    st.sidebar.warning("No se encontró insurance.csv. Se usará una base demo sintética para que la app funcione en la exposición.")
    return make_demo_data(), "Base demo sintética"


try:
    raw_df, source = load_dataset()
    df = clean_data(raw_df)
except Exception as exc:
    st.error(f"No se pudo cargar la base de datos: {exc}")
    st.stop()

st.sidebar.header("Parámetros del modelo")
k_final = st.sidebar.slider("Número de clusters K-Means", 2, 6, 3)
dbscan_eps = st.sidebar.slider("DBSCAN eps", 0.5, 5.0, 1.7, 0.1)
dbscan_min_samples = st.sidebar.slider("DBSCAN min_samples", 3, 20, 8)

artifacts = train_models(df, k_final, dbscan_eps, dbscan_min_samples)

st.sidebar.markdown("---")
st.sidebar.write(f"Fuente: **{source}**")
st.sidebar.write(f"Registros limpios: **{len(df):,}**")

intro, datos, kmeans_tab, pca_tab, dbscan_tab, svm_tab, simulador, salida = st.tabs([
    "Inicio", "Datos", "K-Means", "PCA", "DBSCAN", "SVM", "Simulador", "Salida"
])

with intro:
    st.subheader("Objetivo de la aplicación")
    st.write(
        "Esta app demuestra aprendizaje no supervisado aplicado a datos de seguros médicos. "
        "El modelo K-Means agrupa clientes sin una etiqueta previa y luego se interpreta cada grupo "
        "como riesgo actuarial bajo, medio o alto según el promedio de cargos médicos."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes analizados", f"{len(df):,}")
    c2.metric("Clusters", k_final)
    c3.metric("Silhouette final", f"{artifacts['silhouette']:.3f}")
    st.info("Nota crítica: usar `charges` ayuda a explicar grupos históricos. Para predecir clientes futuros antes de conocer cargos reales, debe entrenarse una variante sin `charges`.")

with datos:
    st.subheader("Vista inicial y descripción")
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Resumen estadístico")
    st.dataframe(df.describe().round(2), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df, x="charges", color="smoker", nbins=35, title="Distribución de cargos por fumador"), use_container_width=True)
    with col2:
        st.plotly_chart(px.scatter(df, x="age", y="charges", color="smoker", size="bmi", hover_data=["sex", "region"], title="Edad, cargos, fumador y BMI"), use_container_width=True)

with kmeans_tab:
    st.subheader("Selección de K: método del codo y Silhouette")
    kr = artifacts["k_results"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(kr, x="k", y="inercia", markers=True, title="Método del codo"), use_container_width=True)
    c2.plotly_chart(px.line(kr, x="k", y="silhouette", markers=True, title="Silhouette score"), use_container_width=True)
    st.write("Resumen de clusters")
    summary = artifacts["summary"].copy()
    summary["riesgo_actuarial"] = [artifacts["risk_map"].get(i, "N/D") for i in summary.index]
    st.dataframe(summary, use_container_width=True)

with pca_tab:
    st.subheader("Visualización con PCA")
    st.write(f"Varianza explicada: PC1={artifacts['pca_variance'][0]:.2%}, PC2={artifacts['pca_variance'][1]:.2%}")
    st.plotly_chart(
        px.scatter(
            artifacts["pca_df"], x="PC1", y="PC2", color="riesgo_actuarial", symbol="cluster",
            hover_data=["smoker", "charges"], title="Clusters K-Means visualizados en dos componentes principales"
        ),
        use_container_width=True,
    )

with dbscan_tab:
    st.subheader("DBSCAN exploratorio")
    st.write("DBSCAN no exige definir K; identifica grupos densos y marca ruido como -1. Es útil para contrastar K-Means, no para reemplazarlo automáticamente.")
    st.plotly_chart(
        px.scatter(artifacts["dbscan_df"], x="PC1", y="PC2", color="dbscan_cluster", title="DBSCAN sobre datos transformados y visualizados con PCA"),
        use_container_width=True,
    )

with svm_tab:
    st.subheader("SVM didáctico sobre etiquetas creadas por K-Means")
    st.write("La SVM no descubre los grupos; aprende a reproducir la etiqueta de riesgo generada por K-Means para comparar kernels.")
    st.dataframe(artifacts["svm_results"], use_container_width=True)
    st.bar_chart(artifacts["svm_results"].set_index("kernel")[["accuracy", "precision_macro"]])

with simulador:
    st.subheader("Evaluar un cliente nuevo")
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Edad", 18, 80, 45)
    sex = c2.selectbox("Sexo", ["male", "female"])
    bmi = c3.number_input("BMI", min_value=10.0, max_value=70.0, value=31.2, step=0.1)
    c4, c5, c6 = st.columns(3)
    children = c4.slider("Hijos", 0, 10, 2)
    smoker = c5.selectbox("Fumador", ["yes", "no"])
    region = c6.selectbox("Región", ["northeast", "northwest", "southeast", "southwest"])
    charges = st.number_input("Cargos médicos observados", min_value=0.0, value=28000.0, step=500.0)

    cliente = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi, "children": children,
        "smoker": smoker, "region": region, "charges": charges,
    }])
    cluster = int(artifacts["pipeline"].predict(cliente)[0])
    riesgo = artifacts["risk_map"].get(cluster, "N/D")
    explicaciones = {
        "Bajo": "Cliente agrupado con perfiles de menor costo médico promedio.",
        "Medio": "Cliente agrupado con perfiles de costo y factores de riesgo intermedios.",
        "Alto": "Cliente agrupado con perfiles de mayor costo médico promedio y/o factores de riesgo relevantes.",
    }
    st.metric("Cluster asignado", cluster)
    st.metric("Riesgo actuarial", riesgo)
    st.success(explicaciones.get(riesgo, "Sin interpretación disponible."))

with salida:
    st.subheader("Descarga de salidas")
    clustered_csv = artifacts["result"].to_csv(index=False).encode("utf-8")
    svm_csv = artifacts["svm_results"].to_csv(index=False).encode("utf-8")
    metadata = {
        "nombre_modelo": "K-Means + SVM para riesgo actuarial",
        "tipo_modelo": "Clustering no supervisado + clasificación supervisada didáctica",
        "n_clusters": k_final,
        "silhouette_score": round(float(artifacts["silhouette"]), 4),
        "variables_numericas": NUMERIC_FEATURES,
        "variables_categoricas": CATEGORICAL_FEATURES,
        "mapa_riesgo": {str(k): v for k, v in artifacts["risk_map"].items()},
    }
    st.download_button("Descargar insurance_con_clusters.csv", clustered_csv, "insurance_con_clusters.csv", "text/csv")
    st.download_button("Descargar svm_resultados_kernels.csv", svm_csv, "svm_resultados_kernels.csv", "text/csv")
    st.download_button("Descargar model_metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False), "model_metadata.json", "application/json")

    # Guardado local opcional cuando la app corre en un entorno con escritura.
    if st.button("Guardar modelo localmente"):
        Path("models").mkdir(exist_ok=True)
        joblib.dump(artifacts["pipeline"], "models/kmeans_riesgo_actuarial.pkl")
        joblib.dump(artifacts["svm_models"], "models/svm_riesgo_actuarial.pkl")
        Path("models/model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        st.success("Modelos guardados en la carpeta models/. En Streamlit Cloud esto puede no persistir después de reiniciar la app.")
