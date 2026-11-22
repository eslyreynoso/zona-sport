import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pronósticos NBA", page_icon="🏀")
st.title("🏀 Pronósticos NBA")
st.caption("Proyecto de prueba. Son probabilidades estimadas por un modelo estadístico, no apuestas seguras.")

pron = pd.read_csv("pronosticos_guardados.csv")
pron["GAME_DATE"] = pd.to_datetime(pron["GAME_DATE"]).dt.date
elegido_local = pron["PROB_LOCAL"] >= 0.5
pron["EQUIPO_ELEGIDO"] = pron["TEAM_NAME_LOC"].where(elegido_local, pron["TEAM_NAME_VIS"])
pron["CONFIANZA"] = pron["PROB_LOCAL"].where(elegido_local, 1 - pron["PROB_LOCAL"])
pron["PROBABILIDAD"] = (pron["CONFIANZA"] * 100).round(0).astype(int).astype(str) + "%"

if "GANO_LOC" not in pron.columns:
    pron["GANO_LOC"] = float("nan")
    pron["ACIERTO"] = float("nan")

proximos = pron[pron["GANO_LOC"].isna()]
jugados = pron[pron["GANO_LOC"].notna()]

columnas = {
    "GAME_DATE": "Fecha",
    "TEAM_NAME_LOC": "Local",
    "TEAM_NAME_VIS": "Visitante",
    "EQUIPO_ELEGIDO": "Pronóstico",
    "PROBABILIDAD": "Probabilidad",
}

st.subheader("Próximos partidos")
if proximos.empty:
    st.info("No hay pronósticos pendientes.")
else:
    t = proximos.sort_values(["GAME_DATE", "CONFIANZA"], ascending=[True, False]).copy()
    t["Nota"] = t["CONFIANZA"].apply(lambda c: "Partido parejo" if c < 0.55 else "")
    st.dataframe(t[list(columnas) + ["Nota"]].rename(columns=columnas), hide_index=True)

st.subheader("Historial real")
st.caption("Pronósticos guardados antes de cada partido y comparados con el resultado.")
if jugados.empty:
    st.info("Todavía no hay partidos jugados con pronóstico guardado. Aparecerán aquí cuando terminen.")
else:
    c1, c2 = st.columns(2)
    c1.metric("Partidos", len(jugados))
    c2.metric("Aciertos", f"{jugados['ACIERTO'].mean():.1%}")
    st.caption("Con pocos partidos el porcentaje varía mucho. Hacen falta varias semanas para sacar conclusiones.")
    h = jugados.sort_values("GAME_DATE", ascending=False).copy()
    h["Resultado"] = h["ACIERTO"].map({1.0: "✅ Acierto", 0.0: "❌ Fallo"})
    st.dataframe(h[list(columnas) + ["Resultado"]].rename(columns=columnas), hide_index=True)

with st.expander("Prueba histórica: temporada 2023-24"):
    hist = pd.read_csv("historial_predicciones.csv")
    elegido = hist["PRONOSTICO"] == 1
    hist["CONFIANZA"] = hist["PROB_LOCAL"].where(elegido, 1 - hist["PROB_LOCAL"])
    hist["RANGO"] = pd.cut(
        hist["CONFIANZA"], [0.5, 0.6, 0.7, 0.8, 1.0],
        labels=["50-60%", "60-70%", "70-80%", "80-100%"],
    )
    tabla = hist.groupby("RANGO", observed=True)["ACIERTO"].agg(["count", "mean"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=tabla, x="RANGO", y="mean", ax=ax)
    for i, fila in tabla.iterrows():
        ax.text(i, fila["mean"] + 0.01, f"{fila['mean']:.0%}\n({fila['count']} partidos)", ha="center")
    ax.axhline(0.5, linestyle="--", color="gray")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confianza del modelo")
    ax.set_ylabel("Porcentaje de aciertos")
    st.pyplot(fig)
    st.caption("Simulación con una temporada ya jugada, no son pronósticos publicados antes de los partidos.")