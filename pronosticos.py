import pandas as pd
import os
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Entrenar el modelo con todas las temporadas históricas
hist = pd.read_csv("partidos_varias_temporadas.csv")
hist["GAME_DATE"] = pd.to_datetime(hist["GAME_DATE"])
hist = hist.sort_values(["TEAM_ID", "GAME_DATE"])
hist["GANO"] = (hist["WL"] == "W").astype(int)
hist["LOCAL"] = hist["MATCHUP"].str.contains("vs.")

grupo = hist.groupby(["TEAM_ID", "SEASON_ID"])
hist["MARGEN10"] = grupo["PLUS_MINUS"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=5).mean()
)
hist["WIN_TEMP"] = grupo["GANO"].transform(
    lambda s: s.shift(1).expanding(min_periods=5).mean()
)
hist["MARGEN_TEMP"] = grupo["PLUS_MINUS"].transform(
    lambda s: s.shift(1).expanding(min_periods=5).mean()
)
hist["DESCANSO"] = grupo["GAME_DATE"].diff().dt.days.clip(upper=4)

base = ["MARGEN10", "WIN_TEMP", "MARGEN_TEMP", "DESCANSO"]
cols = ["GAME_ID", "GAME_DATE", "SEASON_ID", "GANO"] + base
local = hist[hist["LOCAL"]][cols]
visitante = hist[~hist["LOCAL"]][cols]
entrenamiento = local.merge(visitante, on=["GAME_ID", "GAME_DATE", "SEASON_ID"], suffixes=("_LOC", "_VIS")).dropna()

for v in base:
    entrenamiento["DIF_" + v] = entrenamiento[v + "_LOC"] - entrenamiento[v + "_VIS"]

variables = ["DIF_" + v for v in base]
modelo = make_pipeline(StandardScaler(), LogisticRegression())
modelo.fit(entrenamiento[variables], entrenamiento["GANO_LOC"])
print("Modelo entrenado con", len(entrenamiento), "partidos")

# 2. Estadísticas actuales de cada equipo (temporada 2026-27)
act = pd.read_csv("partidos_2026_27.csv")
act["GAME_DATE"] = pd.to_datetime(act["GAME_DATE"])
act["GANO"] = (act["WL"] == "W").astype(int)
act = act.sort_values(["TEAM_ID", "GAME_DATE"])

stats = {}
for tid, g in act.groupby("TEAM_ID"):
    if len(g) >= 5:
        stats[tid] = {
            "MARGEN10": g["PLUS_MINUS"].tail(10).mean(),
            "WIN_TEMP": g["GANO"].mean(),
            "MARGEN_TEMP": g["PLUS_MINUS"].mean(),
            "FECHAS": list(g["GAME_DATE"]),
        }

# 3. Partidos que vienen (se excluyen los que ya aparecen como jugados)
prox = pd.read_csv("partidos_proximos.csv")
prox = prox.drop_duplicates(subset="GAME_ID")
prox["GAME_DATE"] = pd.to_datetime(prox["GAME_DATE"])
prox = prox[~prox["GAME_ID"].isin(act["GAME_ID"])]


def descanso(tid, fecha):
    programados = prox.loc[(prox["LOCAL_ID"] == tid) | (prox["VISITANTE_ID"] == tid), "GAME_DATE"]
    fechas = stats[tid]["FECHAS"] + list(programados)
    anteriores = [f for f in fechas if f < fecha]
    if not anteriores:
        return None
    return min((fecha - max(anteriores)).days, 4)


filas = []
for _, p in prox.iterrows():
    l, v = p["LOCAL_ID"], p["VISITANTE_ID"]
    if l not in stats or v not in stats:
        continue
    dl, dv = descanso(l, p["GAME_DATE"]), descanso(v, p["GAME_DATE"])
    if dl is None or dv is None:
        continue
    filas.append({
        "GAME_ID": p["GAME_ID"],
        "GAME_DATE": p["GAME_DATE"],
        "TEAM_NAME_LOC": p["TEAM_NAME_LOC"],
        "TEAM_NAME_VIS": p["TEAM_NAME_VIS"],
        "DIF_MARGEN10": stats[l]["MARGEN10"] - stats[v]["MARGEN10"],
        "DIF_WIN_TEMP": stats[l]["WIN_TEMP"] - stats[v]["WIN_TEMP"],
        "DIF_MARGEN_TEMP": stats[l]["MARGEN_TEMP"] - stats[v]["MARGEN_TEMP"],
        "DIF_DESCANSO": dl - dv,
    })

nuevos = pd.DataFrame(filas)
nuevos["PROB_LOCAL"] = modelo.predict_proba(nuevos[variables])[:, 1]
nuevos["FECHA_PRONOSTICO"] = pd.Timestamp(date.today())
nuevos = nuevos[["GAME_ID", "GAME_DATE", "TEAM_NAME_LOC", "TEAM_NAME_VIS", "PROB_LOCAL", "FECHA_PRONOSTICO"]]

# 4. Mostrar
vista = nuevos.copy()
vista["ELEGIDO"] = vista["TEAM_NAME_LOC"].where(vista["PROB_LOCAL"] >= 0.5, vista["TEAM_NAME_VIS"])
vista["CONFIANZA"] = (vista[["PROB_LOCAL"]].assign(o=1 - vista["PROB_LOCAL"]).max(axis=1) * 100).round(0).astype(int)
print(vista[["GAME_DATE", "TEAM_NAME_LOC", "TEAM_NAME_VIS", "ELEGIDO", "CONFIANZA"]].to_string(index=False))

archivo = "pronosticos_guardados.csv"
if os.path.exists(archivo):
    viejos = pd.read_csv(archivo)
    viejos["GAME_DATE"] = pd.to_datetime(viejos["GAME_DATE"])
    viejos = viejos[~viejos["GAME_ID"].isin(nuevos["GAME_ID"])]
    nuevos = pd.concat([viejos, nuevos], ignore_index=True)
nuevos.to_csv(archivo, index=False)
print("Guardado", archivo, "con", len(nuevos), "pronósticos")