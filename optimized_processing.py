"""File to process football match data and compute KPIs."""
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

# Se rellenará en tiempo de ejecución si el usuario no lo define manualmente
# home_team: Optional[str] = None
# away_team: Optional[str] = None


# --------- SETTINGS
class Settings:
    events_root: Path = Path("data/events")
    epv_file: Path = Path("epv_data.csv")
    output_dir: Path = Path("processed_data")

# --------- MAIN PIPELINE
# def process_season_folder(season_dir: Path, epv_array: np.ndarray, output_file: Path):
#     header_written = False
#     for match_file in sorted(season_dir.glob("*.csv")):
#         match_id = match_file.stem
#         try:
#             # 1. Leer partido
#             events = pd.read_csv(match_file)
#             events["match_id"] = match_id

#             # 2. Procesar eventos enriquecidos con tu pipeline (adaptar si alguna función cambia nombre)
#             enriched = procesar_partido_con_xt(events, epv_array)
#             enriched = add_kpi_features(enriched)
#             enriched["match_id"] = match_id  # asegurar columna en salida

#             # 3. Guardar incrementalmente
#             enriched.to_csv(output_file, mode='a', index=False, header=not header_written)
#             header_written = True

#         except Exception as e:
#             logging.error(f"Error processing {match_file}: {e}")

def process_season_folder(season_dir: Path, epv_array: np.ndarray, output_file: Path):
    # DEDUCE EL NOMBRE DEL ARCHIVO DE CALENDARIO EN BASE AL NOMBRE DE TEMPORADA
    season = season_dir.name
    calendar_file = Path(f"data/calendars/{season}.csv")
    calendar = pd.read_csv(calendar_file)

    header_written = False
    for match_file in sorted(season_dir.glob("*.csv")):
        match_id = match_file.stem
        logging.info(f"Procesando partido {match_id} en {match_file}")
        try:
            # Leer partido
            events = pd.read_csv(match_file)

            # --- IDENTIFICAR EL GAME_ID PARA CRUZARLO CON EL CALENDARIO ---
            # Suponemos que tienes la columna 'game_id' en los eventos o puedes obtenerla del nombre del archivo.
            # Si tu columna es diferente, AJUSTA el nombre aquí.
            if "game_id" in events.columns:
                game_id = events["game_id"].iloc[0]
            else:
                # Si no tienes columna, intenta extraerlo del nombre
                game_id = match_id  # adaptar según cómo esté en tus ficheros

            # Buscar los datos del calendario
            cal_row = calendar[calendar["game_id"] == int(game_id)]
            if cal_row.empty:
                raise ValueError(f"game_id {game_id} no encontrado en calendario {calendar_file}")

            home_team_id = cal_row["home_team_id"].iloc[0]
            away_team_id = cal_row["away_team_id"].iloc[0]

            # Añadir estas columnas A TODO EL DF DE EVENTOS
            events["home_team_id"] = home_team_id
            events["away_team_id"] = away_team_id

            # Procesar eventos enriquecidos con tu pipeline
            enriched = procesar_partido_con_xt(events, epv_array)
            enriched = add_kpi_features(enriched)

            # Guardar incrementalmente
            enriched.to_csv(output_file, mode='a', index=False, header=not header_written)
            header_written = True

        except Exception as e:
            logging.error(f"Error processing {match_file}: {e}")

def _infer_home_away(df: pd.DataFrame) -> Tuple[int, int]:
    if {"home_team_id", "away_team_id"}.issubset(df.columns):
        return int(df["home_team_id"].iloc[0]), int(df["away_team_id"].iloc[0])
    else:
        raise ValueError("No se encuentran las columnas home_team_id y away_team_id en el DataFrame de eventos")



def reconstruct_carries(events, min_dist=5, max_time=10):
    # Supone que events ya tiene 'possession_id_team' y está ordenado
    nxt = events.shift(-1)

    same_player = events.player_id == nxt.player_id
    same_team   = events.team_id   == nxt.team_id
    same_period = events.period    == nxt.period
    same_poss   = events.possession_id_team == nxt.possession_id_team

    dt   = (nxt.minute*60 + nxt.second) - (events.minute*60 + events.second)
    dist = np.hypot(nxt.x - events.x, nxt.y - events.y)

    mask = same_player & same_team & same_period & same_poss & \
           dt.between(0, max_time) & (dist >= min_dist)

    carries = events.loc[mask].copy()
    carries['duration'] = dt[mask].values
    carries['distance'] = dist[mask].values
    carries['end_x'] = nxt.loc[mask, 'x'].values
    carries['end_y'] = nxt.loc[mask, 'y'].values
    carries['type']  = 'Carry'
    carries['outcome_type'] = 'Successful'   # sigue en la misma posesión

    # Asegura columnas faltantes
    for col in events.columns:
        if col not in carries.columns:
            carries[col] = pd.NA

    # Combina y devuélvelo ordenado
    out = pd.concat([events, carries], ignore_index=True)\
             .sort_values(['period','minute','second'])\
             .reset_index(drop=True)
    return out


def load_all_events_with_carries(events_file: str) -> pd.DataFrame:
    """
    Carga todos los eventos del partido y añade los eventos 'Carry' reconstruidos, sin eliminar ningún otro tipo de evento.
    """
    events = pd.read_csv(events_file)

    carries = reconstruct_carries(events)

    # Concatenar carries como eventos nuevos al DataFrame original
    all_events = pd.concat([events, carries], ignore_index=True)
    all_events = all_events.sort_values(by=['minute', 'second']).reset_index(drop=True)

    return all_events


def compute_xt_from_epv(actions: pd.DataFrame, epv_array: np.ndarray) -> pd.DataFrame:
    """
    Añade el valor de xT a cada acción en base a los valores de EPV del punto inicial y final.

    Parámetros:
    -----------
    actions : DataFrame
        Acciones de tipo 'Pass' o 'Carry' con coordenadas iniciales y finales.
    epv_array : ndarray
        Matriz 2D con los valores de EPV en formato [y, x].

    Retorna:
    --------
    DataFrame original con columnas nuevas: 'start_zone_value', 'end_zone_value', 'xt_value'
    """
    actions = actions.copy()
    n_rows, n_cols = epv_array.shape

    # Bin de coordenadas (0-100 -> índices en matriz EPV)
    actions['x1_bin'] = np.clip((actions['x'] // (100 / n_cols)).astype(int), 0, n_cols-1)
    actions['x2_bin'] = np.clip((actions['end_x'] // (100 / n_cols)).astype(int), 0, n_cols-1)
    actions['y1_bin'] = np.clip((actions['y'] // (100 / n_rows)).astype(int), 0, n_rows-1)
    actions['y2_bin'] = np.clip((actions['end_y'] // (100 / n_rows)).astype(int), 0, n_rows-1)

    # Asignar valores EPV
    actions['start_zone_value'] = [epv_array[y, x] for y, x in zip(actions['y1_bin'], actions['x1_bin'])]
    actions['end_zone_value'] = [epv_array[y, x] for y, x in zip(actions['y2_bin'], actions['x2_bin'])]
    actions['xt_value'] = actions['end_zone_value'] - actions['start_zone_value']

    return actions


def annotate_pass_recipients_next_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna 'pass_recipient_name' en los pases exitosos, tomando
    exclusivamente el jugador que participa en el primer evento posterior
    del mismo equipo y mismo periodo, sin considerar aún las posesiones.

    Requisitos:
    • El DataFrame debe estar ordenado cronológicamente por ['period','minute','second'].
    """
    df = df.copy()
    df['pass_recipient_name'] = pd.NA

    # Índices de pases exitosos
    passes_idx = df[(df['type'] == 'Pass') &
                    (df['outcome_type'] == 'Successful')].index

    for idx in passes_idx:
        row = df.loc[idx]
        team   = row['team']
        period = row['period']

        # Primer evento posterior (idx+1, idx+2, …) del mismo equipo y periodo
        nxt_idx = idx + 1
        while nxt_idx < len(df):
            nxt = df.loc[nxt_idx]
            if (nxt['period'] != period):          # cambió de tiempo
                break
            if (nxt['team'] == team):
                # ¡Encontrado receptor!
                df.at[idx, 'pass_recipient_name'] = nxt['player']
                break
            nxt_idx += 1

    return df


def etiquetar_zonas(df: pd.DataFrame) -> pd.DataFrame:
    def obtener_zona(x, y):
        if pd.isna(x) or pd.isna(y):
            return np.nan
        col = int(x // (100 / 6))
        row = int(y // (100 / 3))
        labels = [[3, 6, 9, 12, 15, 18], [2, 5, 8, 11, 14, 17], [1, 4, 7, 10, 13, 16]]
        row = min(row, 2)
        col = min(col, 5)
        return labels[row][col]

    def obtener_tercio(x):
        if pd.isna(x):
            return np.nan
        if x < 100/3:
            return 'Defensivo'
        elif x < 2*100/3:
            return 'Medio'
        else:
            return 'Ofensivo'

    def obtener_intermedio(y):
        if pd.isna(y):
            return np.nan
        if y < 21.1:
            return 'Banda Derecha'
        elif y < 36.8:
            return 'Intermedio Derecho'
        elif y < 63.2:
            return 'Zona Central'
        elif y < 78.9:
            return 'Intermedio Izquierdo'
        else:
            return 'Banda Izquierda'

    for pos in ['start', 'end']:
        df[f'zona_{pos}'] = df.apply(lambda row: obtener_zona(row['x'] if pos == 'start' else row['end_x'],
                                                              row['y'] if pos == 'start' else row['end_y']), axis=1)
        df[f'tercio_{pos}'] = df.apply(lambda row: obtener_tercio(row['x'] if pos == 'start' else row['end_x']), axis=1)
        df[f'intermedio_{pos}'] = df.apply(lambda row: obtener_intermedio(row['y'] if pos == 'start' else row['end_y']), axis=1)

    return df

def assign_possession_ids_by_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Similar a assign_possession_ids, pero fuerza que al cambiar de period
    arranque una nueva posesión (aunque sea el mismo equipo).
    """
    df = df.sort_values(['period', 'minute', 'second', 'total_minute']).reset_index(drop=True)

    prev_team   = df['team'].shift(1)
    prev_period = df['period'].shift(1)
    df['possession_change'] = (
        (df['team']   != prev_team) |
        (df['period'] != prev_period)
    ).astype(int)

    # ID global (1,2,3…)
    df['possession_id'] = df['possession_change'].cumsum()
    # ID por equipo: cada vez que ese equipo “inicia” una posesión (+1)
    df['possession_id_team'] = df.groupby('team')['possession_change'].cumsum()

    df.drop(columns='possession_change', inplace=True)
    return df




def procesar_partido_con_xt(events: pd.DataFrame, epv_array: np.ndarray) -> pd.DataFrame:
    # Ya no leer desde CSV, solo procesar

    events = events.sort_values(['period','minute','second']).reset_index(drop=True)

    # 3  tiempo total
    events['total_minute'] = events['minute'] + events['second'] / 60

    # 4  posesiones
    events = assign_possession_ids_by_period(events)

    # 5  carries (misma posesión, mismo equipo, ≥5 m, ≤10 s)
    events = reconstruct_carries(events)          # usa la versión con same_possession

    # 6  xT por acción
    mask = events[['x','y','end_x','end_y']].notnull().all(axis=1)
    events.loc[mask, ['start_zone_value','end_zone_value','xt_value']] = \
        compute_xt_from_epv(events.loc[mask], epv_array)[
            ['start_zone_value','end_zone_value','xt_value']]

    # 7  receptores
    events = annotate_pass_recipients_next_event(events)

    # 8  zonas / tercios
    events = etiquetar_zonas(events)

    # 9  cuantiles xT
    if 'xt_value' in events.columns:
        ok = events['xt_value'].notnull()
        events.loc[ok, 'xT_quantile'] = pd.qcut(events.loc[ok, 'xt_value'],
                                                q=4, labels=False, duplicates='drop')

    pos_xt = (events
                .groupby(['period', 'possession_id'], as_index=False)['xt_value']
                .sum()
                .rename(columns={'xt_value': 'xt_possession'}))

    events = events.merge(pos_xt,
                            on=['period', 'possession_id'],
                            how='left')

    # ---------- 11  xT por tercio dentro de la posesión ----------
    tercio_xt = (events
        .groupby(['period', 'possession_id', 'tercio_start'], as_index=False)['xt_value']
        .sum()
        .pivot_table(index=['period', 'possession_id'],
                        columns='tercio_start', values='xt_value',
                        fill_value=0)
        .reset_index()
        .rename(columns={'Defensivo': 'xt_Defensivo',
                         'Medio':      'xt_Medio',
                         'Ofensivo':   'xt_Ofensivo'}))

    for col in ['xt_Defensivo', 'xt_Medio', 'xt_Ofensivo']:
        if col not in tercio_xt.columns:
            tercio_xt[col] = 0

    events = events.merge(tercio_xt,
                          on=['period', 'possession_id'],
                          how='left')

    # 12  elimina PreMatch / PostGame
    events = events[~events['period'].isin(['PreMatch', 'PostGame'])]


    equipos = events['team'].unique()
    team_ids = events['team_id'].unique()
    events['rival_team'] = events['team'].apply(lambda t: [e for e in equipos if e != t][0] if len(equipos) == 2 else pd.NA)
    events['rival_team_id'] = events['team_id'].apply(lambda tid: [i for i in team_ids if i != tid][0] if len(team_ids) == 2 else pd.NA)

    return events



# ----------------------------------------------------------------------------
# 1. EVENT‑LEVEL KPIs GEOMÉTRICOS / DIRECCIONALES
# ----------------------------------------------------------------------------

def _event_level_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve copia de *df* con KPIs evento‑a‑evento añadidos."""

    ev = df.copy()

    # --- coordenadas y distancias
    ev["dx"] = ev["end_x"] - ev["x"]
    ev["dy"] = ev["end_y"] - ev["y"]
    ev["distance_xy"] = np.hypot(ev["dx"], ev["dy"])

    # --- ángulo y verticalidad (0 rad → portería rival)
    ev["pass_angle"] = np.mod(np.arctan2(ev["dy"], ev["dx"]), 2 * np.pi)
    ev["verticality"] = (ev["dx"].abs() / ev["distance_xy"]).where(ev["distance_xy"] > 0)

    # --- switch_flag
    lado_der = {"Banda Derecha", "Intermedio Derecho"}
    lado_izq = {"Banda Izquierda", "Intermedio Izquierdo"}
    ev["switch_flag"] = (
        (ev["intermedio_start"].isin(lado_der) & ev["intermedio_end"].isin(lado_izq))
        | (ev["intermedio_start"].isin(lado_izq) & ev["intermedio_end"].isin(lado_der))
    ).astype("int8")

    # --- progresión y velocidad de carries
    ev["progressive_dx"] = ev["dx"].clip(lower=0)
    if {"distance", "duration"}.issubset(ev.columns):
        ev["progressive_speed"] = ev["distance"] / ev["duration"]
    else:
        ev["progressive_speed"] = np.nan

    ev["tercio_changed"] = (ev["tercio_start"] != ev["tercio_end"]).astype("int8")

    return ev

# ----------------------------------------------------------------------------
# 2. MARCADOR VIVO → arrays de score
# ----------------------------------------------------------------------------

# def _infer_home_away(df: pd.DataFrame) -> Tuple[str, str]:
#     """Determina (home_team, away_team).
#     Estrategia:
#       1. Si el usuario lo fijó manualmente -> usarlo.
#       2. Si existen columnas "home_team" y "away_team" en df -> usar.
#       3. Por convención, el primer equipo que aparece en el archivo es local.
#     """

#     global home_team, away_team

#     if home_team and away_team:
#         return home_team, away_team

#     if {"home_team", "away_team"}.issubset(df.columns):
#         home_team = df["home_team"].iloc[0]
#         away_team = df["away_team"].iloc[0]
#     else:
#         teams = list(df["team"].unique())
#         if len(teams) != 2:
#             raise ValueError("No se pueden determinar los equipos; se han encontrado {}".format(teams))
#         home_team, away_team = teams[0], teams[1]

#     return home_team, away_team


def _running_score_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_team_id, a_team_id = _infer_home_away(df)
    score = {h_team_id: 0, a_team_id: 0}
    n = len(df)
    home_g = np.empty(n, dtype="int8")
    away_g = np.empty(n, dtype="int8")
    gf = np.empty(n, dtype="int8")
    ga = np.empty(n, dtype="int8")

    for i, row in enumerate(df.itertuples(index=False)):
        team_id = row.team_id
        if team_id not in score:
            raise ValueError(f"Evento con team_id desconocido: {team_id}. Esperados: {list(score.keys())}")

        opp_id  = a_team_id if team_id == h_team_id else h_team_id

        home_g[i] = score[h_team_id]
        away_g[i] = score[a_team_id]
        gf[i] = score[team_id]
        ga[i] = score[opp_id]

        is_goal = (getattr(row, "type_id", np.nan) == 16) or (row.type == "Goal")
        if is_goal:
            score[team_id] += 1

    return home_g, away_g, gf, ga


# ----------------------------------------------------------------------------
# 3. KPIs DE POSESIÓN
# ----------------------------------------------------------------------------

def _possession_kpis(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["period", "possession_id"], sort=False)

    agg = grp.agg(
        events_in_possession=("type", "size"),
        num_passes=("type", lambda s: (s == "Pass").sum()),
        num_carries=("type", lambda s: (s == "Carry").sum()),
        successful_passes=("outcome_type", lambda s: (s == "Successful").sum()),
        total_progressive_dx=("progressive_dx", "sum"),
        duels_won=(
            "type",
            lambda s: (
                (s.isin(["Aerial", "Challenge"]) & (df.loc[s.index, "outcome_type"] == "Successful")).sum()
            ),
        ),
        duels_lost=(
            "type",
            lambda s: (
                (s.isin(["Aerial", "Challenge"]) & (df.loc[s.index, "outcome_type"] == "Unsuccessful")).sum()
            ),
        ),
        possession_start_min=("total_minute", "min"),
        possession_end_min=("total_minute", "max"),
    )

    agg["pct_passes_completed"] = agg["successful_passes"] / agg["num_passes"].replace(0, np.nan)
    agg["duels_lost_won_ratio"] = agg["duels_lost"] / (agg["duels_won"] + 1)
    agg["possession_duration"] = agg["possession_end_min"] - agg["possession_start_min"]

    agg["minute_bucket"] = pd.cut(
        agg["possession_start_min"],
        bins=[0, 15, 30, 45, 60, 75, 90, 120],
        labels=["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"],
        right=False,
    )

    # ---- marcador al comienzo de la posesión (home/away y relativo)
    first_evt = grp[
        [
            "home_goals_event",
            "away_goals_event",
            "score_state_event",
            "score_diff_event",
            "goals_for_event",
            "goals_against_event",
        ]
    ].first().reset_index()

    first_evt["scoreline_pos_match"] = (
        first_evt["home_goals_event"].astype(str) + "-" + first_evt["away_goals_event"].astype(str)
    )

    rename = {
        "home_goals_event": "home_goals_pos",
        "away_goals_event": "away_goals_pos",
        "score_state_event": "score_state",
        "score_diff_event": "score_diff_pos",
        "goals_for_event": "goals_for_pos",
        "goals_against_event": "goals_against_pos",
    }

    agg = agg.merge(first_evt.rename(columns=rename), on=["period", "possession_id"], how="left")

    return agg

# ----------------------------------------------------------------------------
# 4. API PRINCIPAL
# ----------------------------------------------------------------------------

def add_kpi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un nuevo DataFrame con KPIs evento‑a‑evento y de posesión.
    Ahora incluye marcador estable Home/Away.
    """

    ev = df.sort_values(["period", "minute", "second", "total_minute"], ignore_index=True).copy()

    # 1) KPIs geométricos
    ev = _event_level_kpis(ev)

    # 2) Marcador vivo
    if {"home_goals_event", "away_goals_event"}.isdisjoint(ev.columns):
        home_g, away_g, gf, ga = _running_score_arrays(ev)
        ev["home_goals_event"] = home_g
        ev["away_goals_event"] = away_g
        ev["scoreline_event_match"] = ev["home_goals_event"].astype(str) + "-" + ev["away_goals_event"].astype(str)

        ev["goals_for_event"] = gf
        ev["goals_against_event"] = ga
        ev["score_diff_event"] = gf - ga
        ev["score_state_event"] = np.sign(ev["score_diff_event"]).astype("int8")

    # 3) KPIs de posesión
    pos = _possession_kpis(ev)
    ev = ev.merge(pos, on=["period", "possession_id"], how="left", suffixes=("", "_dup"))
    ev.drop(columns=[c for c in ev.columns if c.endswith("_dup")], inplace=True)

    return ev

# EOF



def main():
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar EPV una sola vez
    epv_array = pd.read_csv(settings.epv_file, header=None).to_numpy()

    for season_dir in sorted(settings.events_root.iterdir()):
        if not season_dir.is_dir():
            continue
        season = season_dir.name
        output_file = settings.output_dir / f"{season}.csv"
        logging.info(f"Procesando temporada {season} → {output_file}")

        # Si quieres limpiar el archivo previo, descomenta:
        # if output_file.exists(): output_file.unlink()

        process_season_folder(season_dir, epv_array, output_file)
        logging.info(f"Terminado {season}, datos en {output_file}")

if __name__ == "__main__":
    main()









# __all__ = [
#     "add_kpi_features",
#     "home_team",
#     "away_team",
# ]


