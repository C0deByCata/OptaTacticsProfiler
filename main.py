import os
import ssl
import time
import random
import logging
from typing import List

from soccerdata.whoscored import WhoScored
import pandas as pd

# Permitir contexto HTTPS sin verificar (si es necesario)
ssl._create_default_https_context = ssl._create_unverified_context

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)

# Parámetros de usuario (ajustar según necesidad)
LEAGUE = "ESP-La Liga"
START_YEAR = 2019
END_YEAR = 2024  # Incluido
HEADLESS = False  # True para modo headless

# Directorios de salida
CAL_DIR = os.path.join("data", "calendars")
EVENTS_DIR = os.path.join("data", "events")


def ensure_dir(path: str) -> None:
    """Crea el directorio si no existe."""
    os.makedirs(path, exist_ok=True)


# Crear directorios base
ensure_dir(CAL_DIR)
ensure_dir(EVENTS_DIR)


def process_season(season: str) -> None:
    """Descarga y guarda el calendario y eventos para una temporada dada."""
    logging.info(f"Processing season {season}")

    # Inicializar cliente solo con la temporada actual\
    ws = WhoScored(leagues=[LEAGUE], seasons=[season], headless=HEADLESS)

    # Descargar calendario\
    schedule = ws.read_schedule()
    cal_path = os.path.join(CAL_DIR, f"{season}.csv")
    if os.path.exists(cal_path):
        logging.info(f"Calendar exists for {season}, skipping.")
    else:
        schedule.to_csv(cal_path, index=False)
        logging.info(f"Saved calendar for {season} to {cal_path}")

    # Descargar eventos\
    season_event_dir = os.path.join(EVENTS_DIR, season)
    ensure_dir(season_event_dir)

    for idx, row in schedule.iterrows():
        match_id = int(row["game_id"])
        raw_name = row["game"] if "game" in row else f"match_{match_id}"
        safe_name = raw_name.replace(" ", "_").replace(":", "-").replace("/", "-")
        event_file = f"{season}_{safe_name}.csv"
        event_path = os.path.join(season_event_dir, event_file)

        if os.path.exists(event_path):
            logging.info(f"[{idx}] Event exists: {event_file}, skipping.")
            continue

        try:
            logging.info(f"[{idx}] Downloading events for match {match_id}...")
            events = ws.read_events(match_id=match_id)
            events.to_csv(event_path, index=False)
            logging.info(f"[{idx}] Saved events to {event_file}")
        except Exception as e:
            logging.error(f"Error downloading events for {match_id}: {e}")
        time.sleep(random.uniform(5, 20))


def main():
    seasons: List[str] = [f"{y}-{y + 1}" for y in range(START_YEAR, END_YEAR + 1)]
    for season in seasons:
        process_season(season)


if __name__ == "__main__":
    main()
