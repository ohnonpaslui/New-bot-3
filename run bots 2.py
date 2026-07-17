"""
run_bots.py — Runner temps réel des 3 bots de scalping (paper trading).
À chaque clôture de bougie 5m :
  1. récupère les données (une seule fois pour les 3 bots)
  2. vérifie SL/TP des positions ouvertes
  3. interroge chaque stratégie pour un nouveau signal
  4. sauvegarde l'état + commit git si quelque chose a changé

Fonctionne à l'identique :
  - en local        : python run_bots.py            (tourne en continu)
  - sur GitHub CI   : GIT_PUSH=1 MAX_RUNTIME=17700 python run_bots.py
"""

import os
import subprocess
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

from paper_engine import load_state, save_state, step
from strategies import add_common, signal_fibo_volume, signal_fvg_ifvg, signal_ote

# ----------------------------- Paramètres -----------------------------------
SYMBOL      = "PAXG/USD"     # proxy XAUUSD sur Kraken ; sinon "BTC/USD"
TIMEFRAME   = "5m"
TF_SEC      = 300
N_CANDLES   = 720            # max Kraken par requête ; > EMA200, suffisant
MAX_RUNTIME = int(os.environ.get("MAX_RUNTIME", "0"))   # secondes ; 0 = infini
GIT_PUSH    = os.environ.get("GIT_PUSH") == "1"

BOTS = {
    "ote":         signal_ote,
    "fvg_ifvg":    signal_fvg_ifvg,
    "fibo_volume": signal_fibo_volume,
}

# ----------------------------- Données --------------------------------------
def fetch_closed_candles(ex):
    """Récupère les bougies et supprime celle en cours de formation."""
    tf_ms = TF_SEC * 1000
    since = ex.milliseconds() - N_CANDLES * tf_ms
    rows = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=N_CANDLES)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol"])
    current_period = (ex.milliseconds() // tf_ms) * tf_ms
    df = df[df["ts"] < current_period]            # bougies clôturées uniquement
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.drop_duplicates("ts").reset_index(drop=True)


# ----------------------------- Git -------------------------------------------
def git_commit(message):
    try:
        paths = [p for p in ("state", "trades") if os.path.isdir(p)]
        if not paths:
            return
        subprocess.run(["git", "add", *paths], check=True)
        r = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if r.returncode != 0:                     # il y a des changements
            subprocess.run(["git", "commit", "-m", message], check=True)
            subprocess.run(["git", "pull", "--rebase"], check=False)
            subprocess.run(["git", "push"], check=True)
    except Exception as e:
        print(f"[git] échec du commit/push : {e}", flush=True)


# ----------------------------- Boucle principale -----------------------------
def main():
    start = time.time()
    ex = ccxt.kraken()
    last_ts = None
    print(f"Runner démarré — {SYMBOL} {TIMEFRAME} — bots : {', '.join(BOTS)}",
          flush=True)

    while True:
        if MAX_RUNTIME and time.time() - start > MAX_RUNTIME:
            print("Durée max atteinte, arrêt propre du runner.", flush=True)
            break

        try:
            df = fetch_closed_candles(ex)
        except Exception as e:
            print(f"[data] erreur de récupération : {e} — retry dans 30s", flush=True)
            time.sleep(30)
            continue

        if len(df) < 250:
            print("[data] pas assez de bougies, retry dans 60s", flush=True)
            time.sleep(60)
            continue

        ts = str(df["ts"].iloc[-1])
        if ts != last_ts:                          # nouvelle bougie clôturée
            last_ts = ts
            dfc = add_common(df)
            candle = df.iloc[-1]
            all_events = []

            for bot, detect in BOTS.items():
                state = load_state(bot)
                signal = None if state.get("position") else detect(dfc)
                state, changed, events = step(bot, state, candle, signal, ts)
                if changed:
                    save_state(bot, state)
                all_events += events

            now = datetime.now(timezone.utc).strftime("%H:%M")
            if all_events:
                for e in all_events:
                    print(f"{now} UTC  {e}", flush=True)
                if GIT_PUSH:
                    git_commit(" | ".join(all_events))
            else:
                print(f"{now} UTC  bougie {ts} — aucun événement", flush=True)

        # attend la prochaine clôture de bougie (+10s de marge pour l'API)
        now_s = time.time()
        sleep_s = TF_SEC - (now_s % TF_SEC) + 10
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
