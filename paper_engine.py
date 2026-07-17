"""
paper_engine.py — Moteur de paper trading partagé par les 3 bots.
Gère : capital simulé, position ouverte, SL/TP, journal des trades.
État persisté dans state/<bot>.json, trades dans trades/<bot>.csv
"""

import csv
import json
import os

STATE_DIR      = "state"
TRADES_DIR     = "trades"
START_CAPITAL  = 1000.0
RISK_PER_TRADE = 1.0     # % du capital risqué par trade
FEES_PCT       = 0.05    # % par côté


def load_state(bot):
    path = os.path.join(STATE_DIR, f"{bot}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"capital": START_CAPITAL, "position": None, "wins": 0, "losses": 0}


def save_state(bot, state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{bot}.json"), "w") as f:
        json.dump(state, f, indent=2)


def append_trade(bot, row):
    os.makedirs(TRADES_DIR, exist_ok=True)
    path = os.path.join(TRADES_DIR, f"{bot}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def step(bot, state, candle, signal, ts):
    """
    Un cycle pour un bot : d'abord la gestion de la position ouverte
    (SL/TP sur la bougie qui vient de clôturer), puis l'ouverture
    éventuelle sur signal. Retourne (state, changed, events).
    """
    changed, events = False, []
    pos = state.get("position")

    # ---- sortie SL / TP (jamais sur la bougie d'entrée) ----
    if pos and pos.get("opened_ts") != ts:
        if pos["side"] == "sell":
            hit_sl = candle["high"] >= pos["sl"]
            hit_tp = candle["low"]  <= pos["tp"]
        else:
            hit_sl = candle["low"]  <= pos["sl"]
            hit_tp = candle["high"] >= pos["tp"]

        if hit_sl or hit_tp:
            r = -1.0 if hit_sl else pos["rr"]        # SL prioritaire si les deux
            pnl = state["capital"] * (RISK_PER_TRADE / 100) * r
            pnl -= state["capital"] * (RISK_PER_TRADE / 100) * FEES_PCT / 100 * 2
            state["capital"] = round(state["capital"] + pnl, 2)
            if r > 0:
                state["wins"] = state.get("wins", 0) + 1
            else:
                state["losses"] = state.get("losses", 0) + 1
            append_trade(bot, {
                "opened": pos["opened_ts"], "closed": ts, "side": pos["side"],
                "entry": pos["entry"], "sl": pos["sl"], "tp": pos["tp"],
                "result": "TP" if r > 0 else "SL", "r": r,
                "pnl": round(pnl, 2), "capital": state["capital"],
            })
            state["position"] = None
            pos = None
            changed = True
            events.append(f"[{bot}] {'TP' if r > 0 else 'SL'} touché "
                          f"({r:+.1f}R) — capital {state['capital']:.2f}$")

    # ---- entrée sur signal ----
    if pos is None and signal:
        risk = abs(signal["entry"] - signal["sl"])
        rr = round(abs(signal["tp"] - signal["entry"]) / risk, 2) if risk > 0 else 0
        state["position"] = {**signal, "rr": rr, "opened_ts": ts}
        changed = True
        events.append(f"[{bot}] {signal['side'].upper()} ouvert @ {signal['entry']:.2f} "
                      f"(SL {signal['sl']:.2f} / TP {signal['tp']:.2f})")

    return state, changed, events
