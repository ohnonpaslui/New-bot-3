"""
backtest-multi-actifs.py : fibo-volume (SL candle et floor) sur plusieurs actifs.

Objectif : verifier si l'edge de fibo-floor, valide sur l'or (PAXG), existe
aussi sur d'autres actifs avant toute adaptation multi-actifs du bot live.
Memes criteres de verdict que les backtests precedents : R total positif,
aucune annee fortement negative, nombre de trades suffisant.

Ne touche pas a backtest-3ans.py (labo OTE/FVG/fibo conserve tel quel).

Donnees : 5m via ccxt (Binance, puis Gate, puis KuCoin en secours),
cache local data-<actif>-5m.csv par actif.

Usage  : python backtest-multi-actifs.py
Sortie : resultats-backtest/
           synthese-multi-actifs.csv
           ventilation-multi-actifs.csv
           trades-fibo-<mode>-<actif>.csv
"""

import os
import sys
import time

import ccxt
import numpy as np
import pandas as pd

# ----------------------------- Parametres ------------------------------------
SYMBOLS   = ["PAXG/USDT", "BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "5m"
TF_MS     = 5 * 60 * 1000
YEARS     = 3
OUT_DIR   = "resultats-backtest"

START_CAPITAL  = 1000.0
RISK_PER_TRADE = 1.0     # % du capital risque par trade
FEES_PCT       = 0.05    # % par cote (identique a paper_engine)
MIN_RISK_PCT   = 0.05    # distance SL minimale en % du prix (filtre micro-trades)

EMA_TREND = 200
PIVOT_N   = 5
FIBO = dict(N_BINS=30, POC_ZONE_BINS=1, BODY_MAX=0.35, SL_BUFFER=0.10,
            MIN_LEG_PCT=0.15, RR=1.5)
ATR_FLOOR = 1.0          # plancher de distance SL du mode "floor", en ATR(14)
SL_MODES  = ["candle", "floor"]


# ----------------------------- Donnees ---------------------------------------
def get_exchange():
    for name in ("binance", "gateio", "kucoin"):
        try:
            ex = getattr(ccxt, name)()
            ex.load_markets()
            if any(s in ex.markets for s in SYMBOLS):
                print(f"[data] exchange utilise : {name}")
                return ex
        except Exception as e:
            print(f"[data] {name} indisponible ({type(e).__name__}), essai suivant...")
    sys.exit("Aucun exchange accessible. Verifie ta connexion.")


def download_data(ex, symbol, slug):
    data_file = f"data-{slug}-5m.csv"
    target_start = ex.milliseconds() - YEARS * 365 * 24 * 3600 * 1000

    rows = []
    since = target_start
    if os.path.exists(data_file):
        cached = pd.read_csv(data_file)
        if len(cached):
            rows = cached.values.tolist()
            since = int(cached["ts"].iloc[-1]) + TF_MS
            print(f"[data] cache {slug} : {len(rows)} bougies, reprise...")

    end = ex.milliseconds()
    while since < end:
        try:
            batch = ex.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
        except Exception as e:
            print(f"[data] erreur ({e}), retry dans 10s")
            time.sleep(10)
            continue
        if not batch:
            break
        rows += batch
        new_since = batch[-1][0] + TF_MS
        if new_since <= since:
            break
        since = new_since
        if len(rows) % 50000 < 1000:
            print(f"[data] {slug} : {len(rows)} bougies "
                  f"(jusqu'a {pd.to_datetime(batch[-1][0], unit='ms')})", flush=True)
        time.sleep(ex.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_csv(data_file, index=False)
    print(f"[data] {slug} : {len(df)} bougies au total")
    return df


# ----------------------------- Indicateurs -----------------------------------
def add_common(df):
    df = df.copy()
    df["ema"] = df["close"].ewm(span=EMA_TREND, adjust=False).mean()
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()   # ATR(14) Wilder
    n = PIVOT_N
    df["pivot_high"] = df["high"][(df["high"] == df["high"].rolling(2 * n + 1, center=True).max())]
    df["pivot_low"]  = df["low"][(df["low"]  == df["low"].rolling(2 * n + 1, center=True).min())]
    return df


# ----------------------------- Moteur (copie de paper_engine) -----------------
class Engine:
    def __init__(self, bot, ts_labels):
        self.bot = bot
        self.ts = ts_labels
        self.capital = START_CAPITAL
        self.pos = None
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.closed_i = -1
        self.rejected = 0

    def on_candle(self, i, high, low):
        p = self.pos
        if p and p["i"] != i:                     # jamais sur la bougie d'entree
            if p["side"] == "sell":
                hit_sl = high >= p["sl"]
                hit_tp = low <= p["tp"]
            else:
                hit_sl = low <= p["sl"]
                hit_tp = high >= p["tp"]
            if hit_sl or hit_tp:
                r = -1.0 if hit_sl else p["rr"]   # SL prioritaire si les deux
                pnl = self.capital * (RISK_PER_TRADE / 100) * r
                pnl -= self.capital * (RISK_PER_TRADE / 100) * FEES_PCT / 100 * 2
                self.capital = round(self.capital + pnl, 2)
                if r > 0:
                    self.wins += 1
                else:
                    self.losses += 1
                self.trades.append({
                    "opened": self.ts[p["i"]], "closed": self.ts[i],
                    "side": p["side"], "entry": p["entry"],
                    "sl": p["sl"], "tp": p["tp"],
                    "result": "TP" if r > 0 else "SL", "r": r,
                    "pnl": round(pnl, 2), "capital": self.capital,
                })
                self.pos = None
                self.closed_i = i

    def try_open(self, i, side, entry, sl, tp):
        if self.pos is not None or self.closed_i == i:
            return
        risk = abs(entry - sl)
        if entry > 0 and risk / entry * 100 < MIN_RISK_PCT:
            self.rejected += 1
            return
        rr = round(abs(tp - entry) / risk, 2) if risk > 0 else 0
        self.pos = {"side": side, "entry": entry, "sl": sl, "tp": tp,
                    "rr": rr, "i": i}


# ----------------------------- Strategie fibo ---------------------------------
def _poc(l, h, v, i0, i1, lo, hi, n_bins, zone_bins):
    if hi <= lo:
        return None, None
    edges = np.linspace(lo, hi, n_bins + 1)
    vols = np.zeros(n_bins)
    for k in range(i0, i1 + 1):
        c_lo = max(l[k], lo)
        c_hi = min(h[k], hi)
        vk = v[k]
        if c_hi <= c_lo or vk <= 0:
            continue
        overlap = np.clip(np.minimum(edges[1:], c_hi) - np.maximum(edges[:-1], c_lo), 0, None)
        vols += vk * overlap / (c_hi - c_lo)
    b = int(vols.argmax())
    size = (hi - lo) / n_bins
    return (lo + max(b - zone_bins, 0) * size,
            lo + min(b + zone_bins + 1, n_bins) * size)


def bt_fibo(df, eng, sl_mode="floor"):
    P = FIBO
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    e = df["ema"].values
    v = df["vol"].values
    atr = df["atr"].values
    PH, PL = df["pivot_high"].values, df["pivot_low"].values
    setup, leg_id = None, None
    ph_i = pl_i = None

    def stop_for(side, i, entry, rng):
        if side == "buy":
            base = l[i] - P["SL_BUFFER"] * rng
            return base if sl_mode == "candle" else min(base, entry - ATR_FLOOR * atr[i])
        base = h[i] + P["SL_BUFFER"] * rng
        return base if sl_mode == "candle" else max(base, entry + ATR_FLOOR * atr[i])

    for i in range(EMA_TREND, len(df)):
        eng.on_candle(i, h[i], l[i])
        j = i - PIVOT_N
        if j >= 0:
            if not np.isnan(PH[j]):
                ph_i = j
            if not np.isnan(PL[j]):
                pl_i = j

        if ph_i is not None and pl_i is not None:
            new_leg = (ph_i, pl_i)
            if new_leg != leg_id:
                leg_id = new_leg
                setup = None
                hi, lo = PH[ph_i], PL[pl_i]
                leg = hi - lo
                if leg > 0 and leg / c[i] * 100 >= P["MIN_LEG_PCT"]:
                    eq = lo + 0.5 * leg
                    if c[i] > e[i] and pl_i < ph_i:
                        p_lo, p_hi = _poc(l, h, v, pl_i, ph_i, lo, hi,
                                          P["N_BINS"], P["POC_ZONE_BINS"])
                        if p_lo is not None and p_hi <= eq:
                            setup = {"side": "buy", "lo": lo, "hi": hi, "eq": eq,
                                     "poc_lo": p_lo, "poc_hi": p_hi}
                    elif c[i] < e[i] and ph_i < pl_i:
                        p_lo, p_hi = _poc(l, h, v, ph_i, pl_i, lo, hi,
                                          P["N_BINS"], P["POC_ZONE_BINS"])
                        if p_lo is not None and p_lo >= eq:
                            setup = {"side": "sell", "lo": lo, "hi": hi, "eq": eq,
                                     "poc_lo": p_lo, "poc_hi": p_hi}

        if setup:
            rng = h[i] - l[i]
            body = abs(c[i] - o[i])
            small = rng > 0 and body / rng <= P["BODY_MAX"]
            touch = l[i] <= setup["poc_hi"] and h[i] >= setup["poc_lo"]

            if setup["side"] == "buy":
                if c[i] < setup["lo"]:
                    setup = None
                elif small and touch and c[i] < setup["eq"]:
                    entry = c[i]
                    sl = stop_for("buy", i, entry, rng)
                    risk = entry - sl
                    if risk > 0:
                        eng.try_open(i, "buy", entry, sl, entry + P["RR"] * risk)
                    setup = None
            else:
                if c[i] > setup["hi"]:
                    setup = None
                elif small and touch and c[i] > setup["eq"]:
                    entry = c[i]
                    sl = stop_for("sell", i, entry, rng)
                    risk = sl - entry
                    if risk > 0:
                        eng.try_open(i, "sell", entry, sl, entry - P["RR"] * risk)
                    setup = None


# ----------------------------- Rapport ----------------------------------------
def summarize(bot, eng):
    t = pd.DataFrame(eng.trades)
    n = len(t)
    if n == 0:
        return {"bot": bot, "trades": 0, "trades_ecartes_filtre": eng.rejected}
    peak, max_dd = START_CAPITAL, 0.0
    streak, worst_streak = 0, 0
    for _, row in t.iterrows():
        peak = max(peak, row["capital"])
        max_dd = max(max_dd, (peak - row["capital"]) / peak * 100)
        streak = streak + 1 if row["r"] < 0 else 0
        worst_streak = max(worst_streak, streak)
    return {
        "bot": bot,
        "trades": n,
        "winrate_pct": round(eng.wins / n * 100, 1),
        "buys": int((t["side"] == "buy").sum()),
        "sells": int((t["side"] == "sell").sum()),
        "total_r": round(t["r"].sum(), 1),
        "avg_r": round(t["r"].mean(), 3),
        "capital_final": eng.capital,
        "rendement_pct": round((eng.capital / START_CAPITAL - 1) * 100, 1),
        "max_drawdown_pct": round(max_dd, 1),
        "pire_serie_pertes": worst_streak,
        "trades_ecartes_filtre": eng.rejected,
    }


def yearly_rows(bot, eng):
    t = pd.DataFrame(eng.trades)
    rows = []
    if not len(t):
        return rows
    t["annee"] = t["closed"].str[:4]
    for y, g in t.groupby("annee"):
        n = len(g)
        w = int((g["r"] > 0).sum())
        rows.append({
            "bot": bot, "annee": y, "trades": n,
            "winrate_pct": round(w / n * 100, 1),
            "total_r": round(g["r"].sum(), 1),
            "pnl_total": round(g["pnl"].sum(), 2),
            "capital_fin_annee": g["capital"].iloc[-1],
        })
    return rows


# ----------------------------- Main -------------------------------------------
def main():
    ex = get_exchange()
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, yrows = [], []

    for sym in SYMBOLS:
        if sym not in ex.markets:
            print(f"\n[skip] {sym} indisponible sur cet exchange")
            continue
        slug = sym.split("/")[0].lower()
        print(f"\n================ {sym} ================")
        df = download_data(ex, sym, slug)
        if len(df) < 5000:
            print(f"[skip] {slug} : historique insuffisant")
            continue
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        print(f"Periode : {df['ts'].iloc[0]} -> {df['ts'].iloc[-1]} ({len(df)} bougies)")
        if (df["ts"].iloc[-1] - df["ts"].iloc[0]).days < YEARS * 365 - 30:
            print(f"ATTENTION : historique {slug} plus court que {YEARS} ans.")

        dfc = add_common(df)
        ts_labels = df["ts"].astype(str).values

        for mode in SL_MODES:
            bot = f"fibo-{mode}"
            print(f"[backtest] {bot} sur {slug}...")
            eng = Engine(bot, ts_labels)
            bt_fibo(dfc, eng, mode)
            rows.append({"actif": slug, **summarize(bot, eng)})
            yrows += [{"actif": slug, **r} for r in yearly_rows(bot, eng)]
            pd.DataFrame(eng.trades).to_csv(
                os.path.join(OUT_DIR, f"trades-fibo-{mode}-{slug}.csv"), index=False)

    synth = pd.DataFrame(rows)
    synth.to_csv(os.path.join(OUT_DIR, "synthese-multi-actifs.csv"), index=False)
    yearly = pd.DataFrame(yrows)
    yearly.to_csv(os.path.join(OUT_DIR, "ventilation-multi-actifs.csv"), index=False)
    print(f"\n===== SYNTHESE MULTI-ACTIFS (filtre {MIN_RISK_PCT}%, plancher {ATR_FLOOR}xATR) =====")
    print(synth.to_string(index=False))
    print("\n===== VENTILATION PAR ANNEE =====")
    print(yearly.to_string(index=False))
    print(f"\nFichiers ecrits dans {OUT_DIR}/")


if __name__ == "__main__":
    main()
