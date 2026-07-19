"""
backtest-3ans.py : backtest des 3 bots (ote, fvg_ifvg, fibo_volume) sur 3 ans.

Reprend a l'identique la logique de strategies.py et de paper_engine.py :
  - memes parametres (OTE / FVG / FIBO), meme EMA200, memes pivots fractals
  - meme moteur : capital 1000$, risque 1 %/trade, frais 0.05 %/cote,
    SL prioritaire si SL et TP touches dans la meme bougie,
    jamais de sortie sur la bougie d'entree,
    pas de nouvelle entree sur la bougie d'une sortie (comme run_bots.py)

Difference assumee (documentee) : le live recalcule l'EMA et les zones sur une
fenetre glissante de 720 bougies ; ici tout est calcule sur l'historique complet
en un seul passage (beaucoup plus rapide, ecarts negligeables apres warm-up).

Donnees : PAXG/USDT 5m via ccxt (Binance, puis Gate, puis KuCoin en secours),
avec cache local dans data-paxg-5m.csv (relance = reprise, pas de re-telechargement).

Usage  : python backtest-3ans.py
Sortie : dossier resultats-backtest/
           synthese-backtest-3ans.csv
           trades-ote.csv, trades-fvg-ifvg.csv, trades-fibo-volume.csv
"""

import os
import sys
import time

import ccxt
import numpy as np
import pandas as pd

# ----------------------------- Parametres (identiques au live) ---------------
SYMBOL_DL   = "PAXG/USDT"
TIMEFRAME   = "5m"
TF_MS       = 5 * 60 * 1000
YEARS       = 3
DATA_FILE   = "data-paxg-5m.csv"
OUT_DIR     = "resultats-backtest"

START_CAPITAL  = 1000.0
RISK_PER_TRADE = 1.0     # % du capital risque par trade
FEES_PCT       = 0.05    # % par cote (calcul identique a paper_engine)

EMA_TREND = 200
PIVOT_N   = 5

OTE  = dict(FIB_LOW=0.618, FIB_HIGH=0.786, SL_BUFFER=0.10, RR=1.5, ALLOW_BUY=True)
FVG  = dict(MIN_GAP_PCT=0.05, SL_BUFFER=0.10, RR=1.5, MAX_AGE=100, MAX_ZONES=10)
FIBO = dict(N_BINS=30, POC_ZONE_BINS=1, BODY_MAX=0.35, SL_BUFFER=0.10,
            MIN_LEG_PCT=0.15, RR=1.5)


# ----------------------------- Donnees ---------------------------------------
def get_exchange():
    for name in ("binance", "gateio", "kucoin"):
        try:
            ex = getattr(ccxt, name)()
            ex.load_markets()
            if SYMBOL_DL in ex.markets:
                print(f"[data] exchange utilise : {name}")
                return ex
        except Exception as e:
            print(f"[data] {name} indisponible ({type(e).__name__}), essai suivant...")
    sys.exit("Aucun exchange accessible avec PAXG/USDT. Verifie ta connexion.")


def download_data():
    ex = get_exchange()
    target_start = ex.milliseconds() - YEARS * 365 * 24 * 3600 * 1000

    rows = []
    since = target_start
    if os.path.exists(DATA_FILE):
        cached = pd.read_csv(DATA_FILE)
        if len(cached):
            rows = cached.values.tolist()
            since = int(cached["ts"].iloc[-1]) + TF_MS
            print(f"[data] cache trouve : {len(rows)} bougies, reprise...")

    end = ex.milliseconds()
    while since < end:
        try:
            batch = ex.fetch_ohlcv(SYMBOL_DL, TIMEFRAME, since=since, limit=1000)
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
        if len(rows) % 20000 < 1000:
            print(f"[data] {len(rows)} bougies "
                  f"(jusqu'a {pd.to_datetime(batch[-1][0], unit='ms')})", flush=True)
        time.sleep(ex.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"[data] total : {len(df)} bougies sauvegardees dans {DATA_FILE}")
    return df


# ----------------------------- Communs (copie de strategies.py) --------------
def add_common(df):
    df = df.copy()
    df["ema"] = df["close"].ewm(span=EMA_TREND, adjust=False).mean()
    n = PIVOT_N
    df["pivot_high"] = df["high"][(df["high"] == df["high"].rolling(2 * n + 1, center=True).max())]
    df["pivot_low"]  = df["low"][(df["low"]  == df["low"].rolling(2 * n + 1, center=True).min())]
    return df


# ----------------------------- Moteur (copie de paper_engine.py) --------------
class Engine:
    def __init__(self, bot, ts_labels):
        self.bot = bot
        self.ts = ts_labels
        self.capital = START_CAPITAL
        self.pos = None
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.closed_i = -1        # pas de re-entree sur la bougie d'une sortie

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
        # la zone est consommee par l'appelant dans tous les cas (comme au live) ;
        # on n'ouvre que si le bot est flat et n'a pas ferme sur cette bougie
        if self.pos is not None or self.closed_i == i:
            return
        risk = abs(entry - sl)
        rr = round(abs(tp - entry) / risk, 2) if risk > 0 else 0
        self.pos = {"side": side, "entry": entry, "sl": sl, "tp": tp,
                    "rr": rr, "i": i}


# ----------------------------- 1. OTE Scalping --------------------------------
def bt_ote(df, eng):
    P = OTE
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    e = df["ema"].values
    PH, PL = df["pivot_high"].values, df["pivot_low"].values
    zone, leg_id = None, None
    ph_i = pl_i = None

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
                hi, lo = PH[ph_i], PL[pl_i]
                leg = hi - lo
                zone = None
                if leg > 0:
                    if c[i] < e[i] and ph_i < pl_i:
                        zone = {"side": "sell",
                                "f618": lo + P["FIB_LOW"] * leg,
                                "sl":   lo + (P["FIB_HIGH"] + P["SL_BUFFER"]) * leg,
                                "touched": False}
                    elif P["ALLOW_BUY"] and c[i] > e[i] and pl_i < ph_i:
                        zone = {"side": "buy",
                                "f618": hi - P["FIB_LOW"] * leg,
                                "sl":   hi - (P["FIB_HIGH"] + P["SL_BUFFER"]) * leg,
                                "touched": False}

        if zone:
            if zone["side"] == "sell":
                if h[i] >= zone["f618"]:
                    zone["touched"] = True
                if c[i] > zone["sl"]:
                    zone = None
                elif zone["touched"] and c[i] < zone["f618"]:
                    entry, sl = c[i], zone["sl"]
                    risk = sl - entry
                    if risk > 0:
                        eng.try_open(i, "sell", entry, sl, entry - P["RR"] * risk)
                    zone = None
            else:
                if l[i] <= zone["f618"]:
                    zone["touched"] = True
                if c[i] < zone["sl"]:
                    zone = None
                elif zone["touched"] and c[i] > zone["f618"]:
                    entry, sl = c[i], zone["sl"]
                    risk = entry - sl
                    if risk > 0:
                        eng.try_open(i, "buy", entry, sl, entry + P["RR"] * risk)
                    zone = None


# ----------------------------- 2. FVG / IFVG ----------------------------------
def bt_fvg(df, eng):
    P = FVG
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    e = df["ema"].values
    fvgs, ifvgs = [], []

    for i in range(2, len(df)):
        if i >= EMA_TREND:
            eng.on_candle(i, h[i], l[i])

        h2, l2 = h[i - 2], l[i - 2]
        gap_min = c[i] * P["MIN_GAP_PCT"] / 100
        if l[i] > h2 and (l[i] - h2) >= gap_min:
            fvgs.append({"top": l[i], "bottom": h2, "dir": +1, "born": i})
        elif h[i] < l2 and (l2 - h[i]) >= gap_min:
            fvgs.append({"top": l2, "bottom": h[i], "dir": -1, "born": i})

        still = []
        for f in fvgs:
            if f["dir"] == +1 and c[i] < f["bottom"]:
                ifvgs.append({"top": f["top"], "bottom": f["bottom"],
                              "side": "sell", "born": i, "touched": False})
            elif f["dir"] == -1 and c[i] > f["top"]:
                ifvgs.append({"top": f["top"], "bottom": f["bottom"],
                              "side": "buy", "born": i, "touched": False})
            elif i - f["born"] <= P["MAX_AGE"]:
                still.append(f)
        fvgs = still[-P["MAX_ZONES"]:]

        if i < EMA_TREND:
            continue

        keep = []
        for z in ifvgs:
            expired = i - z["born"] > P["MAX_AGE"]
            if z["side"] == "sell":
                invalid = c[i] > z["top"]
                if h[i] >= z["bottom"]:
                    z["touched"] = True
                sig = z["touched"] and c[i] < z["bottom"]
                trend = c[i] < e[i]
            else:
                invalid = c[i] < z["bottom"]
                if l[i] <= z["top"]:
                    z["touched"] = True
                sig = z["touched"] and c[i] > z["top"]
                trend = c[i] > e[i]

            if sig and trend and not invalid:
                height = z["top"] - z["bottom"]
                entry = c[i]
                if z["side"] == "sell":
                    sl = z["top"] + P["SL_BUFFER"] * height
                    risk = sl - entry
                    tp = entry - P["RR"] * risk
                else:
                    sl = z["bottom"] - P["SL_BUFFER"] * height
                    risk = entry - sl
                    tp = entry + P["RR"] * risk
                if risk > 0:
                    eng.try_open(i, z["side"], entry, sl, tp)
                continue                          # zone consommee
            if not expired and not invalid:
                keep.append(z)
        ifvgs = keep[-P["MAX_ZONES"]:]


# ------------------ 3. Fibo Premium/Discount + Volume + Pivot -----------------
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


def bt_fibo(df, eng):
    P = FIBO
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    e = df["ema"].values
    v = df["vol"].values
    PH, PL = df["pivot_high"].values, df["pivot_low"].values
    setup, leg_id = None, None
    ph_i = pl_i = None

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
                    sl = l[i] - P["SL_BUFFER"] * rng
                    risk = entry - sl
                    if risk > 0:
                        eng.try_open(i, "buy", entry, sl, entry + P["RR"] * risk)
                    setup = None
            else:
                if c[i] > setup["hi"]:
                    setup = None
                elif small and touch and c[i] > setup["eq"]:
                    entry = c[i]
                    sl = h[i] + P["SL_BUFFER"] * rng
                    risk = sl - entry
                    if risk > 0:
                        eng.try_open(i, "sell", entry, sl, entry - P["RR"] * risk)
                    setup = None


# ----------------------------- Rapport ----------------------------------------
def summarize(bot, eng):
    t = pd.DataFrame(eng.trades)
    n = len(t)
    if n == 0:
        return {"bot": bot, "trades": 0}
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
        "wins": eng.wins,
        "losses": eng.losses,
        "winrate_pct": round(eng.wins / n * 100, 1),
        "buys": int((t["side"] == "buy").sum()),
        "sells": int((t["side"] == "sell").sum()),
        "total_r": round(t["r"].sum(), 1),
        "avg_r": round(t["r"].mean(), 3),
        "capital_final": eng.capital,
        "rendement_pct": round((eng.capital / START_CAPITAL - 1) * 100, 1),
        "max_drawdown_pct": round(max_dd, 1),
        "pire_serie_pertes": worst_streak,
    }


def main():
    df = download_data()
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    print(f"\nPeriode couverte : {df['ts'].iloc[0]} -> {df['ts'].iloc[-1]} "
          f"({len(df)} bougies 5m)")
    if (df["ts"].iloc[-1] - df["ts"].iloc[0]).days < YEARS * 365 - 30:
        print("ATTENTION : historique disponible plus court que 3 ans.")

    dfc = add_common(df)
    ts_labels = df["ts"].astype(str).values

    os.makedirs(OUT_DIR, exist_ok=True)
    bots = {"ote": bt_ote, "fvg-ifvg": bt_fvg, "fibo-volume": bt_fibo}
    rows = []
    for bot, fn in bots.items():
        print(f"\n[backtest] {bot} en cours...")
        eng = Engine(bot, ts_labels)
        fn(dfc, eng)
        # position encore ouverte a la fin : ignoree (non comptee)
        res = summarize(bot, eng)
        rows.append(res)
        pd.DataFrame(eng.trades).to_csv(
            os.path.join(OUT_DIR, f"trades-{bot}.csv"), index=False)
        print(f"[backtest] {bot} : {res.get('trades', 0)} trades, "
              f"capital final {res.get('capital_final', START_CAPITAL)}$")

    synth = pd.DataFrame(rows)
    synth.to_csv(os.path.join(OUT_DIR, "synthese-backtest-3ans.csv"), index=False)
    print("\n===== SYNTHESE 3 ANS =====")
    print(synth.to_string(index=False))
    print(f"\nFichiers ecrits dans {OUT_DIR}/")


if __name__ == "__main__":
    main()
