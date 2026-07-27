"""
backtest-refonte.py : labo de refonte OTE v2 et FVG v2 sur PAXG (5m, 3 ans).

Diagnostic des versions d'origine (backtests precedents) :
  OTE : trade des jambes de bruit sans taille minimale, entre des 0.618 sans
        confirmation, juge la tendance sur ~17h (EMA200 5m) — broye en 2024.
  FVG : gaps a seuil fixe 0.05% (du bruit en 5m), ne trade que l'invalidation
        (IFVG/continuation), jamais le comblement classique.

Variantes testees (criteres de verdict fixes AVANT le run : R total positif,
aucune annee fortement negative, volume de trades suffisant) :
  ote-v1        : version d'origine — reference
  ote-v2        : jambe >= 0.15% + tendance ~1h + zone 0.705-0.786
                  + bougie de rejet + plancher SL 1xATR
  ote-v2-noconf : v2 sans la bougie de rejet          (ablation confirmation)
  ote-v2-nohtf  : v2 avec tendance 5m d'origine       (ablation tendance 1h)
  fvg-v1        : version d'origine — reference
  fvg-cont      : continuation (IFVG) avec gap >= 0.75xATR, tendance ~1h,
                  plancher SL 1xATR
  fvg-fill      : comblement classique : retest d'un FVG valide, rebond dans
                  le sens du gap, tendance ~1h, plancher SL 1xATR
  fibo-floor    : etalon (champion actuel)

Usage  : python backtest-refonte.py
Sortie : resultats-backtest/synthese-refonte.csv, ventilation-refonte.csv,
         trades-<variante>.csv
"""

import os
import sys
import time

import ccxt
import numpy as np
import pandas as pd

# ----------------------------- Parametres ------------------------------------
SYMBOL    = "PAXG/USDT"
TIMEFRAME = "5m"
TF_MS     = 5 * 60 * 1000
YEARS     = 3
OUT_DIR   = "resultats-backtest"
DATA_FILE = "data-paxg-5m.csv"          # meme cache que les autres backtests

START_CAPITAL  = 1000.0
RISK_PER_TRADE = 1.0
FEES_PCT       = 0.05
MIN_RISK_PCT   = 0.05

EMA_TREND   = 200      # tendance 5m (versions v1)
HTF_SPAN    = 2400     # EMA200 equivalente 1h, calculee sur les bougies 5m
HTF_WARMUP  = 2880     # bougies ignorees le temps que l'EMA 1h soit fiable
PIVOT_N     = 5
ATR_FLOOR   = 1.0      # plancher de distance SL, en ATR(14)
MIN_LEG_PCT = 0.15     # taille minimale de jambe OTE v2 (comme fibo)
FIB_ENTRY   = 0.705    # debut de la vraie zone OTE (v2) ; v1 entrait des 0.618
CONF_RATIO  = 0.6      # bougie de rejet : cloture dans les 40% favorables
GAP_ATR     = 0.75     # taille minimale d'un FVG v2, en ATR(14)

OTE = dict(FIB_LOW=0.618, FIB_HIGH=0.786, SL_BUFFER=0.10, RR=1.5, ALLOW_BUY=True)
FVG = dict(MIN_GAP_PCT=0.05, SL_BUFFER=0.10, RR=1.5, MAX_AGE=100, MAX_ZONES=10)
FIBO = dict(N_BINS=30, POC_ZONE_BINS=1, BODY_MAX=0.35, SL_BUFFER=0.10,
            MIN_LEG_PCT=0.15, RR=1.5)


# ----------------------------- Donnees ---------------------------------------
def get_exchange():
    for name in ("binance", "gateio", "kucoin"):
        try:
            ex = getattr(ccxt, name)()
            ex.load_markets()
            if SYMBOL in ex.markets:
                print(f"[data] exchange utilise : {name}")
                return ex
        except Exception as e:
            print(f"[data] {name} indisponible ({type(e).__name__}), essai suivant...")
    sys.exit("Aucun exchange accessible. Verifie ta connexion.")


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
            print(f"[data] cache : {len(rows)} bougies, reprise...")
    end = ex.milliseconds()
    while since < end:
        try:
            batch = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
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
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"[data] total : {len(df)} bougies")
    return df


# ----------------------------- Indicateurs -----------------------------------
def add_common(df):
    df = df.copy()
    df["ema"] = df["close"].ewm(span=EMA_TREND, adjust=False).mean()
    df["ema_htf"] = df["close"].ewm(span=HTF_SPAN, adjust=False).mean()
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    n = PIVOT_N
    df["pivot_high"] = df["high"][(df["high"] == df["high"].rolling(2 * n + 1, center=True).max())]
    df["pivot_low"]  = df["low"][(df["low"]  == df["low"].rolling(2 * n + 1, center=True).min())]
    return df


# ----------------------------- Moteur ------------------------------------------
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
        if p and p["i"] != i:
            if p["side"] == "sell":
                hit_sl = high >= p["sl"]
                hit_tp = low <= p["tp"]
            else:
                hit_sl = low <= p["sl"]
                hit_tp = high >= p["tp"]
            if hit_sl or hit_tp:
                r = -1.0 if hit_sl else p["rr"]
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


# ----------------------------- OTE v1 (reference) ------------------------------
def bt_ote_v1(df, eng):
    P = OTE
    h, l, c = (df[k].values for k in ("high", "low", "close"))
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
                        zone = {"side": "sell", "f": lo + P["FIB_LOW"] * leg,
                                "sl": lo + (P["FIB_HIGH"] + P["SL_BUFFER"]) * leg,
                                "touched": False}
                    elif c[i] > e[i] and pl_i < ph_i:
                        zone = {"side": "buy", "f": hi - P["FIB_LOW"] * leg,
                                "sl": hi - (P["FIB_HIGH"] + P["SL_BUFFER"]) * leg,
                                "touched": False}
        if zone:
            if zone["side"] == "sell":
                if h[i] >= zone["f"]:
                    zone["touched"] = True
                if c[i] > zone["sl"]:
                    zone = None
                elif zone["touched"] and c[i] < zone["f"]:
                    entry, sl = c[i], zone["sl"]
                    risk = sl - entry
                    if risk > 0:
                        eng.try_open(i, "sell", entry, sl, entry - P["RR"] * risk)
                    zone = None
            else:
                if l[i] <= zone["f"]:
                    zone["touched"] = True
                if c[i] < zone["sl"]:
                    zone = None
                elif zone["touched"] and c[i] > zone["f"]:
                    entry, sl = c[i], zone["sl"]
                    risk = entry - sl
                    if risk > 0:
                        eng.try_open(i, "buy", entry, sl, entry + P["RR"] * risk)
                    zone = None


# ----------------------------- OTE v2 ------------------------------------------
def bt_ote_v2(df, eng, use_conf=True, use_htf=True):
    P = OTE
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    trend_e = df["ema_htf"].values if use_htf else df["ema"].values
    atr = df["atr"].values
    PH, PL = df["pivot_high"].values, df["pivot_low"].values
    start = HTF_WARMUP if use_htf else EMA_TREND
    zone, leg_id = None, None
    ph_i = pl_i = None

    def confirmed(side, i):
        if not use_conf:
            return True
        rng = h[i] - l[i]
        if rng <= 0:
            return False
        if side == "buy":
            return (c[i] - l[i]) / rng >= CONF_RATIO      # cloture haute : rejet du bas
        return (h[i] - c[i]) / rng >= CONF_RATIO          # cloture basse : rejet du haut

    for i in range(start, len(df)):
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
                # jambe qualifiee : taille minimale comme fibo
                if leg > 0 and leg / c[i] * 100 >= MIN_LEG_PCT:
                    if c[i] < trend_e[i] and ph_i < pl_i:
                        zone = {"side": "sell",
                                "f": lo + FIB_ENTRY * leg,
                                "sl": lo + (P["FIB_HIGH"] + P["SL_BUFFER"]) * leg,
                                "touched": False}
                    elif c[i] > trend_e[i] and pl_i < ph_i:
                        zone = {"side": "buy",
                                "f": hi - FIB_ENTRY * leg,
                                "sl": hi - (P["FIB_HIGH"] + P["SL_BUFFER"]) * leg,
                                "touched": False}
        if zone:
            if zone["side"] == "sell":
                if h[i] >= zone["f"]:
                    zone["touched"] = True
                if c[i] > zone["sl"]:
                    zone = None
                elif zone["touched"] and c[i] < zone["f"] and confirmed("sell", i):
                    entry = c[i]
                    sl = max(zone["sl"], entry + ATR_FLOOR * atr[i])
                    risk = sl - entry
                    if risk > 0:
                        eng.try_open(i, "sell", entry, sl, entry - P["RR"] * risk)
                    zone = None
            else:
                if l[i] <= zone["f"]:
                    zone["touched"] = True
                if c[i] < zone["sl"]:
                    zone = None
                elif zone["touched"] and c[i] > zone["f"] and confirmed("buy", i):
                    entry = c[i]
                    sl = min(zone["sl"], entry - ATR_FLOOR * atr[i])
                    risk = entry - sl
                    if risk > 0:
                        eng.try_open(i, "buy", entry, sl, entry + P["RR"] * risk)
                    zone = None


# ----------------------------- FVG v1 (reference) -------------------------------
def bt_fvg_v1(df, eng):
    P = FVG
    h, l, c = (df[k].values for k in ("high", "low", "close"))
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
                continue
            if not expired and not invalid:
                keep.append(z)
        ifvgs = keep[-P["MAX_ZONES"]:]


# ----------------------------- FVG v2 continuation ------------------------------
def bt_fvg_cont(df, eng):
    P = FVG
    h, l, c = (df[k].values for k in ("high", "low", "close"))
    eh = df["ema_htf"].values
    atr = df["atr"].values
    fvgs, ifvgs = [], []
    for i in range(2, len(df)):
        if i >= HTF_WARMUP:
            eng.on_candle(i, h[i], l[i])
        h2, l2 = h[i - 2], l[i - 2]
        gap_min = GAP_ATR * atr[i]                       # gap significatif en ATR
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
        if i < HTF_WARMUP:
            continue
        keep = []
        for z in ifvgs:
            expired = i - z["born"] > P["MAX_AGE"]
            if z["side"] == "sell":
                invalid = c[i] > z["top"]
                if h[i] >= z["bottom"]:
                    z["touched"] = True
                sig = z["touched"] and c[i] < z["bottom"]
                trend = c[i] < eh[i]
            else:
                invalid = c[i] < z["bottom"]
                if l[i] <= z["top"]:
                    z["touched"] = True
                sig = z["touched"] and c[i] > z["top"]
                trend = c[i] > eh[i]
            if sig and trend and not invalid:
                height = z["top"] - z["bottom"]
                entry = c[i]
                if z["side"] == "sell":
                    sl = max(z["top"] + P["SL_BUFFER"] * height,
                             entry + ATR_FLOOR * atr[i])
                    risk = sl - entry
                    tp = entry - P["RR"] * risk
                else:
                    sl = min(z["bottom"] - P["SL_BUFFER"] * height,
                             entry - ATR_FLOOR * atr[i])
                    risk = entry - sl
                    tp = entry + P["RR"] * risk
                if risk > 0:
                    eng.try_open(i, z["side"], entry, sl, tp)
                continue
            if not expired and not invalid:
                keep.append(z)
        ifvgs = keep[-P["MAX_ZONES"]:]


# ----------------------------- FVG v2 comblement --------------------------------
def bt_fvg_fill(df, eng):
    P = FVG
    h, l, c = (df[k].values for k in ("high", "low", "close"))
    eh = df["ema_htf"].values
    atr = df["atr"].values
    fvgs = []
    for i in range(2, len(df)):
        if i >= HTF_WARMUP:
            eng.on_candle(i, h[i], l[i])
        h2, l2 = h[i - 2], l[i - 2]
        gap_min = GAP_ATR * atr[i]
        if l[i] > h2 and (l[i] - h2) >= gap_min:
            fvgs.append({"top": l[i], "bottom": h2, "dir": +1, "born": i,
                         "touched": False})
        elif h[i] < l2 and (l2 - h[i]) >= gap_min:
            fvgs.append({"top": l2, "bottom": h[i], "dir": -1, "born": i,
                         "touched": False})
        if i < HTF_WARMUP:
            fvgs = fvgs[-P["MAX_ZONES"]:]
            continue
        keep = []
        for f in fvgs:
            if f["born"] == i:                      # pas de retest le jour meme
                keep.append(f)
                continue
            expired = i - f["born"] > P["MAX_AGE"]
            if f["dir"] == +1:                      # gap haussier sous le prix
                invalid = c[i] < f["bottom"]        # gap entierement traverse
                if l[i] <= f["top"]:
                    f["touched"] = True             # le prix est revenu combler
                sig = f["touched"] and c[i] > f["top"] and c[i] > eh[i]
                if sig and not invalid:
                    height = f["top"] - f["bottom"]
                    entry = c[i]
                    sl = min(f["bottom"] - P["SL_BUFFER"] * height,
                             entry - ATR_FLOOR * atr[i])
                    risk = entry - sl
                    if risk > 0:
                        eng.try_open(i, "buy", entry, sl, entry + P["RR"] * risk)
                    continue                        # zone consommee
            else:                                   # gap baissier au-dessus du prix
                invalid = c[i] > f["top"]
                if h[i] >= f["bottom"]:
                    f["touched"] = True
                sig = f["touched"] and c[i] < f["bottom"] and c[i] < eh[i]
                if sig and not invalid:
                    height = f["top"] - f["bottom"]
                    entry = c[i]
                    sl = max(f["top"] + P["SL_BUFFER"] * height,
                             entry + ATR_FLOOR * atr[i])
                    risk = sl - entry
                    if risk > 0:
                        eng.try_open(i, "sell", entry, sl, entry - P["RR"] * risk)
                    continue
            if not expired and not invalid:
                keep.append(f)
        fvgs = keep[-P["MAX_ZONES"]:]


# ----------------------------- Fibo-floor (etalon) ------------------------------
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


def bt_fibo_floor(df, eng):
    P = FIBO
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    e = df["ema"].values
    v = df["vol"].values
    atr = df["atr"].values
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
                    sl = min(l[i] - P["SL_BUFFER"] * rng, entry - ATR_FLOOR * atr[i])
                    risk = entry - sl
                    if risk > 0:
                        eng.try_open(i, "buy", entry, sl, entry + P["RR"] * risk)
                    setup = None
            else:
                if c[i] > setup["hi"]:
                    setup = None
                elif small and touch and c[i] > setup["eq"]:
                    entry = c[i]
                    sl = max(h[i] + P["SL_BUFFER"] * rng, entry + ATR_FLOOR * atr[i])
                    risk = sl - entry
                    if risk > 0:
                        eng.try_open(i, "sell", entry, sl, entry - P["RR"] * risk)
                    setup = None


# ----------------------------- Rapport ------------------------------------------
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
    tb = t[t["side"] == "buy"]
    tsell = t[t["side"] == "sell"]
    return {
        "bot": bot,
        "trades": n,
        "winrate_pct": round(eng.wins / n * 100, 1),
        "buys": int(len(tb)),
        "sells": int(len(tsell)),
        "winrate_buy_pct": round((tb["r"] > 0).mean() * 100, 1) if len(tb) else None,
        "winrate_sell_pct": round((tsell["r"] > 0).mean() * 100, 1) if len(tsell) else None,
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


# ----------------------------- Main ---------------------------------------------
def main():
    df = download_data()
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    print(f"\nPeriode : {df['ts'].iloc[0]} -> {df['ts'].iloc[-1]} ({len(df)} bougies)")
    dfc = add_common(df)
    ts_labels = df["ts"].astype(str).values

    os.makedirs(OUT_DIR, exist_ok=True)
    runs = [
        ("ote-v1",        lambda d, e: bt_ote_v1(d, e)),
        ("ote-v2",        lambda d, e: bt_ote_v2(d, e, True, True)),
        ("ote-v2-noconf", lambda d, e: bt_ote_v2(d, e, False, True)),
        ("ote-v2-nohtf",  lambda d, e: bt_ote_v2(d, e, True, False)),
        ("fvg-v1",        lambda d, e: bt_fvg_v1(d, e)),
        ("fvg-cont",      lambda d, e: bt_fvg_cont(d, e)),
        ("fvg-fill",      lambda d, e: bt_fvg_fill(d, e)),
        ("fibo-floor",    lambda d, e: bt_fibo_floor(d, e)),
    ]
    rows, yrows = [], []
    for bot, fn in runs:
        print(f"[backtest] {bot} en cours...")
        eng = Engine(bot, ts_labels)
        fn(dfc, eng)
        rows.append(summarize(bot, eng))
        yrows += yearly_rows(bot, eng)
        pd.DataFrame(eng.trades).to_csv(
            os.path.join(OUT_DIR, f"trades-{bot}.csv"), index=False)

    synth = pd.DataFrame(rows)
    synth.to_csv(os.path.join(OUT_DIR, "synthese-refonte.csv"), index=False)
    yearly = pd.DataFrame(yrows)
    yearly.to_csv(os.path.join(OUT_DIR, "ventilation-refonte.csv"), index=False)
    print("\n===== SYNTHESE REFONTE =====")
    print(synth.to_string(index=False))
    print("\n===== VENTILATION PAR ANNEE =====")
    print(yearly.to_string(index=False))
    print(f"\nFichiers ecrits dans {OUT_DIR}/")


if __name__ == "__main__":
    main()
