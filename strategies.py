"""
strategies.py — Détecteurs de signaux temps réel pour les 3 stratégies.
Chaque fonction rejoue la logique sur l'historique et retourne un signal
UNIQUEMENT si la condition d'entrée se déclenche sur la DERNIÈRE bougie clôturée.
Signal = {"side": "buy"/"sell", "entry": float, "sl": float, "tp": float}
"""

import numpy as np

# ----------------------------- Communs ---------------------------------------
EMA_TREND = 200
PIVOT_N   = 5

# Paramètres par stratégie (identiques aux backtests)
OTE  = dict(FIB_LOW=0.618, FIB_HIGH=0.786, SL_BUFFER=0.10, RR=1.5, ALLOW_BUY=True)
FVG  = dict(MIN_GAP_PCT=0.05, SL_BUFFER=0.10, RR=1.5, MAX_AGE=100, MAX_ZONES=10)
FIBO = dict(N_BINS=30, POC_ZONE_BINS=1, BODY_MAX=0.35, SL_BUFFER=0.10,
            MIN_LEG_PCT=0.15, RR=1.5)


def add_common(df):
    """Ajoute EMA et pivots fractals. À appeler une fois par cycle."""
    df = df.copy()
    df["ema"] = df["close"].ewm(span=EMA_TREND, adjust=False).mean()
    n = PIVOT_N
    df["pivot_high"] = df["high"][(df["high"] == df["high"].rolling(2 * n + 1, center=True).max())]
    df["pivot_low"]  = df["low"][(df["low"]  == df["low"].rolling(2 * n + 1, center=True).min())]
    return df


def _pivots_at(df, i, last_ph_i, last_pl_i):
    j = i - PIVOT_N
    if j >= 0:
        if not np.isnan(df["pivot_high"].iloc[j]):
            last_ph_i = j
        if not np.isnan(df["pivot_low"].iloc[j]):
            last_pl_i = j
    return last_ph_i, last_pl_i


# ----------------------------- 1. OTE Scalping -------------------------------
def signal_ote(df):
    P = OTE
    last = len(df) - 1
    zone, leg_id = None, None
    last_ph_i = last_pl_i = None

    for i in range(EMA_TREND, len(df)):
        last_ph_i, last_pl_i = _pivots_at(df, i, last_ph_i, last_pl_i)
        row = df.iloc[i]

        if last_ph_i is not None and last_pl_i is not None:
            new_leg = (last_ph_i, last_pl_i)
            if new_leg != leg_id:
                leg_id = new_leg
                hi = df["pivot_high"].iloc[last_ph_i]
                lo = df["pivot_low"].iloc[last_pl_i]
                leg = hi - lo
                zone = None
                if leg > 0:
                    if row["close"] < row["ema"] and last_ph_i < last_pl_i:
                        zone = {"side": "sell",
                                "f618": lo + P["FIB_LOW"] * leg,
                                "sl":   lo + (P["FIB_HIGH"] + P["SL_BUFFER"]) * leg,
                                "touched": False}
                    elif P["ALLOW_BUY"] and row["close"] > row["ema"] and last_pl_i < last_ph_i:
                        zone = {"side": "buy",
                                "f618": hi - P["FIB_LOW"] * leg,
                                "sl":   hi - (P["FIB_HIGH"] + P["SL_BUFFER"]) * leg,
                                "touched": False}

        if zone:
            if zone["side"] == "sell":
                if row["high"] >= zone["f618"]:
                    zone["touched"] = True
                if row["close"] > zone["sl"]:
                    zone = None
                elif zone["touched"] and row["close"] < zone["f618"]:
                    entry, sl = row["close"], zone["sl"]
                    risk = sl - entry
                    if risk > 0 and i == last:
                        return {"side": "sell", "entry": entry, "sl": sl,
                                "tp": entry - P["RR"] * risk}
                    zone = None
            else:
                if row["low"] <= zone["f618"]:
                    zone["touched"] = True
                if row["close"] < zone["sl"]:
                    zone = None
                elif zone["touched"] and row["close"] > zone["f618"]:
                    entry, sl = row["close"], zone["sl"]
                    risk = entry - sl
                    if risk > 0 and i == last:
                        return {"side": "buy", "entry": entry, "sl": sl,
                                "tp": entry + P["RR"] * risk}
                    zone = None
    return None


# ----------------------------- 2. FVG / IFVG ---------------------------------
def signal_fvg_ifvg(df):
    P = FVG
    last = len(df) - 1
    fvgs, ifvgs = [], []

    for i in range(2, len(df)):
        row = df.iloc[i]
        h2, l2 = df["high"].iloc[i - 2], df["low"].iloc[i - 2]
        gap_min = row["close"] * P["MIN_GAP_PCT"] / 100

        if row["low"] > h2 and (row["low"] - h2) >= gap_min:
            fvgs.append({"top": row["low"], "bottom": h2, "dir": +1, "born": i})
        elif row["high"] < l2 and (l2 - row["high"]) >= gap_min:
            fvgs.append({"top": l2, "bottom": row["high"], "dir": -1, "born": i})

        still = []
        for f in fvgs:
            if f["dir"] == +1 and row["close"] < f["bottom"]:
                ifvgs.append({"top": f["top"], "bottom": f["bottom"],
                              "side": "sell", "born": i, "touched": False})
            elif f["dir"] == -1 and row["close"] > f["top"]:
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
                invalid = row["close"] > z["top"]
                if row["high"] >= z["bottom"]:
                    z["touched"] = True
                sig = z["touched"] and row["close"] < z["bottom"]
                trend = row["close"] < row["ema"]
            else:
                invalid = row["close"] < z["bottom"]
                if row["low"] <= z["top"]:
                    z["touched"] = True
                sig = z["touched"] and row["close"] > z["top"]
                trend = row["close"] > row["ema"]

            if sig and trend and not invalid:
                height = z["top"] - z["bottom"]
                entry = row["close"]
                if z["side"] == "sell":
                    sl = z["top"] + P["SL_BUFFER"] * height
                    risk = sl - entry
                    tp = entry - P["RR"] * risk
                else:
                    sl = z["bottom"] - P["SL_BUFFER"] * height
                    risk = entry - sl
                    tp = entry + P["RR"] * risk
                if risk > 0 and i == last:
                    return {"side": z["side"], "entry": entry, "sl": sl, "tp": tp}
                continue                      # zone consommée
            if not expired and not invalid:
                keep.append(z)
        ifvgs = keep[-P["MAX_ZONES"]:]
    return None


# ------------------ 3. Fibo Premium/Discount + Volume + Pivot ----------------
def _poc(df, i0, i1, lo, hi, n_bins, zone_bins):
    if hi <= lo:
        return None, None
    edges = np.linspace(lo, hi, n_bins + 1)
    vols = np.zeros(n_bins)
    for k in range(i0, i1 + 1):
        c_lo = max(df["low"].iloc[k], lo)
        c_hi = min(df["high"].iloc[k], hi)
        v = df["vol"].iloc[k]
        if c_hi <= c_lo or v <= 0:
            continue
        overlap = np.clip(np.minimum(edges[1:], c_hi) - np.maximum(edges[:-1], c_lo), 0, None)
        vols += v * overlap / (c_hi - c_lo)
    b = int(vols.argmax())
    size = (hi - lo) / n_bins
    return (lo + max(b - zone_bins, 0) * size,
            lo + min(b + zone_bins + 1, n_bins) * size)


def signal_fibo_volume(df):
    P = FIBO
    last = len(df) - 1
    setup, leg_id = None, None
    last_ph_i = last_pl_i = None

    for i in range(EMA_TREND, len(df)):
        last_ph_i, last_pl_i = _pivots_at(df, i, last_ph_i, last_pl_i)
        row = df.iloc[i]

        if last_ph_i is not None and last_pl_i is not None:
            new_leg = (last_ph_i, last_pl_i)
            if new_leg != leg_id:
                leg_id = new_leg
                setup = None
                hi = df["pivot_high"].iloc[last_ph_i]
                lo = df["pivot_low"].iloc[last_pl_i]
                leg = hi - lo
                if leg > 0 and leg / row["close"] * 100 >= P["MIN_LEG_PCT"]:
                    eq = lo + 0.5 * leg
                    if row["close"] > row["ema"] and last_pl_i < last_ph_i:
                        p_lo, p_hi = _poc(df, last_pl_i, last_ph_i, lo, hi,
                                          P["N_BINS"], P["POC_ZONE_BINS"])
                        if p_lo is not None and p_hi <= eq:
                            setup = {"side": "buy", "lo": lo, "hi": hi, "eq": eq,
                                     "poc_lo": p_lo, "poc_hi": p_hi}
                    elif row["close"] < row["ema"] and last_ph_i < last_pl_i:
                        p_lo, p_hi = _poc(df, last_ph_i, last_pl_i, lo, hi,
                                          P["N_BINS"], P["POC_ZONE_BINS"])
                        if p_lo is not None and p_lo >= eq:
                            setup = {"side": "sell", "lo": lo, "hi": hi, "eq": eq,
                                     "poc_lo": p_lo, "poc_hi": p_hi}

        if setup:
            rng = row["high"] - row["low"]
            body = abs(row["close"] - row["open"])
            small = rng > 0 and body / rng <= P["BODY_MAX"]
            touch = row["low"] <= setup["poc_hi"] and row["high"] >= setup["poc_lo"]

            if setup["side"] == "buy":
                if row["close"] < setup["lo"]:
                    setup = None
                elif small and touch and row["close"] < setup["eq"]:
                    entry = row["close"]
                    sl = row["low"] - P["SL_BUFFER"] * rng
                    risk = entry - sl
                    if risk > 0 and i == last:
                        return {"side": "buy", "entry": entry, "sl": sl,
                                "tp": entry + P["RR"] * risk}
                    setup = None
            else:
                if row["close"] > setup["hi"]:
                    setup = None
                elif small and touch and row["close"] > setup["eq"]:
                    entry = row["close"]
                    sl = row["high"] + P["SL_BUFFER"] * rng
                    risk = sl - entry
                    if risk > 0 and i == last:
                        return {"side": "sell", "entry": entry, "sl": sl,
                                "tp": entry - P["RR"] * risk}
                    setup = None
    return None
