"""
ALGOX SMT Scanner — ICT / Smart Money (SMT + FVG + OB + CSD) on Streamlit.
Single-file app: engine + data + UI.  Deploy: GitHub repo with app.py + requirements.txt
-> share.streamlit.io -> main file = app.py

Crypto = Binance SPOT klines (REST, with geo-block fallback hosts). Metals = Yahoo COMEX futures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ═══════════════════════════ 1) SMT ENGINE ═══════════════════════════
@dataclass
class Params:
    swing_len: int = 10          # SMT lookback in HTF candles
    arm_bars: int = 4            # setup window (HTF candles) after the SMT candle
    conf_mode: str = "FVG or OB"  # "FVG or OB" | "FVG and OB" | "None"
    disp_mult: float = 1.0       # CSD displacement body >= disp_mult * ATR(LTF)
    rr: float = 2.0
    atr_buf: float = 0.25        # SL buffer = atr_buf * ATR(HTF)
    trade_leg: str = "strong"    # "strong" | "weak" | "both"
    fee_pct: float = 0.1         # per side, in %
    atr_len: int = 14


# ───────────────────────────── helpers ─────────────────────────────
def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def prep_htf(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    d = df.copy()
    d["atr"] = _atr(d, p.atr_len)
    d["hh"] = d["high"] > d["high"].shift(1).rolling(p.swing_len).max()
    d["ll"] = d["low"] < d["low"].shift(1).rolling(p.swing_len).min()
    return d


def prep_ltf(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    d = df.copy()
    d["atr"] = _atr(d, p.atr_len)
    dn = d["close"] < d["open"]
    up = d["close"] > d["open"]
    dn_start = dn & ~dn.shift(1, fill_value=False)
    up_start = up & ~up.shift(1, fill_value=False)
    d["dn_run_open"] = d["open"].where(dn_start).ffill()   # open of 1st candle of last down series
    d["up_run_open"] = d["open"].where(up_start).ffill()   # open of 1st candle of last up series
    return d


def smt_events(name_a: str, a: pd.DataFrame, name_b: str, b: pd.DataFrame):
    """Return list of (time, direction, swept_leg_name) for one pair. direction: +1 bullish, -1 bearish."""
    idx = a.index.intersection(b.index)
    a_, b_ = a.loc[idx], b.loc[idx]
    ll_a, ll_b = a_["ll"].to_numpy(), b_["ll"].to_numpy()
    hh_a, hh_b = a_["hh"].to_numpy(), b_["hh"].to_numpy()
    bull = ll_a != ll_b
    bear = hh_a != hh_b
    out = []
    for i in np.flatnonzero(bull ^ bear):
        if bull[i]:
            out.append((idx[i], 1, name_a if ll_a[i] else name_b))
        else:
            out.append((idx[i], -1, name_a if hh_a[i] else name_b))
    return out


def _simulate(direction, entry, sl, tp, lo, hi, times, start, rr):
    for m in range(start, len(lo)):
        if direction == 1:
            hit_sl, hit_tp = lo[m] <= sl, hi[m] >= tp
        else:
            hit_sl, hit_tp = hi[m] >= sl, lo[m] <= tp
        if hit_sl:                       # SL first if both touched in the same candle (conservative)
            return "LOSS", times[m], -1.0
        if hit_tp:
            return "WIN", times[m], rr
    return "OPEN", pd.NaT, np.nan


def _conf_ok(mode: str, has_fvg: bool, has_ob: bool) -> bool:
    if mode == "None":
        return True
    if mode == "FVG and OB":
        return has_fvg and has_ob
    return has_fvg or has_ob


# ───────────────────────── single setup builder ─────────────────────────
def build_setup(pair, leg, direction, t, swept, htf, ltf, p: Params, htf_td, now):
    i = htf.index.get_loc(t)
    n = len(htf)
    o, h, l, c = (htf[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr_h = htf["atr"].to_numpy()[i]

    # Order Block at / just before the SMT candle
    ob = None
    if direction == 1:
        if c[i] < o[i]:
            ob = (o[i], l[i], htf.index[i])
        elif i > 0 and c[i - 1] < o[i - 1]:
            ob = (o[i - 1], l[i - 1], htf.index[i - 1])
    else:
        if c[i] > o[i]:
            ob = (h[i], o[i], htf.index[i])
        elif i > 0 and c[i - 1] > o[i - 1]:
            ob = (h[i - 1], o[i - 1], htf.index[i - 1])

    # FVGs on the SMT candle or the following candles inside the window
    fvgs = []   # (top, bottom, start_time, known_at)
    for j in range(max(i, 2), min(i + p.arm_bars, n - 1) + 1):
        if direction == 1 and l[j] > h[j - 2]:
            fvgs.append((l[j], h[j - 2], htf.index[j - 2], htf.index[j] + htf_td))
        elif direction == -1 and h[j] < l[j - 2]:
            fvgs.append((l[j - 2], h[j], htf.index[j - 2], htf.index[j] + htf_td))

    start = t + htf_td                              # SMT candle has closed
    end = t + (p.arm_bars + 2) * htf_td             # setup window end
    times = ltf.index
    s = times.searchsorted(start, side="left")
    e = times.searchsorted(end, side="left")
    lo, hi = ltf["low"].to_numpy(), ltf["high"].to_numpy()
    op, cl = ltf["open"].to_numpy(), ltf["close"].to_numpy()
    atr_l = ltf["atr"].to_numpy()
    dn_run, up_run = ltf["dn_run_open"].to_numpy(), ltf["up_run_open"].to_numpy()

    res = dict(
        pair=pair, leg=leg, direction=direction, smt_time=t, swept_leg=swept, window_end=end,
        ob=ob, fvgs=fvgs, has_ob=ob is not None, has_fvg=False,
        status="ARMED" if end > now else "EXPIRED",
        csd_time=pd.NaT, entry=np.nan, sl=np.nan, tp=np.nan,
        outcome="", exit_time=pd.NaT, r_net=np.nan,
    )

    sweep = l[i] if direction == 1 else h[i]
    for k in range(s, e):
        sweep = min(sweep, lo[k]) if direction == 1 else max(sweep, hi[k])
        tk = times[k]
        has_fvg = any(f[3] <= tk for f in fvgs)
        if not _conf_ok(p.conf_mode, has_fvg, ob is not None):
            continue
        a_l = atr_l[k]
        if np.isnan(a_l) or abs(cl[k] - op[k]) < p.disp_mult * a_l:
            continue
        if direction == 1:
            trig = cl[k] > op[k] and not np.isnan(dn_run[k]) and cl[k] > dn_run[k]
        else:
            trig = cl[k] < op[k] and not np.isnan(up_run[k]) and cl[k] < up_run[k]
        if not trig:
            continue

        entry = cl[k]
        buf = p.atr_buf * (atr_h if not np.isnan(atr_h) else a_l)
        sl = sweep - buf if direction == 1 else sweep + buf
        risk = (entry - sl) if direction == 1 else (sl - entry)
        if risk <= entry * 1e-6:
            continue
        tp = entry + direction * p.rr * risk
        outcome, exit_t, r_gross = _simulate(direction, entry, sl, tp, lo, hi, times, k + 1, p.rr)
        cost_r = 2 * (p.fee_pct / 100) * entry / risk
        res.update(
            status="TRIGGERED", csd_time=tk, entry=entry, sl=sl, tp=tp, has_fvg=has_fvg,
            outcome=outcome, exit_time=exit_t,
            r_net=(r_gross - cost_r) if outcome in ("WIN", "LOSS") else np.nan,
        )
        break
    return res


# ───────────────────────────── public API ─────────────────────────────
def scan_pair(pair, name_a, name_b, data, p: Params, htf_minutes: int, now):
    """
    data[name] = (htf_df, ltf_df) of CLOSED candles, UTC index, columns open/high/low/close.
    Returns a list of setup dicts (all SMT setups in the loaded history).
    """
    htf_td = pd.Timedelta(minutes=htf_minutes)
    ha = prep_htf(data[name_a][0], p)
    hb = prep_htf(data[name_b][0], p)
    htfs = {name_a: ha, name_b: hb}
    ltfs = {name_a: prep_ltf(data[name_a][1], p), name_b: prep_ltf(data[name_b][1], p)}

    setups = []
    for t, direction, swept in smt_events(name_a, ha, name_b, hb):
        other = name_b if swept == name_a else name_a
        strong, weak = other, swept          # strong = leg that FAILED to sweep
        legs = {"strong": [strong], "weak": [weak], "both": [strong, weak]}[p.trade_leg]
        for leg in legs:
            setups.append(build_setup(pair, leg, direction, t, swept, htfs[leg], ltfs[leg], p, htf_td, now))
    return setups


def no_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one trade at a time per (pair, leg). Expects only TRIGGERED rows."""
    keep = []
    last_exit = {}
    for idx, r in df.sort_values("csd_time").iterrows():
        key = (r["pair"], r["leg"])
        busy_until = last_exit.get(key)
        if busy_until is not None and (pd.isna(busy_until) or r["csd_time"] <= busy_until):
            continue
        keep.append(idx)
        last_exit[key] = r["exit_time"]      # NaT (OPEN trade) blocks everything after it
    return df.loc[keep]


def stats(trades: pd.DataFrame) -> dict:
    """trades: closed trades (WIN/LOSS) with r_net."""
    if trades.empty:
        return dict(trades=0, wins=0, win_rate=0.0, avg_r=0.0, total_r=0.0, profit_factor=0.0, max_dd_r=0.0)
    r = trades.sort_values("exit_time")["r_net"].to_numpy()
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(
        trades=len(r), wins=int((trades["outcome"] == "WIN").sum()),
        win_rate=float((trades["outcome"] == "WIN").mean() * 100),
        avg_r=float(r.mean()), total_r=float(r.sum()),
        profit_factor=float(gp / gl) if gl > 0 else float("inf"),
        max_dd_r=float((peak - eq).max()),
    )


# ═══════════════════════════ 2) DATA LAYER ═══════════════════════════
TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}

BINANCE_HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api.binance.us",
]

# name -> (source, id)
BASE_ASSETS = {
    "BTC": ("binance", "BTCUSDT"),
    "ETH": ("binance", "ETHUSDT"),
    "SOL": ("binance", "SOLUSDT"),
    "AVAX": ("binance", "AVAXUSDT"),
    "GOLD": ("yf", "GC=F"),
    "SILVER": ("yf", "SI=F"),
}

PAIRS = {
    "BTC vs ETH": ("BTC", "ETH"),
    "ETH vs SOL": ("ETH", "SOL"),
    "BTC vs SOL": ("BTC", "SOL"),
    "SOL vs AVAX": ("SOL", "AVAX"),
    "GOLD vs SILVER": ("GOLD", "SILVER"),
}

# yfinance: (base interval, period). 3m and 4h are resampled.
_YF_BASE = {
    "1m": ("1m", "7d"), "3m": ("1m", "7d"), "5m": ("5m", "60d"), "15m": ("15m", "60d"),
    "30m": ("30m", "60d"), "1h": ("60m", "730d"), "4h": ("60m", "730d"),
}
_RESAMPLE = {"3m": "3min", "4h": "4h"}


def _binance_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["t", "open", "high", "low", "close", "volume"]
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("t")
    return df.astype(float)[["open", "high", "low", "close"]]


def fetch_binance(symbol: str, interval: str, bars: int):
    """Returns (df, host_used). Pages backwards with endTime, 1000 candles per request."""
    last_err = None
    for host in BINANCE_HOSTS:
        try:
            rows, end = [], None
            while len(rows) < bars:
                lim = min(1000, bars - len(rows))
                params = {"symbol": symbol, "interval": interval, "limit": lim}
                if end is not None:
                    params["endTime"] = end
                r = requests.get(f"{host}/api/v3/klines", params=params, timeout=15)
                r.raise_for_status()
                chunk = r.json()
                if not chunk:
                    break
                rows = chunk + rows
                end = chunk[0][0] - 1
                if len(chunk) < lim:
                    break
            if not rows:
                raise RuntimeError("empty response")
            df = _binance_df(rows)
            return df[~df.index.duplicated()].sort_index(), host.replace("https://", "")
        except Exception as e:   # try next host (451 geo-block, DNS, timeouts ...)
            last_err = f"{host}: {e}"
    raise RuntimeError(f"Binance fetch failed for {symbol} {interval} -> {last_err}")


def fetch_yf(ticker: str, interval: str, bars: int):
    import yfinance as yf

    base, period = _YF_BASE[interval]
    h = yf.Ticker(ticker).history(period=period, interval=base, auto_adjust=False)
    if h is None or h.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker} {interval}")
    h = h.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    h.index = pd.to_datetime(h.index)
    h.index = h.index.tz_localize("UTC") if h.index.tz is None else h.index.tz_convert("UTC")
    if interval in _RESAMPLE:
        h = (
            h.resample(_RESAMPLE[interval], origin="start_day", label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )
    return h.tail(bars), "yahoo finance"


def closed_only(df: pd.DataFrame, minutes: int, now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(minutes=minutes) <= now]


def load_asset(src: str, ident: str, htf: str, ltf: str, ltf_bars: int, swing_len: int):
    """Returns (htf_df, ltf_df, note) with CLOSED candles only."""
    htf_bars = math.ceil(ltf_bars * TF_MIN[ltf] / TF_MIN[htf]) + swing_len + 30
    fetch = fetch_binance if src == "binance" else fetch_yf
    h, note = fetch(ident, htf, htf_bars)
    l, _ = fetch(ident, ltf, ltf_bars)
    return closed_only(h, TF_MIN[htf]), closed_only(l, TF_MIN[ltf]), note


# ═══════════════════════════ 3) STREAMLIT UI ═══════════════════════════
st.set_page_config(page_title="ALGOX SMT Scanner", page_icon="📈", layout="wide")

TF_SETS = {"15m → 1m": ("15m", "1m"), "30m → 3m": ("30m", "3m"), "1h → 5m": ("1h", "5m"), "4h → 15m": ("4h", "15m")}
GREEN, RED = "#00c853", "#ff1744"


def ist(ts):
    """UTC timestamp -> naive IST timestamp for display."""
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts).tz_convert("Asia/Kolkata").tz_localize(None)


# ─────────────────────────────── sidebar ───────────────────────────────
with st.sidebar:
    st.title("📈 ALGOX SMT")
    tf_label = st.selectbox("Timeframe set (SMT read → CSD entry)", list(TF_SETS), index=0)
    HTF, LTF = TF_SETS[tf_label]
    pair_sel = st.selectbox("Pair (Chart & Backtest tabs)", list(PAIRS))
    st.subheader("Logic")
    swing_len = st.slider("SMT lookback (HTF candles)", 3, 30, 10)
    arm_bars = st.slider("Setup window after SMT (HTF candles)", 1, 10, 4)
    conf_mode = st.selectbox("Required confluence", ["FVG or OB", "FVG and OB", "None"])
    disp_mult = st.slider("CSD displacement (× ATR)", 0.3, 3.0, 1.0, 0.1)
    trade_leg = st.selectbox("Trade leg", ["strong", "weak", "both"],
                             help="strong = leg that FAILED to sweep (ICT preference). weak = leg that swept.")
    st.subheader("Risk")
    rr = st.slider("Reward : Risk", 0.5, 5.0, 2.0, 0.25)
    atr_buf = st.slider("SL buffer (× HTF ATR)", 0.0, 1.0, 0.25, 0.05)
    fee = st.number_input("Fee per side (%)", 0.0, 1.0, 0.10, 0.01)
    st.subheader("Data")
    ltf_bars = st.select_slider("LTF candles to load", [500, 1000, 2000, 3000, 5000], value=2000)
    with st.expander("Metals tickers (Yahoo)"):
        gold_t = st.text_input("Gold", "GC=F")
        silver_t = st.text_input("Silver", "SI=F")
    if st.button("🔄 Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

P = Params(swing_len=swing_len, arm_bars=arm_bars, conf_mode=conf_mode, disp_mult=disp_mult,
           rr=rr, atr_buf=atr_buf, trade_leg=trade_leg, fee_pct=fee)
ASSETS = dict(BASE_ASSETS)
ASSETS["GOLD"] = ("yf", gold_t.strip() or "GC=F")
ASSETS["SILVER"] = ("yf", silver_t.strip() or "SI=F")


# ─────────────────────────────── data ───────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def get_asset(src, ident, htf, ltf, bars, swing):
    return load_asset(src, ident, htf, ltf, bars, swing)


def load_all(names):
    data, notes, errs = {}, {}, {}
    for n in names:
        src, ident = ASSETS[n]
        try:
            h, l, note = get_asset(src, ident, HTF, LTF, ltf_bars, swing_len)
            if len(h) < swing_len + 5 or len(l) < 50:
                raise RuntimeError("not enough candles returned")
            data[n], notes[n] = (h, l), note
        except Exception as e:
            errs[n] = str(e)
    return data, notes, errs


def scan(pairs, data):
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for pname in pairs:
        a, b = PAIRS[pname]
        if a in data and b in data:
            rows += scan_pair(pname, a, b, data, P, TF_MIN[HTF], now)
    return rows


def to_df(setups):
    return pd.DataFrame(setups) if setups else pd.DataFrame()


# ─────────────────────────────── page ───────────────────────────────
st.title("ALGOX SMT Scanner")
st.caption(f"SMT read on **{HTF}** · CSD entry on **{LTF}** · crypto = Binance spot · metals = COMEX (Yahoo) · times in IST")

tab_scan, tab_chart, tab_bt, tab_help = st.tabs(["🔎 Scanner", "📊 Chart", "🧪 Backtest", "ℹ️ Help"])

with st.spinner("Loading market data…"):
    data, notes, errs = load_all(list(ASSETS))

for n, e in errs.items():
    st.warning(f"{n}: {e}")
if notes:
    st.caption("Data source: " + " · ".join(f"{k}→{v}" for k, v in notes.items()))

all_setups = scan(list(PAIRS), data) if data else []
all_df = to_df(all_setups)

# ───────────────────────────── scanner ─────────────────────────────
with tab_scan:
    if all_df.empty:
        st.info("No SMT setups found in the loaded data (or data failed to load).")
    else:
        live = all_df[(all_df["status"] == "ARMED") | ((all_df["status"] == "TRIGGERED") & (all_df["outcome"] == "OPEN"))]
        st.subheader("🔔 Live setups")
        if live.empty:
            st.write("Nothing armed or open right now.")
        else:
            show = pd.DataFrame({
                "Pair": live["pair"], "Trade": live["leg"],
                "Bias": np.where(live["direction"] == 1, "🟢 BUY", "🔴 SELL"),
                "State": np.where(live["status"] == "ARMED", "ARMED – wait CSD", "ENTERED"),
                "SMT (IST)": live["smt_time"].map(ist),
                "OB": live["has_ob"].map({True: "✓", False: "—"}),
                "FVG": np.where(live["status"] == "TRIGGERED", live["has_fvg"].map({True: "✓", False: "—"}),
                                live["fvgs"].map(lambda f: "✓" if len(f) else "—")),
                "CSD (IST)": live["csd_time"].map(ist),
                "Entry": live["entry"], "SL": live["sl"], "TP": live["tp"],
                "Window ends (IST)": live["window_end"].map(ist),
            }).sort_values("SMT (IST)", ascending=False)
            st.dataframe(show, hide_index=True)

        st.subheader("Latest SMT per pair")
        last = all_df.sort_values("smt_time").groupby(["pair", "leg"]).tail(1)
        view = pd.DataFrame({
            "Pair": last["pair"], "Leg": last["leg"],
            "SMT": np.where(last["direction"] == 1, "🟢 Bullish", "🔴 Bearish"),
            "SMT time (IST)": last["smt_time"].map(ist),
            "State": last["status"], "Outcome": last["outcome"],
            "Entry": last["entry"], "SL": last["sl"], "TP": last["tp"],
        }).sort_values("SMT time (IST)", ascending=False)
        st.dataframe(view, hide_index=True)
        st.caption("Press 🔄 Refresh data in the sidebar to re-scan (data cache = 60 s).")

# ───────────────────────────── chart ─────────────────────────────
with tab_chart:
    a, b = PAIRS[pair_sel]
    if a not in data or b not in data:
        st.info("Data for this pair did not load.")
    else:
        leg_pick = st.radio("Chart leg", [a, b], horizontal=True)
        view_bars = st.slider("Candles shown", 100, 1500, 500, 50)
        sel_p = Params(**{**P.__dict__, "trade_leg": "both"})
        setups = [s for s in scan_pair(pair_sel, a, b, data, sel_p, TF_MIN[HTF], pd.Timestamp.now(tz="UTC"))
                  if s["leg"] == leg_pick]
        d = data[leg_pick][1].tail(view_bars)
        x = d.index.tz_convert("Asia/Kolkata").tz_localize(None)
        fig = go.Figure(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                       increasing_line_color=GREEN, decreasing_line_color=RED, name=leg_pick))
        t_min = d.index[0]
        for s in setups:
            if s["window_end"] < t_min:
                continue
            col = GREEN if s["direction"] == 1 else RED
            x0, x1 = ist(s["smt_time"]), ist(s["window_end"])
            if s["ob"] is not None:
                fig.add_shape(type="rect", x0=ist(s["ob"][2]), x1=x1, y0=s["ob"][1], y1=s["ob"][0],
                              line=dict(color=col, width=1), fillcolor=col, opacity=0.15, layer="below")
            for top, bot, st_t, _ in s["fvgs"]:
                fig.add_shape(type="rect", x0=ist(st_t), x1=x1, y0=bot, y1=top,
                              line=dict(color=col, width=1, dash="dot"), fillcolor=col, opacity=0.10, layer="below")
            seg = d[(d.index >= s["smt_time"]) & (d.index < s["smt_time"] + pd.Timedelta(minutes=TF_MIN[HTF]))]
            if not seg.empty:
                y_pt = seg["low"].min() if s["direction"] == 1 else seg["high"].max()
                fig.add_annotation(x=x0, y=y_pt, text="SMT▲" if s["direction"] == 1 else "SMT▼",
                                   showarrow=True, arrowhead=2, ay=30 if s["direction"] == 1 else -30,
                                   font=dict(color=col, size=11), arrowcolor=col)
            if s["status"] == "TRIGGERED":
                xe = ist(s["csd_time"])
                fig.add_trace(go.Scatter(x=[xe], y=[s["entry"]], mode="markers+text",
                                         marker=dict(symbol="triangle-up" if s["direction"] == 1 else "triangle-down",
                                                     size=14, color=col),
                                         text=["BUY" if s["direction"] == 1 else "SELL"], textposition="top center",
                                         showlegend=False))
                x_end = ist(s["exit_time"]) if not pd.isna(s["exit_time"]) else x[-1]
                for y, dash in ((s["sl"], "dot"), (s["tp"], "dash")):
                    fig.add_shape(type="line", x0=xe, x1=x_end, y0=y, y1=y, line=dict(color=col, width=1, dash=dash))
        fig.update_layout(height=650, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                          title=f"{leg_pick} · {LTF} · {pair_sel} (SMT on {HTF})")
        st.plotly_chart(fig, width="stretch")
        st.caption("Solid box = Order Block · dotted box = FVG · triangle = CSD entry · dashed = TP · dotted line = SL")

# ───────────────────────────── backtest ─────────────────────────────
with tab_bt:
    scope = st.radio("Scope", ["Selected pair", "All pairs"], horizontal=True)
    bt = all_df if scope == "All pairs" else (all_df[all_df["pair"] == pair_sel] if not all_df.empty else all_df)
    if bt.empty:
        st.info("No setups to backtest in the loaded data.")
    else:
        trig = bt[bt["status"] == "TRIGGERED"]
        trig = no_overlap(trig) if not trig.empty else trig
        closed = trig[trig["outcome"].isin(["WIN", "LOSS"])] if not trig.empty else trig
        s = stats(closed)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trades", s["trades"])
        c2.metric("Win rate", f"{s['win_rate']:.1f}%")
        c3.metric("Total R (after fees)", f"{s['total_r']:.2f}")
        c4.metric("Profit factor", "∞" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}")
        c5.metric("Max DD (R)", f"{s['max_dd_r']:.2f}")
        st.caption(f"{len(bt)} SMT setups → {len(trig)} triggered (one trade at a time per pair/leg) → "
                   f"{len(closed)} closed. Window covers only the {ltf_bars} {LTF} candles loaded.")
        if not closed.empty:
            ordered = closed.sort_values("exit_time")
            eq = ordered["r_net"].cumsum().to_numpy()
            fig = go.Figure(go.Scatter(x=ordered["exit_time"].map(ist), y=eq, mode="lines+markers",
                                       line=dict(color=GREEN)))
            fig.update_layout(height=320, title="Equity curve (R)", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")
            tbl = pd.DataFrame({
                "Pair": ordered["pair"], "Leg": ordered["leg"],
                "Side": np.where(ordered["direction"] == 1, "BUY", "SELL"),
                "SMT (IST)": ordered["smt_time"].map(ist), "Entry (IST)": ordered["csd_time"].map(ist),
                "Exit (IST)": ordered["exit_time"].map(ist),
                "Entry": ordered["entry"], "SL": ordered["sl"], "TP": ordered["tp"],
                "Result": ordered["outcome"], "R": ordered["r_net"].round(2),
            })
            st.dataframe(tbl, hide_index=True)
        if not trig.empty and (trig["outcome"] == "OPEN").any():
            st.info(f"{int((trig['outcome'] == 'OPEN').sum())} trade(s) still open – not counted.")

# ───────────────────────────── help ─────────────────────────────
with tab_help:
    st.markdown(f"""
**Logic**
1. **SMT** – on a closed **{HTF}** candle exactly one leg of the pair takes the high/low of the previous *N* candles, the other leg refuses.
2. **Confluence** – Order Block at the SMT candle and/or FVG on the SMT candle or inside the setup window.
3. **CSD** – on **{LTF}**, a displacement candle closes beyond the open of the last opposite-close candle series.
4. **Trade** – SL beyond the sweep extreme (+ATR buffer), TP = RR × risk. Signals use closed candles only (no repaint).

**Limits**
- Backtests only cover the loaded candles (e.g. 2000 × 1m ≈ 33 hours). Yahoo 1m data is limited to the last 7 days.
- Results are a sanity check, not proof. Forward-test on demo before real or funded-account money.
- Not financial advice.
""")
