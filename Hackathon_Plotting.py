import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# INPUT PATHS
# =============================================================================
HISTORICAL_XLSX = Path(r"C:\Users\mhasa\Desktop\Hackathon\HackathonDataset.xlsx")
FWD_ERCOT       = Path(r"C:\Users\mhasa\Desktop\Hackathon\Forward price_ERCOT.xlsx")
FWD_MISO        = Path(r"C:\Users\mhasa\Desktop\Hackathon\Forward price_MISO.xlsx")
FWD_CAISO       = Path(r"C:\Users\mhasa\Desktop\Hackathon\Forward price_CAISO.xlsx")

# =============================================================================
# MODEL SETTINGS
# =============================================================================
TECH_TYPES      = {"ERCOT": "Wind", "MISO": "Wind", "CAISO": "Solar"}
DEGRADATION     = {"Wind": 0.007, "Solar": 0.005}            # 0.7% / 0.5% per year
YEARS           = [2026, 2027, 2028, 2029, 2030]
WACC            = 0.07
W_DA, W_RT      = 0.8, 0.2
SIMS            = 5000
QUANTILES       = (0.5, 0.75, 0.9)
OUTDIR          = Path("results")

# =============================================================================
# UTILITIES
# =============================================================================
def ensure_exists(p: Path, label="file"):
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")

def build_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\n"," ").replace("  "," ") for c in df.columns]

    def find_col(tokens):
        for c in df.columns:
            lc = c.lower()
            if any(tok in lc for tok in tokens):
                return c
        return None

    ts_col = find_col(["timestamp","datetime","interval ending","settlement interval"])
    if ts_col:
        df["ts"] = pd.to_datetime(df[ts_col], errors="coerce")
        if df["ts"].notna().any():
            return df

    date_col = find_col(["date","delivery date","trade date"])
    he_col   = find_col(["he","hour ending","hr ending","hour"])
    if date_col and he_col:
        d = pd.to_datetime(df[date_col], errors="coerce")
        he_raw = pd.to_numeric(df[he_col], errors="coerce")
        hour = (he_raw - 1).clip(lower=0, upper=23).fillna(0).astype(int)
        df["ts"] = d + pd.to_timedelta(hour, unit="h")
        return df

    time_col = find_col(["time"])
    if date_col and time_col:
        d = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype("datetime64[ns]")
        t = pd.to_timedelta(df[time_col].astype(str), errors="coerce")
        df["ts"] = d + t
        return df

    raise ValueError("Could not build timestamps. Expect 'Timestamp' OR ('Date','HE').")

def compute_peak_flag(ts: pd.Series, he: pd.Series, market: str) -> pd.Series:
    dow = ts.dt.dayofweek  # Mon=0 ... Sun=6

    market = market.upper()
    if market == "ERCOT":
        is_peak = (dow <= 4) & (he.between(7, 22))          # Mon–Fri, HE 7–22
    elif market == "MISO":
        is_peak = (dow <= 4) & (he.between(8, 23))          # Mon–Fri, HE 8–23
    elif market == "CAISO":
        is_peak = (dow <= 5) & (he.between(7, 22))          # Mon–Sat, HE 7–22
    else:
        # safe default (was your old rule)
        is_peak = (dow <= 4) & (he.between(7, 22))
    return np.where(is_peak, "P", "OP")

def tidy_hourly(raw: pd.DataFrame, market: str) -> pd.DataFrame:
    df = build_ts(raw)

    ren = {
        "Gen":"Gen_MWh", "Gen ":"Gen_MWh", "Generation (MWh)":"Gen_MWh", "Output":"Gen_MWh",
        "DA Hub":"DA_Hub","RT Hub":"RT_Hub","DA Busbar":"DA_Busbar","RT Busbar":"RT_Busbar",
        "P/OP":"P_OP"
    }
    for k,v in ren.items():
        if k in df.columns: df = df.rename(columns={k:v})

    for col in ["Gen_MWh","DA_Hub","RT_Hub","DA_Busbar","RT_Busbar"]:
        if col not in df.columns: df[col] = pd.NA
        df[col] = (df[col].astype(str)
                        .str.replace('[$,]','',regex=True)
                        .str.replace(r'\((\d+\.?\d*)\)', r'-\1', regex=True))
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ts"] = pd.to_datetime(df["ts"])
    df["Year"] = df["ts"].dt.year
    df["Month"] = df["ts"].dt.month
    df["HE"] = df["ts"].dt.hour + 1

    if "P_OP" in df.columns and df["P_OP"].notna().any():
        p = df["P_OP"].astype(str).str.upper().str.replace(" ", "").str.replace("-", "")
        p = p.replace({"PEAK": "P", "ONPEAK": "P", "ON": "P", "OFFPEAK": "OP", "OFF": "OP"})
        df["P_OP"] = np.where(p == "P", "P", "OP")
    else:
        df["P_OP"] = compute_peak_flag(df["ts"], df["HE"], market)

    df["Gen_MWh"] = df["Gen_MWh"].fillna(0.0)
    df.loc[df["Gen_MWh"]<0, "Gen_MWh"] = 0.0

    keep = ["ts","Year","Month","P_OP","Gen_MWh","DA_Hub","RT_Hub","DA_Busbar","RT_Busbar"]
    return df[keep].sort_values("ts").reset_index(drop=True)

def load_sheet(history_path: Path, sheet: str) -> pd.DataFrame:
    ensure_exists(history_path, "historical workbook")
    raw0 = pd.read_excel(history_path, sheet_name=sheet, header=None)
    hdr = 0
    for i, row in raw0.iterrows():
        row_s = row.astype(str).str.lower()
        if row_s.str.contains("date|timestamp|datetime|interval").any():
            hdr = i
            break
    raw = pd.read_excel(history_path, sheet_name=sheet, header=hdr)
    return tidy_hourly(raw, market=sheet)

def load_forwards(forward_path: Path) -> pd.DataFrame:
    ensure_exists(forward_path, "forward file")
    if forward_path.suffix.lower() == ".csv":
        df = pd.read_csv(forward_path)
    else:
        df = pd.read_excel(forward_path)
    df.columns = [str(c).strip() for c in df.columns]

    time_col = None
    for c in df.columns:
        if c.lower().startswith("time") or c.lower() in ("date","month","period"):
            time_col = c
            break
    if time_col is None:
        raise KeyError(f"Could not find a 'Time' column in {forward_path}")

    peak_col = next((c for c in df.columns if "peak" in c.lower() and "off" not in c.lower()), None)
    off_col  = next((c for c in df.columns if "off" in c.lower() and "peak" in c.lower()), None)
    if peak_col is None or off_col is None:
        raise KeyError(f"Could not find 'Peak' and 'Off Peak' columns in {forward_path}")

    t = pd.to_datetime(df[time_col], errors="coerce")
    bad = t.isna()
    if bad.any():
        t2 = pd.to_datetime(df.loc[bad, time_col].astype(str), format="%b-%y", errors="coerce")
        t.loc[bad] = t2
    bad = t.isna()
    if bad.any():
        t3 = pd.to_datetime(df.loc[bad, time_col].astype(str), errors="coerce")
        t.loc[bad] = t3
    if t.isna().any():
        bad_vals = df.loc[t.isna(), time_col].astype(str).unique().tolist()
        raise ValueError(f"Could not parse Time values like: {bad_vals[:5]}")

    def clean_price(x):
        s = str(x).strip().replace("$", "")
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
        s = re.sub(r"\((\s*\d+\.?\d*)\)", r"-\1", s)
        try:
            return float(s)
        except Exception:
            return np.nan

    peak_vals = df[peak_col].apply(clean_price).astype(float)
    off_vals  = df[off_col].apply(clean_price).astype(float)

    fwd_peak = pd.DataFrame({
        "Year":  t.dt.year.astype(int),
        "Month": t.dt.month.astype(int),
        "Period": "peak",
        "Forward_Hub_Price": peak_vals.values
    })
    fwd_off = pd.DataFrame({
        "Year":  t.dt.year.astype(int),
        "Month": t.dt.month.astype(int),
        "Period": "offpeak",
        "Forward_Hub_Price": off_vals.values
    })
    fwd = pd.concat([fwd_peak, fwd_off], ignore_index=True).dropna(subset=["Forward_Hub_Price"])
    return fwd

# =============================================================================
# VOL DRIVERS + SIM ENGINE
# =============================================================================
def build_driver_pools(hourly_df: pd.DataFrame):
    df = hourly_df.copy()
    df["month"] = df["Month"]
    df["is_peak"] = df["P_OP"].astype(str).str.upper().eq("P")

    seas = (df.groupby(["month","is_peak"])[["DA_Hub","RT_Hub"]]
              .mean().rename(columns={"DA_Hub":"DA_mu","RT_Hub":"RT_mu"}))

    df = df.join(seas, on=["month","is_peak"])
    df["DA_shock"] = df["DA_Hub"] / df["DA_mu"]
    df["RT_shock"] = df["RT_Hub"] / df["RT_mu"]
    df["Basis_DA"] = df["DA_Busbar"] - df["DA_Hub"]
    df["Basis_RT"] = df["RT_Busbar"] - df["RT_Hub"]

    pools = {}
    for m in range(1,13):
        for pk in [False, True]:
            sub = df[(df["month"]==m)&(df["is_peak"]==pk)][["Gen_MWh","DA_shock","RT_shock","Basis_DA","Basis_RT"]]
            pools[(m,pk)] = sub.dropna().reset_index(drop=True)
    return pools, seas

def monthly_gen_forecast(hourly_df: pd.DataFrame, tech: str, years: list, degradation_map: dict):
    df = hourly_df[hourly_df["Gen_MWh"]>0].copy()
    grp = (df.groupby(["Month","P_OP"])["Gen_MWh"]
             .agg(mean="mean", std="std", hours="count").reset_index())
    base_year = int(df["Year"].max())
    degr_rate = float(degradation_map.get(tech, 0.0))

    rows = []
    for y in years:
        factor = (1 - degr_rate)**(y - base_year)
        for _, r in grp.iterrows():
            rows.append({
                "Year": y,
                "Month": int(r["Month"]),
                "Period": "peak" if str(r["P_OP"]).upper()=="P" else "offpeak",
                "Expected_Gen_MW": float(r["mean"]) * factor,
                "Gen_Std": float(r["std"]) * factor,
                "Hours": int(r["hours"])
            })
    return pd.DataFrame(rows)

def simulate_term(
    pools, seas, forwards_long: pd.DataFrame, gen_monthly: pd.DataFrame,
    start_year: int, months: int, w_da=0.8, w_rt=0.2, rate=0.07,
    sims=5000, seed=2026, dmb_mode: str = "both",
    no_take_negative: bool = True,          # <— NEW FLAG
    exclude_denominator_when_not_taken: bool = False  # <— OPTIONAL: exclude curtailed MWh from PV denom
):
    rng = np.random.default_rng(seed)
    ym = pd.period_range(f"{start_year}-01", periods=months, freq="M")
    cal = []
    for per in ym:
        cal.append((per, per.month, True))   # peak
        cal.append((per, per.month, False))  # offpeak

    gm = gen_monthly.copy()
    gm["is_peak"] = gm["Period"].eq("peak")
    g_idx = {(int(r.Year), int(r.Month), bool(r.is_peak)):(float(r.Expected_Gen_MW), int(r.Hours))
             for _,r in gm.iterrows()}

    fw = forwards_long.copy()
    fw["is_peak"] = fw["Period"].eq("peak")
    f_idx = {(int(r.Year), int(r.Month), bool(r.is_peak)): float(r.Forward_Hub_Price) for _,r in fw.iterrows()}

    def d(t): return (1.0 + rate)**(-(t/12.0))
    dfs = np.array([d(t) for t in range(1, months+1)])
    dfs = np.repeat(dfs, 2)

    Z_RT_Bus, Z_DA_Bus, Z_RT_Hub, Z_DA_Hub = [], [], [], []
    Z_DMB_Bus, Z_DMB_Hub = [], []

    for _ in range(sims):
        pv_rev_rt_bus = pv_rev_da_bus = pv_rev_rt_hub = pv_rev_da_hub = 0.0
        pv_rev_dmb_bus = pv_rev_dmb_hub = 0.0
        pv_e = 0.0

        for k,(per, m, pk) in enumerate(cal):
            pool = pools.get((m,pk))
            if pool is None or pool.empty:
                continue
            row = pool.iloc[rng.integers(0, len(pool))]

            fwd = f_idx.get((int(per.year), int(m), bool(pk)), np.nan)
            if not np.isfinite(fwd):
                mu = seas.loc[(m,pk)]
                da_mu, rt_mu = float(mu["DA_mu"]), float(mu["RT_mu"])
            else:
                da_mu = rt_mu = float(fwd)

            DA_hub = da_mu * float(row["DA_shock"])
            RT_hub = rt_mu * float(row["RT_shock"])
            DA_bus = DA_hub + float(row["Basis_DA"])
            RT_bus = RT_hub + float(row["Basis_RT"])

            g_key = (int(per.year), int(m), bool(pk))
            if g_key not in g_idx:
                continue
            g_mw, hours = g_idx[g_key]
            gen_mwh = g_mw * hours

            disc = dfs[k]

            # ---------- NEW: no-take-at-negative logic ----------
            prices = {"RT_Hub": RT_hub, "DA_Hub": DA_hub, "RT_Bus": RT_bus, "DA_Bus": DA_bus}

            def eff_energy(price):
                """If no_take_negative, zero the MWh for that product when price < 0."""
                return 0.0 if (no_take_negative and price < 0.0) else gen_mwh

            # Per-product revenues
            e_rt_bus = eff_energy(prices["RT_Bus"])
            e_da_bus = eff_energy(prices["DA_Bus"])
            e_rt_hub = eff_energy(prices["RT_Hub"])
            e_da_hub = eff_energy(prices["DA_Hub"])

            pv_rev_rt_bus += prices["RT_Bus"] * e_rt_bus * disc
            pv_rev_da_bus += prices["DA_Bus"] * e_da_bus * disc
            pv_rev_rt_hub += prices["RT_Hub"] * e_rt_hub * disc
            pv_rev_da_hub += prices["DA_Hub"] * e_da_hub * disc

            # DMB products (use local w_da/w_rt, not globals)
            dmb_bus_price = (w_da * prices["DA_Bus"] + w_rt * prices["RT_Bus"])
            dmb_hub_price = (w_da * prices["DA_Hub"] + w_rt * prices["RT_Hub"])
            e_dmb_bus = eff_energy(dmb_bus_price)
            e_dmb_hub = eff_energy(dmb_hub_price)

            pv_rev_dmb_bus += dmb_bus_price * e_dmb_bus * disc
            pv_rev_dmb_hub += dmb_hub_price * e_dmb_hub * disc

            # PV energy denominator — choose policy
            if exclude_denominator_when_not_taken:
                # Denominator reflects curtailed hours (only energy actually "taken")
                pv_e += max(e_rt_bus, e_da_bus, e_rt_hub, e_da_hub, e_dmb_bus, e_dmb_hub) * disc
            else:
                # Denominator keeps physical generation (even if not taken/paid)
                pv_e += gen_mwh * disc
            # ----------------------------------------------------

        if pv_e > 0:
            Z_RT_Bus.append(pv_rev_rt_bus / pv_e)
            Z_DA_Bus.append(pv_rev_da_bus / pv_e)
            Z_RT_Hub.append(pv_rev_rt_hub / pv_e)
            Z_DA_Hub.append(pv_rev_da_hub / pv_e)
            Z_DMB_Bus.append(pv_rev_dmb_bus / pv_e)
            Z_DMB_Hub.append(pv_rev_dmb_hub / pv_e)

    cols = {
        "Z_RT_Busbar": Z_RT_Bus,
        "Z_DA_Busbar": Z_DA_Bus,
        "Z_RT_Hub":    Z_RT_Hub,
        "Z_DA_Hub":    Z_DA_Hub,
    }
    if dmb_mode in ("bus","both"):
        cols["Z_DMB_Bus"] = Z_DMB_Bus
    if dmb_mode in ("hub","both"):
        cols["Z_DMB_Hub"] = Z_DMB_Hub

    return pd.DataFrame(cols)


def summarize_prices(dist: pd.DataFrame, w_da=0.8, w_rt=0.2, quantiles=(0.5,0.75,0.9)) -> pd.DataFrame:
    rows = []
    label_map = [
        ("Z_RT_Busbar", "RT_Busbar"),
        ("Z_DA_Busbar", "DA_Busbar"),
        ("Z_RT_Hub",    "RT_Hub"),
        ("Z_DA_Hub",    "DA_Hub"),
    ]
    if "Z_DMB_Bus" in dist.columns:
        label_map.insert(0, ("Z_DMB_Bus", f"DMB(DA{w_da*100:.0f}/RT{w_rt*100:.0f})-Busbar"))
    if "Z_DMB_Hub" in dist.columns:
        label_map.insert(0, ("Z_DMB_Hub", f"DMB(DA{w_da*100:.0f}/RT{w_rt*100:.0f})-Hub"))

    for col, name in label_map:
        s = dist[col].dropna()
        row = {"Product": name}
        for q in quantiles:
            row[f"P{int(q*100)}"] = float(np.percentile(s, q*100))
        rows.append(row)
    return pd.DataFrame(rows)

def volatility_driver_summaries(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()
    df["Spread_Bus"] = df["DA_Busbar"] - df["RT_Busbar"]
    df["Spread_Hub"] = df["DA_Hub"] - df["RT_Hub"]
    df["Basis_RT"]   = df["RT_Busbar"] - df["RT_Hub"]
    df["Basis_DA"]   = df["DA_Busbar"] - df["DA_Hub"]
    df["date"] = df["ts"].dt.date
    daily = df.groupby("date")[["RT_Hub","DA_Hub"]].agg(["min","max"])
    daily.columns = ["RT_min","RT_max","DA_min","DA_max"]
    daily["RT_peak2trough"] = daily["RT_max"] - daily["RT_min"]
    daily["DA_peak2trough"] = daily["DA_max"] - daily["DA_min"]

    def qdesc(s):
        s = s.dropna()
        if s.empty:
            return pd.Series({"mean":0,"p10":0,"p50":0,"p90":0,"std":0})
        return pd.Series({"mean":s.mean(),"p10":s.quantile(0.1),"p50":s.quantile(0.5),"p90":s.quantile(0.9),"std":s.std()})

    corr_rt = df[["Gen_MWh","RT_Hub"]].dropna()
    corr_da = df[["Gen_MWh","DA_Hub"]].dropna()
    rho_rt  = float(corr_rt.corr().iloc[0,1]) if corr_rt.shape[0] > 2 else 0.0
    rho_da  = float(corr_da.corr().iloc[0,1]) if corr_da.shape[0] > 2 else 0.0

    out = pd.concat({
        "DA/RT Busbar Spread": df["Spread_Bus"].pipe(qdesc),
        "DA/RT Hub Spread":    df["Spread_Hub"].pipe(qdesc),
        "RT Node-Hub Basis":   df["Basis_RT"].pipe(qdesc),
        "DA Node-Hub Basis":   df["Basis_DA"].pipe(qdesc),
        "Daily RT Peak-Trough": daily["RT_peak2trough"].pipe(qdesc),
        "Daily DA Peak-Trough": daily["DA_peak2trough"].pipe(qdesc),
        "Corr(Gen, RT_Hub)":   pd.Series({"mean": rho_rt}),
        "Corr(Gen, DA_Hub)":   pd.Series({"mean": rho_da})
    }, axis=1).T.reset_index().rename(columns={"index":"Metric"})
    return out

# =============================================================================
# EXCEL REPORT EXPORTER
# =============================================================================
def _ym_label(df):
    out = df.copy()
    out["YM"] = pd.to_datetime(dict(year=out["Year"], month=out["Month"], day=1))
    out["YM_label"] = out["YM"].dt.strftime("%Y-%m")
    return out

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def plot_price_matrix_compare(market: str, pm_base: pd.DataFrame, pm_no: pd.DataFrame, outdir: Path):
    """
    Bar plot of P50/P75/P90 for each product; baseline vs no-take.
    """
    out = _ensure_dir(outdir / "plots")
    # align rows by 'Product'
    j = pm_base.set_index("Product")[["P50","P75","P90"]].join(
        pm_no.set_index("Product")[["P50","P75","P90"]], lsuffix="_BASE", rsuffix="_NO", how="inner"
    )
    if j.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    idx = np.arange(len(j.index))
    w = 0.35
    # P75 is the “risk appetite” – show that pair emphasized
    ax.bar(idx - w/2, j["P75_BASE"].values, width=w, label="P75 (Baseline)")
    ax.bar(idx + w/2, j["P75_NO"].values,   width=w, label="P75 (No-take)")

    ax.set_xticks(idx)
    ax.set_xticklabels(j.index, rotation=15, ha="right")
    ax.set_ylabel("$/MWh")
    ax.set_title(f"{market} – P75 Fair Prices: Baseline vs No-take (<0$ not taken)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / f"{market}_P75_compare.png", dpi=160)
    plt.close(fig)

    # Optional: also plot P50/P90 comparisons as small multiples
    for p in ("P50","P90"):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(idx - w/2, j[f"{p}_BASE"].values, width=w, label=f"{p} (Baseline)")
        ax.bar(idx + w/2, j[f"{p}_NO"].values,   width=w, label=f"{p} (No-take)")
        ax.set_xticks(idx)
        ax.set_xticklabels(j.index, rotation=15, ha="right")
        ax.set_ylabel("$ / MWh")
        ax.set_title(f"{market} – {p} Prices: Baseline vs No-take")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out / f"{market}_{p}_compare.png", dpi=160)
        plt.close(fig)

def plot_scenario_hist_compare(market: str,
                               dist_base: pd.DataFrame,
                               dist_no: pd.DataFrame,
                               products=("Z_DMB_Hub","Z_RT_Busbar"),
                               outdir: Path=None):
    """
    Overlay histograms of scenario unit-PV prices for key products.
    """
    out = _ensure_dir(outdir / "plots")
    for p in products:
        if (p not in dist_base.columns) or (p not in dist_no.columns):
            continue
        s0 = dist_base[p].dropna()
        s1 = dist_no[p].dropna()
        if s0.empty or s1.empty:
            continue
        fig, ax = plt.subplots(figsize=(10,6))
        ax.hist(s0, bins=40, alpha=0.6, label="Baseline")
        ax.hist(s1, bins=40, alpha=0.6, label="No-take")
        ax.set_title(f"{market} – Scenario Distribution: {p}")
        ax.set_xlabel("$ / MWh (unit PV)")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out / f"{market}_{p}_scenario_compare.png", dpi=160)
        plt.close(fig)

def plot_generation_time_series(market: str, gen_fore: pd.DataFrame, outdir: Path):
    out = _ensure_dir(outdir / "plots")
    g = gen_fore.copy()
    g["YM"] = pd.to_datetime(dict(year=g["Year"], month=g["Month"], day=1))
    # sum MWh by month/period
    g["MWh"] = g["Expected_Gen_MW"] * g["Hours"]
    piv = g.pivot_table(index="YM", columns="Period", values="MWh", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(12,6))
    if "peak" in piv.columns:
        ax.plot(piv.index, piv["peak"].values, label="Peak")
    if "offpeak" in piv.columns:
        ax.plot(piv.index, piv["offpeak"].values, label="Off-peak")
    ax.set_title(f"{market} – Monthly Expected Generation (MWh)")
    ax.set_ylabel("MWh")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / f"{market}_ExpectedGen.png", dpi=160)
    plt.close(fig)

def plot_forwards_time_series(market: str, forwards_long: pd.DataFrame, outdir: Path):
    out = _ensure_dir(outdir / "plots")
    f = forwards_long.copy()
    f["YM"] = pd.to_datetime(dict(year=f["Year"], month=f["Month"], day=1))
    piv = f.pivot_table(index="YM", columns="Period", values="Forward_Hub_Price", aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(12,6))
    if "peak" in piv.columns:
        ax.plot(piv.index, piv["peak"].values, label="Peak")
    if "offpeak" in piv.columns:
        ax.plot(piv.index, piv["offpeak"].values, label="Off-peak")
    ax.set_title(f"{market} – Hub Forwards ($/MWh)")
    ax.set_ylabel("$/MWh")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / f"{market}_Forwards.png", dpi=160)
    plt.close(fig)

def plot_vol_drivers_quickview(market: str, drivers: pd.DataFrame, outdir: Path):
    """
    Small bar chart of mean spread/basis for quick visual.
    """
    out = _ensure_dir(outdir / "plots")
    d = drivers.copy()
    focus = d[d["Metric"].isin(["DA/RT Busbar Spread","DA/RT Hub Spread",
                                "RT Node-Hub Basis","DA Node-Hub Basis"])]
    if focus.empty:
        return
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(focus["Metric"], focus["mean"])
    ax.set_title(f"{market} – Mean Spreads & Basis")
    ax.set_ylabel("$ / MWh")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(out / f"{market}_VolDrivers.png", dpi=160)
    plt.close(fig)


def export_market_excel(
    market: str,
    outdir: Path,
    price_matrix: pd.DataFrame,
    gen_fore: pd.DataFrame,
    drivers: pd.DataFrame,
    dist: pd.DataFrame,
    forwards_long: pd.DataFrame,
    wacc: float,
    w_da: float,
    w_rt: float,
    years: list[int],
):
    outdir.mkdir(parents=True, exist_ok=True)
    xlsx = outdir / f"Valuation_Report_{market}.xlsx"
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
        book  = writer.book
        fmt_money = book.add_format({"num_format":"$#,##0.00"})
        fmt_pct   = book.add_format({"num_format":"0.0%"})
        fmt_int   = book.add_format({"num_format":"#,##0"})

        readme = pd.DataFrame({
            "Item": ["Market","Years","Discount rate (WACC)","DMB weights","Quantiles reported","Notes"],
            "Value":[market, f"{years[0]}–{years[-1]}", f"{wacc:.2%}", f"DA {w_da:.0%} / RT {w_rt:.0%}",
                     "P50 / P75 / P90",
                     "Prices are risk-indifference unit PV capture prices ($/MWh). "
                     "Generation forecast includes tech degradation; forwards anchor hub level."]
        })
        readme.to_excel(writer, sheet_name="ReadMe", index=False)
        ws = writer.sheets["ReadMe"]; ws.set_column("A:A", 28); ws.set_column("B:B", 90)

        pm = price_matrix.copy()
        pm.to_excel(writer, sheet_name="FairPrices", index=False)
        ws = writer.sheets["FairPrices"]
        ws.set_column("A:A", 28)
        ws.set_column("B:D", 14, fmt_money)

        chart = book.add_chart({"type":"column"})
        cat_range = ["FairPrices", 1, 0, len(pm), 0]
        for j, q in enumerate(["P50","P75","P90"], start=1):
            chart.add_series({
                "name":       ["FairPrices", 0, j],
                "categories": cat_range,
                "values":     ["FairPrices", 1, j, len(pm), j],
            })
        chart.set_title({"name": f"{market} – Risk-Adjusted Fair Prices"})
        chart.set_y_axis({"name":"$/MWh"}); chart.set_legend({"position":"bottom"})
        ws.insert_chart("F2", chart, {"x_scale":1.4, "y_scale":1.2})

        gf = gen_fore.copy()
        gf["MWh"] = gf["Expected_Gen_MW"] * gf["Hours"]
        gf = _ym_label(gf)
        gf = gf[["Year","Month","YM_label","Period","Expected_Gen_MW","Hours","MWh"]]
        gf.sort_values(["Year","Month","Period"], inplace=True)
        gf.to_excel(writer, sheet_name="ExpectedGen", index=False)
        ws = writer.sheets["ExpectedGen"]
        ws.set_column("A:C", 10); ws.set_column("D:D", 10)
        ws.set_column("E:E", 16); ws.set_column("F:F", 10, fmt_int); ws.set_column("G:G", 14, fmt_int)

        piv = gf.pivot(index="YM_label", columns="Period", values="MWh").reset_index()
        piv.to_excel(writer, sheet_name="GenPivot", index=False)
        ws2 = writer.sheets["GenPivot"]; ws2.set_column("A:A", 10); ws2.set_column("B:Z", 14, fmt_int)
        gen_chart = book.add_chart({"type":"line"})
        cat = ["GenPivot", 1, 0, 1 + len(piv) - 1, 0]
        for c in [c for c in ["offpeak","peak"] if c in piv.columns]:
            col_idx = piv.columns.get_loc(c)
            gen_chart.add_series({
                "name":       ["GenPivot", 0, col_idx],
                "categories": cat,
                "values":     ["GenPivot", 1, col_idx, 1 + len(piv) - 1, col_idx],
            })
        gen_chart.set_title({"name": f"{market} – Monthly Expected Generation (MWh)"})
        gen_chart.set_legend({"position":"bottom"})
        ws.insert_chart("I2", gen_chart, {"x_scale":1.4, "y_scale":1.2})

        fw = _ym_label(forwards_long.copy())
        fwpv = fw.pivot_table(index="YM_label", columns="Period",
                              values="Forward_Hub_Price", aggfunc="mean").reset_index()
        fwpv.to_excel(writer, sheet_name="Forwards", index=False)
        ws = writer.sheets["Forwards"]; ws.set_column("A:A", 10); ws.set_column("B:Z", 12, fmt_money)
        fw_chart = book.add_chart({"type":"line"})
        cat = ["Forwards", 1, 0, 1 + len(fwpv) - 1, 0]
        for c in [col for col in ["peak","offpeak"] if c in fwpv.columns]:
            col_idx = fwpv.columns.get_loc(c)
            fw_chart.add_series({
                "name":       ["Forwards", 0, col_idx],
                "categories": cat,
                "values":     ["Forwards", 1, col_idx, 1 + len(fwpv) - 1, col_idx],
            })
        fw_chart.set_title({"name": f"{market} – Hub Forwards ($/MWh)"}); fw_chart.set_legend({"position":"bottom"})
        ws.insert_chart("G2", fw_chart, {"x_scale":1.4, "y_scale":1.2})

        drivers.to_excel(writer, sheet_name="VolatilityDrivers", index=False)
        ws = writer.sheets["VolatilityDrivers"]; ws.set_column("A:A", 28); ws.set_column("B:F", 14); ws.set_column("G:H", 18)

        dist.to_excel(writer, sheet_name="Scenarios", index=False)
        ws = writer.sheets["Scenarios"]; ws.set_column("A:Z", 14, fmt_money)

        comp_rows = [
            ["Anchor: Hub fwd (avg, 2026–2030)", fw["Forward_Hub_Price"].mean()],
            ["Basis: mean hist (DA)", (drivers.loc[drivers["Metric"]=="DA Node-Hub Basis","mean"].values[0]
                                       if "DA Node-Hub Basis" in drivers["Metric"].values else np.nan)],
            ["Basis: mean hist (RT)", (drivers.loc[drivers["Metric"]=="RT Node-Hub Basis","mean"].values[0]
                                       if "RT Node-Hub Basis" in drivers["Metric"].values else np.nan)],
            ["Weights: DA", w_da],
            ["Weights: RT", w_rt],
            ["Discount rate (WACC)", WACC],
        ]
        comp_df = pd.DataFrame(comp_rows, columns=["Component","Value"])
        comp_df.to_excel(writer, sheet_name="Components", index=False)
        ws = writer.sheets["Components"]; ws.set_column("A:A", 34); ws.set_column("B:B", 18)
        money_rows = [0,1,2]
        for r in range(len(comp_rows)):
            if r in money_rows:
                ws.write_number(r+1, 1, comp_rows[r][1], fmt_money)
            else:
                ws.write_number(r+1, 1, comp_rows[r][1], fmt_pct)

    print(f"✓ Excel report written: {xlsx.resolve()}")
    return xlsx

# =============================================================================
# PIPELINE PER MARKET
# =============================================================================
def run_market(market, tech, history_path: Path, forward_path: Path, years, degradation_map,
               w_da, w_rt, wacc, sims, quantiles, outdir: Path):
    hourly = load_sheet(history_path, market)
    fwd = load_forwards(forward_path)

    gen_fore = monthly_gen_forecast(hourly, tech, years, degradation_map)
    outdir.mkdir(parents=True, exist_ok=True)
    gen_fore.to_csv(outdir / "expected_generation_2026_2030.csv", index=False)

    pools, seas = build_driver_pools(hourly)
    start_year = int(min(years))
    months = (max(years) - start_year + 1) * 12

    # ----- Baseline (includes negative prices) -----
    dist_base = simulate_term(
        pools, seas, fwd, gen_fore,
        start_year=start_year, months=months,
        w_da=w_da, w_rt=w_rt, rate=wacc, sims=sims, seed=2026,
        dmb_mode="both",
        no_take_negative=False
    )
    pm_base = summarize_prices(dist_base, w_da=w_da, w_rt=w_rt, quantiles=quantiles)
    pm_base.to_csv(outdir / "fixed_price_matrix_P50_P75_P90_BASELINE.csv", index=False)
    dist_base.to_csv(outdir / "scenario_unit_PV_distribution_BASELINE.csv", index=False)

    # ----- No-take at negative prices -----
    dist_no = simulate_term(
        pools, seas, fwd, gen_fore,
        start_year=start_year, months=months,
        w_da=w_da, w_rt=w_rt, rate=wacc, sims=sims, seed=2026,
        dmb_mode="both",
        no_take_negative=True,
        exclude_denominator_when_not_taken=False
    )
    pm_no = summarize_prices(dist_no, w_da=w_da, w_rt=w_rt, quantiles=quantiles)
    pm_no.to_csv(outdir / "fixed_price_matrix_P50_P75_P90_NOTAKE.csv", index=False)
    dist_no.to_csv(outdir / "scenario_unit_PV_distribution_NOTAKE.csv", index=False)

    # Keep your original names pointing to the policy you care about most (optional)
    pm_base.to_csv(outdir / "fixed_price_matrix_P50_P75_P90.csv", index=False)
    dist_base.to_csv(outdir / "scenario_unit_PV_distribution.csv", index=False)

    # Volatility drivers (same for both)
    drivers = volatility_driver_summaries(hourly)
    drivers.to_csv(outdir / "volatility_driver_summaries.csv", index=False)

    # Export Excel for the BASELINE case (you can also export both if you want)
    export_market_excel(
        market=market, outdir=outdir, price_matrix=pm_base, gen_fore=gen_fore,
        drivers=drivers, dist=dist_base, forwards_long=fwd, wacc=wacc, w_da=w_da, w_rt=w_rt, years=years
    )

    # ---- PLOTS ----
    plot_price_matrix_compare(market, pm_base, pm_no, outdir)
    plot_scenario_hist_compare(market, dist_base, dist_no,
                               products=("Z_DMB_Hub","Z_RT_Busbar","Z_DA_Hub","Z_DMB_Bus"),
                               outdir=outdir)
    plot_generation_time_series(market, gen_fore, outdir)
    plot_forwards_time_series(market, fwd, outdir)
    plot_vol_drivers_quickview(market, drivers, outdir)

    return pm_base, pm_no, gen_fore, drivers, dist_base, dist_no

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*80)
    print("RENEWABLE ASSET VALUATION")
    print("="*80)
    print(f"Years: {YEARS} | WACC: {WACC:.2%} | DMB Weights: DA {W_DA:.0%} / RT {W_RT:.0%} | Sims: {SIMS}")

    ensure_exists(HISTORICAL_XLSX, "historical workbook")
    ensure_exists(FWD_ERCOT, "ERCOT forward file")
    ensure_exists(FWD_MISO,  "MISO forward file")
    ensure_exists(FWD_CAISO, "CAISO forward file")

    OUTDIR.mkdir(exist_ok=True)
    markets = {"ERCOT": FWD_ERCOT, "MISO": FWD_MISO, "CAISO": FWD_CAISO}

    for mkt, fwdp in markets.items():
        outdir = OUTDIR / mkt
        print(f"\n--- Running {mkt} ({TECH_TYPES[mkt]}) ---")
        pm_base, pm_no, gen_fore, drivers, dist_base, dist_no = run_market(
            market=mkt, tech=TECH_TYPES[mkt],
            history_path=HISTORICAL_XLSX, forward_path=fwdp,
            years=YEARS, degradation_map=DEGRADATION,
            w_da=W_DA, w_rt=W_RT, wacc=WACC, sims=SIMS, quantiles=QUANTILES,
            outdir=outdir
        )
        print("Baseline P75:")
        print(pm_base.set_index("Product")["P75"].round(2).to_string())
        print("No-take P75:")
        print(pm_no.set_index("Product")["P75"].round(2).to_string())
        print(f"Outputs → {outdir.resolve()}")

    # --- Master workbook aggregating all markets (BASELINE set) ---
    master = OUTDIR / "Valuation_Report_ALL_MARKETS.xlsx"
    with pd.ExcelWriter(master, engine="xlsxwriter") as w:
        for mkt in ["ERCOT","MISO","CAISO"]:
            base = OUTDIR / mkt
            for name, fn in [
                (f"{mkt}_FairPrices_BASE", base / "fixed_price_matrix_P50_P75_P90_BASELINE.csv"),
                (f"{mkt}_FairPrices_NOTAKE", base / "fixed_price_matrix_P50_P75_P90_NOTAKE.csv"),
                (f"{mkt}_ExpectedGen", base / "expected_generation_2026_2030.csv"),
                (f"{mkt}_VolDrivers", base / "volatility_driver_summaries.csv"),
                (f"{mkt}_Scenarios_BASE", base / "scenario_unit_PV_distribution_BASELINE.csv"),
                (f"{mkt}_Scenarios_NOTAKE", base / "scenario_unit_PV_distribution_NOTAKE.csv"),
            ]:
                if fn.exists():
                    pd.read_csv(fn).to_excel(w, sheet_name=name, index=False)
    print(f"✓ Master Excel written: {master.resolve()}")


if __name__ == "__main__":
    main()
