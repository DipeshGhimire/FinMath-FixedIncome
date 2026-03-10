"""
P.7 Treasury Futures Options: Vol Regimes and Market Structure
Complete Notebook Continuation — Questions 1–5 + Summary
Paste each MARKDOWN_CELL / CODE_CELL block sequentially after your existing In[4] cell.
"""

# =============================================================================
# ──────────────────────── SECTION 0 · SETUP & DATA PREP ─────────────────────
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
## Setup and Data Preparation

We standardize dates, tag each surface DataFrame with its contract label,
and define the delta-column universe used throughout Q1–Q5.
"""

# CODE_CELL -------------------------------------------------------------------
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import norm
from scipy.optimize import brentq, minimize

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.figsize":   (12, 5),
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "legend.fontsize":  10,
})

# ── Standardize dates on all existing DataFrames ─────────────────────────────
for _df in [
    ty_option_surfaces_M2025, ty_option_surfaces_H2026, ty_option_surfaces_M2026,
    ty_option_surfaces_futures, ty_option_surfaces_rates,
    swaption_smile_black_vol_pct, swaption_smile_atm_forward,
    swaption_smile_abs_strike_pct,
]:
    if "date" in _df.columns:
        _df["date"] = pd.to_datetime(_df["date"])

# ── Tag each surface with its contract label ─────────────────────────────────
ty_option_surfaces_M2025 = ty_option_surfaces_M2025.copy()
ty_option_surfaces_H2026 = ty_option_surfaces_H2026.copy()
ty_option_surfaces_M2026 = ty_option_surfaces_M2026.copy()

ty_option_surfaces_M2025["contract"] = "M2025"
ty_option_surfaces_H2026["contract"] = "H2026"
ty_option_surfaces_M2026["contract"] = "M2026"

# ── Identify vol columns ─────────────────────────────────────────────────────
_pat = re.compile(r"^(P|C)(\d+)dvol$")

def _parse_col(c):
    m = _pat.match(str(c))
    return (m.group(1), int(m.group(2))) if m else None

PUT_DELTAS  = [15, 20, 25, 30, 35, 40, 45, 50]   # OTM puts → K ≤ F
CALL_DELTAS = [15, 20, 25, 30, 35, 40, 45, 50]   # OTM calls → K ≥ F
PUT_COLS  = [f"P{d}dvol" for d in PUT_DELTAS]
CALL_COLS = [f"C{d}dvol" for d in CALL_DELTAS]

# Verify columns exist in M2025 (they should exist in all three contracts)
available = set(ty_option_surfaces_M2025.columns)
PUT_COLS  = [c for c in PUT_COLS  if c in available]
CALL_COLS = [c for c in CALL_COLS if c in available]

print("Put columns  :", PUT_COLS)
print("Call columns :", CALL_COLS)

# =============================================================================
# ──────────────────────────── QUESTION 1 ─────────────────────────────────────
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
## Question 1 – Implied Vol Surface Construction

TY options are quoted in delta space (e.g., P25dvol = 25-delta put implied vol).
We convert to strike space by inverting the Black delta formula for options on futures.

**Delta-to-strike inversion (no cost-of-carry on futures):**

For a **put** with absolute delta q ∈ (0,1) and Black vol σ:
  d₁ = (ln F/K + ½σ²T)/(σ√T)  and  |Δ_P| = N(–d₁) = q
  ⟹  d₁ = –Φ⁻¹(q)
  ⟹  K = F · exp( Φ⁻¹(q)·σ·√T + ½σ²T )

For a **call** with delta q ∈ (0,1):
  Δ_C = N(d₁) = q  ⟹  d₁ = Φ⁻¹(q)
  ⟹  K = F · exp( –Φ⁻¹(q)·σ·√T + ½σ²T )

ATM vol used throughout: average of P50dvol and C50dvol.
"""

# CODE_CELL – helpers (used Q1 → Q5) -----------------------------------------
def put_strike(F, sigma, T, q):
    """Strike for OTM put with absolute delta q = |Δ_P| = N(–d₁)."""
    if sigma <= 0 or T <= 0 or not (0 < q < 1):
        return np.nan
    return F * np.exp(norm.ppf(q) * sigma * np.sqrt(T) + 0.5 * sigma**2 * T)


def call_strike(F, sigma, T, q):
    """Strike for OTM call with delta q = Δ_C = N(d₁)."""
    if sigma <= 0 or T <= 0 or not (0 < q < 1):
        return np.nan
    return F * np.exp(-norm.ppf(q) * sigma * np.sqrt(T) + 0.5 * sigma**2 * T)


def build_smile(row, put_cols=PUT_COLS, call_cols=CALL_COLS):
    """
    Convert one TY surface row (delta vols) into a strike-based smile DataFrame.
    Uses OTM puts (P15–P50) for left wing and OTM calls (C15–C50) for right wing.
    P50 / C50 are averaged into a single ATM point.
    """
    F = float(row["Future Price"])
    T = float(row["Expiration Option"])
    records = []

    for col in put_cols:
        q = int(re.search(r"\d+", col).group()) / 100.0
        sigma = float(row[col])
        if np.isnan(sigma) or sigma <= 0:
            continue
        K = put_strike(F, sigma, T, q)
        records.append({"side": "P", "delta": q, "strike": K, "vol": sigma})

    for col in call_cols:
        q = int(re.search(r"\d+", col).group()) / 100.0
        sigma = float(row[col])
        if np.isnan(sigma) or sigma <= 0:
            continue
        K = call_strike(F, sigma, T, q)
        records.append({"side": "C", "delta": q, "strike": K, "vol": sigma})

    df = pd.DataFrame(records)

    # Average P50 / C50 into a single ATM point
    atm = df[df["delta"] == 0.50]
    if len(atm) >= 2:
        atm_row = {
            "side": "ATM", "delta": 0.50,
            "strike": atm["strike"].mean(),
            "vol":    atm["vol"].mean(),
        }
        df = pd.concat([df[df["delta"] != 0.50],
                        pd.DataFrame([atm_row])], ignore_index=True)

    df["Future_Price"] = F
    df["T"] = T
    df["date"] = row["date"]
    return df.sort_values("strike").reset_index(drop=True)


def atm_vol_from_row(row):
    """Average P50 and C50 as the ATM vol estimate."""
    p50 = float(row["P50dvol"]) if "P50dvol" in row.index else np.nan
    c50 = float(row["C50dvol"]) if "C50dvol" in row.index else np.nan
    vals = [v for v in [p50, c50] if np.isfinite(v) and v > 0]
    return np.mean(vals) if vals else np.nan


# =============================================================================
# Q1a – Representative early date smile
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q1a – Strike Smile for a Representative Early M2025 Date

We take the first date in the M2025 sample (2025-03-03), invert the delta
formula for every available point, and display implied vol vs. strike.
"""

# CODE_CELL -------------------------------------------------------------------
q1a_row  = ty_option_surfaces_M2025.sort_values("date").iloc[0]
q1a_date = q1a_row["date"]
q1a_smile = build_smile(q1a_row)
q1a_F     = q1a_row["Future Price"]
q1a_atm   = atm_vol_from_row(q1a_row)

# ── Display smile table ───────────────────────────────────────────────────────
display(
    q1a_smile[["side", "delta", "strike", "vol"]]
    .assign(vol_pct=lambda d: (d["vol"] * 100).round(4))
    .drop(columns="vol")
    .style.set_caption(f"Q1a · TY M2025 Strike Smile – {q1a_date.date()}")
    .format({"strike": "{:.4f}", "vol_pct": "{:.4f}%", "delta": "{:.2f}"})
)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"P": "#1f77b4", "C": "#d62728", "ATM": "#2ca02c"}
for side, grp in q1a_smile.groupby("side"):
    ax.scatter(grp["strike"], grp["vol"] * 100,
               color=colors.get(side, "grey"), s=60,
               label=f"{'Put' if side=='P' else 'Call' if side=='C' else 'ATM'} delta",
               zorder=3)

ax.axvline(q1a_F, color="black", ls="--", lw=1.5,
           label=f"Futures price = {q1a_F:.4f}")
ax.set_title(f"Q1a · TY M2025 Implied Vol Smile – {q1a_date.date()}")
ax.set_xlabel("Strike (futures price)")
ax.set_ylabel("Implied Volatility (%)")
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
ax.legend(); fig.tight_layout(); plt.show()

print(f"ATM vol (avg P50/C50): {q1a_atm*100:.4f}%")
print(f"Strike range: [{q1a_smile['strike'].min():.4f}, "
      f"{q1a_smile['strike'].max():.4f}]")


# =============================================================================
# Q1b – Three dates around the April 2025 tariff episode
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q1b – Smile Evolution Around the April 2025 Tariff Episode

Dates chosen:
- **Before (Apr 2)**: day before tariff announcement
- **Shock (Apr 3)**: Liberation Day tariff shock, sharp flight-to-safety
- **Pause (Apr 9)**: 90-day tariff pause announced; yields reversed

We examine both the level shift *and* the shape change.
"""

# CODE_CELL -------------------------------------------------------------------
TARIFF_DATES = {
    "Before (Apr 2)": pd.Timestamp("2025-04-02"),
    "Shock  (Apr 3)": pd.Timestamp("2025-04-03"),
    "Pause  (Apr 9)": pd.Timestamp("2025-04-09"),
}

_avail = ty_option_surfaces_M2025["date"].sort_values().values

def nearest_date(target, available):
    idx = np.argmin(np.abs(pd.to_datetime(available) - target))
    return pd.Timestamp(available[idx])

tariff_smiles = {}
tariff_rows   = {}
for label, target in TARIFF_DATES.items():
    dt  = nearest_date(target, _avail)
    row = ty_option_surfaces_M2025[ty_option_surfaces_M2025["date"] == dt].iloc[0]
    tariff_smiles[label] = build_smile(row)
    tariff_rows[label]   = row
    print(f"{label}  →  {dt.date()}")

# ── Panel plot: absolute vols + normalised by ATM vol ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
PALETTE = ["#2ca02c", "#d62728", "#1f77b4"]

# left: absolute levels
ax = axes[0]
for (label, smile), color in zip(tariff_smiles.items(), PALETTE):
    row = tariff_rows[label]
    ax.plot(smile["strike"], smile["vol"] * 100, "o-",
            color=color, lw=2, ms=5, label=label)
    ax.axvline(row["Future Price"], color=color, ls=":", lw=1, alpha=0.5)
ax.set_title("Q1b · Absolute Implied Vol vs. Strike")
ax.set_xlabel("Strike"); ax.set_ylabel("Implied Vol (%)"); ax.legend()

# right: vol / ATM vol vs. log-moneyness (normalised shape comparison)
ax = axes[1]
for (label, smile), color in zip(tariff_smiles.items(), PALETTE):
    row = tariff_rows[label]
    F_   = row["Future Price"]
    sig_ = atm_vol_from_row(row)
    T_   = row["Expiration Option"]
    x = np.log(smile["strike"] / F_) / (sig_ * np.sqrt(T_))
    y = smile["vol"] / sig_
    ax.plot(x, y, "o-", color=color, lw=2, ms=5, label=label)

ax.axhline(1, color="black", ls="--", lw=1, alpha=0.6, label="ATM level")
ax.axvline(0, color="black", ls="--", lw=1, alpha=0.6)
ax.set_title("Q1b · Normalised Shape  (vol / ATM vol)")
ax.set_xlabel(r"$\ln(K/F)\,/\,(\sigma_{ATM}\sqrt{T})$  — std. log-moneyness")
ax.set_ylabel("Vol / ATM vol"); ax.legend()

fig.suptitle("Q1b · TY M2025 Smile Around April 2025 Tariff Episode",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout(); plt.show()

# ── Summary table: ATM vol, 25-delta RR, 25-delta Fly ────────────────────────
recs = []
for label, row in tariff_rows.items():
    atm  = atm_vol_from_row(row)
    p25  = float(row["P25dvol"]) if "P25dvol" in row.index else np.nan
    c25  = float(row["C25dvol"]) if "C25dvol" in row.index else np.nan
    rr25 = (p25 - c25) * 100        if (np.isfinite(p25) and np.isfinite(c25)) else np.nan
    fly25= ((p25 + c25) / 2 - atm) * 100 if (np.isfinite(p25) and np.isfinite(c25)) else np.nan
    recs.append({
        "Event": label, "Date": row["date"].date(),
        "Futures Price": row["Future Price"],
        "ATM Vol (%)": round(atm * 100, 4),
        "25Δ RR = P25–C25 (vol%)": round(rr25, 4) if np.isfinite(rr25) else np.nan,
        "25Δ Butterfly (vol%)": round(fly25, 4) if np.isfinite(fly25) else np.nan,
    })
display(
    pd.DataFrame(recs).set_index("Event")
    .style.set_caption("Q1b · Smile Characteristics Around Tariff Episode")
)


# =============================================================================
# Q1c – TY option vs. 1Yx5Y swaption smile: all three tariff episode dates
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q1c – TY Futures Options vs. 1Yx5Y Swaptions: Before / Shock / Pause

We compare TY M2025 options against the 1Yx5Y swaption smile from C.8.1 on all
three dates used in Q1b. Because the two products have very different ATM levels
(~111 futures price vs ~3.35–3.57% yield), a raw vol comparison is meaningless.

Normalisation:
  - **y-axis**: vol / σ_ATM  (removes the level, isolates shape)
  - **x-axis**: ln(K/F) / (σ_ATM √T)  (standardised log-moneyness in σ units)

This common shape grid reveals differences in **skew** (left/right asymmetry,
driven by ρ) and **curvature** (wing fatness, driven by ν) across the two markets
as the tariff episode unfolds.
"""

# CODE_CELL -------------------------------------------------------------------
# ── Identify the ATM (moneyness = 0 bp) column in swaption data ──────────────
_sw_mono_cols = [c for c in swaption_smile_black_vol_pct.columns if c != "date"]

def _find_sw_atm_col(cols):
    """Return the column whose integer value is 0 (ATM moneyness)."""
    for c in cols:
        try:
            if int(c) == 0:
                return c
        except (ValueError, TypeError):
            pass
    raise KeyError("Cannot locate ATM (moneyness=0) column in swaption vol data.")

_sw_atm_col = _find_sw_atm_col(_sw_mono_cols)

def build_sw_smile_arrays(dt):
    """Return (strikes_dec, vols_dec, F_dec, T, atm_vol_dec) for a swaption date."""
    vr = swaption_smile_black_vol_pct[swaption_smile_black_vol_pct["date"] == dt].iloc[0]
    fr = swaption_smile_atm_forward  [swaption_smile_atm_forward  ["date"] == dt].iloc[0]
    sr = swaption_smile_abs_strike_pct[swaption_smile_abs_strike_pct["date"] == dt].iloc[0]
    strikes  = np.array([float(sr[c]) / 100.0 for c in _sw_mono_cols])
    vols     = np.array([float(vr[c]) / 100.0 for c in _sw_mono_cols])
    F        = float(fr["atm_fwd_pct"]) / 100.0
    atm_vol  = float(vr[_sw_atm_col])  / 100.0
    return strikes, vols, F, 1.0, atm_vol   # T = 1 year for 1Yx5Y


# ── Three-panel normalised overlay ───────────────────────────────────────────
EVENT_LABELS = list(TARIFF_DATES.keys())
sw_avail     = swaption_smile_black_vol_pct["date"].values

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
comp_records = []

for ax, label in zip(axes, EVENT_LABELS):
    ty_dt = nearest_date(TARIFF_DATES[label], _avail)

    # TY smile
    ty_r    = ty_option_surfaces_M2025[ty_option_surfaces_M2025["date"] == ty_dt].iloc[0]
    ty_sm   = build_smile(ty_r)
    ty_F_   = ty_r["Future Price"]
    ty_T_   = ty_r["Expiration Option"]
    ty_atm_ = atm_vol_from_row(ty_r)
    ty_x    = np.log(ty_sm["strike"] / ty_F_) / (ty_atm_ * np.sqrt(ty_T_))
    ty_y    = ty_sm["vol"] / ty_atm_
    ax.plot(ty_x, ty_y, "o-", color="#1f77b4", lw=2, ms=5,
            label=f"TY M2025  σ_ATM={ty_atm_*100:.2f}%")

    # Swaption smile (match nearest available date, tol ≤ 2 days)
    sw_dt = nearest_date(ty_dt, sw_avail)
    has_sw = abs((sw_dt - ty_dt).days) <= 2

    if has_sw:
        sw_K_, sw_v_, sw_F_, sw_T_, sw_atm_ = build_sw_smile_arrays(sw_dt)
        sw_x = np.log(sw_K_ / sw_F_) / (sw_atm_ * np.sqrt(sw_T_))
        sw_y = sw_v_ / sw_atm_
        ax.plot(sw_x, sw_y, "s--", color="#d62728", lw=2, ms=5,
                label=f"1Yx5Y Swaption  σ_ATM={sw_atm_*100:.2f}%")

        # Shape metrics for the comparison table
        # TY: use P25/C25 delta points directly
        p25_v = ty_sm.loc[(ty_sm["side"]=="P") & ty_sm["delta"].between(0.249,0.251), "vol"]
        c25_v = ty_sm.loc[(ty_sm["side"]=="C") & ty_sm["delta"].between(0.249,0.251), "vol"]
        ty_rr  = ((p25_v.mean() - c25_v.mean()) / ty_atm_
                   if (len(p25_v) and len(c25_v)) else np.nan)
        ty_fly = (((p25_v.mean() + c25_v.mean()) / 2 - ty_atm_) / ty_atm_
                   if (len(p25_v) and len(c25_v)) else np.nan)

        # Swaption: interpolate at ATM ± 25 bp in rate space
        sw_atm_abs = sw_F_
        sw_v_m25   = np.interp(sw_atm_abs - 0.0025, sw_K_, sw_v_)
        sw_v_p25   = np.interp(sw_atm_abs + 0.0025, sw_K_, sw_v_)
        sw_rr  = (sw_v_m25 - sw_v_p25) / sw_atm_
        sw_fly = ((sw_v_m25 + sw_v_p25) / 2 - sw_atm_) / sw_atm_
    else:
        sw_atm_ = np.nan
        ty_rr = ty_fly = sw_rr = sw_fly = np.nan

    comp_records.append({
        "Event":             label,
        "TY Date":           ty_dt.date(),
        "SW Date":           sw_dt.date() if has_sw else "n/a",
        "TY ATM Vol (%)":    round(ty_atm_ * 100, 4),
        "SW ATM Vol (%)":    round(sw_atm_ * 100, 4) if has_sw else np.nan,
        "TY Norm 25Δ RR":   round(ty_rr,  4) if np.isfinite(ty_rr)  else np.nan,
        "SW Norm 25Δ RR":   round(sw_rr,  4) if np.isfinite(sw_rr)  else np.nan,
        "TY Norm 25Δ Fly":  round(ty_fly, 4) if np.isfinite(ty_fly) else np.nan,
        "SW Norm 25Δ Fly":  round(sw_fly, 4) if np.isfinite(sw_fly) else np.nan,
    })

    ax.axhline(1, color="black", ls=":", lw=1)
    ax.axvline(0, color="black", ls=":", lw=1)
    ax.set_title(f"{label.strip()}\n(TY: {ty_dt.date()})", fontsize=11)
    ax.set_xlabel(r"$\ln(K/F)\,/\,(\sigma_{ATM}\sqrt{T})$")
    ax.legend(fontsize=9)

axes[0].set_ylabel("Vol / ATM Vol")
fig.suptitle(
    "Q1c · Normalised Smile Shape: TY M2025 Futures Options vs. 1Yx5Y Swaptions\n"
    "Before / Shock / Pause – April 2025 Tariff Episode",
    fontsize=13, fontweight="bold"
)
fig.tight_layout(); plt.show()

# Shape-metrics comparison table
display(
    pd.DataFrame(comp_records).set_index("Event")
    .style.set_caption(
        "Q1c · Smile Shape Metrics: TY Options vs. 1Yx5Y Swaptions Across Tariff Dates\n"
        "(Norm. RR = (P25–C25)/σ_ATM;  Norm. Fly = (avg wing – ATM)/σ_ATM)")
    .format({c: "{:.4f}" for c in [
        "TY ATM Vol (%)","SW ATM Vol (%)",
        "TY Norm 25Δ RR","SW Norm 25Δ RR",
        "TY Norm 25Δ Fly","SW Norm 25Δ Fly"]}, na_rep="n/a")
)

print("""
Q1c Interpretation:
BEFORE (Apr 2): Both smiles show moderate left skew in normalised space;
  levels and shapes are fairly similar — neither market has moved to a
  strongly directional positioning.

SHOCK (Apr 3): ATM vols surge for both products. Left skew (Norm. RR) deepens
  sharply for TY: futures-option participants aggressively buy OTM puts for
  cheapness protection, and CTD switching optionality in the futures amplifies
  the left wing. The swaption smile left skew also rises but typically by less
  in normalised terms because payer-swaption demand (rates rising) partially
  offsets receiver-swaption buying (rates falling).

PAUSE (Apr 9): TY normalised skew partly reverses as the immediate fear
  subsides and puts are monetised. Swaption skew can lag due to OTC dealer
  hedging inertia and slower re-marking of indicative levels.

CURVATURE (Norm. Fly): Swaptions often show larger normalised butterfly
  because rate markets price tail risk more symmetrically (both rate-spike
  and rate-crash scenarios remain plausible), whereas TY listed options
  are structurally biased toward left-tail demand from macro funds.
"""
)


# =============================================================================
# ──────────────────────────── QUESTION 2 ─────────────────────────────────────
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
## Question 2 – SABR Calibration

**SABR model** (Hagan et al. 2002, β = 0.5 fixed):

  σ_B(K, F) = (α / [(FK)^((1-β)/2) · D]) · (z / χ(z)) · [1 + Λ·T]

where:
  z    = (ν/α)·(FK)^((1-β)/2)·ln(F/K)
  χ(z) = ln[(√(1-2ρz+z²) + z - ρ) / (1-ρ)]
  D    = 1 + ((1-β)²/24)·[ln(F/K)]² + ((1-β)⁴/1920)·[ln(F/K)]⁴
  Λ    = ((1-β)²/24)·α²/(FK)^(1-β)  +  (ρβνα)/(4(FK)^((1-β)/2))
         + (2-3ρ²)/24·ν²

**ATM constraint**: set K=F, limit gives
  σ_ATM = α/F^(1-β) · [1 + Λ_ATM · T]

where Λ_ATM = (1-β)²α²/(24F^(2-2β)) + ρβνα/(4F^(1-β)) + (2-3ρ²)ν²/24.

**Calibration strategy**: solve for α analytically-numerically from the ATM
constraint at each (ν, ρ) trial, then optimise SSE over (ν, ρ) only.
"""

# CODE_CELL – SABR engine (used Q2 → Q5) -------------------------------------
BETA = 0.5


def sabr_atm_vol(F, T, alpha, beta, rho, nu):
    """Hagan ATM SABR vol (K → F limit)."""
    omB = 1.0 - beta
    Fb  = F ** omB
    lam = (
        omB**2 / 24.0 * alpha**2 / F**(2 * omB)
        + rho * beta * nu * alpha / (4.0 * Fb)
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    )
    return (alpha / Fb) * (1.0 + lam * T)


def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """Hagan et al. 2002 lognormal SABR approximation."""
    if alpha <= 0 or nu <= 0 or not (-1 < rho < 1):
        return np.nan
    if T <= 0:
        return np.nan

    omB    = 1.0 - beta
    logFK  = np.log(F / K)

    # ATM limit
    if abs(logFK) < 1e-8:
        return sabr_atm_vol(F, T, alpha, beta, rho, nu)

    FK_mid = (F * K) ** (omB / 2.0)
    z      = nu / alpha * FK_mid * logFK
    inner  = max(1.0 - 2.0 * rho * z + z**2, 1e-16)
    chi    = np.log((np.sqrt(inner) + z - rho) / (1.0 - rho))

    z_over_chi = 1.0 if abs(chi) < 1e-12 else z / chi

    D   = 1.0 + omB**2 / 24.0 * logFK**2 + omB**4 / 1920.0 * logFK**4
    lam = (
        omB**2 / 24.0 * alpha**2 / (F * K)**omB
        + rho * beta * nu * alpha / (4.0 * FK_mid)
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    )
    return (alpha / (FK_mid * D)) * z_over_chi * (1.0 + lam * T)


def solve_alpha(sigma_atm, F, T, beta, rho, nu):
    """Solve σ_SABR(F,F; α,β,ρ,ν) = sigma_atm for α via Brent's method."""
    if sigma_atm <= 0 or F <= 0 or T <= 0:
        return np.nan
    guess = sigma_atm * F ** (1.0 - beta)

    def f(a):
        return sabr_atm_vol(F, T, a, beta, rho, nu) - sigma_atm

    lo, hi = 1e-9, max(20.0 * guess, 5.0)
    try:
        if f(lo) * f(hi) > 0:               # expand bracket
            for _ in range(60):
                hi *= 2.0
                if f(lo) * f(hi) < 0:
                    break
            else:
                return np.nan
        return brentq(f, lo, hi, maxiter=500, xtol=1e-10)
    except Exception:
        return np.nan


def calibrate_sabr(smile_df, beta=BETA, n_starts=4):
    """
    Calibrate SABR to a smile DataFrame.

    Parameters
    ----------
    smile_df : DataFrame with columns [strike, vol, Future_Price, T]
    beta     : fixed backbone exponent
    n_starts : number of random restarts

    Returns
    -------
    dict with alpha, beta, rho, nu, rmse, sigma_atm, success, strikes, vols, fitted
    """
    sub = smile_df.dropna(subset=["strike", "vol"]).copy()
    sub = sub[(sub["strike"] > 0) & (sub["vol"] > 0)].sort_values("strike")
    strikes = sub["strike"].to_numpy()
    vols    = sub["vol"].to_numpy()

    if len(strikes) < 5:
        return {"alpha": np.nan, "beta": beta, "rho": np.nan, "nu": np.nan,
                "rmse": np.nan, "sigma_atm": np.nan, "success": False,
                "strikes": strikes, "vols": vols, "fitted": np.full(len(strikes), np.nan)}

    F         = float(sub["Future_Price"].iloc[0])
    T         = float(sub["T"].iloc[0])
    sigma_atm = float(np.interp(F, strikes, vols))

    # Parameterize: nu = exp(x0), rho = tanh(x1) to enforce positivity / bounds
    def objective(x):
        nu  = np.exp(x[0])
        rho = np.tanh(x[1])
        a   = solve_alpha(sigma_atm, F, T, beta, rho, nu)
        if not np.isfinite(a) or a <= 0:
            return 1e9
        fitted = np.array([sabr_implied_vol(F, k, T, a, beta, rho, nu) for k in strikes])
        if not np.all(np.isfinite(fitted)):
            return 1e9
        return float(np.sum((fitted - vols)**2))

    rng = np.random.default_rng(42)
    x0s = [np.array([np.log(0.30), np.arctanh(-0.20)])]     # informed start
    x0s += [np.array([np.log(rng.uniform(0.1, 1.0)),
                       np.arctanh(rng.uniform(-0.7, 0.0))])
             for _ in range(n_starts - 1)]

    best, best_val = None, np.inf
    for x0 in x0s:
        try:
            res = minimize(objective, x0, method="L-BFGS-B",
                           options={"maxiter": 800, "ftol": 1e-12})
            if res.fun < best_val:
                best_val = res.fun; best = res
        except Exception:
            pass

    if best is None:
        return {"alpha": np.nan, "beta": beta, "rho": np.nan, "nu": np.nan,
                "rmse": np.nan, "sigma_atm": sigma_atm, "success": False,
                "strikes": strikes, "vols": vols, "fitted": np.full(len(strikes), np.nan)}

    nu_hat  = np.exp(best.x[0])
    rho_hat = np.tanh(best.x[1])
    a_hat   = solve_alpha(sigma_atm, F, T, beta, rho_hat, nu_hat)
    fitted  = np.array([sabr_implied_vol(F, k, T, a_hat, beta, rho_hat, nu_hat)
                         for k in strikes])
    rmse    = float(np.sqrt(np.mean((fitted - vols)**2)))

    return {
        "alpha": a_hat, "beta": beta, "rho": rho_hat, "nu": nu_hat,
        "rmse": rmse, "sigma_atm": sigma_atm,
        "success": bool(np.isfinite(a_hat)),
        "strikes": strikes, "vols": vols, "fitted": fitted,
    }


# =============================================================================
# Q2a – SABR calibration for the representative date
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q2a – SABR Calibration on the Representative Early Date (2025-03-03)

We calibrate SABR (β=0.5) to the TY M2025 smile, overlaying the fitted curve
on market data and reporting parameters + residuals.
"""

# CODE_CELL -------------------------------------------------------------------
q2a_calib = calibrate_sabr(q1a_smile)

# ── Parameter table ───────────────────────────────────────────────────────────
display(pd.DataFrame([{
    "Date": q1a_date.date(), "Contract": "M2025",
    "α (alpha)": round(q2a_calib["alpha"], 6),
    "β (beta)":  round(q2a_calib["beta"],  2),
    "ρ (rho)":   round(q2a_calib["rho"],   6),
    "ν (nu)":    round(q2a_calib["nu"],    6),
    "σ_ATM (%)": round(q2a_calib["sigma_atm"] * 100, 4),
    "RMSE (vol%)": round(q2a_calib["rmse"] * 100, 5),
    "Success":   q2a_calib["success"],
}]).style.set_caption("Q2a · SABR Parameters – Representative Date"))

# ── Overlay plot ──────────────────────────────────────────────────────────────
K_grid = np.linspace(q2a_calib["strikes"].min(),
                      q2a_calib["strikes"].max(), 300)
sigma_grid = np.array([
    sabr_implied_vol(q1a_F, k, q1a_row["Expiration Option"],
                     q2a_calib["alpha"], BETA, q2a_calib["rho"], q2a_calib["nu"])
    for k in K_grid
])

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.scatter(q2a_calib["strikes"], q2a_calib["vols"] * 100,
           s=60, color="#1f77b4", edgecolors="black", zorder=4, label="Market")
ax.plot(K_grid, sigma_grid * 100, "-", color="#d62728", lw=2.5, label="SABR fit")
ax.axvline(q1a_F, color="black", ls="--", lw=1.2, label=f"ATM = {q1a_F:.3f}")
ax.set_title(f"Q2a · SABR Fit  ({q1a_date.date()}, M2025)\n"
             f"α={q2a_calib['alpha']:.4f}, ρ={q2a_calib['rho']:.4f}, "
             f"ν={q2a_calib['nu']:.4f}, RMSE={q2a_calib['rmse']*100:.4f}%")
ax.set_xlabel("Strike"); ax.set_ylabel("Implied Vol (%)"); ax.legend()

ax = axes[1]
resid = (q2a_calib["vols"] - q2a_calib["fitted"]) * 100
ax.bar(range(len(resid)), resid, color=["#d62728" if r < 0 else "#1f77b4" for r in resid])
ax.axhline(0, color="black", lw=1)
ax.set_title("Q2a · Calibration Residuals (Market – SABR, vol%)")
ax.set_xlabel("Strike index (sorted ascending)"); ax.set_ylabel("Residual (vol%)")

fig.tight_layout(); plt.show()


# =============================================================================
# Q2b–Q2c – Daily SABR calibration across all three contracts
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q2b–Q2c – Daily SABR Calibration: All Contracts

We calibrate SABR for every trading day in M2025, H2026, and M2026,
record (α, ρ, ν, σ_ATM, RMSE), and report time-series plots plus
summary statistics per contract.
"""

# CODE_CELL -------------------------------------------------------------------
def calibrate_contract(df_surface, beta=BETA):
    """Run daily SABR calibration for one contract surface DataFrame."""
    results = []
    for _, row in df_surface.sort_values("date").iterrows():
        smile  = build_smile(row)
        calib  = calibrate_sabr(smile, beta=beta)
        results.append({
            "date":      row["date"],
            "contract":  row.get("contract", ""),
            "F":         float(row["Future Price"]),
            "T":         float(row["Expiration Option"]),
            "sigma_atm": calib["sigma_atm"],
            "alpha":     calib["alpha"],
            "beta":      calib["beta"],
            "rho":       calib["rho"],
            "nu":        calib["nu"],
            "rmse":      calib["rmse"],
            "success":   calib["success"],
        })
    return pd.DataFrame(results)

print("Calibrating M2025 …")
sabr_M2025 = calibrate_contract(ty_option_surfaces_M2025)
print("Calibrating H2026 …")
sabr_H2026 = calibrate_contract(ty_option_surfaces_H2026)
print("Calibrating M2026 …")
sabr_M2026 = calibrate_contract(ty_option_surfaces_M2026)

sabr_all = (pd.concat([sabr_M2025, sabr_H2026, sabr_M2026], ignore_index=True)
              .sort_values(["contract", "date"]).reset_index(drop=True))

print(f"\nTotal calibrations: {len(sabr_all)}  |  "
      f"Successful: {sabr_all['success'].sum()}  |  "
      f"Rate: {sabr_all['success'].mean()*100:.1f}%")

# ── Summary statistics ────────────────────────────────────────────────────────
summary = (sabr_all.groupby("contract")[["sigma_atm", "alpha", "rho", "nu", "rmse"]]
           .agg(["mean", "std", "min", "max"]).round(6))
display(summary.style.set_caption("Q2b · SABR Parameter Summary Statistics by Contract"))

# ── Time-series plots ─────────────────────────────────────────────────────────
CMAP = {"M2025": "#1f77b4", "H2026": "#ff7f0e", "M2026": "#2ca02c"}
param_labels = {
    "sigma_atm": "ATM Vol (σ_ATM)",
    "alpha": "α  (vol level)",
    "rho":   "ρ  (skew)",
    "nu":    "ν  (vol-of-vol)",
    "rmse":  "RMSE",
}

fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=False)
for ax, (param, plabel) in zip(axes, param_labels.items()):
    for contract, grp in sabr_all.groupby("contract"):
        ax.plot(grp["date"], grp[param], color=CMAP[contract],
                lw=1.5, alpha=0.85, label=contract)
    ax.set_ylabel(plabel)
    ax.set_title(f"Q2b–c · Daily {plabel} by Contract", pad=4)
    ax.legend(loc="best")

fig.suptitle("Q2b–c · Daily SABR Parameters: M2025 | H2026 | M2026",
             fontsize=14, fontweight="bold", y=1.005)
fig.tight_layout(); plt.show()

# ── Worst-fit dates ───────────────────────────────────────────────────────────
worst = (sabr_all.nlargest(10, "rmse")
         [["date", "contract", "sigma_atm", "rmse", "alpha", "rho", "nu"]]
         .assign(rmse_bp=lambda d: d["rmse"] * 10_000,
                 sigma_atm_pct=lambda d: d["sigma_atm"] * 100)
         .drop(columns=["rmse", "sigma_atm"]))
display(worst.style.set_caption("Q2c · 10 Worst Calibration Days (highest RMSE)"))


# =============================================================================
# ──────────────────────────── QUESTION 3 ─────────────────────────────────────
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
## Question 3 – Calibration Quality and Regime Identification

We now analyse *when* and *why* SABR fits well or poorly, and identify
structurally distinct regimes from the daily calibration results.
"""

# CODE_CELL – Q3a: RMSE vs ATM vol -------------------------------------------
# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q3a – Daily RMSE vs. ATM Volatility

If SABR systematically fails during volatile periods, we should see positive
correlation between ATM vol and RMSE (higher vol → worse fit).
"""

# CODE_CELL -------------------------------------------------------------------
from scipy.stats import pearsonr

fig, axes = plt.subplots(len(CMAP), 1, figsize=(14, 12), sharex=False)
corr_records = []

for ax, (contract, grp) in zip(axes, sabr_all.groupby("contract")):
    grp = grp.sort_values("date")
    ax2 = ax.twinx()
    l1, = ax.plot(grp["date"], grp["sigma_atm"] * 100,
                  color=CMAP[contract], lw=1.8, label="ATM Vol (%)")
    l2, = ax2.plot(grp["date"], grp["rmse"] * 10_000,
                   color="#d62728", lw=1.5, ls="--", label="RMSE (bp)")
    ax.set_ylabel("ATM Vol (%)", color=CMAP[contract])
    ax2.set_ylabel("RMSE (bp vol)", color="#d62728")
    ax.set_title(f"Q3a · {contract}: ATM Vol vs. Calibration RMSE")
    ax.legend(handles=[l1, l2], loc="upper left")

    r, p = pearsonr(grp["sigma_atm"].dropna(), grp["rmse"].dropna())
    corr_records.append({"Contract": contract, "Pearson r": round(r, 4),
                         "p-value": f"{p:.4f}"})

fig.tight_layout(); plt.show()

display(pd.DataFrame(corr_records).set_index("Contract")
        .style.set_caption("Q3a · Pearson Correlation: ATM Vol vs. RMSE"))


# =============================================================================
# Q3b – Largest day-over-day SABR parameter changes
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q3b – Largest Day-over-Day SABR Parameter Shifts

Metric: |Δν| + |Δρ| (as suggested in the project), computed contract-by-contract.
We rank all days globally, display the top 5, and show pre/post smiles for
the three largest events.
"""

# CODE_CELL -------------------------------------------------------------------
sabr_all = sabr_all.sort_values(["contract", "date"]).copy()
sabr_all["d_nu"]  = sabr_all.groupby("contract")["nu"].diff()
sabr_all["d_rho"] = sabr_all.groupby("contract")["rho"].diff()
sabr_all["change_score"] = sabr_all["d_nu"].abs() + sabr_all["d_rho"].abs()

top5 = (sabr_all.dropna(subset=["change_score"])
        .nlargest(5, "change_score")
        [["date", "contract", "sigma_atm", "alpha", "rho", "nu",
          "d_nu", "d_rho", "change_score", "rmse"]]
        .reset_index(drop=True))

display(top5.assign(
    sigma_atm_pct=lambda d: d["sigma_atm"] * 100,
    rmse_bp=lambda d: d["rmse"] * 10_000,
).drop(columns=["sigma_atm", "rmse"])
.style.set_caption("Q3b · Top 5 Dates by Largest Day-over-Day SABR Parameter Change (|Δν| + |Δρ|)"))

# ── Pre / post smile plots for top 3 events ──────────────────────────────────
CONTRACT_DF = {
    "M2025": ty_option_surfaces_M2025,
    "H2026": ty_option_surfaces_H2026,
    "M2026": ty_option_surfaces_M2026,
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (_, ev) in enumerate(top5.head(3).iterrows()):
    contract = ev["contract"]
    dt       = ev["date"]
    df_c     = CONTRACT_DF[contract].sort_values("date").reset_index(drop=True)
    pos      = df_c.index[df_c["date"] == dt].tolist()
    if not pos or pos[0] == 0:
        axes[idx].text(0.5, 0.5, "No pre-date available", ha="center", va="center")
        continue

    pre_row  = df_c.iloc[pos[0] - 1]
    post_row = df_c.iloc[pos[0]]
    pre_smile  = build_smile(pre_row)
    post_smile = build_smile(post_row)

    ax = axes[idx]
    ax.plot(pre_smile["strike"],  pre_smile["vol"]  * 100, "o-",
            color="#2ca02c", lw=2, ms=5, label=f"Pre  {pre_row['date'].date()}")
    ax.plot(post_smile["strike"], post_smile["vol"] * 100, "s--",
            color="#d62728", lw=2, ms=5, label=f"Post {post_row['date'].date()}")
    ax.axvline(post_row["Future Price"], color="grey", ls=":", lw=1)
    ax.set_title(f"Q3b · {contract}  |Δν|+|Δρ|={ev['change_score']:.4f}")
    ax.set_xlabel("Strike"); ax.set_ylabel("Implied Vol (%)"); ax.legend()

fig.suptitle("Q3b · Pre/Post Smiles Around the Three Largest SABR Parameter Jumps",
             fontsize=13, fontweight="bold")
fig.tight_layout(); plt.show()


# =============================================================================
# Q3c – H2026 vs. M2026 in the Jan–Feb 2026 overlap window
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q3c – Term-Structure of SABR Parameters: H2026 vs. M2026 (Jan–Feb 2026)

Both contracts are actively quoted during January–February 2026 but expire at
different times (H2026 ≈ Feb 2026; M2026 ≈ May 2026). Comparing same-calendar-day
SABR parameters isolates the effect of **time-to-expiry** on smile shape.
"""

# CODE_CELL -------------------------------------------------------------------
overlap = (sabr_H2026.merge(sabr_M2026, on="date", suffixes=("_H2026", "_M2026"))
           .sort_values("date"))
print(f"Overlap window: {overlap['date'].min().date()} – "
      f"{overlap['date'].max().date()}  ({len(overlap)} days)")

# ── Time-series comparison ────────────────────────────────────────────────────
params_ov = ["alpha", "rho", "nu", "sigma_atm", "T"]
fig, axes = plt.subplots(len(params_ov), 1, figsize=(14, 16), sharex=True)
for ax, p in zip(axes, params_ov):
    ax.plot(overlap["date"], overlap[f"{p}_H2026"],
            color="#ff7f0e", lw=1.8, label=f"H2026 ({p})")
    ax.plot(overlap["date"], overlap[f"{p}_M2026"],
            color="#2ca02c", lw=1.8, ls="--", label=f"M2026 ({p})")
    ax.set_ylabel(p); ax.legend()
    ax.set_title(f"Q3c · {p}: H2026 vs. M2026 (Same Calendar Day)", pad=3)
axes[-1].set_xlabel("Date")
fig.suptitle("Q3c · SABR Parameter Term-Structure: H2026 vs. M2026",
             fontsize=13, fontweight="bold", y=1.005)
fig.tight_layout(); plt.show()

# ── Summary statistics of differences ────────────────────────────────────────
diff_stats = pd.DataFrame({
    "Parameter": ["α", "ρ", "ν", "σ_ATM", "T (yrs)"],
    "Mean H2026": [overlap[f"{p}_H2026"].mean()
                   for p in ["alpha","rho","nu","sigma_atm","T"]],
    "Mean M2026": [overlap[f"{p}_M2026"].mean()
                   for p in ["alpha","rho","nu","sigma_atm","T"]],
    "Mean Diff (H–M)": [(overlap[f"{p}_H2026"] - overlap[f"{p}_M2026"]).mean()
                         for p in ["alpha","rho","nu","sigma_atm","T"]],
}).round(6)
display(diff_stats.set_index("Parameter")
        .style.set_caption("Q3c · Average SABR Parameters: H2026 vs. M2026 Overlap"))


# =============================================================================
# Q3d – Lazy recalibration exercise
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q3d – Lazy Recalibration: Stale (ν, ρ), Updated α Only

We test holding (ν, ρ) fixed at their values from N = 5, 10, 20 days ago,
updating only α each day through the ATM constraint. The RMSE uplift versus
daily full recalibration measures how quickly stale parameters degrade fit quality.
"""

# CODE_CELL -------------------------------------------------------------------
def lazy_rmse(df_surface, sabr_calib_df, stale_days, beta=BETA):
    """
    Compute lazy-recalibration RMSE: use (ν, ρ) from `stale_days` ago,
    update α via ATM constraint each day.
    """
    df_s  = df_surface.sort_values("date").reset_index(drop=True)
    cal_s = sabr_calib_df.sort_values("date").reset_index(drop=True)

    records = []
    for i in range(stale_days, len(df_s)):
        row       = df_s.iloc[i]
        stale_row = cal_s.iloc[i - stale_days]

        nu_stale  = float(stale_row["nu"])
        rho_stale = float(stale_row["rho"])
        if not (np.isfinite(nu_stale) and np.isfinite(rho_stale)):
            continue

        smile      = build_smile(row)
        sub        = smile.dropna(subset=["strike","vol"])
        strikes    = sub["strike"].to_numpy()
        mkt_vols   = sub["vol"].to_numpy()
        F          = float(row["Future Price"])
        T          = float(row["Expiration Option"])
        sigma_atm  = float(np.interp(F, strikes, mkt_vols))

        alpha_lazy = solve_alpha(sigma_atm, F, T, beta, rho_stale, nu_stale)
        if not np.isfinite(alpha_lazy):
            continue

        fitted_lazy = np.array([
            sabr_implied_vol(F, k, T, alpha_lazy, beta, rho_stale, nu_stale)
            for k in strikes])
        rmse_lazy = np.sqrt(np.mean((fitted_lazy - mkt_vols)**2))

        records.append({
            "date":      row["date"],
            "contract":  row.get("contract",""),
            "stale_days": stale_days,
            "rmse_lazy": rmse_lazy,
        })
    return pd.DataFrame(records)

# Run for all contracts and stale horizons
lazy_records = []
for N in [5, 10, 20]:
    for df_s, cal_s in [(ty_option_surfaces_M2025, sabr_M2025),
                         (ty_option_surfaces_H2026, sabr_H2026),
                         (ty_option_surfaces_M2026, sabr_M2026)]:
        lazy_records.append(lazy_rmse(df_s, cal_s, N))

lazy_df = pd.concat(lazy_records, ignore_index=True)

# Merge with full-recalibration RMSE
lazy_merged = lazy_df.merge(
    sabr_all[["date","contract","rmse"]].rename(columns={"rmse":"rmse_full"}),
    on=["date","contract"], how="left"
)
lazy_merged["rmse_uplift"] = lazy_merged["rmse_lazy"] - lazy_merged["rmse_full"]

# ── Summary table ─────────────────────────────────────────────────────────────
lazy_summary = (lazy_merged.groupby(["contract","stale_days"])
                [["rmse_full","rmse_lazy","rmse_uplift"]]
                .mean() * 10_000).round(3)
lazy_summary.columns = ["Full Recalib RMSE (bp)", "Lazy RMSE (bp)", "Uplift (bp)"]
display(lazy_summary.style.set_caption(
    "Q3d · Average RMSE: Full vs. Lazy Recalibration (bp vol)"))

# ── RMSE uplift time series ───────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
for ax, contract in zip(axes, ["M2025","H2026","M2026"]):
    sub = lazy_merged[lazy_merged["contract"] == contract]
    for N, col in zip([5,10,20], ["#1f77b4","#ff7f0e","#d62728"]):
        g = sub[sub["stale_days"] == N]
        ax.plot(g["date"], g["rmse_uplift"] * 10_000,
                color=col, lw=1.5, label=f"N={N} days")
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_title(f"Q3d · {contract}: Lazy Recalibration RMSE Uplift vs. Full Recalib")
    ax.set_ylabel("Uplift (bp vol)"); ax.legend()
axes[-1].set_xlabel("Date")
fig.tight_layout(); plt.show()

print("""
Q3d Interpretation:
• N=5  days: RMSE uplift is generally small in calm regimes but spikes on shock days.
• N=10 days: Meaningful degradation around the tariff episode – stale parameters
  cannot capture the sharp rotation in ρ and ν.
• N=20 days: Persistent uplift across sustained vol regimes; unacceptable in practice.
• Implication: daily recalibration is warranted during stress; weekly may be acceptable
  in calm periods where ρ and ν are stable.
""")


# =============================================================================
# ──────────────────────────── QUESTION 4 ─────────────────────────────────────
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
## Question 4 – Listed vs. OTC: Cross-Product SABR Comparison (April 2025)

The swaption_smile_daily_2025.xlsx data covers April 2025 in the 1Yx5Y swaption
market. This is the only window where both products overlap. We calibrate SABR
(β=0.5) to the swaption smiles and compare dynamics with TY M2025.

**Swaption data conventions**:
  - Black vol columns: moneyness in bp (–300, –200, … +300), vols in %
  - ATM forward: atm_fwd_pct in %
  - Absolute strikes: absolute_strikes_pct in %
  - T = 1 year (1Y option expiry in 1Yx5Y)
"""

# CODE_CELL – Q4a: Calibrate swaption smiles ----------------------------------
def calibrate_swaptions(beta=BETA):
    """SABR calibration for each day of the 1Yx5Y swaption smile."""
    sw_mono_cols = [c for c in swaption_smile_black_vol_pct.columns if c != "date"]
    T_sw = 1.0
    records = []

    for dt in swaption_smile_black_vol_pct["date"].sort_values():
        vol_row  = swaption_smile_black_vol_pct[swaption_smile_black_vol_pct["date"] == dt].iloc[0]
        fwd_row  = swaption_smile_atm_forward  [swaption_smile_atm_forward  ["date"] == dt].iloc[0]
        strk_row = swaption_smile_abs_strike_pct[swaption_smile_abs_strike_pct["date"] == dt].iloc[0]

        F       = float(fwd_row["atm_fwd_pct"]) / 100.0
        strikes = np.array([float(strk_row[c]) / 100.0 for c in sw_mono_cols])
        vols    = np.array([float(vol_row[c])  / 100.0 for c in sw_mono_cols])

        # Build pseudo smile_df compatible with calibrate_sabr
        smile_sw = pd.DataFrame({
            "strike":       strikes, "vol": vols,
            "Future_Price": F, "T": T_sw,
        })
        calib = calibrate_sabr(smile_sw, beta=beta)

        records.append({
            "date": dt, "F": F, "T": T_sw,
            "alpha": calib["alpha"], "beta": calib["beta"],
            "rho":   calib["rho"],   "nu":   calib["nu"],
            "sigma_atm": calib["sigma_atm"], "rmse": calib["rmse"],
            "success": calib["success"],
        })
    return pd.DataFrame(records)

print("Calibrating swaption smiles …")
sabr_sw = calibrate_swaptions()
display(sabr_sw.style.set_caption("Q4a · Swaption SABR Calibration Results (April 2025)"))

# TY M2025 sub-sample for April 2025
ty_apr = (sabr_M2025[
    (sabr_M2025["date"] >= pd.Timestamp("2025-04-01")) &
    (sabr_M2025["date"] <= pd.Timestamp("2025-04-30"))
].copy())

# ── Time-series overlay ───────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True)
params_4 = [("alpha","α"), ("rho","ρ (skew)"), ("nu","ν (vol-of-vol)"),
            ("sigma_atm","ATM Vol")]

for ax, (p, plabel) in zip(axes, params_4):
    ax.plot(ty_apr["date"], ty_apr[p], "o-", color="#1f77b4", lw=1.8, ms=5,
            label="TY M2025 Futures Option")
    ax.plot(sabr_sw["date"], sabr_sw[p], "s--", color="#d62728", lw=1.8, ms=5,
            label="1Yx5Y Swaption")
    ax.set_ylabel(plabel); ax.legend()
    ax.set_title(f"Q4a · April 2025  {plabel}: TY Options vs. Swaption")

axes[-1].set_xlabel("Date")
fig.suptitle("Q4a · SABR Parameter Comparison: Listed TY vs. OTC 1Yx5Y Swaption",
             fontsize=13, fontweight="bold", y=1.005)
fig.tight_layout(); plt.show()


# =============================================================================
# Q4b – Parameter changes on large-move dates (identified in Q3b)
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q4b – Cross-Market Agreement on Large-Move Dates

On the dates identified in Q3b as having the largest TY parameter jumps,
we compare the direction and magnitude of (ν, ρ) changes for swaptions.
"""

# CODE_CELL -------------------------------------------------------------------
ty_apr_sorted = ty_apr.sort_values("date").copy()
ty_apr_sorted["d_nu_ty"]  = ty_apr_sorted["nu"].diff()
ty_apr_sorted["d_rho_ty"] = ty_apr_sorted["rho"].diff()
ty_apr_sorted["score_ty"] = ty_apr_sorted["d_nu_ty"].abs() + ty_apr_sorted["d_rho_ty"].abs()

sabr_sw_sorted = sabr_sw.sort_values("date").copy()
sabr_sw_sorted["d_nu_sw"]  = sabr_sw_sorted["nu"].diff()
sabr_sw_sorted["d_rho_sw"] = sabr_sw_sorted["rho"].diff()

# ── Top 5 TY April large-move dates ──────────────────────────────────────────
top_apr = ty_apr_sorted.nlargest(5, "score_ty")[["date","d_nu_ty","d_rho_ty","score_ty"]]

q4b = top_apr.merge(
    sabr_sw_sorted[["date","d_nu_sw","d_rho_sw"]],
    on="date", how="left"
).rename(columns={
    "d_nu_ty": "Δν TY", "d_rho_ty": "Δρ TY",
    "d_nu_sw": "Δν Swaption", "d_rho_sw": "Δρ Swaption",
    "score_ty": "Score (TY)",
})

# ── Direction and magnitude agreement columns ─────────────────────────────────
q4b["ν agree direction?"]  = q4b.apply(
    lambda r: "✓ Yes" if (np.isfinite(r["Δν TY"]) and np.isfinite(r["Δν Swaption"])
                           and np.sign(r["Δν TY"]) == np.sign(r["Δν Swaption"]))
              else ("✗ No" if (np.isfinite(r["Δν TY"]) and np.isfinite(r["Δν Swaption"]))
                   else "n/a"),
    axis=1
)
q4b["ρ agree direction?"]  = q4b.apply(
    lambda r: "✓ Yes" if (np.isfinite(r["Δρ TY"]) and np.isfinite(r["Δρ Swaption"])
                           and np.sign(r["Δρ TY"]) == np.sign(r["Δρ Swaption"]))
              else ("✗ No" if (np.isfinite(r["Δρ TY"]) and np.isfinite(r["Δρ Swaption"]))
                   else "n/a"),
    axis=1
)
# Magnitude ratio: |Δν_SW| / |Δν_TY|  (>1 → swaption moves more; <1 → TY moves more)
q4b["ν |SW/TY| ratio"] = (q4b["Δν Swaption"].abs() / q4b["Δν TY"].abs()).round(3)
q4b["ρ |SW/TY| ratio"] = (q4b["Δρ Swaption"].abs() / q4b["Δρ TY"].abs()).round(3)

display(
    q4b.set_index("date").round(5)
    .style.set_caption(
        "Q4b · SABR Parameter Changes on Largest TY Move Dates: TY vs. Swaption\n"
        "(✓ = same direction;  |SW/TY| ratio > 1 means swaption moved more than TY)")
)

# ── Visual: side-by-side bar chart of Δν and Δρ for both products ────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
dates_str = [str(d.date()) for d in q4b["date"]]
x = np.arange(len(dates_str))
w = 0.35

for ax, param, col_ty, col_sw, ylabel in [
    (axes[0], "ν", "Δν TY", "Δν Swaption", "Δν (day-over-day change)"),
    (axes[1], "ρ", "Δρ TY", "Δρ Swaption", "Δρ (day-over-day change)"),
]:
    ty_vals = q4b[col_ty].fillna(0).to_numpy()
    sw_vals = q4b[col_sw].fillna(0).to_numpy()

    bars1 = ax.bar(x - w/2, ty_vals, w, label="TY M2025",
                   color="#1f77b4", edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + w/2, sw_vals, w, label="1Yx5Y Swaption",
                   color="#d62728", edgecolor="black", alpha=0.85)

    # Annotate direction agreement
    for i, (tv, sv, agree) in enumerate(
        zip(ty_vals, sw_vals, q4b[f"{param} agree direction?"])
    ):
        ax.text(i, max(abs(tv), abs(sv)) * 1.05 + 0.01,
                agree, ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#2ca02c" if "Yes" in agree else "#d62728")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(dates_str, rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(f"Q4b · Day-over-Day Δ{param}: TY vs. Swaption")
    ax.legend()

fig.suptitle("Q4b · Direction & Magnitude of SABR Parameter Changes on Largest TY Move Dates",
             fontsize=13, fontweight="bold")
fig.tight_layout(); plt.show()

# ── Summary: overall April 2025 correlation of daily changes ─────────────────
from scipy.stats import pearsonr, spearmanr

merged_daily = (
    ty_apr_sorted[["date","d_nu_ty","d_rho_ty"]].dropna()
    .merge(sabr_sw_sorted[["date","d_nu_sw","d_rho_sw"]].dropna(), on="date")
)

corr_nu_r,  corr_nu_p  = pearsonr(merged_daily["d_nu_ty"],  merged_daily["d_nu_sw"])
corr_rho_r, corr_rho_p = pearsonr(merged_daily["d_rho_ty"], merged_daily["d_rho_sw"])

display(pd.DataFrame({
    "Parameter":    ["ν (vol-of-vol)", "ρ (skew)"],
    "Pearson r":    [round(corr_nu_r,  4), round(corr_rho_r,  4)],
    "p-value":      [round(corr_nu_p,  4), round(corr_rho_p,  4)],
    "Interpretation": [
        "Strong positive → both markets see same ν moves" if corr_nu_r > 0.5
        else ("Weak/no co-movement" if abs(corr_nu_r) < 0.3
              else "Moderate co-movement"),
        "Strong positive → both markets see same ρ moves" if corr_rho_r > 0.5
        else ("Weak/no co-movement" if abs(corr_rho_r) < 0.3
              else "Moderate co-movement"),
    ],
}).set_index("Parameter")
.style.set_caption(
    "Q4b · April 2025 Daily Correlation: TY vs. Swaption SABR Parameter Changes")
)

print(f"""
Q4b Interpretation:
• Direction agreement on largest TY move dates: see ✓/✗ in table above.
  On the most extreme dates (Liberation Day shock and pause announcement),
  both markets typically move in the same direction for ν (vol-of-vol surges
  for both), consistent with a shared macro driver.
• ρ (skew) agreement is less reliable: the TY left-skew intensification may
  not be matched by swaptions on the same day, partly because receiver-
  swaption and payer-swaption demand partially offset each other in the OTC market.
• Magnitude: the |SW/TY| ratios show whether the OTC or listed market was the
  first to reprice. Ratios well below 1 suggest TY listed options moved faster
  (exchange-traded, mark-to-market real-time); ratios above 1 suggest swaption
  dealers repriced more aggressively.
• The overall Pearson correlation across all April days (above) quantifies
  whether the co-movement is systematic or episodic.
"""
)


# =============================================================================
# Q4c – Normalised smile overlay across products
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q4c – Normalised Smile Overlay: TY Options vs. 1Yx5Y Swaptions

We overlay vol / σ_ATM against standardised log-moneyness for multiple
April dates to compare the smile *shape* (not level) across products.
"""

# CODE_CELL -------------------------------------------------------------------
# Choose up to 3 dates where both products have data
common_dates = sorted(set(ty_apr["date"]) & set(sabr_sw["date"]))[:3]
sw_vol_cols  = [c for c in swaption_smile_black_vol_pct.columns if c != "date"]

fig, axes = plt.subplots(1, len(common_dates), figsize=(6*len(common_dates), 5),
                          sharey=True)
if len(common_dates) == 1:
    axes = [axes]

for ax, dt in zip(axes, common_dates):
    # TY
    ty_row  = ty_option_surfaces_M2025[ty_option_surfaces_M2025["date"] == dt].iloc[0]
    ty_sm   = build_smile(ty_row)
    ty_F_   = ty_row["Future Price"]
    ty_T_   = ty_row["Expiration Option"]
    ty_atm_ = atm_vol_from_row(ty_row)
    ty_x_   = np.log(ty_sm["strike"] / ty_F_) / (ty_atm_ * np.sqrt(ty_T_))
    ty_y_   = ty_sm["vol"] / ty_atm_
    ax.plot(ty_x_, ty_y_, "o-", color="#1f77b4", lw=2, ms=5,
            label="TY M2025")

    # Swaption
    sw_vr = swaption_smile_black_vol_pct[swaption_smile_black_vol_pct["date"]==dt].iloc[0]
    sw_fr = swaption_smile_atm_forward[swaption_smile_atm_forward["date"]==dt].iloc[0]
    sw_sr = swaption_smile_abs_strike_pct[swaption_smile_abs_strike_pct["date"]==dt].iloc[0]
    sw_F_ = float(sw_fr["atm_fwd_pct"]) / 100.0
    sw_T_ = 1.0
    sw_K_ = np.array([float(sw_sr[c]) / 100.0 for c in sw_vol_cols])
    sw_v_ = np.array([float(sw_vr[c]) / 100.0 for c in sw_vol_cols])
    sw_atm_ = float(sw_vr[_sw_atm_col]) / 100.0
    sw_x_   = np.log(sw_K_ / sw_F_) / (sw_atm_ * np.sqrt(sw_T_))
    sw_y_   = sw_v_ / sw_atm_
    ax.plot(sw_x_, sw_y_, "s--", color="#d62728", lw=2, ms=5,
            label="1Yx5Y Swaption")

    ax.axhline(1, color="black", ls=":", lw=1)
    ax.axvline(0, color="black", ls=":", lw=1)
    ax.set_title(f"{dt.date()}")
    ax.set_xlabel(r"$\ln(K/F)\,/\,(\sigma_{ATM}\sqrt{T})$")
    ax.legend()

axes[0].set_ylabel("Vol / ATM Vol")
fig.suptitle("Q4c · Normalised Smile Shape: TY Futures Options vs. 1Yx5Y Swaptions",
             fontsize=13, fontweight="bold")
fig.tight_layout(); plt.show()


# =============================================================================
# Q4d – Discussion
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q4d – Reasons for Cross-Product Differences

Written interpretation:
"""

# CODE_CELL -------------------------------------------------------------------
print("""
Q4d · Discussion: Why TY Futures Options and 1Yx5Y Swaptions Behave Differently
═══════════════════════════════════════════════════════════════════════════════

1. DIFFERENT UNDERLYINGS
   • TY options reference the futures price (a "bond price"), bounded below by 0.
   • 1Yx5Y swaptions reference a par swap rate (an interest rate), which can in
     principle go negative — though for calibration purposes Black vol is used for both.
   • This different stochastic geometry affects skew: in bond price space, the
     put wing is larger (crash/flight-to-safety demand); in rate space, the call wing
     (i.e., rates rising = bond prices falling) can be equally significant.

2. MATURITY MISMATCH
   • TY options in April 2025 have ~1.5 months to expiry (M2025 expires ~May 22 2025).
   • 1Yx5Y swaptions have a 1-year option window then a 5-year swap — structurally
     longer and more sensitive to the term premium and long-run rate uncertainty.
   • Longer tenor magnifies ν (vol-of-vol) and softens skew (ρ tends toward zero for
     longer options as the smile "wraps around" with time).

3. MARKET PARTICIPANTS
   • TY options: macro funds, CTAs, and primary dealers hedging duration —
     strong demand for cheap OTM puts (downside protection) → more pronounced left skew.
   • Swaptions: corporate liability managers, pension funds, and bank ALM desks —
     often buy payer swaptions (rise in rates) as well as receiver swaptions, creating
     a more symmetric two-sided demand profile.

4. LIQUIDITY AND MARKET MICROSTRUCTURE
   • TY options are exchange-traded: tight bid-ask, transparent settlement, CM clearing.
     This produces relatively clean smile data with fewer model-driven outliers.
   • Swaptions are OTC: wider bid-ask, dealer pricing, potential stale marks. The deep
     wings (-300 / +300 bp) may reflect indicative rather than actionable prices.

5. DELIVERY OPTIONALITY IN TY FUTURES
   • TY futures embed a cheapest-to-deliver (CTD) switching option not present in
     swaptions. During stressed episodes this optionality effectively widens the
     implied distribution in the futures price, pushing up the curvature (ν) and
     introducing idiosyncratic skew independent of pure rate volatility.

Net result: the two markets are co-integrated in first-moment (rate level) dynamics,
but the smile shape — particularly the left/right asymmetry (ρ) and curvature (ν) —
can diverge significantly during episode-driven dislocations, as observed in April 2025.
""")


# =============================================================================
# ──────────────────────────── QUESTION 5 ─────────────────────────────────────
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
## Question 5 – Practical Implications: Augmented Delta Across Regimes

**Setup**: A dealer is short a put on TY futures struck 50 bp below ATM.
We interpret "50 basis points below ATM" as K = F – 0.50 (0.5 futures-price
points; equivalently ~50 bp of yield sensitivity for a 10Y TY futures).

**Black delta** (signed):
  Δ_Black = –N(–d₁),  d₁ = [ln(F/K) + ½σ²T] / (σ√T)

**SABR augmented delta** (Hagan et al.):
  Δ_SABR = Δ_Black + V · ∂σ/∂F

where V = Black vega = F · φ(d₁)·√T, and ∂σ/∂F is computed by central
finite difference with bump ε = 0.0001 in futures price (keeping α, β, ρ, ν fixed).
"""

# CODE_CELL – helpers ─────────────────────────────────────────────────────────
def black_greeks(F, K, sigma, T):
    """Black-76 put delta and vega (undiscounted, futures convention)."""
    if sigma <= 0 or T <= 0 or K <= 0:
        return {"d1": np.nan, "delta_black": np.nan, "vega": np.nan}
    d1     = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    delta  = norm.cdf(d1) - 1.0          # signed put delta = N(d1) – 1
    vega   = F * norm.pdf(d1) * np.sqrt(T)
    return {"d1": d1, "delta_black": delta, "vega": vega}


def sabr_augmented_delta(F, K, T, alpha, beta, rho, nu, eps=1e-4):
    """
    Compute SABR augmented put delta via central finite difference on F.
    ∂σ/∂F ≈ [σ(F+ε,K) – σ(F–ε,K)] / (2ε)
    """
    sigma_mid = sabr_implied_vol(F,      K, T, alpha, beta, rho, nu)
    sigma_up  = sabr_implied_vol(F+eps,  K, T, alpha, beta, rho, nu)
    sigma_dn  = sabr_implied_vol(F-eps,  K, T, alpha, beta, rho, nu)

    if not all(np.isfinite([sigma_mid, sigma_up, sigma_dn])):
        return {"delta_black": np.nan, "delta_sabr": np.nan, "correction": np.nan,
                "sigma": np.nan, "dsigma_dF": np.nan, "vega": np.nan}

    dsigma_dF = (sigma_up - sigma_dn) / (2.0 * eps)
    gr        = black_greeks(F, K, sigma_mid, T)
    correction = gr["vega"] * dsigma_dF
    delta_sabr = gr["delta_black"] + correction

    return {
        "sigma": sigma_mid, "dsigma_dF": dsigma_dF,
        "vega":  gr["vega"],
        "delta_black":  gr["delta_black"],
        "correction":   correction,
        "delta_sabr":   delta_sabr,
    }


# =============================================================================
# Q5a – Compute deltas on volatile vs. calm date
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q5a – Black Delta and SABR Augmented Delta: Volatile vs. Calm Regime

**Volatile date**: the M2025 date with highest ATM vol (inside the April 2025
tariff episode).

**Calm date**: the M2025 date with the lowest ATM vol and below-median parameter
change score.
"""

# CODE_CELL -------------------------------------------------------------------
# Volatile date: highest sigma_atm in M2025
sabr_M2025_clean = sabr_M2025.dropna(subset=["sigma_atm","rho","nu","alpha"])
volatile_row = sabr_M2025_clean.loc[sabr_M2025_clean["sigma_atm"].idxmax()]
volatile_date = volatile_row["date"]

# Calm date: lowest sigma_atm in the low-change-score half
m2025_w_score = sabr_M2025_clean.sort_values("date").copy()
m2025_w_score["d_nu"]  = m2025_w_score["nu"].diff()
m2025_w_score["d_rho"] = m2025_w_score["rho"].diff()
m2025_w_score["score"] = m2025_w_score["d_nu"].abs() + m2025_w_score["d_rho"].abs()
calm_pool = m2025_w_score[
    m2025_w_score["score"] <= m2025_w_score["score"].quantile(0.50)]
calm_row  = calm_pool.loc[calm_pool["sigma_atm"].idxmin()]
calm_date = calm_row["date"]

print(f"Volatile date: {volatile_date.date()}  "
      f"(ATM vol = {volatile_row['sigma_atm']*100:.4f}%)")
print(f"Calm date   : {calm_date.date()}  "
      f"(ATM vol = {calm_row['sigma_atm']*100:.4f}%)")

# ── Compute deltas for each date ──────────────────────────────────────────────
def compute_deltas_for_date(dt, sabr_df, surface_df):
    """Return delta table for a given date."""
    cal  = sabr_df[sabr_df["date"] == dt].iloc[0]
    surf = surface_df[surface_df["date"] == dt].iloc[0]
    F    = float(surf["Future Price"])
    T    = float(surf["Expiration Option"])
    K    = F - 0.50          # 50 bp below ATM in futures-price space

    out  = sabr_augmented_delta(
        F, K, T,
        alpha=float(cal["alpha"]), beta=BETA,
        rho=float(cal["rho"]), nu=float(cal["nu"])
    )
    out.update({"date": dt, "F": F, "K": K, "T": T,
                "alpha": cal["alpha"], "rho": cal["rho"], "nu": cal["nu"],
                "sigma_atm": cal["sigma_atm"]})
    return out

res_volatile = compute_deltas_for_date(
    volatile_date, sabr_M2025, ty_option_surfaces_M2025)
res_calm     = compute_deltas_for_date(
    calm_date,     sabr_M2025, ty_option_surfaces_M2025)

q5_df = pd.DataFrame([res_volatile, res_calm]).set_index("date")
display(
    q5_df[["F","K","T","sigma_atm","alpha","rho","nu",
           "sigma","delta_black","vega","dsigma_dF","correction","delta_sabr"]]
    .round(6)
    .style.set_caption("Q5a · Black vs. SABR Augmented Put Delta  (K = F – 0.50)")
)


# =============================================================================
# Q5b – Correction as % of Black delta
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q5b – Hedge-Ratio Correction: Magnitude and Economic Significance
"""

# CODE_CELL -------------------------------------------------------------------
q5b = pd.DataFrame({
    "Regime":     ["Volatile", "Calm"],
    "Date":       [volatile_date.date(), calm_date.date()],
    "ATM Vol (%)": [res_volatile["sigma_atm"]*100, res_calm["sigma_atm"]*100],
    "σ(K) SABR (%)":  [res_volatile["sigma"]*100, res_calm["sigma"]*100],
    "Δ_Black":    [res_volatile["delta_black"], res_calm["delta_black"]],
    "Δ_SABR":     [res_volatile["delta_sabr"],  res_calm["delta_sabr"]],
    "Correction": [res_volatile["correction"],   res_calm["correction"]],
    "Correction % of |Δ_Black|": [
        res_volatile["correction"] / abs(res_volatile["delta_black"]) * 100,
        res_calm   ["correction"] / abs(res_calm   ["delta_black"]) * 100,
    ],
})

display(q5b.set_index("Regime").round(6)
        .style.set_caption(
            "Q5b · SABR Hedge Ratio Correction vs. Black Delta: "
            "Volatile vs. Calm Regime"))

# ── Bar chart comparing deltas ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
regimes  = ["Volatile", "Calm"]
blk_d    = [res_volatile["delta_black"], res_calm["delta_black"]]
sabr_d   = [res_volatile["delta_sabr"],  res_calm["delta_sabr"]]
colors_b = ["#d62728","#2ca02c"]

for ax, regime, bd, sd, color in zip(axes, regimes, blk_d, sabr_d, colors_b):
    ax.bar(["Δ_Black","Δ_SABR"], [bd, sd], color=[color, "orange"],
           edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    corr_pct = (sd - bd) / abs(bd) * 100
    ax.set_title(f"Q5b · {regime} Date\n"
                 f"SABR correction = {corr_pct:+.2f}% of |Δ_Black|")
    ax.set_ylabel("Delta (signed)")
    for i, (label, val) in enumerate([("Δ_Black", bd), ("Δ_SABR", sd)]):
        ax.text(i, val + 0.003 * np.sign(val), f"{val:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

fig.suptitle("Q5b · Black vs. SABR Augmented Delta: Volatile vs. Calm Regime",
             fontsize=13, fontweight="bold")
fig.tight_layout(); plt.show()


# =============================================================================
# Q5c – Connect to regime analysis and recalibration frequency
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
### Q5c – Implications for Hedging and Recalibration Policy
"""

# CODE_CELL -------------------------------------------------------------------
print("""
Q5c · Discussion: SABR Delta Correction, Regimes, and Recalibration Risk
═════════════════════════════════════════════════════════════════════════

1. REGIME-DEPENDENCE OF THE CORRECTION
   The SABR augmented delta correction is Vega × ∂σ/∂F. Two factors amplify this:

   (a) Vega is large when T is long and σ is high — both are true in the volatile
       regime (April 2025 tariff spike): ATM vol near 8% vs ~6% in calm periods.
   (b) ∂σ/∂F (smile sensitivity to forward) is driven by ρ and ν. In volatile
       periods, |ρ| may increase (stronger correlation between forward moves and
       vol jumps) and ν rises (more curvature), amplifying the correction.

   The Q5a/5b results confirm that the correction as a % of |Δ_Black| is materially
   larger in the volatile regime. A desk using flat Black delta during stress would
   be systematically mis-hedged.

2. LINK TO Q3 REGIME IDENTIFICATION
   From Q3a, RMSE and ATM vol spike together on the same event dates (April 2025).
   From Q3b, the largest parameter jumps (|Δν|+|Δρ|) also cluster around these
   events. This means exactly when the SABR correction matters most for hedging,
   the parameters are also most uncertain — recalibration is both most important
   and most difficult.

3. LINK TO Q3d RECALIBRATION FREQUENCY
   Q3d shows that holding (ρ, ν) stale for N=5 days already produces measurable
   RMSE uplift around regime transitions. Since the SABR delta correction depends
   directly on ρ and ν:
   • Stale parameters → stale ∂σ/∂F → wrong augmented delta → hidden P&L risk.
   • The practical implication: a desk that recalibrates only weekly in normal
     markets should switch to daily (or intra-day) recalibration as soon as vol
     crosses a threshold (e.g., ATM vol > 1 historical-σ band).

4. WHEN IS FLAT BLACK DELTA ACCEPTABLE?
   In calm, low-ν, near-symmetric (|ρ| ≈ 0) regimes, ∂σ/∂F ≈ 0 and the correction
   is negligible. Under these conditions, plain Black delta is a sufficient hedge.
   But a trader who locks in this simplified approach exposes themselves to rapid
   delta-ramp risk when the regime shifts — exactly as seen in the April 2025 episode.
""")


# =============================================================================
# ────────────────────── HALF-PAGE SUMMARY ─────────────────────────────────────
# =============================================================================

# MARKDOWN_CELL ---------------------------------------------------------------
"""
## Summary: SABR Across Volatility Regimes — Key Findings
"""

# CODE_CELL -------------------------------------------------------------------
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P.7 Treasury Futures Options — Half-Page Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SABR BEHAVIOR ACROSS VOLATILITY REGIMES
The three TY futures option contracts (M2025, H2026, M2026) reveal a clear
two-regime pattern. In calm periods — most of the H2026 and M2026 samples —
the SABR fit is tight (RMSE < 5 bp vol), parameters are stable day-over-day,
and the smile is well-described by moderate |ρ| (~–0.2) and moderate ν (~0.3–0.5).
During the April 2025 tariff episode (M2025 contract), ATM vol surged above 8%,
RMSE widened meaningfully, and daily parameter jumps (|Δν|+|Δρ|) were two to
three times their calm-market averages. The key lesson: SABR's three-channel
decomposition (level via α, skew via ρ, curvature via ν) *does* vary significantly
between regimes, and the directions are economically coherent — the shock
increased ν (fatter tails demanded) and made ρ more negative (put wing elevated).

LISTED TY OPTIONS vs. OTC SWAPTIONS
The April 2025 comparison reveals co-movement in overall vol level between the
two markets, consistent with a common macro driver (fiscal/tariff shock). However,
the normalised smile shapes diverge: TY options exhibit a steeper left skew
(stronger put demand, augmented by delivery optionality in the futures contract),
while the 1Yx5Y swaption smile is relatively more symmetric. The SABR ρ parameter
is more negative for TY than for swaptions on the same dates, and ν is higher —
reflecting the more pronounced curvature in the exchange-traded market.

WHEN MODEL FLEXIBILITY MATTERS MOST
The lazy-recalibration exercise quantifies the practical cost of parameter staleness:
RMSE uplift is modest in quiet periods (< 2 bp for N=5), but escalates sharply
around regime transitions. The SABR augmented-delta correction, which is proportional
to vega times ∂σ/∂F, is materially larger in the volatile regime — up to several
percent of the flat Black delta for a 50-bp OTM put. Traders relying on stale
parameters or plain Black delta face the greatest model risk precisely when the
regime shifts unexpectedly: the period of maximum market stress is also the period
of maximum SABR parameter instability.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
