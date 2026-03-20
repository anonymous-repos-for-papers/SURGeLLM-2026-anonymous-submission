# ============================================================
# 01_data_cleansing_and_transformation.py - Load, cleanse, merge
#
# KDD-structured data preparation pipeline:
#   Section 1 - Data Loading (CSV)
#   Section 2 - Exploratory Data Analysis (EDA)
#   Section 3 - Data Cleansing
#   Section 4 - Data Transformation (Merge + Feature Engineering)
# ============================================================
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score

from . import config
from .utils import downcast_df, mem


# --------------------------------------------------------------
# SECTION 1 — Data Loading (CSV)
# --------------------------------------------------------------

def load_source_a():
    """Load and downcast Source A data from CSV."""
    df = pd.read_csv(config.CSV_SOURCE_A, sep=config.CSV_SEP, low_memory=False)
    df.rename(columns=config.CSV_RENAME_SOURCE_A, inplace=True)
    df = downcast_df(df)
    print(f"Source A loaded: {df.shape}")
    mem()
    return df


def load_source_b():
    """Load and downcast Source B data from CSV."""
    df = pd.read_csv(config.CSV_SOURCE_B, sep=config.CSV_SEP, low_memory=False)
    df.columns = df.columns.str.lower()
    df.rename(columns=config.CSV_RENAME_SOURCE_B, inplace=True)
    df = downcast_df(df)
    print(f"Source B loaded: {df.shape}")
    mem()
    return df


# --------------------------------------------------------------
# SECTION 2 � Exploratory Data Analysis (EDA)
# --------------------------------------------------------------

# -- Consistent thesis palette (same as visualization.py) -----
_PALETTE = list(mcolors.TABLEAU_COLORS.values())
_FIGSIZE_WIDE = (14, 8)


# -- Exploratory Data Analysis (EDA) -------------------------
#
#   1. dataset_overview          � size, columns, missing-value bar chart
#   2. plot_feature_distributions � histogram grid of key numeric features
#   3. plot_target_distribution  � binary imbalance + multiclass frequency
#   4. plot_temporal_trends      � monthly retrofit volumes (total + by type)
#   5. descriptive_statistics    � mean / std / quartile table
#   6. run_full_eda              � convenience: runs 1-5 in one call
# -------------------------------------------------------------

# 1. Dataset Overview
def dataset_overview(df, label=""):
    """
    Print dataset size, column count, dtypes breakdown,
    and plot a horizontal bar chart of missing-value percentages.
    """
    tag = f" ({label})" if label else ""
    print(f"\n{'=' * 70}")
    print(f"DATASET OVERVIEW{tag}")
    print(f"{'=' * 70}")
    print(f"  Rows   : {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Memory : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Dtype breakdown
    dt = df.dtypes.value_counts()
    print(f"\n  Dtypes:")
    for dtype, cnt in dt.items():
        print(f"    {str(dtype):20s} {cnt}")

    # Missing values
    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("\n  No missing values detected.")
        return

    print(f"\n  Columns with missing values: {len(missing)} / {df.shape[1]}")

    # Bar chart of top-20 missing columns
    top = missing.head(20)
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
    bars = ax.barh(top.index[::-1], top.values[::-1],
                   color=_PALETTE[3], edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Missing (%)", fontsize=11)
    ax.set_title(f"Missing Values{tag}", fontsize=13, fontweight="bold")
    ax.set_xlim(0, min(top.max() * 1.15, 100))
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()

# 2. Distribution of Key Features
def plot_feature_distributions(df, feature_list=None):
    """
    Grid of histograms + KDE for key numeric features.

    Parameters
    ----------
    df : DataFrame
        The merged DataFrame (df_binary or df_multi).
    feature_list : list[str], optional
        Columns to plot. If None, auto-detects the standard engineered
        features: vehicle_age_months, derivat_fleet_size, plus any
        other *_fleet_size columns present.
    """
    # Auto-detect if not provided
    if feature_list is None:
        candidates = [
            "vehicle_age_months", "derivat_fleet_size",
            "motortyp_fleet_size", "pl_fleet_size",
            "puk_werk_fleet_size", "bo_fleet_size",
            "days_to_deadline", "car_ready_age_months",
        ]
        feature_list = [c for c in candidates if c in df.columns]

    # Drop any that don't exist
    feature_list = [c for c in feature_list if c in df.columns]
    if not feature_list:
        print("  No numeric features found to plot.")
        return

    n = len(feature_list)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, feat in enumerate(feature_list):
        color = _PALETTE[i % len(_PALETTE)]
        sns.histplot(df[feat].dropna(), kde=True, ax=axes[i],
                     color=color, edgecolor="white", linewidth=0.5)
        axes[i].set_title(feat.replace("_", " ").title(),
                          fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].grid(axis="y", alpha=0.25, linestyle="--")

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution of Key Numeric Features",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()

# 3. Target Variable Distribution
def plot_target_distribution(df_binary, df_multi=None):
    """
    Side-by-side panels showing binary imbalance + multiclass frequencies.

    Parameters
    ----------
    df_binary : DataFrame   � must contain 'came_for_retrofit'
    df_multi  : DataFrame   � optional, must contain 'umr_art_clean'
    """
    panels = 2 if df_multi is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(7 * panels, 5))
    if panels == 1:
        axes = [axes]

    # -- Binary --
    _t = config.COL_CAME_FOR_RETROFIT
    vc = df_binary[_t].value_counts().sort_index()
    labels = ["No Retrofit", "Retrofit"]
    colors = [_PALETTE[0], _PALETTE[1]]
    axes[0].bar(labels, vc.values, color=colors, edgecolor="black")
    for i, v in enumerate(vc.values):
        axes[0].text(i, v + max(vc) * 0.01, f"{v:,}", ha="center", fontsize=10)
    ratio = vc.min() / vc.max()
    axes[0].set_title(f"Binary Target  (imbalance ratio {ratio:.3f})",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    # -- Multiclass --
    if df_multi is not None:
        _u = config.COL_MULTICLASS_TARGET if config.COL_MULTICLASS_TARGET in df_multi.columns else config.COL_UMRUESTART
        vc2 = df_multi[_u].value_counts().sort_values(ascending=True)
        c = [_PALETTE[i % len(_PALETTE)] for i in range(len(vc2))]
        axes[1].barh(vc2.index, vc2.values, color=c, edgecolor="black", linewidth=0.4)
        for bar, v in zip(axes[1].patches, vc2.values):
            axes[1].text(bar.get_width() + max(vc2) * 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         f"{v:,}", va="center", fontsize=9)
        axes[1].set_title(f"Multiclass Target  ({len(vc2)} types)",
                          fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Count")
        axes[1].grid(axis="x", alpha=0.25, linestyle="--")

    fig.suptitle("Target Variable Distribution",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()

# 4. Temporal Trends in Historical Retrofit Volumes
def plot_temporal_trends(df_source_a_clean):
    """
    Monthly retrofit volumes over time, with optional breakdown by type.

    Parameters
    ----------
    df_source_a_clean : DataFrame
        Cleaned Source A data (output of cleanse_source_a). Must contain
        'umr_start' (datetime) and 'umr_art' (retrofit type).
    """
    _s = config.COL_START_UR
    _u = config.COL_UMRUESTART

    if _s not in df_source_a_clean.columns:
        print(f"  Column '{_s}' not found � cannot plot temporal trends.")
        return

    df = df_source_a_clean.dropna(subset=[_s]).copy()
    df["_month"] = df[_s].dt.to_period("M").dt.to_timestamp()

    # -- Panel 1: total monthly volume  |  Panel 2: stacked by type --
    fig, axes = plt.subplots(1, 2, figsize=_FIGSIZE_WIDE)

    # Total
    monthly = df.groupby("_month").size()
    axes[0].fill_between(monthly.index, monthly.values,
                         alpha=0.35, color=_PALETTE[0])
    axes[0].plot(monthly.index, monthly.values,
                 color=_PALETTE[0], linewidth=1.5)
    axes[0].set_title("Total Retrofit Volume / Month",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Retrofits")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].tick_params(axis="x", rotation=45)

    # Stacked by type (top-8 types, rest grouped as "Other")
    if _u in df.columns:
        top_types = df[_u].value_counts().head(8).index.tolist()
        df["_type"] = df[_u].where(df[_u].isin(top_types), "Other")
        pivot = df.groupby(["_month", "_type"]).size().unstack(fill_value=0)
        # Sort columns by total descending
        pivot = pivot[pivot.sum().sort_values(ascending=False).index]
        pivot.plot.area(ax=axes[1], stacked=True, alpha=0.7, linewidth=0.5,
                        color=_PALETTE[:pivot.shape[1]])
        axes[1].set_title("Monthly Volume by Retrofit Type (top 8)",
                          fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Retrofits")
        axes[1].legend(fontsize=7, loc="upper left", framealpha=0.8)
        axes[1].grid(axis="y", alpha=0.25, linestyle="--")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        axes[1].set_visible(False)

    fig.suptitle("Historical Retrofit Temporal Trends",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.show()
    plt.close("all"); gc.collect()

# 5. Descriptive Statistics Table
def descriptive_statistics(df, label=""):
    """
    Print and return a descriptive-statistics table for all numeric columns.

    Parameters
    ----------
    df : DataFrame
    label : str � optional label shown in the header
    """
    tag = f" ({label})" if label else ""
    num = df.select_dtypes(include="number")
    if num.empty:
        print(f"  No numeric columns found{tag}.")
        return pd.DataFrame()

    stats = num.describe().T
    stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats["count"] = stats["count"].astype(int)
    stats = stats.round(3)

    print(f"\n{'=' * 70}")
    print(f"DESCRIPTIVE STATISTICS{tag}  �  {len(stats)} numeric features")
    print(f"{'=' * 70}")
    print(stats.to_string())
    return stats


# 6. Convenience: run full EDA
def run_full_eda(df_binary, df_multi=None, df_source_a_clean=None):
    """
    Run all 5 EDA steps in sequence.

    Parameters
    ----------
    df_binary     : DataFrame — LEFT-joined dataset (binary target)
    df_multi      : DataFrame — INNER-joined dataset (multiclass target), optional
    df_source_a_clean : DataFrame — cleaned Source A data for temporal trends, optional
    """
    print("\n" + "#" * 70)
    print("#  EXPLORATORY DATA ANALYSIS (EDA)")
    print("#" * 70)

    # 1. Overview
    dataset_overview(df_binary, label="Binary � df_binary")
    if df_multi is not None:
        dataset_overview(df_multi, label="Multiclass � df_multi")

    # 2. Feature distributions (from binary, which has the Source B features)
    plot_feature_distributions(df_binary)

    # 3. Target distributions
    plot_target_distribution(df_binary, df_multi)

    # 4. Temporal trends (needs cleaned Source A with umr_start dates)
    if df_source_a_clean is not None:
        plot_temporal_trends(df_source_a_clean)

    # 5. Descriptive statistics
    descriptive_statistics(df_binary, label="Binary")

    print("\n? EDA complete.\n")
    gc.collect()


# --------------------------------------------------------------
# SECTION 3 — Data Cleansing (Source A)
# --------------------------------------------------------------

def cleanse_source_a(df_source_a_raw):
    """
    Clean Source A data: remove cancelled, deduplicate, parse dates,
    expand retrofit types, add temporal features.

    Returns only the feature-engineered dataset (no redundant copies).
    """
    print("\n" + "=" * 80)
    print("DATA CLEANSING: SOURCE A")
    print("=" * 80 + "\n")

    df = df_source_a_raw.copy()
    print(f"Step 1: Raw rows: {len(df):,}")

    # Remove cancelled
    before = len(df)
    df = df[df[config.COL_UMRUSTSTATUS] != config.CANCELLED_STATUS]
    print(f"Step 2: Removed cancelled: {before:,} -> {len(df):,}")

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates()
    print(f"Step 3: Deduplicated: {before:,} -> {len(df):,}")

    # Normalise umr_art: sort comma-separated components alphabetically
    # e.g. "TypeB, TypeA" → "TypeA, TypeB"
    _u = config.COL_UMRUESTART
    if _u in df.columns:
        n_types_before = df[_u].nunique()
        df[_u] = (
            df[_u]
            .astype(str)
            .apply(lambda x: ", ".join(sorted(p.strip() for p in str(x).split(",")))
                   if isinstance(x, str) and x != "nan" else str(x))
        )
        n_types_after = df[_u].nunique()
        print(f"Step 3b: Normalised {_u}: {n_types_before} ? {n_types_after} unique types")

        # Remove invalid umr_art combinations
        if config.INVALID_UMR_ART:
            _inv_mask = df[_u].isin(config.INVALID_UMR_ART)
            n_inv = _inv_mask.sum()
            if n_inv:
                df = df[~_inv_mask]
                print(f"Step 3c: Removed {n_inv:,} rows with invalid umr_art "
                      f"({config.INVALID_UMR_ART})")

        # Remove rows with missing / empty umr_art (NaN → "nan" after astype(str))
        _empty_mask = df[_u].isin(["nan", "<NA>", ""]) | df[_u].str.strip().eq("")
        n_empty = _empty_mask.sum()
        if n_empty:
            df = df[~_empty_mask]
            print(f"Step 3d: Removed {n_empty:,} rows with missing/empty umr_art")

    # Parse dates
    for col in config.DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Step 4 complete - dates parsed
    print(f"Step 4: Dates parsed")

    # Step 5: Temporal features
    _s = config.COL_START_UR
    _a = config.COL_AV_UR
    df["month"] = df[_s].dt.month
    df["quarter"] = df[_s].dt.quarter
    df["year"] = df[_s].dt.year
    df["day_of_week"] = df[_s].dt.dayofweek
    df["week_of_year"] = df[_s].dt.isocalendar().week  # Keep as nullable Int64

    # Compute duration (umr_av - umr_start) and derive lead_time_category
    if _a in df.columns and _s in df.columns:
        dur = (df[_a] - df[_s]).dt.days
        df["lead_time_category"] = pd.cut(
            dur,
            bins=[-float("inf"), -5, 5, 15, float("inf")],
            labels=["early", "on_time", "slight_delay", "major_delay"],
        )

    _u = config.COL_UMRUESTART
    monthly_counts = df.groupby(
        [df[_s].dt.to_period("M"), _u]
    ).size()
    df["monthly_retrofit_count"] = df.apply(
        lambda row: monthly_counts.get(
            (row[_s].to_period("M"), row[_u]), 0
        )
        if pd.notna(row[_s])
        else 0,
        axis=1,
    )

    print(f"Step 5: Feature engineering done. Rows: {len(df):,}")

    # Step 6: Composite join key (at_nummer -> bt_nummer -> v_nummer)
    build_join_key(df)

    gc.collect()
    mem()
    return df


def build_join_key(df):
    """Create composite ``_join_key`` column using the fallback chain
    at_nummer -> bt_nummer -> v_nummer.  Operates in-place."""
    key = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in reversed(config.JOIN_KEY_FALLBACK):
        if col not in df.columns:
            continue
        vals = df[col].astype(str).replace({"nan": pd.NA, "": pd.NA, "0": pd.NA, "0.0": pd.NA})
        key = key.where(key.notna(), vals)
    df[config.COL_JOIN_KEY] = key
    # Stats
    _counts = {}
    _used = pd.Series("none", index=df.index)
    for col in config.JOIN_KEY_FALLBACK:
        if col not in df.columns:
            continue
        valid = df[col].astype(str).replace({"nan": pd.NA, "": pd.NA, "0": pd.NA, "0.0": pd.NA}).notna()
        mask = valid & (_used == "none")
        _used = _used.where(~mask, col)
        _counts[col] = int(mask.sum())
    n_null = int(df[config.COL_JOIN_KEY].isna().sum())
    parts = ", ".join(f"{c}={n:,}" for c, n in _counts.items())
    print(f"  Join key fallback: {parts}, still null={n_null:,}")


def visualize_cleansing(df_source_a_clean):
    """Section 3 visualization: cleansing results."""
    print("\n" + "#" * 70)
    print("#  SECTION 3 VIZ — Data Cleansing")
    print("#" * 70)
    plot_temporal_trends(df_source_a_clean)
    _u = config.COL_UMRUESTART
    if _u in df_source_a_clean.columns:
        vc = df_source_a_clean[_u].value_counts()
        fig, ax = plt.subplots(figsize=(10, max(4, len(vc) * 0.35)))
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(vc))]
        ax.barh(vc.index[::-1], vc.values[::-1], color=colors[::-1],
                edgecolor="black", linewidth=0.4)
        for bar, v in zip(ax.patches, vc.values[::-1]):
            ax.text(bar.get_width() + max(vc) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:,}", va="center", fontsize=9)
        ax.set_title("Retrofit Type Distribution (after cleansing)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        plt.tight_layout(); plt.show()
        plt.close("all"); gc.collect()
    print("? Cleansing visualization complete.\n")


# --------------------------------------------------------------
# SECTION 4 � Data Transformation (Merge + Shared Feature Engineering)
# --------------------------------------------------------------

def merge_left_join(df_source_a, df_source_b_raw):
    """
    LEFT JOIN: Source B (all vehicles) ← Source A.
    For BINARY classification: came_for_retrofit (0/1).
    Only Source B features used (no Source A columns) to avoid leakage.
    """
    print("\n" + "=" * 80)
    print("DATA MERGE: LEFT JOIN (binary classification)")
    print("=" * 80 + "\n")

    df_source_b = df_source_b_raw.copy()
    build_join_key(df_source_b)
    print(f"Source B vehicles: {len(df_source_b):,}  |  Source A visits: {len(df_source_a):,}")

    # Use composite join key for matching
    _k = config.COL_JOIN_KEY
    source_a_vehicles = set(df_source_a.loc[df_source_a[_k].notna(), _k].unique())
    print(f"Source A unique vehicles (via fallback key): {len(source_a_vehicles):,}")

    df = df_source_b.copy()
    del df_source_b
    gc.collect()

    # Target: did this vehicle come for retrofit?
    df[config.COL_CAME_FOR_RETROFIT] = df[_k].isin(source_a_vehicles).astype("int32")

    pos = df[config.COL_CAME_FOR_RETROFIT].sum()
    print(f"Rows: {len(df):,}  |  Positive: {pos:,}  |  Negative: {len(df) - pos:,}")

    # Feature engineering (Source B features only)
    _engineer_features(df)

    # Auto-create _clean for all categorical columns (incl. string[python/pyarrow])
    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        df[f"{col}_clean"] = df[col].astype(str).replace("nan", config.FILLNA_CATEGORICAL)

    df = downcast_df(df)
    print(f"Final shape: {df.shape}")

    gc.collect()
    mem()
    return df


def merge_inner_join(df_source_a, df_source_b_raw):
    """
    INNER JOIN: Source B ← Source A on v_nummer.
    For MULTICLASS classification: predict retrofit type.
    All features from both sources available.
    """
    print("\n" + "=" * 80)
    print("DATA MERGE: INNER JOIN (multiclass)")
    print("=" * 80 + "\n")

    df_source_b = df_source_b_raw.copy()
    build_join_key(df_source_b)

    # Use composite join key for matching
    _k = config.COL_JOIN_KEY
    source_a_valid = df_source_a[df_source_a[_k].notna()].copy()
    print(f"Source B vehicles: {len(df_source_b):,}  |  Source A visits: {len(source_a_valid):,}")

    # INNER JOIN — only vehicles in both
    df = source_a_valid.merge(df_source_b, on=_k, how="inner", suffixes=("", "_src_b"))
    del df_source_b, source_a_valid
    gc.collect()

    # Deduplicate columns (prefer Source A, fallback Source B)
    src_b_suffix_cols = [c for c in df.columns if c.endswith("_src_b")]
    merged_count = 0
    for src_b_col in src_b_suffix_cols:
        base = src_b_col.removesuffix("_src_b")
        if base in df.columns:
            df[base] = df[base].fillna(df[src_b_col])
            merged_count += 1
        df.drop(columns=src_b_col, inplace=True)
    if merged_count:
        print(f"  Merged {merged_count} duplicate cols (Source A preferred, Source B fallback)")

    print(f"Joined: {len(df):,} rows")

    # Feature engineering (all features)
    _engineer_features(df)

    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        df[f"{col}_clean"] = df[col].astype(str).replace("nan", config.FILLNA_CATEGORICAL)

    df = downcast_df(df)
    print(f"Final shape: {df.shape}")

    gc.collect()
    mem()
    return df


# -- Internal helpers (used by both merge functions) ----------

def _get_col(df, base_name):
    for suffix in ("_src_b", "_src_a", ""):
        candidate = f"{base_name}{suffix}" if suffix else base_name
        if candidate in df.columns:
            return candidate
    return None


def _engineer_features(df):
    """Add all derived features in-place."""
    now = pd.Timestamp.now()

    anlauf = _get_col(df, config.COL_ANLAUF_SOP)
    if anlauf:
        df[anlauf] = pd.to_datetime(df[anlauf], errors="coerce").dt.tz_localize(None)
        # vehicle_age_months: positive = SOP in future (pre-production), negative = already in production
        df["vehicle_age_months"] = ((df[anlauf] - now).dt.days / 30.44).fillna(0).astype("float32")

    # av_fzg (Source B): date the whole car must be ready
    # positive = deadline in future, negative = already done
    av_fzg_col = _get_col(df, "av_fzg")
    if av_fzg_col:
        df[av_fzg_col] = pd.to_datetime(df[av_fzg_col], errors="coerce").dt.tz_localize(None)
        df["car_ready_age_months"] = ((df[av_fzg_col] - now).dt.days / 30.44).fillna(0).astype("float32")

    # umr_av (Source A): date the retrofit must be ready
    # positive = deadline in future, negative = already done
    umr_av_col = _get_col(df, config.COL_AV_UR)
    if umr_av_col:
        df[umr_av_col] = pd.to_datetime(df[umr_av_col], errors="coerce").dt.tz_localize(None)
        df["retrofit_ready_age_months"] = ((df[umr_av_col] - now).dt.days / 30.44).fillna(0).astype("float32")

    # Separate categories for car-ready and retrofit-ready
    if "car_ready_age_months" in df.columns:
        df["car_ready_category"] = pd.cut(
            df["car_ready_age_months"], bins=[-999, 0, 3, 6, 12, 24, 999],
            labels=["done", "0-3m", "3-6m", "6-12m", "12-24m", "24m+"],
        ).astype(str)
    if "retrofit_ready_age_months" in df.columns:
        df["retrofit_ready_category"] = pd.cut(
            df["retrofit_ready_age_months"], bins=[-999, 0, 3, 6, 12, 24, 999],
            labels=["done", "0-3m", "3-6m", "6-12m", "12-24m", "24m+"],
        ).astype(str)

    # Recently completed (within last 6 months): value between -6 and 0
    if "car_ready_age_months" in df.columns:
        df["is_recent_av_fzg"] = ((df["car_ready_age_months"] < 0) & (df["car_ready_age_months"] > -6)).astype("int32")
    if "retrofit_ready_age_months" in df.columns:
        df["is_recent_av_retrofit"] = ((df["retrofit_ready_age_months"] < 0) & (df["retrofit_ready_age_months"] > -6)).astype("int32")

    derivat = _get_col(df, config.COL_DERIVAT)
    basistyp = _get_col(df, "basistyp")
    if derivat and basistyp:
        df["derivat_x_basistyp"] = (
            df[derivat].astype(str).replace("nan", config.FILLNA_CATEGORICAL)
            + "_"
            + df[basistyp].astype(str).replace("nan", config.FILLNA_CATEGORICAL)
        )

    if derivat:
        counts = df[derivat].value_counts()
        df["derivat_fleet_size"] = df[derivat].map(counts).fillna(0).astype("float32")
        df["is_common_derivat"] = (df["derivat_fleet_size"] > counts.median()).astype("int32")

    hybrid = _get_col(df, "hybrid")
    antrieb = _get_col(df, "antriebsart")
    if hybrid and antrieb:
        df["hybrid_x_antrieb"] = (
            df[hybrid].astype(str).replace("nan", config.FILLNA_CATEGORICAL)
            + "_"
            + df[antrieb].astype(str).replace("nan", config.FILLNA_CATEGORICAL)
        )

    prio = _get_col(df, config.COL_PRIO)
    istufe = _get_col(df, "istufe") or _get_col(df, "current_istufe")
    if prio and istufe:
        df["prio_x_istufe"] = (
            df[prio].astype(str).replace("nan", config.FILLNA_CATEGORICAL)
            + "_"
            + df[istufe].astype(str).replace("nan", config.FILLNA_CATEGORICAL)
        )

    bauphase = _get_col(df, "pep_bauphase")
    if derivat and bauphase:
        df["derivat_x_bauphase"] = (
            df[derivat].astype(str).replace("nan", config.FILLNA_CATEGORICAL)
            + "_"
            + df[bauphase].astype(str).replace("nan", config.FILLNA_CATEGORICAL)
        )

    if "vehicle_age_months" in df.columns:
        df["vehicle_age_category"] = pd.cut(
            df["vehicle_age_months"], bins=[-999, 0, 12, 24, 36, 60, 999],
            labels=["in-prod", "0-1y", "1-2y", "2-3y", "3-5y", "5y+"],
        ).astype(str)
        # New vehicle = SOP in the future (positive after flip)
        df["is_new_vehicle"] = (df["vehicle_age_months"] > 0).astype("int32")

    ksps = _get_col(df, "ksps") or _get_col(df, "ist_ksps")
    if prio and ksps:
        df["prio_x_ksps"] = (
            df[prio].astype(str).replace("nan", config.FILLNA_CATEGORICAL)
            + "_"
            + df[ksps].astype(str).replace("nan", config.FILLNA_CATEGORICAL)
        )

    # Fleet-size features
    for name, getter in [
        ("motortyp", lambda: _get_col(df, "motortyp") or _get_col(df, "motortype")),
        ("pl", lambda: _get_col(df, "pl")),
        ("puk_werk", lambda: _get_col(df, "puk_werk")),
        ("bo", lambda: _get_col(df, "bo")),
    ]:
        src = getter()
        if src:
            counts = df[src].value_counts()
            df[f"{name}_fleet_size"] = df[src].map(counts).fillna(0).astype("float32")

    # rueckgabe_soll (Source B deadline column)
    rs = _get_col(df, config.COL_RUECKGABE_SOLL)
    if rs:
        df[rs] = pd.to_datetime(df[rs], errors="coerce").dt.tz_localize(None)
        df["days_to_deadline"] = ((df[rs] - now).dt.days).fillna(999).astype("float32")
        df["has_deadline"] = df[rs].notna().astype("int32")

    # -- Additional engineered features (v2) ------------------
    # Frequency-encoded: how common each categorical value is (ratio 0-1)
    n_rows = len(df)
    for name, col in [
        ("derivat_freq", _get_col(df, config.COL_DERIVAT)),
        ("basistyp_freq", _get_col(df, "basistyp")),
        ("motortyp_freq", _get_col(df, "motortyp") or _get_col(df, "motortype")),
        ("bo_freq", _get_col(df, "bo")),
        ("puk_werk_freq", _get_col(df, "puk_werk")),
        ("istufe_freq", istufe),
    ]:
        if col and col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[name] = df[col].map(freq).fillna(0).astype("float32")

    # Numeric interactions
    if "vehicle_age_months" in df.columns and "derivat_fleet_size" in df.columns:
        df["age_x_fleet"] = (
            df["vehicle_age_months"] * df["derivat_fleet_size"]
        ).astype("float32")

    if "days_to_deadline" in df.columns and "vehicle_age_months" in df.columns:
        df["deadline_x_age"] = (
            df["days_to_deadline"] * df["vehicle_age_months"]
        ).astype("float32")

    if prio and "derivat_fleet_size" in df.columns:
        # Higher prio + larger fleet ? more likely retrofit
        _prio_num = pd.to_numeric(df[prio], errors="coerce").fillna(0)
        df["prio_x_fleet"] = (_prio_num * df["derivat_fleet_size"]).astype("float32")

    if "has_deadline" in df.columns and "is_new_vehicle" in df.columns:
        df["deadline_x_new"] = (
            df["has_deadline"] * df["is_new_vehicle"]
        ).astype("float32")


def visualize_transformation(df_binary, df_multi=None, df_source_a_clean=None):
    """Section 4 visualization: post-merge data quality."""
    run_full_eda(df_binary, df_multi, df_source_a_clean)


