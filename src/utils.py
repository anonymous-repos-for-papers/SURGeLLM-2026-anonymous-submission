# ============================================================
# utils.py - Memory helpers, caching, downcast
# ============================================================
import gc
import hashlib
import os
import shutil
import psutil
import joblib
import pandas as pd
from . import config


def mem():
    """Print current process memory usage in MB."""
    used = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"Memory: {used:.0f} MB")


def downcast_df(df):
    """Downcast int64->int32, float64->float32 to halve memory."""
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    return df


def free(*objs):
    """Delete objects and force garbage collection."""
    for o in objs:
        del o
    gc.collect()


# ── Data-aware cache invalidation ────────────────────────────

def _data_fingerprint(*dfs):
    """Compute a fast hash from shape + a sample of each DataFrame.

    Combines: number of rows, number of columns, column names, dtypes,
    and a deterministic sample of up to 200 rows per DF.  This is
    enough to detect new rows, new columns, or schema changes without
    hashing the entire (potentially huge) dataset.
    """
    h = hashlib.sha256()
    for df in dfs:
        if df is None or not isinstance(df, pd.DataFrame):
            continue
        h.update(f"shape={df.shape}".encode())
        h.update(",".join(df.columns).encode())
        h.update(",".join(str(d) for d in df.dtypes).encode())
        # Deterministic sample (first + last 100 rows)
        n = min(100, len(df))
        sample = pd.concat([df.head(n), df.tail(n)], ignore_index=True)
        h.update(sample.to_csv(index=False).encode(errors="replace"))
    return h.hexdigest()[:16]   # 16 hex chars = 64 bits — collision-free enough


def validate_cache(*dfs):
    """Check if the current data matches the cached fingerprint.

    If the data has changed (new rows, new columns, etc.), all
    cached checkpoints are deleted automatically so models are
    retrained from scratch.

    Call this once *after* loading the raw DataFrames and *before*
    the first ``load_checkpoint`` call.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    fp_path = os.path.join(config.CACHE_DIR, config.CACHE_FINGERPRINT_FILE)
    current_fp = _data_fingerprint(*dfs)

    if os.path.exists(fp_path):
        saved_fp = open(fp_path).read().strip()
        if saved_fp == current_fp:
            print(f"  Cache fingerprint OK ({current_fp}) — reusing checkpoints")
            return
        # Data changed → wipe everything
        print(f"  ⚠ Data changed!  old={saved_fp}  new={current_fp}")
        print(f"  → Deleting all cached checkpoints in {config.CACHE_DIR}")
        shutil.rmtree(config.CACHE_DIR)
        os.makedirs(config.CACHE_DIR, exist_ok=True)
    else:
        print(f"  No previous fingerprint — fresh run")

    # Save new fingerprint
    with open(fp_path, "w") as f:
        f.write(current_fp)
    print(f"  Fingerprint saved: {current_fp}")


def _reconstruct_clf(module_path, class_name, state):
    """Reconstruct a classifier from its module path, class name, and state.

    Used by LLM-based classifiers whose module names start with digits
    (03_data_modeling_and_evaluation_binary,
     04_data_modeling_and_evaluation_multiclass).  Pickle cannot resolve
    class identity for those modules in Jupyter notebooks because
    re-execution creates fresh class objects while the trained model's
    __class__ still references the old one.  By routing through this
    function (which lives in a 'normal' module), pickle can serialise
    and deserialise the classifiers without the identity check.
    """
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    return obj


def save_checkpoint(obj, name):
    """Save an intermediate result to disk so we can resume after a crash."""
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    path = os.path.join(config.CACHE_DIR, f"{name}.pkl")
    joblib.dump(obj, path, compress=3)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(name):
    """Load a previously saved checkpoint. Returns None if not found."""
    path = os.path.join(config.CACHE_DIR, f"{name}.pkl")
    if os.path.exists(path):
        try:
            obj = joblib.load(path)
            print(f"  Checkpoint loaded: {path}")
            return obj
        except Exception as exc:
            print(f"  ⚠ Checkpoint corrupt/incompatible ({exc.__class__.__name__}), "
                  f"deleting {path} — will recompute")
            os.remove(path)
    return None


# ── KDD Phase Summaries ──────────────────────────────────────

def kdd_summary(problem, phase, **metrics):
    """Print a standardised KDD phase summary and return a result dict.

    Parameters
    ----------
    problem : str   – "Binary", "Multiclass", "Temporal", "Time Series"
    phase   : str   – "Cleansing", "Transformation", "Modeling", "Evaluation"
    **metrics       – key/value pairs to display (e.g. n_rows=1000, auc=0.95)

    Returns
    -------
    dict with keys: problem, phase, and all metrics.
    """
    hdr = f"  KDD [{problem}] {phase}"
    print(f"\n{'─'*60}")
    print(hdr)
    print(f"{'─'*60}")
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            print(f"    {label:<30s} {v:.4f}")
        else:
            print(f"    {label:<30s} {v}")
    print(f"{'─'*60}")
    mem()
    return {"problem": problem, "phase": phase, **metrics}
