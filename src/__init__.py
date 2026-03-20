# ============================================================
# src/__init__.py – Package initialiser
#
# This file is intentionally kept empty / lightweight.
# run_pipeline.py imports each submodule explicitly via
# importlib.import_module(), so __init__.py does NOT need
# to eagerly import anything.
#
# Keeping it minimal avoids cascading ImportErrors when
# third-party packages (seaborn, xgboost, …) are not yet
# installed at the time `import src` is first called.
# ============================================================
