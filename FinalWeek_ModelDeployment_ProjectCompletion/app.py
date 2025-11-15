# app.py — Flask App Entry Point (corrected for Flask 3+)
from flask import Flask, render_template, request, redirect, url_for, send_file, abort, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.exceptions import NotFittedError
from datetime import datetime
import logging
import traceback

# CONFIG
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "co2_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
DATA_CSV = BASE_DIR / "data" / "processed" / "processed_co2_data.csv"

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET', 'dev-secret')
# app.py (add these imports at top)
from flask import Flask, request, jsonify, render_template, url_for
import pandas as pd
import io

app = Flask(__name__)

# Helper: try to find column matching candidates (case-insensitive)
def find_column(df_columns, candidates):
    lc = [c.lower() for c in df_columns]
    for cand in candidates:
        if cand.lower() in lc:
            return df_columns[lc.index(cand.lower())]
    # try partial match
    for i, col in enumerate(lc):
        for cand in candidates:
            if cand.lower() in col:
                return df_columns[i]
    return None

@app.route('/upload_emissions', methods=['POST'])
def upload_emissions():
    """
    Expects: form-data with key 'file' containing a CSV.
    Returns: JSON with labels (years) and values (emissions).
    """
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify(error='No selected file'), 400

    filename = f.filename.lower()
    if not filename.endswith('.csv'):
        return jsonify(error='Only CSV files are supported'), 400

    try:
        # read CSV into pandas
        contents = f.read()
        # handle different encodings / whitespace
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        return jsonify(error=f'Failed to read CSV: {str(e)}'), 400

    if df.shape[0] == 0:
        return jsonify(error='CSV has no rows'), 400

    # heuristics for year and emissions column names
    year_candidates = ['year', 'Year', 'yr']
    emission_candidates = [
        'annual co₂ emissions (tonnes )',
        'annual co2 emissions (tonnes )',
        'emissions',
        'emission',
        'co2',
        'co₂',
        'annual co2',
        'annual emissions'
    ]

    # Try to find columns
    df_cols = list(df.columns)
    year_col = find_column(df_cols, year_candidates)
    emis_col = find_column(df_cols, emission_candidates)

    if not year_col or not emis_col:
        # give user helpful message including available columns
        return jsonify(error='Could not locate year and emission columns in CSV. Available columns: ' + ', '.join(df_cols)), 400

    # attempt to coerce year to numeric and emission to numeric
    try:
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df[emis_col] = pd.to_numeric(df[emis_col], errors='coerce')
    except Exception:
        pass

    # drop rows with missing year or emis
    df = df.dropna(subset=[year_col, emis_col])
    if df.shape[0] == 0:
        return jsonify(error='No numeric rows found after parsing year and emission columns'), 400

    # group by year if multiple entries per year (sum or mean — choose sum)
    grouped = df.groupby(year_col, as_index=False)[emis_col].sum()

    # sort by year ascending
    grouped = grouped.sort_values(by=year_col)

    # prepare labels/values as lists (convert years to strings for chart labels)
    labels = grouped[year_col].astype(int).astype(str).tolist()
    values = grouped[emis_col].tolist()

    return jsonify(labels=labels, values=values, label='Annual CO₂ Emissions (tonnes)')


import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

# Path constants (adjust if needed)
BASE_DIR = Path(r"C:\Users\Giri V\Downloads\CO2_Emission_Estimation_Full\CO2_Emission_Estimation_Full\flask_app")
MODELS_DIR = BASE_DIR / "models"
FEATURES_PATH = MODELS_DIR / "feature_list.pkl"   # optional, created if you save it in training
PIPELINE_PATH = MODELS_DIR / "co2_pipeline.pkl"   # optional (recommended)
MODEL_PATH = MODELS_DIR / "co2_model.pkl"

# Load saved feature_list if exists
def load_feature_list():
    if FEATURES_PATH.exists():
        try:
            with open(FEATURES_PATH, "rb") as f:
                feature_list = pickle.load(f)
            app.logger.debug("Loaded feature_list from %s", FEATURES_PATH)
            return feature_list
        except Exception as e:
            app.logger.warning("Could not load feature_list.pkl: %s", e)
    # fallback: None (we'll infer from model if possible)
    return None

# A small helper to compute simple imputations from your processed CSV (median global or per-entity)
def compute_imputations(processed_csv_path=BASE_DIR / "data" / "processed" / "processed_co2_data.csv"):
    impute = {}
    try:
        if processed_csv_path.exists():
            df = pd.read_csv(processed_csv_path)
            # global medians
            for col in ["Emission_Intensity", "Renewable_Share"]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    if vals.notna().sum() > 0:
                        impute[col] = float(vals.median())
                    else:
                        impute[col] = None
                else:
                    impute[col] = None
            # optional: per-entity medians (for better imputation)
            # build map: impute_entity[entity][col] = median
            impute_entity = {}
            if "entity" in df.columns:
                for ent, g in df.groupby("entity"):
                    impute_entity[ent] = {}
                    for col in ["Emission_Intensity", "Renewable_Share"]:
                        vals = pd.to_numeric(g[col], errors="coerce")
                        if vals.notna().sum() > 0:
                            impute_entity[ent][col] = float(vals.median())
                        else:
                            impute_entity[ent][col] = None
                impute["per_entity"] = impute_entity
            return impute
    except Exception as e:
        app.logger.warning("compute_imputations failed: %s", e)
    # fallback defaults
    return {"Emission_Intensity": None, "Renewable_Share": None, "per_entity": {}}

IMPUTE_CACHE = compute_imputations()

# prepare features robustly
def prepare_features_from_form(form):
    """
    Build a single-row DataFrame with same columns & order as training.
    Priorities:
    1) If models/feature_list.pkl exists, use that order and column set.
    2) Else, if model.feature_names_in_ exists, use that.
    3) Else, build from common columns we expect and fill missing with sensible values.
    """
    # 1) read input dict
    input_dict = {k: (v if v != "" else None) for k, v in form.items()}
    # 2) load feature_list if available
    feature_list = load_feature_list()

    # 3) If we have a pipeline saved, prefer that (handled in predict)
    # Build a row from input for now
    if feature_list is None:
        # default feature candidates (common columns found in your CSV)
        feature_list = ["entity", "code", "year", "Emission_Intensity", "Renewable_Share",
                        "missing_emission_intensity", "missing_renewable_share"]
        app.logger.debug("No feature_list.pkl found — using default feature candidates: %s", feature_list)

    # build row dict with keys = feature_list
    row = {}
    for col in feature_list:
        raw = input_dict.get(col, None)

        # Coerce common typed columns
        if raw is None:
            # try imputing numeric columns
            if col in ["Emission_Intensity", "Renewable_Share"]:
                # try per-entity imputation first
                ent = input_dict.get("entity", None)
                val = None
                if ent and IMPUTE_CACHE.get("per_entity", {}).get(ent):
                    val = IMPUTE_CACHE["per_entity"][ent].get(col)
                if val is None:
                    val = IMPUTE_CACHE.get(col, None)
                # fallback to 0.0 if no median available - but prefer NaN so the pipeline can handle it
                row[col] = val if val is not None else np.nan
            elif col in ["year"]:
                row[col] = int(input_dict.get("year")) if input_dict.get("year") not in [None, ""] else np.nan
            elif col.startswith("missing_"):
                # treat as boolean flag: if user didn't supply, compute from inputs
                if col == "missing_emission_intensity":
                    row[col] = pd.isna(row.get("Emission_Intensity", np.nan))
                elif col == "missing_renewable_share":
                    row[col] = pd.isna(row.get("Renewable_Share", np.nan))
                else:
                    row[col] = False
            else:
                row[col] = None
        else:
            # convert numeric-like strings to numbers for known numeric columns
            if col in ["Emission_Intensity", "Renewable_Share", "year"]:
                try:
                    row[col] = float(raw) if col != "year" else int(float(raw))
                except Exception:
                    row[col] = np.nan
            elif col.startswith("missing_"):
                # user may send 'True' or 'False' strings
                if str(raw).lower() in ["true", "1", "yes"]:
                    row[col] = True
                elif str(raw).lower() in ["false", "0", "no"]:
                    row[col] = False
                else:
                    # compute from values if possible
                    if col == "missing_emission_intensity":
                        row[col] = pd.isna(input_dict.get("Emission_Intensity"))
                    elif col == "missing_renewable_share":
                        row[col] = pd.isna(input_dict.get("Renewable_Share"))
                    else:
                        row[col] = False
            else:
                # keep as string for categorical columns like entity, code
                row[col] = str(raw)

    # create DataFrame in exact order
    X_df = pd.DataFrame([row], columns=feature_list)
    # final housekeeping: if some columns are all None, convert to numeric NaN
    for c in X_df.columns:
        if c in ["year", "Emission_Intensity", "Renewable_Share"]:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    return X_df

# configure simple logging
if not app.logger.handlers:
    logging.basicConfig(level=logging.INFO)

# global model objects
model = None
scaler = None
_expected_features = None  # will hold a list of feature names model expects (if available)


def load_model():
    """
    Loads model and scaler into global variables if they are not already loaded.
    Also attempts to populate _expected_features from model.feature_names_in_ (preferred).
    Safe to call multiple times because of guards.
    """
    global model, scaler, _expected_features

    # Load model
    try:
        if model is None:
            if MODEL_PATH.exists():
                app.logger.info("Loading model from %s", MODEL_PATH)
                with open(MODEL_PATH, "rb") as f:
                    model = pickle.load(f)
                app.logger.info("Model loaded: %s", type(model))
            else:
                app.logger.warning("Model file not found at %s", MODEL_PATH)
    except Exception as e:
        app.logger.exception("Failed to load model: %s", e)
        model = None

    # Load scaler
    try:
        if scaler is None:
            if SCALER_PATH.exists():
                app.logger.info("Loading scaler from %s", SCALER_PATH)
                with open(SCALER_PATH, "rb") as f:
                    scaler = pickle.load(f)
                app.logger.info("Scaler loaded: %s", type(scaler))
            else:
                app.logger.warning("Scaler file not found at %s", SCALER_PATH)
    except Exception as e:
        app.logger.exception("Failed to load scaler: %s", e)
        scaler = None

    # Determine expected feature names if possible
    try:
        if model is not None and hasattr(model, "feature_names_in_"):
            _expected_features = list(model.feature_names_in_)
            app.logger.info("Model feature names detected: %s", _expected_features)
        else:
            # Fall back to scaler feature names (sometimes scaler saved the preprocessor columns)
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                _expected_features = list(scaler.feature_names_in_)
                app.logger.info("Scaler feature names detected (used as expected_features): %s", _expected_features)
            else:
                _expected_features = None
                # We can still use model.n_features_in_ to verify number of features expected
                if model is not None and hasattr(model, "n_features_in_"):
                    app.logger.info("Model expects %d features (names not available).", int(model.n_features_in_))
    except Exception as e:
        app.logger.exception("Error determining expected features: %s", e)
        _expected_features = None


# load model at import time
load_model()


def _get_from_form(form: dict, keys):
    """
    Utility: return first non-empty value from form for a list of possible keys (case-insensitive).
    """
    for k in keys:
        # try exact key
        if k in form and form.get(k) not in (None, ''):
            return form.get(k)
    # try case-insensitive match
    lowered = {kk.lower(): kk for kk in form.keys()}
    for k in keys:
        lk = k.lower()
        if lk in lowered and form.get(lowered[lk]) not in (None, ''):
            return form.get(lowered[lk])
    return None


def prepare_features_from_form(form: dict):
    """
    Build a pandas.DataFrame (1 row) that matches the model's expected input columns (names & order).
    - Uses model.feature_names_in_ if available.
    - If feature names are unavailable, attempts to create numeric inputs from common form fields.
    - Ensures returned object is a DataFrame with named columns (not a numpy array).
    """
    global scaler, model, _expected_features

    try:
        # Basic candidate raw inputs gathered from form (expand as needed)
        # We'll try several common keys/names to be robust.
        raw = {}
        # numeric candidates
        raw['year'] = _get_from_form(form, ['year', 'Year'])
        raw['Emission_Intensity'] = _get_from_form(form, ['emission_intensity', 'Emission_Intensity', 'emissionIntensity', 'emission'])
        raw['Renewable_Share'] = _get_from_form(form, ['renewable_share', 'Renewable_Share', 'renewableShare', 'renewable'])
        raw['population'] = _get_from_form(form, ['population', 'Population', 'pop', 'popn'])
        # entity and actual included for saving results but not used as numeric features by default
        raw['entity'] = _get_from_form(form, ['entity', 'Entity', 'country', 'region'])
        raw['actual'] = _get_from_form(form, ['actual', 'Actual'])

        # Convert available numeric strings to floats or np.nan
        numeric_cols = {}
        for k in ['year', 'Emission_Intensity', 'Renewable_Share', 'population']:
            val = raw.get(k)
            if val is None or val == '':
                numeric_cols[k] = np.nan
            else:
                try:
                    numeric_cols[k] = float(val)
                except Exception:
                    # if conversion fails, set NaN
                    numeric_cols[k] = np.nan

        # start DataFrame with the numeric columns we found
        X = pd.DataFrame([numeric_cols])

        # If your trained model used additional engineered features, compute them here.
        # Example (uncomment/modify if used in training):
        # X['year_sq'] = X['year'] ** 2
        # X['energy_per_capita'] = X['some_energy_col'] / X['population']

        # Fill missing numeric values with median or sensible defaults (match training imputation if possible)
        if not X.empty:
            # numeric_only median fill - for a single-row df, median of row is the value itself or NaN
            # it's better to use a fixed default if you know one; here we fallback to 0 for safety
            X = X.fillna(X.median(numeric_only=True))
            X = X.fillna(0.0)  # final fallback; adjust if you prefer np.nan or a different strategy

        # If model exposes feature names, add missing columns and reorder
        if _expected_features:
            # Add any missing expected columns with default 0.0
            for col in _expected_features:
                if col not in X.columns:
                    X[col] = 0.0
            # Reorder columns to expected_features order
            X = X[_expected_features]
        else:
            # If no expected names but we know how many features model expects, pad/reorder as needed
            if model is not None and hasattr(model, "n_features_in_"):
                expected_n = int(model.n_features_in_)
                # If we currently have fewer columns than expected, add dummy columns f_0.. to match count.
                if X.shape[1] < expected_n:
                    for i in range(expected_n - X.shape[1]):
                        X[f"filler_{i}"] = 0.0
                # If too many, drop extras (shouldn't usually happen)
                if X.shape[1] > expected_n:
                    X = X.iloc[:, :expected_n]

        # Apply scaler if present: try to transform the columns the scaler knows about; operate on DataFrame
        if scaler is not None:
            try:
                # If scaler saved feature names, use them
                if hasattr(scaler, "feature_names_in_"):
                    scaler_cols = [c for c in scaler.feature_names_in_ if c in X.columns]
                else:
                    # fallback: choose numeric columns in X
                    scaler_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]

                if scaler_cols:
                    # Copy so we don't ruin X's other columns
                    X_to_scale = X[scaler_cols].astype(float)
                    scaled_vals = scaler.transform(X_to_scale)
                    # scaler.transform returns numpy array; place back into DataFrame with same column names
                    X_scaled = X.copy()
                    X_scaled[scaler_cols] = scaled_vals
                    X = X_scaled
                else:
                    app.logger.warning("Scaler available but no overlapping columns found to scale.")
            except Exception as e:
                app.logger.exception("Scaler transform failed — returning unscaled DataFrame. %s", e)

        app.logger.debug("Prepared feature DataFrame columns: %s", list(X.columns))
        app.logger.debug("Prepared feature DataFrame shape: %s", X.shape)
        return X

    except Exception as e:
        app.logger.exception("Error preparing features from form: %s", e)
        raise


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    # Load data for charts and stats
    if DATA_CSV.exists():
        try:
            df = pd.read_csv(DATA_CSV)
        except Exception as e:
            app.logger.warning("Failed to read DATA_CSV: %s", e)
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # Load recent predictions to get total count
    total_predictions = 0
    recent_predictions = []
    recent_file = BASE_DIR / "data" / "processed" / "predictions_recent.csv"
    if recent_file.exists():
        try:
            rp_df = pd.read_csv(recent_file)
            total_predictions = len(rp_df)
            recent_predictions = rp_df.to_dict(orient='records')[-10:] # Keep showing last 10
        except Exception as e:
            app.logger.warning("Failed to read recent predictions file: %s", e)

    # compute quick stats (guarded)
    stats = {
        "total_samples": int(len(df)) if not df.empty else 0,
        "total_predictions": total_predictions
    }

    # Placeholder metrics - if you saved model evaluation metrics somewhere, load them instead
    metrics = {"mae": "-", "rmse": "-", "r2": "-"}

    # Chart data examples (aggregate)
    if not df.empty:
        # ensure columns exist - adapt column names to your CSV
        if 'year' in df.columns and 'annual co₂ emissions (tonnes )' in df.columns:
            try:
                emissions_by_year = df.groupby('year')['annual co₂ emissions (tonnes )'].sum().sort_index()
                chart_emissions = {
                    "labels": emissions_by_year.index.astype(str).tolist(),
                    "values": emissions_by_year.values.tolist()
                }
            except Exception as e:
                app.logger.warning("Failed to aggregate emissions_by_year: %s", e)
                chart_emissions = {"labels": [], "values": []}
        else:
            chart_emissions = {"labels": [], "values": []}

        # feature importance if available
        feat_labels, feat_values = [], []
        global model
        try:
            if model is not None and hasattr(model, "feature_importances_"):
                feat_values = model.feature_importances_.tolist()
                feat_labels = getattr(model, "feature_names_in_", None)
                if feat_labels is None:
                    feat_labels = [f"f{i}" for i in range(len(feat_values))]
        except Exception:
            feat_labels, feat_values = [], []
    else:
        chart_emissions = {"labels": [], "values": []}
        feat_labels, feat_values = [], []


    return render_template(
        "dashboard.html",
        stats=stats,
        metrics=metrics,
        chart={"emissions": chart_emissions, "features": {"labels": feat_labels, "values": feat_values}},
        recent_predictions=recent_predictions,
        last_updated=(datetime.now().strftime("%Y-%m-%d %H:%M"))
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("prediction.html")

    # POST: form submitted
    try:
        # try to load full pipeline first (if you later create it)
        pipeline = None
        if PIPELINE_PATH.exists():
            try:
                with open(PIPELINE_PATH, "rb") as f:
                    pipeline = pickle.load(f)
                app.logger.debug("Loaded pipeline from %s", PIPELINE_PATH)
            except Exception as e:
                app.logger.warning("Couldn't load pipeline: %s", e)

        # If pipeline exists, use it (this avoids manual scaling/encoding)
        if pipeline is not None:
            X_df = prepare_features_from_form(request.form)  # prepare same column names
            # If pipeline was trained on a specific feature set, it will drop/ignore unknown columns
            # but it expects columns with correct dtypes; we try to pass the columns pipeline expects
            try:
                y_pred = pipeline.predict(X_df)
                pred_value = float(y_pred[0])
            except Exception as e:
                app.logger.exception("Pipeline.predict failed: %s", e)
                return render_template("prediction.html", prediction="Error",
                                       explanation="Pipeline predict failed: " + str(e))
        else:
            # fallback to legacy model variable loaded by your load_model() function
            load_model()  # ensure global `model` is loaded as before
            if model is None:
                return render_template("prediction.html", prediction="Model not available",
                                       explanation="Please ensure models/co2_model.pkl exists on server.")

            # Build DataFrame using saved feature_list if possible, else try model.feature_names_in_
            feature_list = load_feature_list()
            if feature_list is None:
                if hasattr(model, "feature_names_in_"):
                    feature_list = list(model.feature_names_in_)
                    app.logger.debug("Using model.feature_names_in_: %s", feature_list)
                else:
                    # last-resort defaults (same as prepare helper)
                    feature_list = ["entity", "code", "year", "Emission_Intensity", "Renewable_Share",
                                    "missing_emission_intensity", "missing_renewable_share"]
                    app.logger.debug("No feature_list or model.feature_names_in_; using fallback list: %s", feature_list)

            X_df = prepare_features_from_form(request.form)

            # Ensure columns exactly match feature_list order; create missing columns as NaN (not zeros)
            for c in feature_list:
                if c not in X_df.columns:
                    X_df[c] = np.nan
            X_df = X_df[feature_list]

            app.logger.debug("Final DataFrame sent to model: columns=%s", list(X_df.columns))
            app.logger.debug("Final DataFrame head:\n%s", X_df.head().to_dict(orient="records"))

            # run prediction
            try:
                y_pred = model.predict(X_df)
                pred_value = float(y_pred[0])
            except Exception as e:
                app.logger.exception("Prediction failed: %s", e)
                return render_template("prediction.html", prediction="Error", explanation=str(e))

        # Save recent prediction record (unchanged from your code)
        recent_file = BASE_DIR / "data" / "processed" / "predictions_recent.csv"
        rec = {
            "timestamp": datetime.now().isoformat(),
            "entity": request.form.get('entity', 'N/A'),
            "year": request.form.get('year', ''),
            "actual": request.form.get('actual', ''),
            "predicted": pred_value
        }
        try:
            df_rec = pd.DataFrame([rec])
            if recent_file.exists():
                df_rec.to_csv(recent_file, mode='a', header=False, index=False)
            else:
                recent_file.parent.mkdir(parents=True, exist_ok=True)
                df_rec.to_csv(recent_file, index=False)
        except Exception as e:
            app.logger.warning("Could not save recent prediction: %s", e)

        pretty_pred = f"{pred_value:,.2f}"
        return render_template("prediction.html", prediction=pretty_pred, explanation="Prediction in tonnes (rounded).")

    except Exception as e:
        app.logger.exception("Prediction failed top-level: %s", e)
        return render_template("prediction.html", prediction="Error", explanation=str(e))


from flask import make_response, Response

@app.route("/download_csv")
def download_csv():
    """
    Serve the processed CSV for export.
    Friendly fallback if the file is missing (does not require templates/error.html).
    """
    candidates = [
        DATA_CSV,
        BASE_DIR / "data" / "processed_co2_data.csv",
        BASE_DIR / "data" / "processed" / "predictions_recent.csv"
    ]
    found = None
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                found = p
                break
        except Exception:
            app.logger.exception("Error checking candidate path %s", p)

    if found:
        try:
            return send_file(str(found), as_attachment=True)
        except Exception as e:
            app.logger.exception("Failed to send CSV %s: %s", found, e)
            html = f"<h3>Download failed</h3><p>Could not send file: {e}</p>"
            return make_response(html, 500)

    app.logger.warning("CSV not found. Candidates: %s", [str(p) for p in candidates])
    # Simple dev-friendly HTML response (no missing template required)
    msg = (
        f"<h3>CSV not found</h3>"
        f"<p>Expected one of these paths on the server:</p>"
        f"<ul>{''.join(f'<li><code>{p}</code></li>' for p in candidates)}</ul>"
        f"<p>Fixes: place the processed CSV at one of these paths, or update DATA_CSV in app.py.</p>"
    )
    return make_response(msg, 404)


# health check
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    # In dev, the reloader imports the module more than once. load_model() guards prevent double loads.
    load_model()
    # set debug=False for production
    app.run(host="0.0.0.0", port=5000, debug=True)
