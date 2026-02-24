import os
import random
import json
import time
import warnings
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -------------------------
# 설정
# -------------------------
CSV_PATH   = "./2026/data/df_final_2026_with_gdelt_and_news.csv"
DATE_COL   = "date"
TARGET_COL = "USD_KRW 종가"

SHIFT_LIST     = [1]
LOOKBACK_LIST  = [5, 10, 20, 30, 60, 90]

MODEL_LIST = ["RandomWalk", "SARIMA", "RandomForest", "XGBoost"]

CASES = [
    "Macro Only",
    "Macro + Event",
    "Macro + Sentiment (Direct)",
    "Macro + Sentiment (Indirect)",
    "ALL",
]

TEST_HOLDOUT_RATIO = 0.20

# TS-CV (train_dev 내부)
CV_N_FOLDS   = 5
CV_TEST_SIZE = 0.12
CV_MIN_TRAIN = 0.55

# Random Search budget
N_TRIALS     = 20
SEED_DEFAULT = 42

OUT_DIR = "./output_ml_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

RUN_TAG = "ML_BASELINE"

TRIALS_PARTIAL_PATH   = os.path.join(OUT_DIR, f"tuning_trials_{RUN_TAG}.csv")
BESTCV_PARTIAL_PATH   = os.path.join(OUT_DIR, f"best_cv_{RUN_TAG}.csv")
HOLDOUT_PARTIAL_PATH  = os.path.join(OUT_DIR, f"holdout_results_{RUN_TAG}.csv")
HOPRED_PARTIAL_PATH   = os.path.join(OUT_DIR, f"holdout_predictions_{RUN_TAG}.csv")
STATUS_PATH           = os.path.join(OUT_DIR, f"run_status_{RUN_TAG}.json")

# 튜닝 조기 종료
TUNE_NO_IMPROVE_PATIENCE = 6
TUNE_MIN_IMPROVEMENT     = 0.005
MIN_TE_SEQS = 5


# -------------------------
# 공통 유틸
# -------------------------
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def eval_metrics(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = mean_absolute_error(y_true, y_pred)

    denom = np.maximum(np.abs(y_true), eps)
    ape   = np.abs(y_true - y_pred) / denom
    spe   = ((y_true - y_pred) / denom) ** 2

    mape   = float(np.mean(ape) * 100.0)
    mspe   = float(np.mean(spe) * 100.0)
    medae  = float(np.median(np.abs(y_true - y_pred)))
    medape = float(np.median(ape) * 100.0)

    return dict(RMSE=rmse, MSPE=mspe, MAE=mae, MAPE=mape, MedAE=medae, MedAPE=medape)


def make_flat_sequences(X, y, dates, lookback, shift=1):
    """시퀀스를 flat feature로 변환 (RF, XGBoost용)."""
    X_flat, y_target, y_prev, y_true_list, target_dates = [], [], [], [], []
    n = len(y)
    max_i = n - lookback - shift
    for i in range(max_i):
        idx_last   = i + lookback - 1
        target_idx = idx_last + shift
        prev_idx   = target_idx - 1

        # lookback 구간의 feature를 flatten
        X_flat.append(X[i:i+lookback].ravel())
        y_target.append(y[target_idx] - y[prev_idx])   # residual (delta)
        y_prev.append(y[prev_idx])
        y_true_list.append(y[target_idx])
        target_dates.append(dates[target_idx])

    return (np.array(X_flat, np.float32),
            np.array(y_target, np.float32),
            np.array(y_prev, np.float32),
            np.array(y_true_list, np.float32),
            np.array(target_dates))


def final_holdout_split(df, ratio=0.2):
    n = len(df)
    n_test = max(1, int(np.floor(n * ratio)))
    split = n - n_test
    tr_dev = df.iloc[:split].reset_index(drop=True)
    te     = df.iloc[split:].reset_index(drop=True)
    return tr_dev, te


def walk_forward_splits(n, n_folds=5, test_size=0.12, min_train=0.55):
    test_len = max(2, int(n * test_size))
    start_test = int(n * min_train)

    splits = []
    for k in range(n_folds):
        te_start = start_test + k * test_len
        te_end   = min(n, te_start + test_len)
        if te_end - te_start < 5:
            break
        tr_end = te_start
        if tr_end < 30:
            continue
        splits.append((slice(0, tr_end), slice(te_start, te_end)))
        if te_end >= n:
            break
    return splits


# -------------------------
# 컬럼 세트 자동 추론
# -------------------------
def infer_gdelt_cols(df):
    return [c for c in df.columns if c.startswith("gkg_") or c.startswith("events_")]


def infer_news_cols_all(df):
    news_like = []
    for c in df.columns:
        if c.startswith("news_") or c.startswith("sent_"):
            news_like.append(c)
    for c in [
        "abs_sent_mean", "pos_ratio", "neg_ratio", "neu_ratio",
        "direct_ratio", "indirect_ratio",
        "sent_net_ratio", "sent_net_count", "sent_std"
    ]:
        if c in df.columns:
            news_like.append(c)
    return sorted(list(set(news_like)))


def infer_news_cols_direct_only(df):
    all_news = infer_news_cols_all(df)
    direct = []
    for c in all_news:
        cl = c.lower()
        if ("direct" in cl) and ("indirect" not in cl):
            direct.append(c)
    return sorted(list(set(direct)))


def infer_news_cols_indirect_only(df):
    all_news = infer_news_cols_all(df)
    indirect = []
    for c in all_news:
        cl = c.lower()
        if "indirect" in cl:
            indirect.append(c)
    return sorted(list(set(indirect)))


def parse_case(case_name: str):
    """Returns (include_event: bool, sent_mode: str).
    sent_mode: NONE / DIRECT / INDIRECT / BOTH
    """
    if case_name == "Macro Only":
        return False, "NONE"
    if case_name == "Macro + Event":
        return True, "NONE"
    if case_name == "Macro + Sentiment (Direct)":
        return False, "DIRECT"
    if case_name == "Macro + Sentiment (Indirect)":
        return False, "INDIRECT"
    if case_name == "ALL":
        return True, "BOTH"
    raise ValueError(f"Unknown case: {case_name}")


def build_case_df(df_raw: pd.DataFrame, case_name: str):
    include_event, sent_mode = parse_case(case_name)

    gdelt_cols = infer_gdelt_cols(df_raw)

    if sent_mode == "NONE":
        news_cols = []
    elif sent_mode == "DIRECT":
        news_cols = infer_news_cols_direct_only(df_raw)
    elif sent_mode == "INDIRECT":
        news_cols = infer_news_cols_indirect_only(df_raw)
    elif sent_mode == "BOTH":
        news_cols = infer_news_cols_all(df_raw)
    else:
        raise ValueError(f"Unknown sent_mode: {sent_mode}")

    numeric_cols = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c != TARGET_COL]
    macro_cols = [c for c in numeric_cols if (c not in gdelt_cols and c not in news_cols)]

    keep = [DATE_COL, TARGET_COL] + macro_cols
    if include_event:
        keep += gdelt_cols
    if sent_mode != "NONE":
        keep += news_cols

    keep = list(dict.fromkeys(keep))

    df_case = df_raw[keep].copy()
    df_case = df_case.sort_values(DATE_COL).dropna(subset=[DATE_COL, TARGET_COL]).reset_index(drop=True)
    df_case = df_case.replace([np.inf, -np.inf], np.nan).dropna()
    return df_case


# -------------------------
# 저장 유틸
# -------------------------
def _safe_append_csv(df: pd.DataFrame, path: str):
    if df is None or len(df) == 0:
        return
    if not os.path.exists(path):
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, mode="a", header=False, encoding="utf-8-sig")


def load_partial(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(path, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        except Exception:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")


def key_str(case, model, lookback, shift):
    return f"{case}||{model}||lb={lookback}||shift={shift}"


def holdout_key(case, model, lookback, shift):
    return f"{key_str(case, model, lookback, shift)}||HOLDOUT"


def save_status(status: Dict[str, Any]):
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def load_status() -> Dict[str, Any]:
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"started_at": None, "updated_at": None}


# -------------------------
# 하이퍼파라미터 탐색 공간
# -------------------------
def sample_hp(model_name: str, rng: np.random.RandomState):
    u = model_name.lower()

    if u == "randomforest":
        return {
            "n_estimators": int(rng.choice([100, 200, 300, 500])),
            "max_depth": int(rng.choice([5, 10, 15, 20, 30])),
            "min_samples_split": int(rng.choice([2, 5, 10])),
            "min_samples_leaf": int(rng.choice([1, 2, 4])),
            "max_features": str(rng.choice(["sqrt", "log2", "0.5", "0.8"])),
        }
    elif u == "xgboost":
        return {
            "n_estimators": int(rng.choice([100, 200, 300, 500])),
            "max_depth": int(rng.choice([3, 5, 7, 9])),
            "learning_rate": float(rng.choice([0.01, 0.03, 0.05, 0.1])),
            "subsample": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
            "colsample_bytree": float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
            "reg_alpha": float(rng.choice([0.0, 0.01, 0.1, 1.0])),
            "reg_lambda": float(rng.choice([0.1, 1.0, 5.0, 10.0])),
            "min_child_weight": int(rng.choice([1, 3, 5, 7])),
        }
    elif u == "sarima":
        return {
            "order_p": int(rng.choice([0, 1, 2])),
            "order_d": int(rng.choice([0, 1])),
            "order_q": int(rng.choice([0, 1, 2])),
            "seasonal_P": int(rng.choice([0, 1])),
            "seasonal_D": int(rng.choice([0, 1])),
            "seasonal_Q": int(rng.choice([0, 1])),
            "seasonal_s": 5,  # 주 5영업일 고정 (속도)
        }
    else:
        return {}


# -------------------------
# 모델별 학습/예측
# -------------------------
def fit_predict_randomwalk(y_train_raw, y_test_raw, dates_test, shift=1):
    """Random Walk: 예측 = 직전 값 (y_{t} = y_{t-1})."""
    # shift=1이면 t-1 값을 그대로 예측으로 사용
    y_pred = y_test_raw[:-shift] if shift > 0 else y_test_raw.copy()
    y_true = y_test_raw[shift:]
    d      = dates_test[shift:]

    if len(y_true) < MIN_TE_SEQS:
        raise ValueError("test sequence too short for RandomWalk")

    metrics = eval_metrics(y_true, y_pred[:len(y_true)])
    pred_df = pd.DataFrame({"date": d, "y_true": y_true, "y_pred": y_pred[:len(y_true)]})
    return metrics, pred_df


def fit_predict_sarima(y_train_raw, y_test_raw, dates_test, hp, shift=1):
    """SARIMA: exog 없이 타겟 시계열만 사용."""
    order = (hp["order_p"], hp["order_d"], hp["order_q"])
    seasonal_order = (hp["seasonal_P"], hp["seasonal_D"], hp["seasonal_Q"], hp["seasonal_s"])

    try:
        model = SARIMAX(
            y_train_raw,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=100, method="lbfgs")
    except Exception:
        # fallback: 단순 (1,1,0) 모델
        model = SARIMAX(
            y_train_raw,
            order=(1, 1, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=50)

    n_test = len(y_test_raw)
    forecast = result.forecast(steps=n_test)
    y_pred = np.asarray(forecast, dtype=np.float64)

    if len(y_test_raw) < MIN_TE_SEQS:
        raise ValueError("test sequence too short for SARIMA")

    metrics = eval_metrics(y_test_raw, y_pred)
    pred_df = pd.DataFrame({"date": dates_test, "y_true": y_test_raw, "y_pred": y_pred})
    return metrics, pred_df


def fit_predict_ml(df_case, model_name, hp, seed, lookback, shift, tr_sl, te_sl):
    """Random Forest / XGBoost: flat-sequence + residual 방식."""
    set_seed(seed)

    feats = [c for c in df_case.select_dtypes(include=[np.number]).columns if c != TARGET_COL]
    X_all = df_case[feats].values.astype(np.float32)
    y_all = df_case[TARGET_COL].values.astype(np.float32)
    dates = df_case[DATE_COL].values

    X_tr_raw, y_tr_raw, d_tr = X_all[tr_sl], y_all[tr_sl], dates[tr_sl]
    X_te_raw, y_te_raw, d_te = X_all[te_sl], y_all[te_sl], dates[te_sl]

    x_scaler = RobustScaler()
    X_tr = x_scaler.fit_transform(X_tr_raw)
    X_te = x_scaler.transform(X_te_raw)

    X_tr_flat, y_d_tr, _, _, _ = make_flat_sequences(X_tr, y_tr_raw, d_tr, lookback, shift)
    X_te_flat, _, p_last_te, y_true_te, target_dates = make_flat_sequences(X_te, y_te_raw, d_te, lookback, shift)

    if len(X_tr_flat) < 25:
        raise ValueError("train sequence too short")
    if len(X_te_flat) < MIN_TE_SEQS:
        raise ValueError("test sequence too short")

    u = model_name.lower()
    if u == "randomforest":
        max_feat = hp.get("max_features", "sqrt")
        if max_feat in ("sqrt", "log2"):
            mf = max_feat
        else:
            mf = float(max_feat)

        model = RandomForestRegressor(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            min_samples_split=hp["min_samples_split"],
            min_samples_leaf=hp["min_samples_leaf"],
            max_features=mf,
            random_state=seed,
            n_jobs=-1,
        )
    elif u == "xgboost":
        model = XGBRegressor(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            subsample=hp["subsample"],
            colsample_bytree=hp["colsample_bytree"],
            reg_alpha=hp["reg_alpha"],
            reg_lambda=hp["reg_lambda"],
            min_child_weight=hp["min_child_weight"],
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown ML model: {model_name}")

    model.fit(X_tr_flat, y_d_tr)
    d_pred = model.predict(X_te_flat)
    yhat = p_last_te + d_pred

    metrics = eval_metrics(y_true_te, yhat)
    pred_df = pd.DataFrame({"date": target_dates, "y_true": y_true_te, "y_pred": yhat})
    return metrics, pred_df


# -------------------------
# Fold 학습/예측 (CV) - 모델 분기
# -------------------------
def fit_predict_fold(df_case, model_name, hp, seed, lookback, shift, tr_sl, te_sl):
    u = model_name.lower()

    if u == "randomwalk":
        y_all = df_case[TARGET_COL].values.astype(np.float32)
        dates = df_case[DATE_COL].values
        y_te = y_all[te_sl]
        d_te = dates[te_sl]
        return fit_predict_randomwalk(y_all[tr_sl], y_te, d_te, shift)

    elif u == "sarima":
        y_all = df_case[TARGET_COL].values.astype(np.float32)
        dates = df_case[DATE_COL].values
        y_tr = y_all[tr_sl]
        y_te = y_all[te_sl]
        d_te = dates[te_sl]
        return fit_predict_sarima(y_tr, y_te, d_te, hp, shift)

    else:
        return fit_predict_ml(df_case, model_name, hp, seed, lookback, shift, tr_sl, te_sl)


def cv_score(df_train_dev, model_name, hp, seed, lookback, shift):
    splits = walk_forward_splits(len(df_train_dev), n_folds=CV_N_FOLDS,
                                  test_size=CV_TEST_SIZE, min_train=CV_MIN_TRAIN)
    if len(splits) == 0:
        raise ValueError("No CV splits produced.")

    fold_metrics = []
    fold_preds = []
    skipped = 0

    for f, (tr_sl, te_sl) in enumerate(splits):
        try:
            m, p = fit_predict_fold(df_train_dev, model_name, hp, seed, lookback, shift, tr_sl, te_sl)
        except ValueError:
            skipped += 1
            continue

        m["fold"] = f
        fold_metrics.append(m)
        p["fold"] = f
        fold_preds.append(p)

    if len(fold_metrics) == 0:
        raise ValueError("All folds skipped.")

    met_df  = pd.DataFrame(fold_metrics)
    pred_df = pd.concat(fold_preds, ignore_index=True) if fold_preds else pd.DataFrame()

    avg = met_df.mean(numeric_only=True).to_dict()
    avg["skipped_folds"] = skipped
    avg["used_folds"] = len(fold_metrics)
    return avg, met_df, pred_df


# -------------------------
# Holdout 평가
# -------------------------
def eval_holdout(df_train_dev, df_test, model_name, hp, seed, lookback, shift):
    u = model_name.lower()

    if u == "randomwalk":
        y_tr = df_train_dev[TARGET_COL].values.astype(np.float32)
        y_te = df_test[TARGET_COL].values.astype(np.float32)
        d_te = df_test[DATE_COL].values
        return fit_predict_randomwalk(y_tr, y_te, d_te, shift)

    elif u == "sarima":
        y_tr = df_train_dev[TARGET_COL].values.astype(np.float32)
        y_te = df_test[TARGET_COL].values.astype(np.float32)
        d_te = df_test[DATE_COL].values
        return fit_predict_sarima(y_tr, y_te, d_te, hp, shift)

    else:
        # RF / XGBoost: 전체 train_dev로 학습, holdout으로 평가
        set_seed(seed)
        feats = [c for c in df_train_dev.select_dtypes(include=[np.number]).columns if c != TARGET_COL]

        X_tr_raw = df_train_dev[feats].values.astype(np.float32)
        y_tr_raw = df_train_dev[TARGET_COL].values.astype(np.float32)
        d_tr     = df_train_dev[DATE_COL].values

        X_te_raw = df_test[feats].values.astype(np.float32)
        y_te_raw = df_test[TARGET_COL].values.astype(np.float32)
        d_te     = df_test[DATE_COL].values

        x_scaler = RobustScaler()
        X_tr = x_scaler.fit_transform(X_tr_raw)
        X_te = x_scaler.transform(X_te_raw)

        X_tr_flat, y_d_tr, _, _, _ = make_flat_sequences(X_tr, y_tr_raw, d_tr, lookback, shift)
        X_te_flat, _, p_last_te, y_true_te, target_dates = make_flat_sequences(X_te, y_te_raw, d_te, lookback, shift)

        if len(X_tr_flat) < 30:
            raise ValueError("train_dev sequence too short")
        if len(X_te_flat) < MIN_TE_SEQS:
            raise ValueError("holdout test sequence too short")

        if u == "randomforest":
            max_feat = hp.get("max_features", "sqrt")
            mf = max_feat if max_feat in ("sqrt", "log2") else float(max_feat)
            model = RandomForestRegressor(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                min_samples_split=hp["min_samples_split"],
                min_samples_leaf=hp["min_samples_leaf"],
                max_features=mf,
                random_state=seed,
                n_jobs=-1,
            )
        elif u == "xgboost":
            model = XGBRegressor(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                learning_rate=hp["learning_rate"],
                subsample=hp["subsample"],
                colsample_bytree=hp["colsample_bytree"],
                reg_alpha=hp["reg_alpha"],
                reg_lambda=hp["reg_lambda"],
                min_child_weight=hp["min_child_weight"],
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_tr_flat, y_d_tr)
        d_pred = model.predict(X_te_flat)
        yhat = p_last_te + d_pred

        metrics = eval_metrics(y_true_te, yhat)
        pred_df = pd.DataFrame({"date": target_dates, "y_true": y_true_te, "y_pred": yhat})
        return metrics, pred_df


# -------------------------
# 튜닝 (RF, XGBoost만 - RandomWalk/SARIMA는 별도 처리)
# -------------------------
def tune_and_select(df_train_dev, case_name, model_name, lookback, shift):
    u = model_name.lower()
    this_key = key_str(case_name, model_name, lookback, shift)
    base_seed = 2026
    rng = np.random.RandomState(base_seed + (abs(hash((case_name, model_name, lookback, shift))) % 100000))

    # RandomWalk: 하이퍼파라미터 없음
    if u == "randomwalk":
        try:
            avg, _, pred_df = cv_score(df_train_dev, model_name, {}, SEED_DEFAULT, lookback, shift)
        except ValueError as e:
            raise RuntimeError(f"RandomWalk CV failed: {e}")

        summary = {
            "key": this_key, "case": case_name, "model": model_name,
            "lookback": lookback, "shift": shift,
            "best_cv_RMSE": avg.get("RMSE", np.nan),
            "best_cv_MSPE": avg.get("MSPE", np.nan),
            "best_cv_MAE": avg.get("MAE", np.nan),
            "best_cv_MAPE": avg.get("MAPE", np.nan),
            "best_cv_MedAE": avg.get("MedAE", np.nan),
            "best_cv_MedAPE": avg.get("MedAPE", np.nan),
            "best_hp_json": "{}",
        }
        return pd.DataFrame([summary]), pred_df

    # SARIMA: 조합 시도 (속도 위해 적게)
    n_trials = N_TRIALS if u in ("randomforest", "xgboost") else 5

    best = None
    best_score = np.inf
    no_improve = 0

    for t in range(n_trials):
        hp = sample_hp(model_name, rng)
        t0 = time.time()

        try:
            avg, _, pred_df = cv_score(df_train_dev, model_name, hp, SEED_DEFAULT, lookback, shift)
            score = avg["RMSE"]

            row = {
                "key": this_key, "trial": t, "status": "OK",
                "case": case_name, "model": model_name,
                "lookback": lookback, "shift": shift,
                "score_RMSE": score,
                "MAE": avg.get("MAE", np.nan),
                "MAPE": avg.get("MAPE", np.nan),
                "MSPE": avg.get("MSPE", np.nan),
                "MedAE": avg.get("MedAE", np.nan),
                "MedAPE": avg.get("MedAPE", np.nan),
                "hp_json": json.dumps(hp, ensure_ascii=False),
                "seconds": float(time.time() - t0),
            }
            _safe_append_csv(pd.DataFrame([row]), TRIALS_PARTIAL_PATH)

            improved = (best is None) or (score < (best_score - TUNE_MIN_IMPROVEMENT))
            if improved:
                best_score = score
                best = (score, hp, avg, pred_df)
                no_improve = 0
            else:
                no_improve += 1

            print(f"  [TUNE] {model_name} lb={lookback} trial={t} RMSE={score:.4f} (best={best_score:.4f})")

            if no_improve >= TUNE_NO_IMPROVE_PATIENCE:
                break

        except Exception as e:
            row = {
                "key": this_key, "trial": t, "status": "FAIL",
                "case": case_name, "model": model_name,
                "lookback": lookback, "shift": shift,
                "error": str(e)[:300],
                "hp_json": json.dumps(hp, ensure_ascii=False),
                "seconds": float(time.time() - t0),
            }
            _safe_append_csv(pd.DataFrame([row]), TRIALS_PARTIAL_PATH)
            print(f"  [TUNE-FAIL] {model_name} lb={lookback} trial={t} -> {str(e)[:100]}")
            continue

    if best is None:
        raise RuntimeError(f"No successful trials for {this_key}")

    best_score, best_hp, best_avg, best_pred_df = best

    summary = {
        "key": this_key, "case": case_name, "model": model_name,
        "lookback": lookback, "shift": shift,
        "best_cv_RMSE": best_avg.get("RMSE", np.nan),
        "best_cv_MSPE": best_avg.get("MSPE", np.nan),
        "best_cv_MAE": best_avg.get("MAE", np.nan),
        "best_cv_MAPE": best_avg.get("MAPE", np.nan),
        "best_cv_MedAE": best_avg.get("MedAE", np.nan),
        "best_cv_MedAPE": best_avg.get("MedAPE", np.nan),
        "best_hp_json": json.dumps(best_hp, ensure_ascii=False),
    }
    return pd.DataFrame([summary]), best_pred_df


# -------------------------
# 전체 실행
# -------------------------
def run_protocol():
    assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"

    df_hold_partial = load_partial(HOLDOUT_PARTIAL_PATH)
    done_hold_set = set(df_hold_partial["holdout_key"].astype(str).tolist()) if len(df_hold_partial) else set()
    print(f"[RESUME] done_holdouts={len(done_hold_set)}")

    status = load_status()
    if not status.get("started_at"):
        status["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_status(status)

    df_raw = pd.read_csv(CSV_PATH, encoding="utf-8-sig").replace([np.inf, -np.inf], np.nan)
    df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL], errors="coerce")
    df_raw = df_raw.sort_values(DATE_COL).dropna(subset=[DATE_COL, TARGET_COL]).reset_index(drop=True)

    print(f"df_raw: {df_raw.shape} | period: {df_raw[DATE_COL].min()} ~ {df_raw[DATE_COL].max()}")
    print("CASES =", CASES)
    print("MODELS =", MODEL_LIST)
    print("LOOKBACKS =", LOOKBACK_LIST)

    for case in CASES:
        df_case = build_case_df(df_raw, case)
        df_train_dev, df_holdout = final_holdout_split(df_case, ratio=TEST_HOLDOUT_RATIO)

        print(f"\n{'='*60}", flush=True)
        print(f"CASE: {case}", flush=True)
        print(f"train_dev: {df_train_dev.shape} | holdout: {df_holdout.shape} | "
              f"holdout period: {df_holdout[DATE_COL].min()} ~ {df_holdout[DATE_COL].max()}", flush=True)
        print(f"{'='*60}", flush=True)

        for shift in SHIFT_LIST:
            # ── RandomWalk & SARIMA: lookback 무관 → 케이스당 1회만 실행, 결과 복사 ──
            for model_name in ["RandomWalk", "SARIMA"]:
                # 이미 모든 lookback에 대해 done인지 확인
                all_done = all(holdout_key(case, model_name, lb, shift) in done_hold_set
                               for lb in LOOKBACK_LIST)
                if all_done:
                    print(f"[SKIP] {model_name} all lookbacks done for {case}", flush=True)
                    continue

                print(f"\n--- {model_name} | shift={shift} (lookback-independent) ---", flush=True)
                try:
                    # lookback=5 는 dummy (SARIMA/RW는 lookback 사용 안 함)
                    dummy_lb = LOOKBACK_LIST[0]
                    bestcv_df, bestcv_pred_df = tune_and_select(
                        df_train_dev, case, model_name, dummy_lb, shift
                    )
                    best_hp = json.loads(bestcv_df.loc[0, "best_hp_json"])

                    ho_met, ho_pred = eval_holdout(
                        df_train_dev, df_holdout, model_name, best_hp,
                        SEED_DEFAULT, dummy_lb, shift
                    )

                    # 모든 lookback에 동일 결과 저장
                    for lb in LOOKBACK_LIST:
                        hk = holdout_key(case, model_name, lb, shift)
                        if hk in done_hold_set:
                            continue

                        cv_row = bestcv_df.iloc[0].to_dict()
                        cv_row["key"] = key_str(case, model_name, lb, shift)
                        cv_row["lookback"] = lb
                        df_bestcv = load_partial(BESTCV_PARTIAL_PATH)
                        df_bestcv = pd.concat([df_bestcv, pd.DataFrame([cv_row])], ignore_index=True)
                        df_bestcv = df_bestcv.drop_duplicates(subset=["key"], keep="last")
                        df_bestcv.to_csv(BESTCV_PARTIAL_PATH, index=False, encoding="utf-8-sig")

                        row = {
                            "holdout_key": hk,
                            "key": key_str(case, model_name, lb, shift),
                            "case": case, "model": model_name,
                            "lookback": lb, "shift": shift,
                            "best_hp_json": json.dumps(best_hp, ensure_ascii=False),
                            "best_cv_RMSE":  float(bestcv_df.loc[0, "best_cv_RMSE"]),
                            "best_cv_MAE":   float(bestcv_df.loc[0, "best_cv_MAE"]),
                            "best_cv_MAPE":  float(bestcv_df.loc[0, "best_cv_MAPE"]),
                            "best_cv_MSPE":  float(bestcv_df.loc[0, "best_cv_MSPE"]),
                            "best_cv_MedAE": float(bestcv_df.loc[0, "best_cv_MedAE"]),
                            "best_cv_MedAPE": float(bestcv_df.loc[0, "best_cv_MedAPE"]),
                            "holdout_RMSE":  ho_met.get("RMSE", np.nan),
                            "holdout_MSPE":  ho_met.get("MSPE", np.nan),
                            "holdout_MAE":   ho_met.get("MAE", np.nan),
                            "holdout_MAPE":  ho_met.get("MAPE", np.nan),
                            "holdout_MedAE": ho_met.get("MedAE", np.nan),
                            "holdout_MedAPE": ho_met.get("MedAPE", np.nan),
                        }
                        _safe_append_csv(pd.DataFrame([row]), HOLDOUT_PARTIAL_PATH)
                        done_hold_set.add(hk)

                    print(f"  [HOLDOUT] {model_name} RMSE={ho_met['RMSE']:.4f} | MAE={ho_met['MAE']:.4f} | "
                          f"MAPE={ho_met['MAPE']:.2f}% | MSPE={ho_met['MSPE']:.4f} | "
                          f"MedAE={ho_met['MedAE']:.4f} | MedAPE={ho_met['MedAPE']:.2f}%", flush=True)

                    status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    status["last_done"] = f"{case}||{model_name}||ALL_LB"
                    save_status(status)

                except Exception as e:
                    print(f"  [ERROR] {model_name} -> {str(e)[:300]}", flush=True)
                    continue

            # ── RandomForest & XGBoost: lookback별로 실행 ──
            for lb in LOOKBACK_LIST:
                for model_name in ["RandomForest", "XGBoost"]:
                    hk = holdout_key(case, model_name, lb, shift)
                    if hk in done_hold_set:
                        print(f"[SKIP] already done -> {hk}", flush=True)
                        continue

                    print(f"\n--- {model_name} | lookback={lb} | shift={shift} ---", flush=True)

                    try:
                        bestcv_df, bestcv_pred_df = tune_and_select(
                            df_train_dev, case, model_name, lb, shift
                        )

                        df_bestcv = load_partial(BESTCV_PARTIAL_PATH)
                        df_bestcv = pd.concat([df_bestcv, bestcv_df], ignore_index=True)
                        df_bestcv = df_bestcv.drop_duplicates(subset=["key"], keep="last")
                        df_bestcv.to_csv(BESTCV_PARTIAL_PATH, index=False, encoding="utf-8-sig")

                        best_hp = json.loads(bestcv_df.loc[0, "best_hp_json"])

                        ho_met, ho_pred = eval_holdout(
                            df_train_dev, df_holdout, model_name, best_hp,
                            SEED_DEFAULT, lb, shift
                        )

                        row = {
                            "holdout_key": hk,
                            "key": key_str(case, model_name, lb, shift),
                            "case": case, "model": model_name,
                            "lookback": lb, "shift": shift,
                            "best_hp_json": json.dumps(best_hp, ensure_ascii=False),
                            "best_cv_RMSE":  float(bestcv_df.loc[0, "best_cv_RMSE"]),
                            "best_cv_MAE":   float(bestcv_df.loc[0, "best_cv_MAE"]),
                            "best_cv_MAPE":  float(bestcv_df.loc[0, "best_cv_MAPE"]),
                            "best_cv_MSPE":  float(bestcv_df.loc[0, "best_cv_MSPE"]),
                            "best_cv_MedAE": float(bestcv_df.loc[0, "best_cv_MedAE"]),
                            "best_cv_MedAPE": float(bestcv_df.loc[0, "best_cv_MedAPE"]),
                            "holdout_RMSE":  ho_met.get("RMSE", np.nan),
                            "holdout_MSPE":  ho_met.get("MSPE", np.nan),
                            "holdout_MAE":   ho_met.get("MAE", np.nan),
                            "holdout_MAPE":  ho_met.get("MAPE", np.nan),
                            "holdout_MedAE": ho_met.get("MedAE", np.nan),
                            "holdout_MedAPE": ho_met.get("MedAPE", np.nan),
                        }

                        _safe_append_csv(pd.DataFrame([row]), HOLDOUT_PARTIAL_PATH)
                        done_hold_set.add(hk)

                        ho_pred = ho_pred.copy()
                        ho_pred["case"] = case
                        ho_pred["model"] = model_name
                        ho_pred["lookback"] = lb
                        ho_pred["shift"] = shift
                        _safe_append_csv(ho_pred, HOPRED_PARTIAL_PATH)

                        print(f"  [HOLDOUT] RMSE={ho_met['RMSE']:.4f} | MAE={ho_met['MAE']:.4f} | "
                              f"MAPE={ho_met['MAPE']:.2f}% | MSPE={ho_met['MSPE']:.4f} | "
                              f"MedAE={ho_met['MedAE']:.4f} | MedAPE={ho_met['MedAPE']:.2f}%", flush=True)

                        status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        status["last_done"] = hk
                        save_status(status)

                    except Exception as e:
                        print(f"  [ERROR] {model_name} lb={lb} -> {str(e)[:300]}", flush=True)
                        continue

    # -------------------------
    # 최종 결과 요약 테이블
    # -------------------------
    print("\n" + "=" * 80)
    print("FINAL HOLDOUT RESULTS SUMMARY")
    print("=" * 80)

    df_final = load_partial(HOLDOUT_PARTIAL_PATH)
    if len(df_final):
        metric_cols = ["holdout_RMSE", "holdout_MSPE", "holdout_MAE",
                       "holdout_MAPE", "holdout_MedAE", "holdout_MedAPE"]
        display_cols = ["case", "model", "lookback", "shift"] + metric_cols
        avail = [c for c in display_cols if c in df_final.columns]

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")

        print(df_final[avail].to_string(index=False))

        # 모델별 평균
        print("\n--- Average by Model ---")
        avail_metrics = [c for c in metric_cols if c in df_final.columns]
        print(df_final.groupby("model")[avail_metrics].mean().to_string())

        # 케이스별 평균
        print("\n--- Average by Case ---")
        print(df_final.groupby("case")[avail_metrics].mean().to_string())

        # 최종 CSV 저장
        final_path = os.path.join(OUT_DIR, f"final_summary_{RUN_TAG}.csv")
        df_final.to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"\nFinal summary saved to: {final_path}")

    print("\n================= DONE =================")
    print("All results saved to:", OUT_DIR)


if __name__ == "__main__":
    run_protocol()
