#!/usr/bin/env python3
"""
Herramientas de auditoria de generalizacion para modelos de trading.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit


EPS = 1e-12


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.values
    return np.asarray(x)


def _higher_is_better(metric_name: str) -> bool:
    lower_is_better_terms = {"loss", "drawdown", "mse", "mae", "rmse"}
    name = metric_name.lower()
    return not any(term in name for term in lower_is_better_terms)


def purged_kfold_cv(
    X,
    y,
    n_splits: int = 5,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.005,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Purged K-Fold para series temporales.
    - Test folds contiguos
    - Purge alrededor del fold de test
    - Embargo posterior al fold de test
    """
    X_np = _as_numpy(X)
    y_np = _as_numpy(y)
    if len(X_np) != len(y_np):
        raise ValueError("X e y deben tener la misma longitud")
    if n_splits < 2:
        raise ValueError("n_splits debe ser >= 2")

    n = len(X_np)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    fold_starts = np.cumsum(np.r_[0, fold_sizes[:-1]])

    for fold_idx, (start, size) in enumerate(zip(fold_starts, fold_sizes)):
        test_start = int(start)
        test_end = int(start + size)  # exclusivo
        test_idx = np.arange(test_start, test_end)

        purge = max(1, int(size * purge_pct))
        embargo = max(1, int(size * embargo_pct))

        left_keep_end = max(0, test_start - purge)
        right_keep_start = min(n, test_end + purge + embargo)

        train_left = np.arange(0, left_keep_end)
        train_right = np.arange(right_keep_start, n)
        train_idx = np.concatenate([train_left, train_right])

        if train_idx.size == 0:
            raise ValueError(
                f"Fold {fold_idx}: train vacio. Reduce purge/embargo o n_splits."
            )

        yield train_idx, test_idx


class TimeSeriesCV:
    """
    Cross-validacion temporal avanzada con gap.
    Soporta expanding window o sliding window.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 60,
        gap: int = 0,
        expanding: bool = True,
        min_train_size: Optional[int] = None,
    ):
        if n_splits < 2:
            raise ValueError("n_splits debe ser >= 2")
        if test_size <= 0:
            raise ValueError("test_size debe ser > 0")
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = max(0, int(gap))
        self.expanding = expanding
        self.min_train_size = min_train_size

    def split(self, X) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        X_np = _as_numpy(X)
        n = len(X_np)
        total_test = self.n_splits * self.test_size
        if total_test + self.gap >= n:
            raise ValueError("Insuficientes muestras para los parametros dados")

        start_test = n - total_test
        if self.min_train_size is None:
            min_train = max(30, start_test)
        else:
            min_train = int(self.min_train_size)
        if min_train <= 0:
            raise ValueError("min_train_size invalido")

        for split_i in range(self.n_splits):
            test_start = start_test + split_i * self.test_size
            test_end = test_start + self.test_size
            train_end = max(0, test_start - self.gap)

            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - min_train)

            if train_end - train_start < min_train:
                train_start = max(0, train_end - min_train)

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            if train_idx.size == 0 or test_idx.size == 0:
                continue
            yield train_idx, test_idx


def detect_overfitting(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    threshold: float = 0.2,
) -> Dict[str, Dict[str, float | str | bool]]:
    """
    Detector simple de overfitting via gap relativo train-vs-val.
    """
    report: Dict[str, Dict[str, float | str | bool]] = {}
    for metric, train_value in train_metrics.items():
        if metric not in val_metrics:
            continue
        val_value = val_metrics[metric]
        hib = _higher_is_better(metric)

        if hib:
            gap = (train_value - val_value) / (abs(train_value) + EPS)
        else:
            gap = (val_value - train_value) / (abs(train_value) + EPS)

        is_overfit = bool(gap > threshold)
        status = "OVERFIT" if is_overfit else "OK"
        severity = "CRITICO" if gap > max(0.5, threshold * 2.0) else ("ALTO" if gap > threshold else "BAJO")

        report[metric] = {
            "train": float(train_value),
            "val": float(val_value),
            "relative_gap": float(gap),
            "threshold": float(threshold),
            "overfit": is_overfit,
            "status": status,
            "severity": severity,
        }
    return report


def plot_learning_curves(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    history: Optional[Dict[str, Sequence[float]]] = None,
    task: str = "regression",
    epochs: int = 100,
    patience: int = 10,
) -> Dict[str, object]:
    """
    Plots:
    1) train vs validation loss por epoch (si hay history o partial_fit)
    2) performance vs dataset size
    3) out-of-sample predictions vs real (scatter)
    """
    X_train_np = _as_numpy(X_train)
    y_train_np = _as_numpy(y_train)
    X_val_np = _as_numpy(X_val)
    y_val_np = _as_numpy(y_val)

    train_loss: List[float] = []
    val_loss: List[float] = []
    epochs_axis: List[int] = []

    # Caso 1: se pasa historial ya entrenado (DL frameworks)
    if history and "train_loss" in history and "val_loss" in history:
        train_loss = list(map(float, history["train_loss"]))
        val_loss = list(map(float, history["val_loss"]))
        epochs_axis = list(range(1, len(train_loss) + 1))
    # Caso 2: modelo incremental con partial_fit
    elif hasattr(model, "partial_fit"):
        classes = np.unique(y_train_np) if task == "classification" else None
        for ep in range(1, epochs + 1):
            if task == "classification":
                model.partial_fit(X_train_np, y_train_np, classes=classes)
            else:
                model.partial_fit(X_train_np, y_train_np)

            pred_tr = model.predict(X_train_np)
            pred_val = model.predict(X_val_np)
            if task == "classification":
                tr = 1.0 - accuracy_score(y_train_np, pred_tr)
                vl = 1.0 - accuracy_score(y_val_np, pred_val)
            else:
                tr = mean_squared_error(y_train_np, pred_tr)
                vl = mean_squared_error(y_val_np, pred_val)
            train_loss.append(float(tr))
            val_loss.append(float(vl))
            epochs_axis.append(ep)
    else:
        # Fallback: una sola observacion de loss tras fit
        model.fit(X_train_np, y_train_np)
        pred_tr = model.predict(X_train_np)
        pred_val = model.predict(X_val_np)
        if task == "classification":
            train_loss = [1.0 - accuracy_score(y_train_np, pred_tr)]
            val_loss = [1.0 - accuracy_score(y_val_np, pred_val)]
        else:
            train_loss = [mean_squared_error(y_train_np, pred_tr)]
            val_loss = [mean_squared_error(y_val_np, pred_val)]
        epochs_axis = [1]

    best_epoch = int(np.argmin(val_loss) + 1)
    early_stop_epoch = best_epoch
    if len(val_loss) > patience:
        best = np.inf
        wait = 0
        for i, v in enumerate(val_loss, start=1):
            if v < best - 1e-12:
                best = v
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    early_stop_epoch = i - patience
                    break

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs_axis, y=train_loss, mode="lines+markers", name="Train loss"))
    fig_loss.add_trace(go.Scatter(x=epochs_axis, y=val_loss, mode="lines+markers", name="Validation loss"))
    fig_loss.add_vline(x=best_epoch, line_dash="dash", line_color="green")
    fig_loss.add_vline(x=early_stop_epoch, line_dash="dot", line_color="orange")
    fig_loss.update_layout(title="Training vs Validation Loss", xaxis_title="Epoch", yaxis_title="Loss")

    # Learning curve por tamano de dataset
    fractions = np.linspace(0.1, 1.0, 8)
    lc_rows = []
    for frac in fractions:
        n = max(20, int(len(X_train_np) * frac))
        x_sub = X_train_np[:n]
        y_sub = y_train_np[:n]
        m = clone(model)
        m.fit(x_sub, y_sub)
        pred_tr_sub = m.predict(x_sub)
        pred_val_sub = m.predict(X_val_np)

        if task == "classification":
            tr_score = accuracy_score(y_sub, pred_tr_sub)
            val_score = accuracy_score(y_val_np, pred_val_sub)
        else:
            tr_score = r2_score(y_sub, pred_tr_sub)
            val_score = r2_score(y_val_np, pred_val_sub)

        lc_rows.append({"n_samples": n, "train_score": tr_score, "val_score": val_score})

    lc_df = pd.DataFrame(lc_rows)
    fig_size = go.Figure()
    fig_size.add_trace(go.Scatter(x=lc_df["n_samples"], y=lc_df["train_score"], mode="lines+markers", name="Train"))
    fig_size.add_trace(go.Scatter(x=lc_df["n_samples"], y=lc_df["val_score"], mode="lines+markers", name="Validation"))
    fig_size.update_layout(title="Performance vs Dataset Size", xaxis_title="Train samples", yaxis_title="Score")

    # Scatter OOS pred vs real
    final_model = clone(model)
    final_model.fit(X_train_np, y_train_np)
    y_pred_oos = final_model.predict(X_val_np)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=y_val_np, y=y_pred_oos, mode="markers", name="Predictions"))
    ymin = float(min(np.min(y_val_np), np.min(y_pred_oos)))
    ymax = float(max(np.max(y_val_np), np.max(y_pred_oos)))
    fig_scatter.add_trace(
        go.Scatter(x=[ymin, ymax], y=[ymin, ymax], mode="lines", name="Ideal", line=dict(dash="dash"))
    )
    fig_scatter.update_layout(title="Out-of-sample Predictions vs Real", xaxis_title="Real", yaxis_title="Pred")

    return {
        "loss_curve_fig": fig_loss,
        "dataset_size_fig": fig_size,
        "oos_scatter_fig": fig_scatter,
        "best_epoch": best_epoch,
        "suggested_early_stop_epoch": early_stop_epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_curve_df": lc_df,
    }


def robustness_matrix(
    model,
    X_test,
    y_test,
    perturbation_levels: Sequence[float] = (0.01, 0.05, 0.1),
    task: str = "regression",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evalua estabilidad ante ruido gaussiano en inputs.
    """
    X_np = _as_numpy(X_test).astype(float)
    y_np = _as_numpy(y_test)
    rng = np.random.default_rng(random_state)

    y_pred = model.predict(X_np)
    if task == "classification":
        baseline = accuracy_score(y_np, y_pred)
    else:
        baseline = r2_score(y_np, y_pred)

    rows = [{"perturbation": 0.0, "score": baseline, "relative_drop": 0.0}]
    x_std = np.std(X_np, axis=0, keepdims=True) + EPS

    for level in perturbation_levels:
        noise = rng.normal(0.0, level, size=X_np.shape) * x_std
        x_noisy = X_np + noise
        y_pred_noisy = model.predict(x_noisy)
        if task == "classification":
            score = accuracy_score(y_np, y_pred_noisy)
        else:
            score = r2_score(y_np, y_pred_noisy)
        drop = (baseline - score) / (abs(baseline) + EPS)
        rows.append({"perturbation": float(level), "score": float(score), "relative_drop": float(drop)})

    return pd.DataFrame(rows)


def feature_importance_stability(
    model,
    X,
    y,
    splitter: Iterable[Tuple[np.ndarray, np.ndarray]],
    n_repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Estabilidad de permutation importance entre folds temporales.
    """
    X_df = pd.DataFrame(X)
    y_np = _as_numpy(y)
    fold_importances: List[np.ndarray] = []
    fold_names: List[str] = []

    for fold_idx, (tr, te) in enumerate(splitter):
        m = clone(model)
        m.fit(X_df.iloc[tr], y_np[tr])
        imp = permutation_importance(
            m,
            X_df.iloc[te],
            y_np[te],
            n_repeats=n_repeats,
            random_state=random_state + fold_idx,
        )
        fold_importances.append(imp.importances_mean)
        fold_names.append(f"fold_{fold_idx}")

    imp_mat = np.vstack(fold_importances)
    imp_df = pd.DataFrame(imp_mat, index=fold_names, columns=list(X_df.columns))
    corr = imp_df.T.corr(method="spearman")
    stability = float(np.nanmean(corr.values[np.triu_indices_from(corr.values, k=1)]))

    return {
        "importance_by_fold": imp_df,
        "spearman_corr_matrix": corr,
        "stability_score": stability,
    }


def adversarial_validation_auc(
    X_train,
    X_test,
    model=None,
    random_state: int = 42,
) -> float:
    """
    AUC para detectar dataset shift entre train y test.
    AUC >> 0.5 implica shift fuerte.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    x_tr = _as_numpy(X_train)
    x_te = _as_numpy(X_test)
    x = np.vstack([x_tr, x_te])
    y = np.r_[np.zeros(len(x_tr)), np.ones(len(x_te))]

    x_fit, x_val, y_fit, y_val = train_test_split(x, y, test_size=0.3, random_state=random_state, stratify=y)
    clf = model or RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(x_fit, y_fit)
    p = clf.predict_proba(x_val)[:, 1]
    return float(roc_auc_score(y_val, p))


def randomized_labels_test(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    task: str = "classification",
    n_iter: int = 30,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Si con etiquetas random el modelo sigue dando buen score, hay leakage/señal espuria.
    """
    rng = np.random.default_rng(random_state)
    X_train_np = _as_numpy(X_train)
    y_train_np = _as_numpy(y_train)
    X_test_np = _as_numpy(X_test)
    y_test_np = _as_numpy(y_test)

    # score real
    m_real = clone(model)
    m_real.fit(X_train_np, y_train_np)
    p_real = m_real.predict(X_test_np)
    if task == "classification":
        real_score = accuracy_score(y_test_np, p_real)
    else:
        real_score = r2_score(y_test_np, p_real)

    random_scores = []
    for _ in range(n_iter):
        y_rand = rng.permutation(y_train_np)
        m = clone(model)
        m.fit(X_train_np, y_rand)
        p = m.predict(X_test_np)
        if task == "classification":
            s = accuracy_score(y_test_np, p)
        else:
            s = r2_score(y_test_np, p)
        random_scores.append(float(s))

    random_scores_np = np.array(random_scores)
    p_value = float(np.mean(random_scores_np >= real_score))
    return {
        "real_score": float(real_score),
        "random_mean": float(np.mean(random_scores_np)),
        "random_std": float(np.std(random_scores_np)),
        "p_value": p_value,
    }


def white_reality_check(
    strategy_returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    n_bootstrap: int = 2000,
    block_size: int = 20,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    White Reality Check (bootstrap max performance under null).
    strategy_returns: shape (n_obs, n_strategies) o (n_obs,)
    """
    rng = np.random.default_rng(random_state)
    r = _as_numpy(strategy_returns).astype(float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    n_obs, n_strat = r.shape

    if benchmark_returns is not None:
        b = _as_numpy(benchmark_returns).astype(float).reshape(-1, 1)
        if len(b) != n_obs:
            raise ValueError("benchmark_returns debe tener igual longitud que strategy_returns")
        excess = r - b
    else:
        excess = r.copy()

    # estadistico observado: max mean excess across strategies
    obs_stats = np.mean(excess, axis=0)
    obs_max = float(np.max(obs_stats))

    # Null: mean zero per strategy (de-meaned)
    centered = excess - np.mean(excess, axis=0, keepdims=True)

    def _block_bootstrap_idx(n: int, bsz: int) -> np.ndarray:
        starts = rng.integers(0, n, size=math.ceil(n / bsz))
        idx = np.concatenate([np.arange(s, min(s + bsz, n)) for s in starts])[:n]
        return idx

    boot_max = []
    for _ in range(n_bootstrap):
        idx = _block_bootstrap_idx(n_obs, block_size)
        sample = centered[idx]
        boot_stat = np.mean(sample, axis=0)
        boot_max.append(np.max(boot_stat))

    boot_max_np = np.array(boot_max)
    p_val = float(np.mean(boot_max_np >= obs_max))
    return {
        "observed_max_mean_excess": obs_max,
        "p_value": p_val,
        "n_strategies": float(n_strat),
        "n_obs": float(n_obs),
    }


def romano_wolf_correction(
    observed_stats: Sequence[float],
    bootstrap_stats: np.ndarray,
) -> pd.DataFrame:
    """
    Step-down Romano-Wolf correction.
    observed_stats: vector length m
    bootstrap_stats: matrix (B, m)
    """
    t_obs = np.asarray(observed_stats, dtype=float)
    t_boot = np.asarray(bootstrap_stats, dtype=float)
    if t_boot.ndim != 2 or t_boot.shape[1] != len(t_obs):
        raise ValueError("bootstrap_stats debe tener shape (B, m)")

    m = len(t_obs)
    order = np.argsort(-np.abs(t_obs))
    adjusted = np.ones(m, dtype=float)
    remaining = list(order)

    for rank, idx in enumerate(order):
        active = remaining[rank:]
        max_boot = np.max(np.abs(t_boot[:, active]), axis=1)
        p = np.mean(max_boot >= abs(t_obs[idx]))
        adjusted[idx] = p

    # monotonicidad step-down
    sorted_adj = adjusted[order].copy()
    for i in range(1, m):
        sorted_adj[i] = max(sorted_adj[i], sorted_adj[i - 1])
    adjusted[order] = sorted_adj

    return pd.DataFrame(
        {
            "hypothesis": np.arange(m),
            "stat": t_obs,
            "romano_wolf_p": adjusted,
        }
    ).sort_values("romano_wolf_p")


def deflated_sharpe_ratio(
    sharpe: float,
    sharpe_trials: Sequence[float],
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> Dict[str, float]:
    """
    Approx de Deflated Sharpe Ratio (Bailey et al.).
    Retorna probabilidad de que el SR observado sea genuino tras multiple testing.
    """
    from scipy.stats import norm

    sr = float(sharpe)
    trials = np.asarray(sharpe_trials, dtype=float)
    if len(trials) < 2:
        raise ValueError("sharpe_trials debe contener >=2 trials")
    if n_obs < 3:
        raise ValueError("n_obs debe ser >=3")

    n_trials = len(trials)
    mu_sr = float(np.mean(trials))
    sigma_sr = float(np.std(trials) + EPS)

    euler_gamma = 0.5772156649015329
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sr_star = mu_sr + sigma_sr * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)

    denom = max(EPS, 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * (sr**2))
    z = (sr - sr_star) * math.sqrt((n_obs - 1.0) / denom)
    dsr = float(norm.cdf(z))

    return {
        "deflated_sharpe_ratio_prob": dsr,
        "sr_threshold_after_deflation": float(sr_star),
        "z_score": float(z),
    }


def probability_of_backtest_overfitting(
    returns_df: pd.DataFrame,
    n_splits: int = 8,
) -> Dict[str, object]:
    """
    PBO via CSCV (Combinatorially Symmetric Cross-Validation).
    returns_df: filas=tiempo, columnas=estrategias.
    """
    if n_splits % 2 != 0:
        raise ValueError("n_splits debe ser par para CSCV")
    if n_splits < 4:
        raise ValueError("n_splits debe ser >= 4")

    r = returns_df.copy()
    n = len(r)
    chunk = n // n_splits
    if chunk < 5:
        raise ValueError("Muy pocas observaciones para n_splits solicitado")

    # recortar para chunks iguales
    r = r.iloc[: chunk * n_splits]
    slices = [r.iloc[i * chunk : (i + 1) * chunk] for i in range(n_splits)]
    half = n_splits // 2
    combinations = list(itertools.combinations(range(n_splits), half))

    logits = []
    selected_strategies = []
    for ins_idx in combinations:
        oos_idx = tuple(i for i in range(n_splits) if i not in ins_idx)
        ins = pd.concat([slices[i] for i in ins_idx], axis=0)
        oos = pd.concat([slices[i] for i in oos_idx], axis=0)

        ins_sharpe = ins.mean() / (ins.std() + EPS) * np.sqrt(252)
        best_col = ins_sharpe.idxmax()
        selected_strategies.append(best_col)

        oos_sharpe = oos.mean() / (oos.std() + EPS) * np.sqrt(252)
        rank_pct = float((oos_sharpe.rank(pct=True))[best_col])
        rank_pct = min(max(rank_pct, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank_pct / (1.0 - rank_pct)))

    logits_np = np.array(logits)
    pbo = float(np.mean(logits_np < 0.0))
    return {
        "pbo": pbo,
        "logit_distribution": logits_np,
        "selected_strategies": selected_strategies,
        "n_paths": len(combinations),
    }


def overfitting_alerts_from_metrics(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> List[str]:
    """
    Alertas directas estilo red-flags.
    """
    alerts: List[str] = []

    tr_acc = train_metrics.get("accuracy")
    te_acc = test_metrics.get("accuracy")
    if tr_acc is not None and te_acc is not None and tr_acc > 0.90 and te_acc < 0.60:
        alerts.append("CRITICO: Train accuracy >90% y Test <60% (overfit severo / leakage).")

    tr_sh = train_metrics.get("sharpe_ratio")
    te_sh = test_metrics.get("sharpe_ratio")
    if tr_sh is not None and te_sh is not None:
        if tr_sh > 3.0 and te_sh < tr_sh * 0.4:
            alerts.append("CRITICO: Sharpe cae >60% de train a test.")
        if tr_sh > 3.0 and te_sh is not None:
            alerts.append("ALERTA: Sharpe de backtest >3.0 requiere auditoria de costos/leakage.")

    tr_r2 = train_metrics.get("r2")
    te_r2 = test_metrics.get("r2")
    if tr_r2 is not None and te_r2 is not None and (tr_r2 - te_r2) > 0.2:
        alerts.append("ALTO: gap R2 train-test > 0.2.")

    tr_dd = train_metrics.get("max_drawdown")
    te_dd = test_metrics.get("max_drawdown")
    if tr_dd is not None and te_dd is not None and te_dd > tr_dd * 1.5:
        alerts.append("ALTO: drawdown fuera de muestra muy superior a train.")

    return alerts


@dataclass
class DiagnosticRow:
    metric: str
    train: float
    val: float
    test: float
    oos: float
    status: str


def build_diagnostic_matrix(rows: Sequence[DiagnosticRow]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows])

