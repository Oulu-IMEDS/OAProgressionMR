import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, average_precision_score,
                             roc_auc_score, precision_recall_curve, roc_curve,
                             recall_score, confusion_matrix)
from tqdm import tqdm


def calc_metrics(prog_target, prog_pred_proba, with_curves=False):
    """
    prog_target: (sample, )
    prog_pred_proba: (sample, class)
    target: {}
    """
    out = dict()

    # Adapted from tiulpin2019multimodal
    prog_target_bin = prog_target > 0
    prog_pred_proba_bin = prog_pred_proba[:, 1:].sum(1)
    prog_pred_multi = np.argmax(prog_pred_proba, axis=1)

    map_freqs = pd.Series(prog_target_bin.ravel()).value_counts(normalize=True).to_dict()

    out["prevalence"] = np.sum(prog_target_bin) / prog_target_bin.shape[0]
    out["roc_auc"] = roc_auc_score(prog_target_bin, prog_pred_proba_bin)

    out["avg_precision"] = average_precision_score(prog_target_bin,
                                                   prog_pred_proba_bin)

    out["b_accuracy"] = balanced_accuracy_score(prog_target_bin,
                                                prog_pred_proba_bin > 0.5)
    # ACTION: comment out
    out["cm"] = confusion_matrix(prog_target, prog_pred_multi,
                                 labels=[0, 2, 1])
    out["cm_norm"] = confusion_matrix(prog_target, prog_pred_multi,
                                      labels=[0, 2, 1], normalize="true")

    if with_curves:
        t = roc_curve(prog_target_bin, prog_pred_proba_bin)
        fpr, tpr, _ = t
        out["roc_curve"] = (fpr, tpr)

        t = precision_recall_curve(y_true=prog_target_bin,
                                   probas_pred=prog_pred_proba_bin,
                                   sample_weight=None)
        prec, rec, thr = t
        out["pr_curve"] = (prec, rec)

    for k, v in out.items():
        if k in ("prevalence", "roc_auc", "avg_precision", "b_accuracy"):
            out[k] = np.round(v, 3)

    return out


def calc_bootstrap(metric, y_true, y_preds, n_bootstrap=100, seed=0, stratified=True,
                   alpha=95):
    """

    Inspired by https://github.com/MIPT-Oulu/OAProgression/blob/master/oaprogression/evaluation/stats.py

    Parameters
    ----------
    metric : fucntion
        Metric to compute
    y_true : ndarray
        Ground truth
    y_preds : ndarray
        Predictions
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed
    stratified : bool
        Whether to do a stratified bootstrapping
    alpha : float
        Confidence intervals width
    """
    if len(np.unique(y_true)) > 2:
        raise ValueError(f"Expected binary target, got: {np.unique(y_true)}")

    np.random.seed(seed)
    metric_vals = []
    ind_pos = np.where(y_true == 1)[0]
    ind_neg = np.where(y_true == 0)[0]

    for _ in tqdm(range(n_bootstrap), total=n_bootstrap, desc='Bootstrap:'):
        if stratified:
            ind_pos_bs = np.random.choice(ind_pos, ind_pos.shape[0])
            ind_neg_bs = np.random.choice(ind_neg, ind_neg.shape[0])
            ind = np.hstack((ind_pos_bs, ind_neg_bs))
        else:
            ind = np.random.choice(y_true.shape[0], y_true.shape[0])

        if y_true[ind].sum() == 0:
            continue
        metric_vals.append(metric(y_true[ind], y_preds[ind]))

    metric_val = metric(y_true, y_preds)
    ci_l = np.percentile(metric_vals, (100 - alpha) // 2)
    ci_h = np.percentile(metric_vals, alpha + (100 - alpha) // 2)
    std_err = np.std(metric_vals)

    return metric_val, std_err, ci_l, ci_h


def calc_bootstrap_multiclass(metric, y_true, y_preds, n_bootstrap=100, seed=0,
                              stratified=True, alpha=95):
    """

    Parameters
    ----------
    metric : fucntion
        Metric to compute
    y_true : ndarray
        Ground truth
    y_preds : ndarray
        Predictions
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed
    stratified : bool
        Whether to do a stratified bootstrapping
    alpha : float
        Confidence intervals width
    """
    if len(np.unique(y_true)) != 3:
        raise ValueError(f"Expected binary target, got: {np.unique(y_true)}")

    np.random.seed(seed)
    metric_vals = []
    ind_2 = np.where(y_true == 2)[0]
    ind_1 = np.where(y_true == 1)[0]
    ind_0 = np.where(y_true == 0)[0]

    for _ in tqdm(range(n_bootstrap), total=n_bootstrap, desc='Bootstrap:'):
        if stratified:
            ind_2_bs = np.random.choice(ind_2, ind_2.shape[0])
            ind_1_bs = np.random.choice(ind_1, ind_1.shape[0])
            ind_0_bs = np.random.choice(ind_0, ind_0.shape[0])
            ind = np.hstack((ind_2_bs, ind_1_bs, ind_0_bs))
        else:
            ind = np.random.choice(y_true.shape[0], y_true.shape[0])

        if y_true[ind].sum() == 0:
            continue
        metric_vals.append(metric(y_true[ind], y_preds[ind]))

    metric_val = metric(y_true, y_preds)
    ci_l = np.percentile(metric_vals, (100 - alpha) // 2)
    ci_h = np.percentile(metric_vals, alpha + (100 - alpha) // 2)
    std_err = np.std(metric_vals)

    return metric_val, std_err, ci_l, ci_h


def calc_metrics_bootstrap(prog_target, prog_pred_proba):
    """
    prog_target: (sample, )
    prog_pred_proba: (sample, class)
    target: {}
    """
    out = dict()

    kws_bootstrap = {"n_bootstrap": 1000, "seed": 0, "stratified": True, "alpha": 95}

    # Adapted from tiulpin2019multimodal
    prog_target_bin = prog_target > 0
    prog_pred_proba_bin = prog_pred_proba[:, 1:].sum(1)

    out['roc_auc'] = calc_bootstrap(metric=roc_auc_score,
                                    y_true=prog_target_bin,
                                    y_preds=prog_pred_proba_bin,
                                    **kws_bootstrap)
    out['avg_precision'] = calc_bootstrap(metric=average_precision_score,
                                          y_true=prog_target_bin,
                                          y_preds=prog_pred_proba_bin,
                                          **kws_bootstrap)

    for k, v in out.items():
        if k in ("prevalence", "roc_auc", "avg_precision", "b_accuracy"):
            out[k] = np.round(v, 3)

    return out
