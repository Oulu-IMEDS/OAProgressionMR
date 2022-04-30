import os
import logging
import time
import pickle
import functools
from pathlib import Path
from collections import defaultdict

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.special import softmax
import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import thop  # see also: https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md

from oaprmr.datasets import prepare_datasets_loaders
from oaprmr.models import dict_models
from oaprmr import preproc
from oaprmr.various import CheckpointHandler, set_ultimate_seed, calc_metrics


# Fix to PyTorch multiprocessing issue: "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logging.basicConfig()
logger = logging.getLogger('eval')
logger.setLevel(logging.DEBUG)

set_ultimate_seed()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class ProgressionPrediction(object):
    def __init__(self, *, config):
        self.config = config

        # Build a list of folds to run on
        if config.testing.folds.idx == -1:
            self.fold_idcs = list(range(config.training.folds.num))
        else:
            self.fold_idcs = [config.testing.folds.idx, ]
        if config.testing.folds.ignore is not None:
            for g in config.testing.folds.ignore:
                self.fold_idcs = [i for i in self.fold_idcs if i != g]

        if self.config.model.downscale:
            logger.warning("Downscaling is enabled!")

        # Initialize datasets, loaders, and transforms
        t = prepare_datasets_loaders(config=config, fold_idx=0)
        self.datasets = t[0]
        self.data_loaders = t[1]

        # Init experiment paths
        self.path_weights = Path(self.config.path_experiment_root, "weights")
        self.path_logs = Path(self.config.path_experiment_root, "logs_eval")

        self.tb = SummaryWriter(str(self.path_logs))

    def eval(self):
        paths_cache = {
            "raw_fold-w": Path(self.path_logs, f"eval_raw_foldw.pkl"),
            "raw_ens": Path(self.path_logs, f"eval_raw_ens.pkl"),
            "metrics_fold-w": Path(self.path_logs, f"eval_metrics_foldw.pkl"),
            "metrics_ens": Path(self.path_logs, f"eval_metrics_ens.pkl"),
        }

        # Only raw predicts are restored from cache
        raw_foldw = dict()
        raw_ens = dict()

        # Fold-w predicts
        if self.config.testing.use_cached and paths_cache["raw_fold-w"].exists():
            with open(paths_cache["raw_fold-w"], "rb") as f:
                raw_foldw = pickle.load(f)
        else:
            for fold_idx in self.fold_idcs:
                # Init fold-wise paths
                paths_weights_fold = dict()
                paths_weights_fold["prog"] = \
                    Path(self.path_weights, "prog", f"fold_{fold_idx}")

                handlers_ckpt = dict()
                handlers_ckpt["prog"] = CheckpointHandler(paths_weights_fold["prog"])

                paths_ckpt_sel = dict()
                paths_ckpt_sel["prog"] = handlers_ckpt["prog"].get_last_ckpt()

                # Initialize and configure model
                models = dict()
                models["prog"] = dict_models[self.config.model.name](
                    config=self.config.model,
                    path_weights=paths_ckpt_sel["prog"])

                models["prog"] = models["prog"].to(device)
                if self.config.testing.profile == "none":
                    models["prog"] = nn.DataParallel(models["prog"])
                # Switch to eval regime
                models["prog"] = models["prog"].eval()

                # Eval model on subset
                t = self.eval_epoch(models=models)
                raw_foldw[fold_idx] = t

            # Save fold-w to cache
            with open(paths_cache["raw_fold-w"], "wb") as f:
                pickle.dump(raw_foldw, f, pickle.HIGHEST_PROTOCOL)

        def _pprint_metrics(d):
            for k, v in d.items():
                if k == "roc_curve":
                    continue
                logger.info(f"{k}: {np.round(v, decimals=3)}")

        # Metrics fold-w
        if self.config.testing.metrics_foldw:
            metrics_foldw = dict()
            for fold_idx in self.fold_idcs:
                if fold_idx in raw_foldw:
                    metrics_foldw[fold_idx] = calc_metrics(
                        prog_target=np.asarray(raw_foldw[fold_idx]["target"]),
                        prog_pred_proba=np.asarray(raw_foldw[fold_idx]["predict_proba"]),
                        target=self.config.data.target)

            with open(paths_cache["metrics_fold-w"], "wb") as f:
                pickle.dump(metrics_foldw, f, pickle.HIGHEST_PROTOCOL)
            logger.info(f"Metrics fold-w:")
            for fold_idx in self.fold_idcs:
                logger.info(f"Fold {fold_idx}:")
                _pprint_metrics(metrics_foldw[fold_idx])

        # Ens predicts
        if self.config.testing.ensemble_foldw and len(raw_foldw) > 0:
            if self.config.testing.use_cached and paths_cache["raw_ens"].exists():
                with open(paths_cache["raw_ens"], "rb") as f:
                    raw_ens = pickle.load(f)
            else:
                raw_ens = self.ensemble_foldw(raw_foldw=raw_foldw)

                # Save ens to cache
                with open(paths_cache["raw_ens"], "wb") as f:
                    pickle.dump(raw_ens, f, pickle.HIGHEST_PROTOCOL)

            # Metrics ens
            if self.config.testing.metrics_ensemble:
                metrics_ens = calc_metrics(
                    prog_target=np.asarray(raw_ens["target"]),
                    prog_pred_proba=np.asarray(raw_ens["predict_proba"]),
                    target=self.config.data.target)

                with open(paths_cache["metrics_ens"], "wb") as f:
                    pickle.dump(metrics_ens, f, pickle.HIGHEST_PROTOCOL)
                logger.info("Metrics ens:")
                _pprint_metrics(metrics_ens)

    @staticmethod
    def _extract_modal(batch, modal):
        assert modal in ("sag_3d_dess", "xr_pa")
        return batch[f"image__{modal}"]

    @staticmethod
    def _downscale_x(modal, x, factor):
        if factor:
            x = preproc.PTInterpolate(scale_factor=factor)(x)
            x = x.contiguous()
        return x

    def eval_epoch(self, models):
        """Evaluation regime"""
        acc = defaultdict(list)

        ds = next(iter(self.config.data.sets.values()))
        dl = self.data_loaders[ds.name]["test"]
        steps_dl = len(dl)

        prog_bar_params = {"total": steps_dl, "desc": "Testing"}

        if self.config.testing.profile == "time":
            sum_time = 0
            sum_samples = 0

        with torch.no_grad(), tqdm(**prog_bar_params) as prog_bar:
            for step_idx, data_batch_ds in enumerate(dl):
                # Select vars from batch
                xs_vec_ds = tuple(self._extract_modal(data_batch_ds, m)
                                  for m in ds.modals)
                xs_vec_ds = (x.to(device) for x in xs_vec_ds)
                ys_true_ds = data_batch_ds["target"]
                ys_true_ds = ys_true_ds.to(device)

                # Last-chance preprocessing
                if self.config.model.downscale:
                    xs_vec_ds = tuple(self._downscale_x(m, x, tuple(f))
                                      for m, x, f in zip(ds.modals, xs_vec_ds,
                                                         self.config.model.downscale))

                # Model inference
                if self.config.testing.profile == "compute":
                    xs_vec_dummy = tuple(x[0:1] for x in xs_vec_ds)
                    macs, params = thop.profile(models["prog"], inputs=xs_vec_dummy)
                    macs, params = thop.clever_format([macs, params], "%.3f")
                    logger.info(f"MACs: {macs}, params: {params}")
                    quit()
                if self.config.testing.profile == "time":
                    time_pre = time.time()

                ys_pred_ds = models["prog"](*xs_vec_ds)["main"]

                if self.config.testing.profile == "time":
                    time_post = time.time()
                    sum_time += (time_post - time_pre)
                    sum_samples += int(xs_vec_ds[0].shape[0])

                # Accumulate the predictions
                ys_true_ds_np = ys_true_ds.detach().to("cpu").numpy()
                t = ys_pred_ds.detach().to("cpu")
                ys_pred_ds_np = torch.argmax(t, dim=1).numpy()
                ys_pred_proba_ds_np = nn.functional.softmax(t, dim=1)

                acc["exam_knee_id"].extend(data_batch_ds[("-", "exam_knee_id")])
                acc["target"].extend(ys_true_ds_np.tolist())
                acc["predict"].extend(ys_pred_ds_np.tolist())
                acc["predict_proba"].extend(ys_pred_proba_ds_np.tolist())

                prog_bar.update(1)

        if self.config.testing.profile == "time":
            logger.info(f"Inference time per sample: {sum_time / sum_samples}")
            quit()

        return acc

    def ensemble_foldw(self, raw_foldw):
        """Merge the predictions over all folds"""
        dfs = []
        for fold_idx, d in raw_foldw.items():
            t = pd.DataFrame.from_dict(d)
            t = t.rename(columns={"predict": f"predict__{fold_idx}",
                                  "predict_proba": f"predict_proba__{fold_idx}"})
            dfs.append(t)

        selectors = ["exam_knee_id", ]
        # Drop repeating columns with dtype not supported by merge
        dfs[1:] = [e.drop(columns="target") for e in dfs[1:]]
        df_ens = functools.reduce(
            lambda l, r: pd.merge(l, r, on=selectors, validate="1:1"), dfs)

        # Average fold predictions
        cols = [c for c in df_ens.columns if c.startswith("predict_proba__")]
        t = np.asarray(df_ens[cols].values.tolist())
        # samples * folds * classes
        t = softmax(np.mean(t, axis=1), axis=-1)
        df_ens["predict_proba"] = t.tolist()
        df_ens["predict"] = np.argmax(t, axis=-1).tolist()

        raw_ens = df_ens.to_dict(orient="list")
        return raw_ens


@hydra.main(config_path="conf", config_name="prog")
def main(config: DictConfig) -> None:
    Path(config.path_logs).mkdir(exist_ok=True, parents=True)
    logging_fh = logging.FileHandler(
        Path(config.path_logs, "eval_prog_{}.log".format(config.testing.folds.idx)))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)

    logger.info(OmegaConf.to_yaml(config, resolve=True))

    prog_pred = ProgressionPrediction(config=config)
    prog_pred.eval()


if __name__ == '__main__':
    main()
