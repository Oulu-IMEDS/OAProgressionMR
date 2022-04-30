"""
Entry point to all available datasets, subsets, folds, and dataloaders.
"""

import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import sklearn.model_selection
from torch.utils.data import DataLoader, WeightedRandomSampler

from oaprmr import preproc
from oaprmr.various import StratifiedGroupKFold
from oaprmr.datasets import DatasetOAI3d
from oaprmr.datasets.oai import index_from_path_oai


logging.basicConfig()
logger = logging.getLogger("provider")
logger.setLevel(logging.DEBUG)


def sources_from_path(*, path_data_root, modals_all, target, fold_num, scheme_train_val,
                      seed_trainval_test, site_test, seed_train_val, ignore_cache=False,
                      select_side=None):
    """

    Args:
        modals_all:
        scheme_train_val:
        site_test:
        ignore_cache:
        select_side:
        path_data_root (str):
        target (str):
        fold_num (int):
            Number of folds.
        seed_trainval_test (int):
            Random state for the trainval/test splitting.
        seed_train_val (int):
            Random state for the train/val splitting.

    Returns:

    """
    assert scheme_train_val in ("strat_target", "one_site_out")

    def _exclude_missing_clin(df_i):
        df_o = df_i.copy()
        df_o = df_o.dropna(axis=0, subset=[("-", "P01BMI"), ])
        logger.info(f"Removed samples with missing BMI")
        logger.info(f"Selected: {len(df_o)}")
        return df_o

    def _select_subjects_target(df_i):
        df_o = df_i.copy()
        logger.info(f"Using target: {target}!")

        if target in ("tiulpin2019_prog",):
            df_o[("-", "target")] = df_o[("-", target)]
            df_o = df_o[df_o[("-", "tiulpin2019_sel")] == 1]
            df_o = df_o[df_o[("-", "target")] != -1]
        else:
            raise ValueError(f"Unsupported target: {target}")

        logger.info(f"Removed samples with missing target")
        logger.info(f"Selected: {len(df_o)}")
        return df_o

    def _exclude_corrupted_imaging(df_i):
        corr = [
            ("9004315", "000m", "RIGHT"),  # "SAG_3D_DESS"
            ("9522128", "000m", "RIGHT"),  # "SAG_3D_DESS"  [352, 352, 29]
            ("9560965", "000m", "RIGHT"),  # "SAG_3D_DESS"
            ("9594253", "000m", "LEFT"),  # "SAG_3D_DESS"  [352, 352, 124]
            ("9617608", "000m", "LEFT"),  # "SAG_3D_DESS"  [352, 288, 160]
            ("9637394", "000m", "RIGHT"),  # "SAG_3D_DESS"  [352, 352, 56]
        ]

        df_o = df_i.copy()
        for c in corr:
            df_o = df_o[~(
                (df_o[("-", "patient")] == c[0]) &
                (df_o[("-", "visit_month")] == c[1]) &
                (df_o[("-", "side")] == c[2])
            )]
        logger.info(f"Excluded samples with corrupted imaging. Selected: {len(df_o)}")
        return df_o

    def _select_side(df_i, side):
        assert side in ("LEFT", "RIGHT", None)
        df_o = df_i.copy()
        if side is None:
            return df_o
        else:
            df_o = df_o[df_o[("-", "side")] == side]
            logger.info(f"Selected side {side}: {len(df_o)}")
            return df_o

    path_data_root = Path(path_data_root).resolve()

    t_df = dict()
    sources = dict()

    t_df["full_df"] = index_from_path_oai(path_root=path_data_root,
                                          modals_all=modals_all,
                                          ignore_cache=ignore_cache)
    logger.info(f"Number of samples, total: {len(t_df['full_df'])}")

    # Select the specific subset
    t_df["sel_df"] = t_df["full_df"].copy()
    t_df["sel_df"] = _select_subjects_target(t_df["sel_df"])
    t_df["sel_df"] = _exclude_missing_clin(t_df["sel_df"])
    t_df["sel_df"] = _exclude_corrupted_imaging(t_df["sel_df"])
    t_df["sel_df"] = _select_side(t_df["sel_df"], select_side)
    logger.info(f"Number of samples, selected: {len(t_df['sel_df'])}")

    # Get trainval/test split
    t_df["trainval_df"] = t_df["sel_df"][t_df["sel_df"][("-", "V00SITE")] != site_test]
    t_df["test_df"] = t_df["sel_df"][t_df["sel_df"][("-", "V00SITE")] == site_test]
    logger.info("Made trainval-test split:")
    logger.info("  number of subjects: "
                f"{len(pd.unique(t_df['trainval_df'][('-', 'patient')]))}, "
                f"{len(pd.unique(t_df['test_df'][('-', 'patient')]))}")
    logger.info("  number of samples: "
                f"{len(t_df['trainval_df'])}, {len(t_df['test_df'])}")

    # Make train_val folds
    if scheme_train_val == "strat_target":
        t_gkf = StratifiedGroupKFold(n_splits=fold_num,
                                     shuffle=True,
                                     random_state=seed_train_val)
        t_grades = t_df["trainval_df"].loc[:, ("-", "target")].values
        t_groups = t_df["trainval_df"].loc[:, ("-", "patient")].values

        t_df["trainval_folds"] = t_gkf.split(X=t_df["trainval_df"],
                                             y=t_grades,
                                             groups=t_groups)
    elif scheme_train_val == "one_site_out":
        t_gkf = sklearn.model_selection.LeaveOneGroupOut()
        t_grades = t_df["trainval_df"].loc[:, ("-", "target")].values
        t_groups = t_df["trainval_df"].loc[:, ("-", "V00SITE")].values
        # Treat low-data sites A and E as one
        t_groups[t_groups == "E"] = "A"

        t_df["trainval_folds"] = t_gkf.split(X=t_df["trainval_df"],
                                             y=t_grades,
                                             groups=t_groups)

    sources["oai"] = t_df
    return sources


def prepare_datasets_loaders(config, fold_idx):
    """

    Returns:
        (datasets, loaders)
    """
    datasets = defaultdict(dict)
    loaders = defaultdict(dict)

    # Collect available sources and make splits
    sources = sources_from_path(
        path_data_root=config.path_data_root,
        modals_all=config.data.modals_all,
        target=config.data.target,
        # select_side=config.data.side,
        fold_num=config.training.folds.num,
        # scheme_trainval_test=config.scheme_trainval_test,
        scheme_train_val=config.scheme_train_val,
        seed_trainval_test=config.seed_trainval_test,
        seed_train_val=config.seed_train_val,
        site_test=config.site_test,
        ignore_cache=config.data.ignore_cache,
    )

    # ds_names = [n for n in sources.keys()]
    ds_names = [d.name for _, d in config.data.sets.items()]

    # Use straightforward fold allocation strategy
    folds_seq = [sources[n]["trainval_folds"] for n in ds_names]
    folds_zip = list(zip(*folds_seq))
    # Select fold
    idcs_subsets = folds_zip[fold_idx]

    for idx, (_, ds) in enumerate(config.data.sets.items()):
        stats_classes = pd.value_counts(
            sources[ds.name]["sel_df"][("-", "target")]).to_dict()
        logger.info(f"Number of class occurrences in selected dataset: {stats_classes}")

        stats_classes = pd.value_counts(
            sources[ds.name]["trainval_df"][("-", "target")]).to_dict()
        logger.info(f"Number of class occurrences in trainval subset: {stats_classes}")

        sources[ds.name]["train_idcs"] = idcs_subsets[idx][0]
        sources[ds.name]["val_idcs"] = idcs_subsets[idx][1]

        sources[ds.name]["train_df"] = \
            sources[ds.name]["trainval_df"].iloc[sources[ds.name]["train_idcs"]]
        sources[ds.name]["val_df"] = \
            sources[ds.name]["trainval_df"].iloc[sources[ds.name]["val_idcs"]]

    # Select fraction of samples keeping balance of targets
    for idx, (_, ds) in enumerate(config.data.sets.items()):
        frac = ds.frac_classw
        if frac != 1.0:
            logger.warning(f"Sampled fraction of {frac} per target class")

            df_tmp = sources[ds.name]["train_df"]
            df_tmp = (df_tmp
                      .sort_values([("-", "target"), ])
                      .groupby(("-", "target"))
                      .sample(frac=frac, random_state=0))
            logger.warning(f"Selected only {len(df_tmp)} samples from train")
            sources[ds.name]["train_df"] = df_tmp

            df_tmp = sources[ds.name]["val_df"]
            df_tmp = (df_tmp
                      .sort_values([("-", "target"), ])
                      .groupby(("-", "target"))
                      .sample(frac=frac, random_state=0))
            logger.warning(f"Selected only {len(df_tmp)} samples from val")
            sources[ds.name]["val_df"] = df_tmp

        for n, s in sources.items():
            logger.info("Made {} train-val split, number of samples: {}, {}"
                        .format(n, len(s["train_df"]), len(s["val_df"])))
            logger.info("Test subset, number of samples: {}".format(len(s["test_df"])))

    # Initialize datasets
    for _, ds in config.data.sets.items():
        transfs = defaultdict(dict)  # (ds_modal, regime): []

        for idx, modal in enumerate(ds.modals):
            # Transforms and augmentations
            transfs["train"][modal] = []
            transfs["val"][modal] = []
            transfs["test"][modal] = []

            if (ds.name, modal) == ("oai", "sag_3d_dess"):
                transfs["train"][modal].extend([
                    preproc.RandomCrop(output_size=list(config.model.input_size[idx]), ndim=3),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                    preproc.PTRotate3DInSlice(degree_range=[-15., 15.], prob=0.5),
                    preproc.PTGammaCorrection(gamma_range=(0.5, 2.0), prob=0.5, clip_to_unit=False),
                ])
            elif (ds.name, modal) == ("oai", "xr_pa"):
                transfs["train"][modal].extend([
                    preproc.RandomCrop(output_size=list(config.model.input_size[idx]), ndim=2),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                    preproc.PTRotate2D(degree_range=[-15., 15.], prob=0.5),
                    preproc.PTGammaCorrection(gamma_range=(0.5, 2.0), prob=0.5, clip_to_unit=False),
                ])
            else:
                raise ValueError(f"Unknown dataset/modality: {ds.name}/{modal}")

            if (ds.name, modal) == ("oai", "sag_3d_dess"):
                transfs["train"][modal].append(
                    preproc.PTNormalize(mean=[0.257, ], std=[0.235, ]))
            elif (ds.name, modal) == ("oai", "xr_pa"):
                transfs["train"][modal].append(
                    preproc.PTNormalize(mean=[0.543, ], std=[0.296, ]))
            else:
                raise ValueError(f"Unknown dataset/modality: {ds.name}/{modal}")

            if (ds.name, modal) == ("oai", "sag_3d_dess"):
                transfs["val"][modal].extend([
                    preproc.CenterCrop(output_size=list(config.model.input_size[idx]), ndim=3),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                ])
            elif (ds.name, modal) == ("oai", "xr_pa"):
                transfs["val"][modal].extend([
                    preproc.CenterCrop(output_size=list(config.model.input_size[idx]), ndim=2),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                ])
            else:
                raise ValueError(f"Unknown dataset/modality: {ds.name}/{modal}")

            if (ds.name, modal) == ("oai", "sag_3d_dess"):
                transfs["val"][modal].append(
                    preproc.PTNormalize(mean=[0.257, ], std=[0.235, ]))
            elif (ds.name, modal) == ("oai", "xr_pa"):
                transfs["val"][modal].append(
                    preproc.PTNormalize(mean=[0.543, ], std=[0.296, ]))
            else:
                raise ValueError(f"Unknown dataset/modality: {ds.name}/{modal}")

            if (ds.name, modal) == ("oai", "sag_3d_dess"):
                transfs["test"][modal].extend([
                    preproc.CenterCrop(output_size=list(config.model.input_size[idx]), ndim=3),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                ])
            elif (ds.name, modal) == ("oai", "xr_pa"):
                transfs["test"][modal].extend([
                    preproc.CenterCrop(output_size=list(config.model.input_size[idx]), ndim=2),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                ])
            else:
                raise ValueError(f"Unknown dataset/modality: {ds.name}/{modal}")

            if (ds.name, modal) == ("oai", "sag_3d_dess"):
                transfs["test"][modal].append(
                    preproc.PTNormalize(mean=[0.257, ], std=[0.235, ]))
            elif (ds.name, modal) == ("oai", "xr_pa"):
                transfs["test"][modal].append(
                    preproc.PTNormalize(mean=[0.543, ], std=[0.296, ]))
            else:
                raise ValueError(f"Unknown dataset/modality: {ds.name}/{modal}")

        if ds.name == "oai":
            cls = DatasetOAI3d
        else:
            raise ValueError(f"Unknown dataset {ds.name}")

        # Instantiate datasets
        datasets[ds.name]["train"] = cls(
            df_meta=sources[ds.name]["train_df"],
            modals=ds.modals,
            transforms=transfs["train"])
        datasets[ds.name]["val"] = cls(
            df_meta=sources[ds.name]["val_df"],
            modals=ds.modals,
            transforms=transfs["val"])
        datasets[ds.name]["test"] = cls(
            df_meta=sources[ds.name]["test_df"],
            modals=ds.modals,
            transforms=transfs["test"])

    # Initialize data loaders
    for _, ds in config.data.sets.items():
        # Configure samplers
        logger.info("Using frequency-based sampler for training subset")
        t_df = sources[ds.name]["train_df"]
        map_freqs = t_df[("-", "target")].value_counts(normalize=True).to_dict()
        sample_weights = [1.0 / map_freqs[e] for e in t_df[("-", "target")].tolist()]
        sampler_train = WeightedRandomSampler(weights=sample_weights,
                                              num_samples=len(sample_weights),
                                              replacement=True)

        # Instantiate dataloaders
        loaders[ds.name]["train"] = DataLoader(
            datasets[ds.name]["train"],
            batch_size=config.training.batch_size,
            sampler=sampler_train,
            # shuffle=True,
            num_workers=config.num_workers,
            drop_last=True)

        logger.warning("Validation balanced sampling is disabled!")
        loaders[ds.name]["val"] = DataLoader(
            datasets[ds.name]["val"],
            batch_size=config.validation.batch_size,
            # sampler=sampler_val,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=True)

        loaders[ds.name]["test"] = DataLoader(
            datasets[ds.name]["test"],
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False)

    return datasets, loaders
