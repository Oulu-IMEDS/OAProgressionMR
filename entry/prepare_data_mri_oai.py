import tempfile
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
import dicom2nifti
import pydicom
import nibabel as nib

from oaprmr.datasets.oai import release_to_prefix_var, release_to_visit_month
from oaprmr.various import numpy_to_nifti


logging.basicConfig()
logger = logging.getLogger("prepare")
logger.setLevel(logging.DEBUG)


def dicom_series_to_numpy_meta(dir_dicom):
    # Read, sort, and orient the 3D scan via `dicom2nifti`
    try:
        with tempfile.TemporaryDirectory() as tmp:
            # Convert to NIfTI and orient in LAS+
            dicom2nifti.convert_directory(dir_dicom, tmp, compression=True, reorient=True)
            # Load the image
            fname_nii_in = str(list(Path(tmp).glob("*.nii*"))[0])
            image = nib.load(fname_nii_in).get_fdata()
    except Exception as e:
        logger.warning(f"Skipped {dir_dicom}")
        return None

    # Read DICOM     tags from the first slice
    path_dicom = str(list(Path(dir_dicom).glob("*"))[0])
    data = pydicom.dcmread(path_dicom)
    meta = dict()

    if hasattr(data, "ImagerPixelSpacing"):
        meta["pixel_spacing_0"] = float(data.ImagerPixelSpacing[0])
        meta["pixel_spacing_1"] = float(data.ImagerPixelSpacing[1])
    elif hasattr(data, "PixelSpacing"):
        meta["pixel_spacing_0"] = float(data.PixelSpacing[0])
        meta["pixel_spacing_1"] = float(data.PixelSpacing[1])
    else:
        msg = f"DICOM {path_dicom} does not contain spacing info"
        raise AttributeError(msg)

    meta["slice_thickness"] = float(data.SliceThickness)
    if hasattr(data, "BodyPartExamined"):
        meta["body_part"] = str.upper(data.BodyPartExamined)
    else:
        meta["body_part"] = "KNEE"

    if "RIGHT" in data.SeriesDescription:
        meta["side"] = "RIGHT"
    elif "LEFT" in data.SeriesDescription:
        meta["side"] = "LEFT"
    else:
        msg = f"DICOM {path_dicom} does not contain side info"
        raise AttributeError(msg)

    meta["series"] = str.upper(data.SeriesDescription)

    supported_seqs = ("SAG_3D_DESS", )
    meta["sequence"] = None
    for seq in supported_seqs:
        if seq in meta["series"]:
            meta["sequence"] = seq

    # Reorient the axes
    if meta["sequence"] == "SAG_3D_DESS":
        # Convert from LAS+ to IPR+
        image = np.moveaxis(image, [0, 1, 2], [2, 1, 0])
        image = np.flip(image)
    else:
        logger.error(f"Unsupported series: {dir_dicom}, {meta['series']}")
        return None

    # Apply the corrections based on the DICOM tags
    if data.PhotometricInterpretation == "MONOCHROME1":
        image = image.max(initial=0) - image

    return image, meta


def preproc_compress_series(image_in, meta, path_stack):
    # Version 2
    if meta["sequence"] == "SAG_3D_DESS":
        image_tmp = image_in
        # Truncate least significant bits
        image_tmp = image_tmp.astype(np.uint16)
        image_tmp = image_tmp >> 3
        # Clip outlier intensities
        percents = np.percentile(image_tmp, q=(0., 99.9))
        if percents[1] > 255:
            raise ValueError(f"Out-of-range intensity after clipping: {path_stack}")
        image_tmp = np.clip(image_tmp, percents[0], percents[1])
        # Discretize
        image_tmp = image_tmp.astype(np.uint8)
        # Crop to exclude registration artefacts on margins
        margin = 16
        image_out = np.ascontiguousarray(image_tmp[margin:-margin, margin:-margin, :])

        return image_out, meta

    else:
        raise NotImplementedError(f"Preprocessing is not available: {meta['sequence']}")


def handle_series(config, path_stack):
    if config.debug:
        print(path_stack)

    if "SAG_3D_DESS" in path_stack:
        ret = dicom_series_to_numpy_meta(path_stack)
    else:
        raise ValueError("Error guessing sequence")
    if ret is None:
        logger.warning(f"Error reading: {path_stack}")
        return None
    else:
        image, meta = ret

    image, meta = preproc_compress_series(image, meta, path_stack)

    meta["release"], meta["patient"] = path_stack.split("/")[-4:-2]
    meta["visit_month"] = release_to_visit_month[meta["release"]]
    meta["prefix_var"] = release_to_prefix_var[meta["release"]]

    # Save image and mask
    protocol = f"{meta['body_part']}__{meta['side']}__{meta['sequence']}"
    dir_out = Path(config.dir_root_output, meta["patient"],
                   meta["visit_month"], protocol)
    dir_out.mkdir(exist_ok=True, parents=True)

    spacings = (meta["pixel_spacing_0"],
                meta["pixel_spacing_1"],
                meta["slice_thickness"])

    path_image = str(Path(dir_out, "image.nii.gz"))

    if meta["sequence"] == "SAG_3D_DESS":
        numpy_to_nifti(image, path_image, spacings=spacings, ipr_to_ras=True)
    else:
        numpy_to_nifti(image, path_image, spacings=spacings)

    sel = (
        "patient", "release", "visit_month", "prefix_var",
        "sequence", "body_part", "side",
        "pixel_spacing_0", "pixel_spacing_1", "slice_thickness",
    )
    return {k: meta[k] for k in sel}


@dataclass
class Config:
    dir_root_oai_mri: str
    path_csv_extract: str
    dir_root_output: str
    num_threads: int
    debug: bool = False
    ignore_cache: bool = False


cs = ConfigStore.instance()
cs.store(name="base", node=Config)


@hydra.main(config_path=None, config_name="base")
def main(config: Config) -> None:
    logger.warning(f"Only SAG_3D_DESS is currently supported!")
    logger.warning(f"Only baseline (00m) images are processed!")

    path_df_images = Path(config.dir_root_output, "meta_images.csv")
    if path_df_images.exists() and not config.ignore_cache:
        logger.info("Cached version of the index exists")
    else:
        # OAI data path structure:
        #   root / examination / release / patient / date / barcode (/ slices)
        df_extract = pd.read_csv(config.path_csv_extract)

        paths_stacks = [str(Path(config.dir_root_oai_mri, "00m", subdir))
                        for subdir in df_extract["Folder"].tolist()]
        paths_stacks.sort(key=lambda x: int(x.split("/")[-3]))

        if config.num_threads == 1:  # Debug mode
            if config.debug:
                # Single series
                metas = [handle_series(config, paths_stacks[0]), ]
            else:
                # All series in 1 thread
                metas = [handle_series(config, path_stack)
                         for path_stack in tqdm(paths_stacks)]
        else:
            metas = Parallel(n_jobs=config.num_threads, verbose=10)(
                delayed(handle_series)(*[config, path_stack])
                for path_stack in tqdm(paths_stacks))

        # Merge meta information from different stacks
        tmp = defaultdict(list)
        for d in metas:
            if d is None:
                continue
            for k, v in d.items():
                tmp[k].append(v)
        df_images = pd.DataFrame.from_dict(tmp)
        dtypes = {"patient": str, "visit_month": str, "side": str, "sequence": str}
        df_images = df_images.astype(dtypes)

        # Save the meta
        df_images.to_csv(path_df_images, index=False)


if __name__ == "__main__":
    main()
