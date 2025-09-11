import functools
import inspect
import logging
import os.path as op
from time import time

import nibabel as nib
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram

try:
    from trx.trx_file_memmap import TrxFile
    from trx.io import save as save_trx
    has_trx = True
except ModuleNotFoundError:
    has_trx = False

import numpy as np

from AFQ.tasks.utils import get_fname
from AFQ.utils.path import drop_extension, write_json


# These should only be used with immlib.calc
__all__ = ["as_file", "as_fit_deriv", "as_img"]


logger = logging.getLogger('AFQ')
logger.setLevel(logging.INFO)


def get_new_signature(og_func, needed_args):
    sig = inspect.signature(og_func)
    param_dict = sig.parameters

    existing_args = set(param_dict)
    new_param_names = [arg for arg in needed_args if arg not in existing_args]

    new_params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in new_param_names]

    parameters = new_params + list(param_dict.values())
    new_sig = sig.replace(parameters=parameters)

    return new_sig, new_param_names


def get_param(kwargs, new_params, arg_name):
    if arg_name in new_params:
        return kwargs.pop(arg_name)
    else:
        return kwargs.get(arg_name)


def as_file(suffix, subfolder=None):
    """
    return img and meta as saved file path, with json,
    and only run if not already found
    """
    def _as_file(func):
        new_signature, new_params = get_new_signature(
            func, ["base_fname", "output_dir", "tracking_params"])

        @functools.wraps(func)
        def wrapper_as_file(*args, **kwargs):
            base_fname = get_param(kwargs, new_params, "base_fname")
            output_dir = get_param(kwargs, new_params, "output_dir")
            tracking_params = get_param(kwargs, new_params, "tracking_params")

            this_file = get_fname(base_fname, suffix, subfolder=subfolder)

            # if file has no extension, we need to determine it
            if not op.splitext(this_file)[1]:
                if tracking_params.get("trx", False):
                    this_file = this_file + ".trx"
                else:
                    this_file = this_file + ".trk"

            if not op.exists(this_file):
                logger.info(f"Calculating {suffix}")

                gen, meta = func(
                    *args, **kwargs)

                logger.info(f"{suffix} completed. Saving to {this_file}")
                if isinstance(gen, nib.Nifti1Image):
                    nib.save(gen, this_file)
                elif isinstance(gen, StatefulTractogram):
                    save_tractogram(
                        gen, this_file, bbox_valid_check=False)
                elif isinstance(gen, np.ndarray):
                    np.save(this_file, gen)
                elif has_trx and isinstance(gen, TrxFile):
                    save_trx(gen, this_file)
                else:
                    gen.to_csv(this_file)

                # these are used to determine dependencies
                # when clobbering derivatives
                if "_desc-profiles" in suffix or\
                        "viz" in inspect.getfile(func):
                    meta["dependent"] = "prof"
                elif "segmentation" in inspect.getfile(func) or\
                        "mapping" in inspect.getfile(func):
                    meta["dependent"] = "rec"
                elif "tractography" in inspect.getfile(func):
                    meta["dependent"] = "trk"
                else:
                    meta["dependent"] = "dwi"

                # modify meta source to be relative
                if "source" in meta:
                    meta["source"] = op.relpath(meta["source"], output_dir)

                meta_fname = get_fname(
                    base_fname, f"{drop_extension(suffix)}.json",
                    subfolder=subfolder)
                write_json(meta_fname, meta)
            return this_file

        wrapper_as_file.__signature__ = new_signature

        return wrapper_as_file
    return _as_file


def as_fit_deriv(tf_name):
    """
    return data as nibabel image, meta with params information
    """
    def _as_fit_deriv(func):
        new_signature, new_params = get_new_signature(
            func, ["dwi_affine", f"{tf_name.lower()}_params"])

        @functools.wraps(func)
        def wrapper_as_fit_deriv(*args, **kwargs):
            dwi_affine = get_param(kwargs, new_params, "dwi_affine")
            params = get_param(kwargs, new_params,
                               f"{tf_name.lower()}_params")

            img = nib.Nifti1Image(
                func(*args, **kwargs), dwi_affine)
            return img, {f"{tf_name}ParamsFile": params}

        wrapper_as_fit_deriv.__signature__ = new_signature

        return wrapper_as_fit_deriv
    return _as_fit_deriv


def as_img(func):
    """
    return data, meta as nibabel image, meta with timing
    """
    new_signature, new_params = get_new_signature(
        func, ["dwi_affine"])

    @functools.wraps(func)
    def wrapper_as_img(*args, **kwargs):
        dwi_affine = get_param(kwargs, new_params, "dwi_affine")
        start_time = time()
        data, meta = func(*args, **kwargs)
        meta['timing'] = time() - start_time
        img = nib.Nifti1Image(data.astype(np.float32), dwi_affine)
        return img, meta

    wrapper_as_img.__signature__ = new_signature

    return wrapper_as_img
