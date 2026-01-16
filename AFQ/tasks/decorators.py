import functools
import inspect
import logging
import os.path as op
from time import time

import nibabel as nib
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram

try:
    from trx.io import save as save_trx
    from trx.trx_file_memmap import TrxFile

    has_trx = True
except ModuleNotFoundError:
    has_trx = False

import numpy as np

from AFQ.tasks.utils import get_fname
from AFQ.utils.path import drop_extension, read_json, write_json

# These should only be used with immlib.calc
__all__ = ["as_file", "as_fit_deriv", "as_img"]


logger = logging.getLogger("AFQ")
logger.setLevel(logging.INFO)


def get_new_signature(og_func, needed_args):
    sig = inspect.signature(og_func)
    param_dict = sig.parameters

    existing_args = set(param_dict)
    new_param_names = [arg for arg in needed_args if arg not in existing_args]

    new_params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in new_param_names
    ]

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
    Decorator to save function outputs (img and meta) as files.
    'suffix' can be:
        - A string (suffix)
        - A list of strings (suffixes)
        - A list of tuples [(suffix, subfolder), ...]
    """
    if isinstance(suffix, str):
        output_specs = [(suffix, subfolder)]
    else:
        output_specs = [
            (spec if isinstance(spec, (list, tuple)) else (spec, subfolder))
            for spec in suffix
        ]

    def _as_file(func):
        new_signature, new_params = get_new_signature(
            func, ["base_fname", "output_dir", "tracking_params"]
        )

        @functools.wraps(func)
        def wrapper_as_file(*args, **kwargs):
            base_fname = get_param(kwargs, new_params, "base_fname")
            output_dir = get_param(kwargs, new_params, "output_dir")
            tracking_params = get_param(kwargs, new_params, "tracking_params")

            resolved_files = []
            calculation_name = ""
            for suffix, sub in output_specs:
                fpath = get_fname(base_fname, suffix, subfolder=sub)
                calculation_name += f"{suffix}, "

                # Determine extension for tractography files if missing
                if not op.splitext(fpath)[1]:
                    ext = ".trx" if tracking_params.get("trx", False) else ".trk"
                    fpath += ext
                resolved_files.append(fpath)
            calculation_name = calculation_name.rstrip(", ")

            if all(op.exists(f) for f in resolved_files):
                return resolved_files if len(resolved_files) > 1 else resolved_files[0]

            logger.info(f"Calculating {calculation_name}...")

            try:
                results = func(*args, **kwargs)

                if len(output_specs) == 1:
                    results = [results]
            except Exception:
                logger.error(f"Error in task: {func.__qualname__}")
                raise

            for i, (data, meta) in enumerate(results):
                this_file = resolved_files[i]
                this_suffix, this_sub = output_specs[i]

                logger.info(f"{this_suffix} completed. Saving to {this_file}")
                if isinstance(data, nib.Nifti1Image):
                    nib.save(data, this_file)
                elif isinstance(data, StatefulTractogram):
                    save_tractogram(data, this_file, bbox_valid_check=False)
                elif isinstance(data, np.ndarray):
                    np.save(this_file, data)
                elif has_trx and isinstance(data, TrxFile):
                    save_trx(data, this_file)
                else:
                    data.to_csv(this_file)

                # these are used to determine dependencies
                # when clobbering derivatives
                if "_desc-profiles" in this_suffix or "viz" in inspect.getfile(func):
                    meta["dependent"] = "prof"
                elif "segmentation" in inspect.getfile(
                    func
                ) or "mapping" in inspect.getfile(func):
                    meta["dependent"] = "rec"
                elif "tractography" in inspect.getfile(func):
                    meta["dependent"] = "trk"
                else:
                    meta["dependent"] = "dwi"

                # modify meta source to be relative
                if "source" in meta:
                    meta["source"] = op.relpath(meta["source"], output_dir)

                meta_fname = get_fname(
                    base_fname,
                    f"{drop_extension(this_suffix)}.json",
                    subfolder=this_sub,
                )
                write_json(meta_fname, meta)
            return resolved_files if len(resolved_files) > 1 else resolved_files[0]

        wrapper_as_file.__signature__ = new_signature

        return wrapper_as_file

    return _as_file


def as_fit_deriv(tf_name):
    """
    return data as nibabel image, meta with params information
    """

    def _as_fit_deriv(func):
        new_signature, new_params = get_new_signature(
            func, ["dwi_affine", f"{tf_name.lower()}_params"]
        )

        @functools.wraps(func)
        def wrapper_as_fit_deriv(*args, **kwargs):
            dwi_affine = get_param(kwargs, new_params, "dwi_affine")
            params = get_param(kwargs, new_params, f"{tf_name.lower()}_params")
            params_meta = read_json(drop_extension(params) + ".json")
            img_meta = {}
            if "Model" in params_meta:
                img_meta["Model"] = params_meta["Model"]
            img_meta["Source"] = params

            results = func(*args, **kwargs)

            if len(results) == 1:
                data = results[0]
            else:
                data, meta = results

            img_meta.update(meta)

            img = nib.Nifti1Image(data, dwi_affine)
            return img, img_meta

        wrapper_as_fit_deriv.__signature__ = new_signature

        return wrapper_as_fit_deriv

    return _as_fit_deriv


def as_img(func):
    """
    Decorator to convert function output (ndarray, meta) into (Nifti1Image, meta).
    Supports functions returning a single tuple or a list of tuples.
    """
    new_signature, new_params = get_new_signature(func, ["dwi_affine"])

    @functools.wraps(func)
    def wrapper_as_img(*args, **kwargs):
        dwi_affine = get_param(kwargs, new_params, "dwi_affine")

        start_time = time()
        results = func(*args, **kwargs)
        elapsed = time() - start_time

        is_single_output = isinstance(results[0], np.ndarray)

        if is_single_output:
            outputs = [results]
        else:
            outputs = results

        processed_outputs = []
        for data, meta in outputs:
            meta["timing"] = elapsed

            img = nib.Nifti1Image(data.astype(np.float32), dwi_affine)
            processed_outputs.append((img, meta))

        return processed_outputs[0] if is_single_output else processed_outputs

    wrapper_as_img.__signature__ = new_signature
    return wrapper_as_img
