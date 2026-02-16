import inspect
import os
import os.path as op

from AFQ.utils.path import drop_extension

__all__ = ["get_tp", "get_fname", "with_name", "get_base_fname"]


def get_tp(tp_name, structural_imap, data_imap, tissue_imap):
    if tp_name in data_imap:
        return data_imap[tp_name]
    elif tp_name in structural_imap:
        return structural_imap[tp_name]
    elif tissue_imap is not None and tp_name in tissue_imap:
        return tissue_imap[tp_name]
    else:
        raise NotImplementedError(f"tp_name {tp_name} not found")


def get_base_fname(output_dir, dwi_data_file):
    # setup up path and base file name for outputs
    # remove suffix and desc from dwi data file name
    used_key_list = ["desc", "space", "to", "from"]
    dwi_dfile_no_ext = op.join(output_dir, drop_extension(op.basename(dwi_data_file)))
    fname = op.dirname(dwi_dfile_no_ext) + "/"
    for key_val_pair in op.basename(dwi_dfile_no_ext).split("_"):
        if "-" in key_val_pair:
            key = key_val_pair.split("-")[0]
            if key not in used_key_list:
                fname = fname + key_val_pair + "_"
    fname = fname[:-1]
    return fname


def _split_path(path):
    parts = []
    while True:
        path, folder = os.path.split(path)
        if folder:
            parts.append(folder)
        else:
            if path:
                parts.append(path)
            break
    return parts[::-1]


def check_onnxruntime(model_name, alternative_text):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            f"onnxruntime is required to run the {model_name} model. "
            f"When we tried to import onnxruntime, we got the "
            f"following error:\n{e}\n"
            f"Please install onnxruntime to use this feature, "
            f"by doing `pip install onnxruntime` or `pip install pyAFQ[nn]`. "
            f"{alternative_text}\n"
            "If there are still issues, post an issue on "
            "https://github.com/tractometry/pyAFQ/issues"
        ) from e
    return ort


def get_fname(base_fname, suffix, subfolder=None):
    if subfolder is None:
        return base_fname + suffix
    elif subfolder == "..":
        base_dir = op.dirname(base_fname)
        base_fname = op.basename(base_fname)
        folders = _split_path(base_dir)
        if folders[-1] == "dwi":
            if len(folders) > 1 and "sub-" in folders[-2] or "ses-" in folders[-2]:
                return op.join(*folders[:-2], folders[-2] + suffix)
            else:
                return op.join(*folders[:-1], base_fname + suffix)
        else:
            return op.join(*folders[:-1], folders[-1] + suffix)
    else:
        base_dir = op.dirname(base_fname)
        base_fname = op.basename(base_fname)
        subfolder = op.join(base_dir, subfolder)
        os.makedirs(subfolder, exist_ok=True)
        return op.join(subfolder, base_fname + suffix)


# Turn list of tasks into dictionary with names for each task
def with_name(task_list):
    return {f"{task.__name__}_res": task for task in task_list}


def get_default_args(func):
    return {
        k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def str_to_desc(string):
    return string.replace("-", "").replace("_", "").replace(" ", "")
