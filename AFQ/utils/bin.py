import ast
import datetime
import os.path as op
import platform

from AFQ.api.bundle_dict import *  # interprets bundle_dicts loaded from command line # noqa F403
from AFQ.api.bundle_dict import BundleDict
from AFQ.api.utils import kwargs_descriptors
from AFQ.definitions.image import *  # interprets masks loaded from command line # noqa F403
from AFQ.definitions.mapping import *  # interprets mappings loaded from command line # noqa F403
from AFQ.definitions.utils import Definition
from AFQ.utils.docstring_parser import parse_numpy_docstring


def pyafq_str_to_val(t):
    if isinstance(t, list):
        return [pyafq_str_to_val(e) for e in t]

    if not isinstance(t, str):
        return t  # already an int, float, bool, etc.

    if isinstance(t, str) and t[0] == "[":
        return eval(t)

    if isinstance(t, str) and t[0] == "{":
        return eval(t)

    t = t.strip()
    if not t:
        return None

    # Strings that construct pyAFQ objects need eval
    if any(k in t for k in ("Image", "Map", "Dict", "_bd(")):
        try:
            val = eval(t)
        except (NameError, SyntaxError, TypeError):
            return t
        return val if isinstance(val, (Definition, BundleDict)) else t

    # Default to literal_eval for other strings
    try:
        return ast.literal_eval(t)
    except (ValueError, SyntaxError):
        return t


def val_to_formal(v):
    if v is None:
        return '""'
    elif isinstance(v, Definition):
        return f'"{v.str_formal()}"'
    elif isinstance(v, str):
        return f'"{v}"'
    elif isinstance(v, bool):
        if v:
            return "true"
        else:
            return "false"
    elif callable(v):
        return f'"{v.__name__}"'
    elif isinstance(v, dict):
        return f'"{v}"'
    elif isinstance(v, list):
        return f'"{v}"'
    else:
        return f"{v}"


def arg_dict_formatted(dictionary):
    desc = "# Use '' to indicate None\n# Wrap dictionaries in quotes\n"
    desc = desc + "# Wrap definition object instantiations in quotes\n\n"
    for section, args in dictionary.items():
        if section == "AFQ_desc":
            desc = "# " + dictionary["AFQ_desc"].replace("\n", "\n# ") + "\n\n" + desc
            continue
        desc = desc + f"[{section}]\n"
        for arg, arg_info in args.items():
            desc = desc + "\n"
            if isinstance(arg_info, dict) and "default" in arg_info:
                if "desc" in arg_info:
                    desc = desc + arg_info["desc"]
                desc = desc + f"{arg} = {val_to_formal(arg_info['default'])}\n"
            else:
                desc = desc + f"{arg} = {val_to_formal(arg_info)}\n"
        desc = desc + "\n"
    return desc + "\n"


# these params are handled internally in the qsiprep pipeline,
# not shown to the user (mostly BIDS filters stuff)
qsi_prep_ignore_params = [
    "bids_path",
    "bids_filters",
    "dwi_preproc_pipeline",
    "participant_labels",
    "output_dir",
]


def dict_to_json(dictionary):
    json = "                "
    local_ignore = qsi_prep_ignore_params.copy()
    for section, args in dictionary.items():
        if section == "AFQ_desc":
            continue
        for arg, arg_info in args.items():
            if arg in local_ignore:
                continue
            local_ignore.append(arg)
            if isinstance(arg_info, dict):
                json = json + f'"{arg}": {val_to_formal(arg_info["default"])}'
            else:
                json = json + f'"{arg}": {val_to_formal(arg_info)}'
            json = json + ",\n                "
    return json[:-18]  # remove trailing ,\n and indent


def func_dict_to_arg_dict(func_dict=None, logger=None):
    if func_dict is None:
        import AFQ.tractography.tractography as aft
        from AFQ.api.group import GroupAFQ
        from AFQ.recognition.recognize import recognize

        func_dict = {
            "BIDS": GroupAFQ.__init__,
            "Tractography": aft.track,
            "Segmentation": recognize,
        }

    arg_dict = {}
    for name, func in func_dict.items():
        docstr_parsed = parse_numpy_docstring(func)
        if name == "BIDS":
            arg_dict["AFQ_desc"] = docstr_parsed["description"]
        for arg, info in docstr_parsed["arguments"].items():
            try:
                section = name.upper() + "_PARAMS"
                desc = info["help"]
                if name != "BIDS" and "positional" in info and info["positional"]:
                    continue
            except (KeyError, IndexError) as error:
                if logger is not None:
                    logger.error(
                        "We are missing a valid description for the "
                        + f"{name} argument {arg}"
                    )
                raise error
            if section not in arg_dict.keys():
                arg_dict[section] = {}
            arg_dict[section][arg] = {}
            if "default" in info:
                default = info["default"]
            else:
                default = None
            arg_dict[section][arg]["default"] = default
            arg_dict[section][arg]["desc"] = desc

    for section, arg_info in kwargs_descriptors.items():
        section = section.upper()
        if section not in arg_dict.keys():
            arg_dict[section] = {}
        for arg, info in arg_info.items():
            if arg not in ["segmentation_params", "tracking_params"]:
                arg_dict[section][arg] = info

    for section, arg_info in arg_dict.items():
        if section == "AFQ_desc":
            continue
        for arg, _info in arg_info.items():
            desc = arg_dict[section][arg]["desc"]
            arg_dict[section][arg]["desc"] = ""
            for desc_line in desc.splitlines():
                f_desc_line = "# " + desc_line.strip() + "\n"
                arg_dict[section][arg]["desc"] = (
                    arg_dict[section][arg]["desc"] + f_desc_line
                )

    return arg_dict


_PARTICIPANT_CLI_ONLY = [
    "dwi",
    "bvec",
    "bval",
    "t1",
    "o_folder",
    "verbose",
    "dry_run",
    "to_call",
]

_BIDS_CLI_ONLY = [
    "bids_dir",
    "analysis_level",
    "participant_label",
    "session_id",
    "dwi_preproc_pipeline",
    "t1_preproc_pipeline",
    "bids_filter_file",
    "nprocs",
    "parallel_engine",
    "skip_bids_validation",
    "verbose",
    "dry_run",
    "to_call",
]


def cli_args_to_kwargs(cli_args, default_arg_dict, cli_only_args):
    """
    Convert parsed argparse arguments into kwargs for the AFQ API.

    Parameters
    ----------
    cli_args : argparse.Namespace
        Parsed command line arguments.
    default_arg_dict : dict
        Output of func_dict_to_arg_dict. Updated in place.
    cli_only_args : list of str
        Arguments for the CLI and mandatory arguments for AFQ
        that are not passed into the AFQ tasks system.

    Returns
    -------
    kwargs : dict
        Keyword arguments to pass to ParticipantAFQ or GroupAFQ.
        Tractography and segmentation parameters are nested under
        ``tracking_params`` and ``segmentation_params`` respectively.
    """
    f_arg_dict = vars(cli_args)

    special_args = {
        "SEGMENTATION_PARAMS": "segmentation_params",
        "TRACTOGRAPHY_PARAMS": "tracking_params",
    }

    special_args_assignment = {}
    for section_name, new_section_name in special_args.items():
        if section_name in default_arg_dict:
            for arg in default_arg_dict[section_name].keys():
                special_args_assignment[arg] = new_section_name

    kwargs = {}
    for arg, default in f_arg_dict.items():
        if arg in cli_only_args:
            continue
        val = pyafq_str_to_val(default)
        if val is None:
            continue
        if arg in special_args_assignment:
            section_name = special_args_assignment[arg]
            if section_name not in kwargs:
                kwargs[section_name] = {}
            kwargs[section_name][arg] = val
        else:
            kwargs[arg] = val
        for section, args in default_arg_dict.items():
            if section == "AFQ_desc" or not isinstance(args, dict):
                continue
            if arg in args and isinstance(args[arg], dict):
                args[arg]["default"] = default
                break

    return kwargs


def _find_gradient_file(dwi, exts, name):
    for nii_ext in (".nii.gz", ".nii"):
        if not dwi.endswith(nii_ext):
            continue
        for ext in exts:
            candidate = dwi[: -len(nii_ext)] + ext
            if op.exists(candidate):
                return candidate
    raise FileNotFoundError(
        f"Could not find {name} file. Please specify the path to the {name} file."
    )


def _start_metadata(default_arg_dict):
    from AFQ import __version__

    default_arg_dict["pyAFQ"] = {}
    default_arg_dict["pyAFQ"]["utc_time_started"] = datetime.datetime.now().isoformat(
        "T"
    )
    default_arg_dict["pyAFQ"]["version"] = __version__
    default_arg_dict["pyAFQ"]["platform"] = platform.system()


def _write_metadata(afq_metadata_file, default_arg_dict):
    with open(afq_metadata_file, "w") as ff:
        ff.write(arg_dict_formatted(default_arg_dict))


def parse_config_run_afq(
    dwi,
    bval,
    bvec,
    t1,
    o_folder,
    default_arg_dict,
    cli_args,
    to_call="export_all",
    logger=None,
    verbose=False,
    dry_run=False,
):
    from AFQ.api.participant import ParticipantAFQ

    if bval is False:
        bval = _find_gradient_file(dwi, (".bval", ".bvals"), "bval")
    if bvec is False:
        bvec = _find_gradient_file(dwi, (".bvec", ".bvecs"), "bvec")

    kwargs = cli_args_to_kwargs(cli_args, default_arg_dict, _PARTICIPANT_CLI_ONLY)

    if logger is not None and (verbose or dry_run):
        logger.info("The following arguments are recognized: " + str(kwargs))

    if dry_run:
        return

    _start_metadata(default_arg_dict)

    myafq = ParticipantAFQ(dwi, bval, bvec, t1, o_folder, **kwargs)

    afq_metadata_file = op.join(o_folder, "afq_metadata.toml")
    _write_metadata(afq_metadata_file, default_arg_dict)

    # call user specified function:
    if to_call == "all":
        myafq.export_all()
    else:
        myafq.export(to_call)

    # If you got this far, you can report on time ended and record that:
    default_arg_dict["pyAFQ"]["utc_time_ended"] = datetime.datetime.now().isoformat("T")
    _write_metadata(afq_metadata_file, default_arg_dict)


def run_bids_afq(
    bids_dir,
    analysis_level,
    default_arg_dict,
    cli_args,
    participant_label=None,
    session_id=None,
    dwi_preproc_pipeline="all",
    t1_preproc_pipeline=None,
    bids_filter_file=None,
    nprocs=1,
    parallel_engine="serial",
    skip_bids_validation=False,
    to_call="all",
    logger=None,
    verbose=False,
    dry_run=False,
):
    """
    BIDS-App style entry point

    Parameters
    ----------
    bids_dir : str
        Root of the BIDS dataset (containing ``derivatives/``).
    analysis_level : {"participant", "group"}
        ``participant`` runs the pipeline for each subject/session.
        ``group`` runs all and combines participant-level tract
        profiles into a single ``tract_profiles.csv``.
    """
    import json

    from AFQ.api.group import GroupAFQ

    bids_filters = {"suffix": "dwi"}
    if bids_filter_file is not None:
        with open(bids_filter_file) as ff:
            user_filters = json.load(ff)
        if not isinstance(user_filters, dict):
            raise TypeError("--bids-filter-file must contain a JSON object")
        # Accept either a flat dict of entities, or a qsiprep/fmriprep-style
        # file keyed by datatype, in which case we use the "dwi" entry.
        if "dwi" in user_filters and isinstance(user_filters["dwi"], dict):
            user_filters = user_filters["dwi"]
        bids_filters.update(user_filters)
    if session_id is not None:
        bids_filters["session"] = [str(s).removeprefix("ses-") for s in session_id]

    if participant_label is not None:
        participant_label = [str(p).removeprefix("sub-") for p in participant_label]

    parallel_params = {"engine": parallel_engine}
    if nprocs is not None and nprocs != 1:
        parallel_params["n_jobs"] = nprocs
        if parallel_engine == "serial":
            parallel_params["engine"] = "joblib"

    bids_layout_kwargs = {}
    if skip_bids_validation:
        bids_layout_kwargs = {"validate": False, "index_metadata": False}

    kwargs = cli_args_to_kwargs(cli_args, default_arg_dict, _BIDS_CLI_ONLY)

    group_kwargs = dict(
        bids_filters=bids_filters,
        dwi_preproc_pipeline=dwi_preproc_pipeline,
        t1_preproc_pipeline=t1_preproc_pipeline,
        participant_labels=participant_label,
        parallel_params=parallel_params,
        bids_layout_kwargs=bids_layout_kwargs,
    )

    if logger is not None and (verbose or dry_run):
        logger.info("The following BIDS arguments are recognized: " + str(group_kwargs))
        logger.info("The following arguments are recognized: " + str(kwargs))

    if dry_run:
        return

    _start_metadata(default_arg_dict)

    myafq = GroupAFQ(bids_dir, **group_kwargs, **kwargs)

    afq_metadata_file = op.join(myafq.afq_path, "afq_metadata.toml")
    _write_metadata(afq_metadata_file, default_arg_dict)

    if analysis_level == "group":
        myafq.combine_profiles()
    elif to_call == "all":
        myafq.export_all()
    else:
        myafq.export(to_call)

    default_arg_dict["pyAFQ"]["utc_time_ended"] = datetime.datetime.now().isoformat("T")
    _write_metadata(afq_metadata_file, default_arg_dict)


def generate_json(json_folder, overwrite=False, logger=None):
    json_file_our_trk = op.join(json_folder, "pyafq.json")
    json_file_their_trk = op.join(json_folder, "pyafq_input_trk.json")
    if not overwrite and (
        op.exists(json_file_our_trk) or op.exists(json_file_their_trk)
    ):
        raise FileExistsError(
            "Config file already exists. "
            + "If you want to overwrite this file,"
            + " add the argument --overwrite-config"
        )
    if logger is not None:
        logger.info("Generating pyAFQ full pipeline QSIprep json file.")
    qsi_spec_intro_our_trk = """{
    "description": "Use pyAFQ to perform the full Tractometry pipeline",
    "space": "T1w",
    "name": "pyAFQ_full",
    "atlases": [],
    "nodes": [
        {
            "name": "pyAFQ_full",
            "software": "pyAFQ",
            "action": "pyAFQ_full",
            "input": "qsiprep",
            "output_suffix": "PYAFQ_FULL",
            "parameters": {
                "use_external_tracking": false,
                "export": "all",
"""
    qsi_spec_intro_their_trk = """{
    "description": "Use pyAFQ to perform the Tractometry pipeline, with tractography from qsiprep",
    "space": "T1w",
    "name": "pyAFQ_import_trk",
    "atlases": [],
    "nodes": [
        {
            "name": "msmt_csd",
            "software": "MRTrix3",
            "action": "csd",
            "output_suffix": "msmtcsd",
            "input": "qsiprep",
            "parameters": {
                "mtnormalize": true,
                "response": {
                "algorithm": "dhollander"
                },
                "fod": {
                "algorithm": "msmt_csd",
                "max_sh": [4, 8, 8]
                }
            }
        },
        {
            "name": "track_ifod2",
            "software": "MRTrix3",
            "action": "tractography",
            "output_suffix": "ifod2",
            "input": "msmt_csd",
            "parameters": {
                "use_5tt": false,
                "use_sift2": true,
                "tckgen":{
                "algorithm": "iFOD2",
                "select": 1e6,
                "maxlen": 250,
                "minlen": 30,
                "power":0.33
                },
                "sift2":{}
            }
        },
        {
            "name": "pyAFQ_full",
            "software": "pyAFQ",
            "action": "pyAFQ_full",
            "input": "track_ifod2",
            "output_suffix": "PYAFQ_FULL_ET",
            "parameters": {
                "use_external_tracking": true,
                "export": "all",
"""  # noqa
    qsi_spec_outro = """
            }
        }
    ]
}"""
    import AFQ.tractography.tractography as aft
    from AFQ.recognition.cleaning import clean_bundle
    from AFQ.recognition.recognize import recognize

    func_dict = {
        "Tractography": aft.track,
        "Segmentation": recognize,
        "Cleaning": clean_bundle,
    }

    arg_dict = func_dict_to_arg_dict(func_dict, logger=logger)

    json_file = open(json_file_our_trk, "w")
    json_file.write(qsi_spec_intro_our_trk)
    json_file.write(dict_to_json(arg_dict))
    json_file.write(qsi_spec_outro)
    json_file.close()

    json_file = open(json_file_their_trk, "w")
    json_file.write(qsi_spec_intro_their_trk)
    json_file.write(dict_to_json(arg_dict))
    json_file.write(qsi_spec_outro)
    json_file.close()
