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
    if isinstance(t, str) and len(t) < 1:
        return None
    elif isinstance(t, list):
        ls = []
        for e in t:
            ls.append(pyafq_str_to_val(e))
        return ls
    elif isinstance(t, str) and t[0] == "[":
        return eval(t)
    elif isinstance(t, str) and t[0] == "{":
        return eval(t)  # interpret as dictionary
    elif isinstance(t, str) and (
        "Image" in t or "Map" in t or "Dict" in t or "_bd(" in t
    ):
        try:
            definition_or_dict = eval(t)
        except NameError:
            return t
        if isinstance(definition_or_dict, Definition):
            return definition_or_dict
        elif isinstance(definition_or_dict, BundleDict):
            return definition_or_dict
        else:
            return t
    else:
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
    from AFQ import __version__
    from AFQ.api.participant import ParticipantAFQ

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

    if bval is False:
        bval = dwi.replace(".nii.gz", ".bval")
        if not op.exists(bval):
            bval = dwi.replace(".nii.gz", ".bvals")
        if not op.exists(bval):
            bval = dwi.replace(".nii", ".bval")
        if not op.exists(bval):
            bval = dwi.replace(".nii", ".bvals")
        if not op.exists(bval):
            raise FileNotFoundError(
                "Could not find bval file. Please specify the path to the bval file."
            )
    if bvec is False:
        bvec = dwi.replace(".nii.gz", ".bvec")
        if not op.exists(bvec):
            bvec = dwi.replace(".nii.gz", ".bvecs")
        if not op.exists(bvec):
            bvec = dwi.replace(".nii", ".bvec")
        if not op.exists(bvec):
            bvec = dwi.replace(".nii", ".bvecs")
        if not op.exists(bvec):
            raise FileNotFoundError(
                "Could not find bvec file. Please specify the path to the bvec file."
            )

    # extract arguments from file
    kwargs = {}

    for arg, default in f_arg_dict.items():
        if arg in [
            "dwi",
            "bvec",
            "bval",
            "t1",
            "o_folder",
            "verbose",
            "dry_run",
            "to_call",
        ]:
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
        if arg not in default_arg_dict:
            default_arg_dict[arg] = {}
        default_arg_dict[arg]["default"] = default

    if logger is not None and (verbose or dry_run):
        logger.info("The following arguments are recognized: " + str(kwargs))

    if dry_run:
        return

    # generate metadata file for this run
    default_arg_dict["pyAFQ"] = {}
    default_arg_dict["pyAFQ"]["utc_time_started"] = datetime.datetime.now().isoformat(
        "T"
    )
    default_arg_dict["pyAFQ"]["version"] = __version__
    default_arg_dict["pyAFQ"]["platform"] = platform.system()

    myafq = ParticipantAFQ(dwi, bval, bvec, t1, o_folder, **kwargs)

    afq_metadata_file = op.join(o_folder, "afq_metadata.toml")
    with open(afq_metadata_file, "w") as ff:
        ff.write(arg_dict_formatted(default_arg_dict))

    # call user specified function:
    if to_call == "all":
        myafq.export_all()
    else:
        myafq.export(to_call)

    # If you got this far, you can report on time ended and record that:
    default_arg_dict["pyAFQ"]["utc_time_ended"] = datetime.datetime.now().isoformat("T")
    with open(afq_metadata_file, "w") as ff:
        ff.write(arg_dict_formatted(default_arg_dict))


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
