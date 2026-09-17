import json
import logging
import os.path as op
from argparse import Namespace

import pytest

import AFQ.utils.bin as afb
from AFQ.definitions.image import ImageFile


def _make_args(arg_dict, **overrides):
    """Build a Namespace as argparse would, with all dynamic defaults."""
    ns = {}
    for section, args in arg_dict.items():
        if section in ("AFQ_desc", "BIDS_PARAMS"):
            continue
        for arg, info in args.items():
            ns[arg] = info["default"]
    ns.update(overrides)
    return Namespace(**ns)


def test_cli_args_to_kwargs_nests_special_sections():
    arg_dict = afb.func_dict_to_arg_dict()
    args = _make_args(
        arg_dict,
        rng_seed="2026",
        return_idx="True",
        seed_mask='ImageFile(suffix="mask")',
        min_bval="500",
        verbose=True,
        dry_run=True,
    )
    kwargs = afb.cli_args_to_kwargs(args, arg_dict, ["verbose", "dry_run"])

    assert kwargs["tracking_params"]["rng_seed"] == 2026
    assert isinstance(kwargs["tracking_params"]["seed_mask"], ImageFile)
    assert kwargs["segmentation_params"]["return_idx"] is True
    assert kwargs["min_bval"] == 500
    assert "verbose" not in kwargs and "dry_run" not in kwargs
    # metadata dict is updated with the value actually used
    assert arg_dict["TRACTOGRAPHY_PARAMS"]["rng_seed"]["default"] == "2026"


def test_run_bids_afq_dry_run(tmp_path, caplog):
    bids_dir = tmp_path / "bids"
    (bids_dir / "derivatives").mkdir(parents=True)
    with open(bids_dir / "dataset_description.json", "w") as ff:
        json.dump({"Name": "test", "BIDSVersion": "1.8.0"}, ff)
    filter_file = tmp_path / "filters.json"
    with open(filter_file, "w") as ff:
        json.dump({"dwi": {"acquisition": "64dir", "space": "T1w"}}, ff)

    arg_dict = afb.func_dict_to_arg_dict()
    args = _make_args(arg_dict, rng_seed="1")
    logger = logging.getLogger("AFQ")

    with caplog.at_level(logging.INFO, logger="AFQ"):
        afb.run_bids_afq(
            str(bids_dir),
            "participant",
            arg_dict,
            args,
            participant_label=["sub-01", "02"],
            session_id=["ses-a"],
            dwi_preproc_pipeline="qsiprep",
            bids_filter_file=str(filter_file),
            nprocs=4,
            skip_bids_validation=True,
            logger=logger,
            dry_run=True,
        )

    msg = caplog.text
    assert "'participant_labels': ['01', '02']" in msg
    assert "'session': ['a']" in msg
    assert "'acquisition': '64dir'" in msg
    assert "'engine': 'joblib', 'n_jobs': 4" in msg
    assert "'validate': False" in msg
    assert "'rng_seed': 1" in msg
    # nothing written on a dry run
    assert not op.exists(tmp_path / "out")


def test_run_bids_afq_bad_filter_file(tmp_path):
    filter_file = tmp_path / "filters.json"
    with open(filter_file, "w") as ff:
        json.dump(["not", "a", "dict"], ff)
    arg_dict = afb.func_dict_to_arg_dict()
    args = _make_args(arg_dict)
    with pytest.raises(TypeError):
        afb.run_bids_afq(
            str(tmp_path),
            "participant",
            arg_dict,
            args,
            bids_filter_file=str(filter_file),
            dry_run=True,
        )
