import logging
from time import time

import immlib
import nibabel as nib
import numpy as np
from trx.trx_file_memmap import TrxFile

import AFQ.tractography.tractography as aft
from AFQ.definitions.image import ScalarImage
from AFQ.definitions.utils import Definition
from AFQ.tasks.decorators import as_file
from AFQ.tasks.utils import get_default_args, with_name

logger = logging.getLogger("AFQ")


def _meta_from_tracking_params(tracking_params, start_time, seed, pve, n_streamlines=0):
    meta_directions = {"det": "deterministic", "prob": "probabilistic"}
    meta = dict(
        TractographyClass="local",
        TractographyMethod=meta_directions.get(
            tracking_params["directions"], tracking_params["directions"]
        ),
        Seeding=dict(
            ROI=seed,
            n_seeds=tracking_params["n_seeds"],
            random_seeds=tracking_params["random_seeds"],
        ),
        Constraints=dict(PVE=pve),
        Parameters=dict(
            Units="mm",
            StepSize=tracking_params["step_size"],
            MinimumLength=tracking_params["minlen"],
            MaximumLength=tracking_params["maxlen"],
            Unidirectional=False,
        ),
        Timing=time() - start_time,
    )
    if n_streamlines != 0:
        meta["Count"] = n_streamlines
    return meta


def _fiber_odf(data_imap, tissue_imap, tracking_params):
    odf_model = tracking_params["odf_model"]
    if isinstance(odf_model, str):
        calc_name = f"{odf_model.lower()}_params"
        if calc_name in data_imap:
            params_file = data_imap[calc_name]
        elif calc_name in tissue_imap:
            params_file = tissue_imap[calc_name]
        else:
            raise ValueError((f"Could not find {odf_model}"))
    else:
        raise TypeError(("odf_model must be a string or Definition"))

    return params_file


@immlib.calc("streamlines")
@as_file("_tractography", subfolder="tractography")
def streamlines(
    structural_imap, data_imap, seed, tissue_imap, citations, tracking_params
):
    """
    full path to the complete, unsegmented tractography file

    Parameters
    ----------
    tracking_params : dict, optional
        The parameters for tracking. Defaults to using the default behavior of
        the aft.track function. Seed mask and seed threshold, if not
        specified, are replaced with scalar masks from scalar[0]
        thresholded to 0.2. The ``seed_mask`` items of
        this dict may be ``AFQ.definitions.image.ImageFile`` instances.
    """
    citations.add("girard2014towards")
    citations.add("smith2012anatomically")

    this_tracking_params = tracking_params.copy()
    fodf = _fiber_odf(data_imap, tissue_imap, tracking_params)

    # get masks
    this_tracking_params["seed_mask"] = nib.load(seed).get_fdata()

    is_trx = this_tracking_params.get("trx", False)

    if is_trx:
        start_time = time()
        dtype_dict = {"positions": np.float32, "offsets": np.uint32}

        lazyt = aft.track(
            fodf,
            tissue_imap["pve_internal"],
            structural_imap["n_threads"],
            **this_tracking_params,
        )

        if (
            this_tracking_params["directions"] == "prob"
            or this_tracking_params["directions"] == "ptt"
        ):
            # We do not count these as we go yet,
            # this needs to be implemented in GPUStreamlines
            n_streamlines = 0
            sft = lazyt
        else:
            # Chunk size is number of streamlines tracked before saving to disk.
            sft = TrxFile.from_lazy_tractogram(
                lazyt,
                seed,
                dtype_dict=dtype_dict,
                chunk_size=1e5,
                extra_buffer=int(1e6),
            )
            n_streamlines = len(sft)

    else:
        start_time = time()
        sft = aft.track(
            fodf,
            tissue_imap["pve_internal"],
            structural_imap["n_threads"],
            **this_tracking_params,
        )
        n_streamlines = len(sft.streamlines)

    if len(sft) == 0:
        raise ValueError(
            "No streamlines were generated. "
            "Please check your tracking parameters and input data."
        )

    return sft, _meta_from_tracking_params(
        tracking_params,
        start_time,
        seed,
        tissue_imap["pve_internal"],
        n_streamlines,
    )


@immlib.calc("streamlines")
def custom_tractography(import_tract=None):
    """
    full path to the complete, unsegmented tractography file

    Parameters
    ----------
    import_tract : dict or str or None, optional
        BIDS filters for inputing a user made tractography file,
        or a path to the tractography file. If None, DIPY is used
        to generate the tractography.
        Default: None
    """
    if not isinstance(import_tract, str):
        raise TypeError("import_tract must be" + " either a dict or a str")
    return import_tract


def get_tractography_plan(kwargs):
    if "tracking_params" in kwargs and not isinstance(kwargs["tracking_params"], dict):
        raise TypeError("tracking_params a dict")

    tractography_tasks = with_name([streamlines])

    # use imported tractography if given
    if "import_tract" in kwargs and kwargs["import_tract"] is not None:
        tractography_tasks["streamlines_res"] = custom_tractography
        if "trx" not in kwargs.get("tracking_params", {}):
            if "tracking_params" not in kwargs:
                kwargs["tracking_params"] = {}
            kwargs["tracking_params"]["trx"] = kwargs["import_tract"][-4:] == ".trx"

    # determine reasonable defaults
    best_scalar = kwargs["scalars"][0]
    fa_found = False
    for scalar in kwargs["scalars"]:
        if isinstance(scalar, str):
            if "fa" in scalar:
                best_scalar = scalar
                fa_found = True
                break
        else:
            if "fa" in scalar.get_name():
                best_scalar = scalar
                fa_found = True
                break
    if not fa_found:
        logger.warning(
            "FA not found in list of scalars, will use first scalar"
            " for visualizations"
            " unless these are also specified"
        )
    kwargs["best_scalar"] = best_scalar

    default_tracking_params = get_default_args(aft.track)

    # Replace the defaults only for kwargs for which a non-default value
    # was given:
    if "tracking_params" in kwargs:
        for k in kwargs["tracking_params"]:
            default_tracking_params[k] = kwargs["tracking_params"][k]

    kwargs["tracking_params"] = default_tracking_params
    if isinstance(kwargs["tracking_params"]["odf_model"], str):
        kwargs["tracking_params"]["odf_model"] = kwargs["tracking_params"][
            "odf_model"
        ].upper()

    if kwargs["tracking_params"]["seed_mask"] is None:
        kwargs["tracking_params"]["seed_mask"] = ScalarImage("wm_gm_interface")
        kwargs["tracking_params"]["seed_threshold"] = 0.5

    seed_mask = kwargs["tracking_params"]["seed_mask"]
    odf_model = kwargs["tracking_params"]["odf_model"]

    if isinstance(seed_mask, Definition):
        tractography_tasks["export_seed_mask_res"] = immlib.calc("seed")(
            as_file("_desc-seed_mask.nii.gz", subfolder="tractography")(
                seed_mask.get_image_getter("tractography")
            )
        )
    else:
        raise TypeError(
            "seed_mask must be an AFQ Definition when using the GroupAFQ or "
            "ParticipantAFQ API. Consider using "
            'ScalarImage("wm_gm_interface"), ThresholdedScalarImage, '
            "RoiImage, or another AFQ Image definition."
        )

    if isinstance(odf_model, Definition):
        tractography_tasks["fiber_odf_res"] = immlib.calc("fodf")(
            odf_model.get_image_getter("tractography")
        )

    n_seeds = kwargs["tracking_params"]["n_seeds"]
    if (
        kwargs["tracking_params"]["random_seeds"]
        and isinstance(n_seeds, int)
        and n_seeds <= 20
    ):
        raise ValueError(
            "Using random seeds with a low number of seeds is not recommended."
            " Please increase n_seeds or set random_seeds to False."
            " A recommended number of seeds when using random seeds is 1e7."
        )

    return immlib.plan(**tractography_tasks)
