import logging
from time import time

import dipy.data as dpd
import immlib
import nibabel as nib
import numpy as np
from trx.trx_file_memmap import TrxFile
from trx.trx_file_memmap import concatenate as trx_concatenate

import AFQ.tractography.tractography as aft
from AFQ.definitions.image import ScalarImage
from AFQ.definitions.utils import Definition
from AFQ.tasks.decorators import as_file
from AFQ.tasks.utils import get_default_args, with_name
from AFQ.tractography.utils import gen_seeds

try:
    import ray

    has_ray = True
except ModuleNotFoundError:
    has_ray = False

try:
    from AFQ.tractography.gputractography import gpu_track

    has_gputrack = True
except ModuleNotFoundError:
    has_gputrack = False

logger = logging.getLogger("AFQ")


def _meta_from_tracking_params(tracking_params, start_time, n_streamlines, seed, pve):
    meta_directions = {"det": "deterministic", "prob": "probabilistic"}
    meta = dict(
        TractographyClass="local",
        TractographyMethod=meta_directions.get(
            tracking_params["directions"], tracking_params["directions"]
        ),
        Count=n_streamlines,
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
    return meta


@immlib.calc("streamlines")
@as_file("_tractography", subfolder="tractography")
def streamlines(data_imap, seed, tissue_imap, fodf, tracking_params):
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
    this_tracking_params = tracking_params.copy()

    # get masks
    this_tracking_params["seed_mask"] = nib.load(seed).get_fdata()
    this_tracking_params["pve"] = tissue_imap["pve_internal"]

    is_trx = this_tracking_params.get("trx", False)

    num_chunks = data_imap["n_cpus"]

    if is_trx:
        start_time = time()
        dtype_dict = {"positions": np.float16, "offsets": np.uint32}
        if num_chunks and num_chunks > 1:
            if not has_ray:
                raise ImportError(
                    "Ray is required to perform tractography in"
                    "parallel, please install ray or remove the"
                    " 'num_chunks' arg"
                )

            @ray.remote
            class TractActor:
                def __init__(self):
                    self.TrxFile = TrxFile
                    self.aft = aft
                    self.objects = {}

                def trx_from_lazy_tractogram(self, lazyt_id, seed, dtype_dict):
                    id = self.objects[lazyt_id]
                    return self.TrxFile.from_lazy_tractogram(
                        id, seed, dtype_dict=dtype_dict
                    )

                def create_lazyt(self, id, *args, **kwargs):
                    self.objects[id] = self.aft.track(*args, **kwargs)
                    return id

                def delete_lazyt(self, id):
                    if id in self.objects:
                        del self.objects[id]

            actors = [TractActor.remote() for _ in range(num_chunks)]
            object_id = 1
            tracking_params_list = []

            # random seeds case
            if isinstance(
                this_tracking_params.get("n_seeds"), int
            ) and this_tracking_params.get("random_seeds"):
                remainder = this_tracking_params["n_seeds"] % num_chunks
                for i in range(num_chunks):
                    # create copy of tracking params
                    copy = this_tracking_params.copy()
                    n_seeds = this_tracking_params["n_seeds"]
                    copy["n_seeds"] = n_seeds // num_chunks
                    # add remainder to 1st list
                    if i == 0:
                        copy["n_seeds"] += remainder
                    tracking_params_list.append(copy)

            elif isinstance(this_tracking_params["n_seeds"], (np.ndarray, list)):
                n_seeds = np.array(this_tracking_params["n_seeds"])
                seed_chunks = np.array_split(n_seeds, num_chunks)
                tracking_params_list = [
                    this_tracking_params.copy() for _ in range(num_chunks)
                ]

                for i in range(num_chunks):
                    tracking_params_list[i]["n_seeds"] = seed_chunks[i]

            else:
                seeds = gen_seeds(
                    this_tracking_params["seed_mask"],
                    this_tracking_params["seed_threshold"],
                    this_tracking_params["n_seeds"],
                    this_tracking_params["thresholds_as_percentages"],
                    this_tracking_params["random_seeds"],
                    this_tracking_params["rng_seed"],
                    data_imap["dwi_affine"],
                )
                seed_chunks = np.array_split(seeds, num_chunks)
                tracking_params_list = [
                    this_tracking_params.copy() for _ in range(num_chunks)
                ]
                for i in range(num_chunks):
                    tracking_params_list[i]["n_seeds"] = seed_chunks[i]

            # create lazyt inside each actor
            tasks = [
                ray_actor.create_lazyt.remote(
                    object_id, fodf, **tracking_params_list[i]
                )
                for i, ray_actor in enumerate(actors)
            ]
            ray.get(tasks)

            # create trx from lazyt
            tasks = [
                ray_actor.trx_from_lazy_tractogram.remote(
                    object_id, seed, dtype_dict=dtype_dict
                )
                for ray_actor in actors
            ]
            sfts = ray.get(tasks)

            # cleanup objects
            tasks = [ray_actor.delete_lazyt.remote(object_id) for ray_actor in actors]
            ray.get(tasks)

            sft = trx_concatenate(sfts)
        else:
            lazyt = aft.track(fodf, **this_tracking_params)
            # Chunk size is number of streamlines tracked before saving to disk.
            sft = TrxFile.from_lazy_tractogram(
                lazyt, seed, dtype_dict=dtype_dict, chunk_size=1e5
            )
        n_streamlines = len(sft)

    else:
        start_time = time()
        sft = aft.track(fodf, **this_tracking_params)
        sft.to_vox()
        n_streamlines = len(sft.streamlines)

    return sft, _meta_from_tracking_params(
        tracking_params, start_time, n_streamlines, seed, tissue_imap["pve_internal"]
    )


@immlib.calc("fodf")
def fiber_odf(data_imap, tissue_imap, tracking_params):
    """
    Nifti Image containing the fiber orientation distribution function
    """
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


@immlib.calc("streamlines")
@as_file("_tractography", subfolder="tractography")
def gpu_tractography(
    data_imap,
    tracking_params,
    fodf,
    seed,
    tissue_imap,
    tractography_ngpus=0,
    chunk_size=100000,
):
    """
    full path to the complete, unsegmented tractography file

    Parameters
    ----------
    tractography_ngpus : int, optional
        Number of GPUs to use in tractography. If non-0,
        this algorithm is used for tractography,
        https://github.com/dipy/GPUStreamlines
        PTT, Prob can be used with any SHM model.
        Bootstrapped can be done with CSA/OPDT.
        Default: 0
    chunk_size : int, optional
        Chunk size for GPU tracking.
        Default: 100000
    """
    start_time = time()
    if tracking_params["directions"] == "boot":
        data = data_imap["data"]
    else:
        if isinstance(fodf, str):
            fodf = nib.load(fodf)
        data = fodf.get_fdata()

    pve = tissue_imap["pve_internal"]

    sphere = tracking_params["sphere"]
    if sphere is None:
        sphere = dpd.default_sphere

    sft = gpu_track(
        data,
        data_imap["gtab"],
        seed,
        pve,
        tracking_params["odf_model"],
        sphere,
        tracking_params["directions"],
        tracking_params["seed_threshold"],
        tracking_params["thresholds_as_percentages"],
        tracking_params["max_angle"],
        tracking_params["step_size"],
        tracking_params["n_seeds"],
        tracking_params["random_seeds"],
        tracking_params["rng_seed"],
        tracking_params["trx"],
        tractography_ngpus,
        chunk_size,
    )

    return sft, _meta_from_tracking_params(tracking_params, start_time, sft, seed, pve)


def get_tractography_plan(kwargs):
    if "tracking_params" in kwargs and not isinstance(kwargs["tracking_params"], dict):
        raise TypeError("tracking_params a dict")

    tractography_tasks = with_name([streamlines, fiber_odf])

    # use GPU accelerated tractography if asked for
    if "tractography_ngpus" in kwargs and kwargs["tractography_ngpus"] != 0:
        if not has_gputrack:
            raise ImportError(
                "Please install from ghcr.io/nrdg/pyafq_gpu"
                " docker file or from "
                "https://github.com/dipy/GPUStreamlines"
                " to use gpu-accelerated"
                " tractography"
            )
        tractography_tasks["streamlines_res"] = gpu_tractography
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

    if isinstance(odf_model, Definition):
        tractography_tasks["fiber_odf_res"] = immlib.calc("fodf")(
            odf_model.get_image_getter("tractography")
        )

    return immlib.plan(**tractography_tasks)
