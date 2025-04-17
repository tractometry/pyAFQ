
.. _kwargs_docs:

The pyAFQ API optional arguments
--------------------------------
You can run pyAFQ on either a subject or participant level
using pyAFQ's API objects, :class:`AFQ.api.group.GroupAFQ`
and :class:`AFQ.api.participant.ParticipantAFQ`. Either way,
these classes take additional optional arguments. These arguments
give the user control over each step of the tractometry pipeline,
allowing customizaiton of tractography, bundle recognition, registration,
etc. Here are all of these arguments and their descriptions, organized
into 5 sections:

Here are the arguments you can pass to kwargs, to customize the tractometry pipeline. They are organized into 5 sections.

==========================================================
DATA
==========================================================
min_bval: float
	Minimum b value you want to use from the dataset (other than b0), inclusive. If None, there is no minimum limit. Default: None

max_bval: float
	Maximum b value you want to use from the dataset (other than b0), inclusive. If None, there is no maximum limit. Default: None

filter_b: bool
	Whether to filter the DWI data based on min or max bvals. Default: True

b0_threshold: int
	The value of b under which it is considered to be b0. Default: 50.

robust_tensor_fitting: bool
	Whether to use robust_tensor_fitting when doing dti. Only applies to dti. Default: False

csd_response: tuple or None
	The response function to be used by CSD, as a tuple with two elements. The first is the eigen-values as an (3,) ndarray and the second is the signal value for the response function without diffusion-weighting (i.e. S0). If not provided, auto_response will be used to calculate these values. Default: None

csd_sh_order: int or None
	default: infer the number of parameters from the number of data volumes, but no larger than 8. Default: None

csd_lambda_: float
	weight given to the constrained-positivity regularization part of the deconvolution equation. Default: 1

csd_tau: float
	threshold controlling the amplitude below which the corresponding fODF is assumed to be zero. Ideally, tau should be set to zero. However, to improve the stability of the algorithm, tau is set to tau*100 percent of the mean fODF amplitude (here, 10 percent by default) (see [1]_). Default: 0.1

csd_fa_thr: float
	The threshold on the FA used to calculate the single shell auto response. Can be useful to reduce for baby subjects. Default: 0.7

gq_sampling_length: float
	Diffusion sampling length. Default: 1.2

rumba_wm_response: 1D or 2D ndarray or AxSymShResponse.
	Able to take response[0] from auto_response_ssst. default: array([0.0017, 0.0002, 0.0002])

rumba_gm_response: float
	Mean diffusivity for GM compartment. If None, then grey matter volume fraction is not computed. Default: 0.8e-3

rumba_csf_response: float
	Mean diffusivity for CSF compartment. If None, then CSF volume fraction is not computed. Default: 3.0e-3

rumba_n_iter: int
	Number of iterations for fODF estimation. Must be a positive int. Default: 600

opdt_sh_order: int
	Spherical harmonics order for OPDT model. Must be even. Default: 8

csa_sh_order: int
	Spherical harmonics order for CSA model. Must be even. Default: 8

sphere: Sphere class instance
	The sphere providing sample directions for the initial search of the maximal value of kurtosis. Default: 'repulsion100'

gtol: float
	This input is to refine kurtosis maxima under the precision of the directions sampled on the sphere class instance. The gradient of the convergence procedure must be less than gtol before successful termination. If gtol is None, fiber direction is directly taken from the initial sampled directions of the given sphere object. Default: 1e-2

brain_mask_definition: instance from `AFQ.definitions.image`
	This will be used to create the brain mask, which gets applied before registration to a template. If you want no brain mask to be applied, use FullImage. If None, use B0Image() Default: None

bundle_info: dict or BundleDict
	A dictionary or BundleDict for use in segmentation. See `Defining Custom Bundle Dictionaries` in the `usage` section of pyAFQ's documentation for details. If None, will get all appropriate bundles for the chosen segmentation algorithm. Default: None

reg_template_spec: str
	The target image data for registration. Can either be a Nifti1Image, a path to a Nifti1Image, or if "mni_T2", "dti_fa_template", "hcp_atlas", or "mni_T1", image data will be loaded automatically. If "hcp_atlas" is used, slr registration will be used and reg_subject should be "subject_sls". Default: "mni_T1"

reg_template_space_name: str
	Name to use in file names for the template space. Default: "mni"


==========================================================
MAPPING
==========================================================
mapping_definition: instance of `AFQ.definitions.mapping`
	This defines how to either create a mapping from each subject space to template space or load a mapping from another software. If creating a map, will register reg_subject and reg_template. If None, use SynMap() Default: None

reg_subject_spec: str
	The source image data to be registered. Can either be a Nifti1Image, an ImageFile, or str. if "b0", "dti_fa_subject", "subject_sls", or "power_map," image data will be loaded automatically. If "subject_sls" is used, slr registration will be used and reg_template should be "hcp_atlas". Default: "power_map"


==========================================================
SEGMENTATION
==========================================================
segmentation_params: dict
	The parameters for segmentation. Default: use the default behavior of the seg.Segmentation object.

profile_weights: str
	How to weight each streamline (1D) or each node (2D) when calculating the tract-profiles. If callable, this is a function that calculates weights. If None, no weighting will be applied. If "gauss", gaussian weights will be used. If "median", the median of values at each node will be used instead of a mean or weighted mean. Default: "gauss"

n_points_profile: int
	Number of points to resample each streamline to before calculating the tract-profiles. Default: 100

scalars: list of strings and/or scalar definitions
	List of scalars to use. Can be any of: "dti_fa", "dti_md", "dki_fa", "dki_md", "dki_awf", "dki_mk". Can also be a scalar from AFQ.definitions.image. Defaults for single shell data to ["dti_fa", "dti_md"], and for multi-shell data to ["dki_fa", "dki_md"]. Default: ['dti_fa', 'dti_md']


==========================================================
TRACTOGRAPHY
==========================================================
tracking_params: dict
	The parameters for tracking. Default: use the default behavior of the aft.track function. Seed mask and seed threshold, if not specified, are replaced with scalar masks from scalar[0] thresholded to 0.2. The ``seed_mask`` and ``stop_mask`` items of this dict may be ``AFQ.definitions.image.ImageFile`` instances. If ``tracker`` is set to "pft" then ``stop_mask`` should be an instance of ``AFQ.definitions.image.PFTImage``.

import_tract: dict or str or None
	BIDS filters for inputing a user made tractography file, or a path to the tractography file. If None, DIPY is used to generate the tractography. Default: None

tractography_ngpus: int
	Number of GPUs to use in tractography. If non-0, this algorithm is used for tractography, https://github.com/dipy/GPUStreamlines PTT, Prob can be used with any SHM model. Bootstrapped can be done with CSA/OPDT. Default: 0

chunk_size: int
	Chunk size for GPU tracking. Default: 100000


==========================================================
VIZ
==========================================================
sbv_lims_bundles: ndarray
	Of the form (lower bound, upper bound). Shading based on shade_by_volume will only differentiate values within these bounds. If lower bound is None, will default to 0. If upper bound is None, will default to the maximum value in shade_by_volume. Default: [None, None]

volume_opacity_bundles: float
	Opacity of volume slices. Default: 0.3

n_points_bundles: int or None
	n_points to resample streamlines to before plotting. If None, no resampling is done. Default: 40

sbv_lims_indiv: ndarray
	Of the form (lower bound, upper bound). Shading based on shade_by_volume will only differentiate values within these bounds. If lower bound is None, will default to 0. If upper bound is None, will default to the maximum value in shade_by_volume. Default: [None, None]

volume_opacity_indiv: float
	Opacity of volume slices. Default: 0.3

n_points_indiv: int or None
	n_points to resample streamlines to before plotting. If None, no resampling is done. Default: 40

virtual_frame_buffer: bool
	Whether to use a virtual fram buffer. This is neccessary if generating GIFs in a headless environment. Default: False

viz_backend_spec: str
	Which visualization backend to use. See Visualization Backends page in documentation for details https://tractometry.org/pyAFQ/usage/viz_backend.html One of {"fury", "plotly", "plotly_no_gif"}. Default: "plotly_no_gif"

