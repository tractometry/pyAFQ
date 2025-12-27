import logging
import math
import os.path as op
import tempfile
from time import time

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from AFQ.api.utils import (
    AFQclass_doc,
    check_attribute,
    export_all_helper,
    kwargs_descriptors,
    used_kwargs_exceptions,
    valid_exports_string,
)
from AFQ.definitions.mapping import SlrMap
from AFQ.tasks.data import get_data_plan
from AFQ.tasks.mapping import get_mapping_plan
from AFQ.tasks.segmentation import get_segmentation_plan
from AFQ.tasks.structural import get_structural_plan
from AFQ.tasks.tissue import get_tissue_plan
from AFQ.tasks.tractography import get_tractography_plan
from AFQ.tasks.utils import get_base_fname
from AFQ.tasks.viz import get_viz_plan
from AFQ.utils.bin import pyafq_str_to_val
from AFQ.utils.path import apply_cmd_to_afq_derivs
from AFQ.viz.utils import BEST_BUNDLE_ORIENTATIONS, get_eye, trim

__all__ = ["ParticipantAFQ"]


class ParticipantAFQ(object):
    f"""{AFQclass_doc}"""

    def __init__(
        self, dwi_data_file, bval_file, bvec_file, t1_file, output_dir, **kwargs
    ):
        """
        Initialize a ParticipantAFQ object.

        Parameters
        ----------
        dwi_data_file : str
            Path to DWI data file.
        bval_file : str
            Path to bval file.
        bvec_file : str
            Path to bvec file.
        t1_file : str
            Path to T1-weighted image file. Must already be registered
            to the DWI data, though not resampled.
        output_dir : str
            Path to output directory.
        kwargs : additional optional parameters
            You can set additional parameters for any step
            of the process. See :ref:`usage/kwargs` for more details.

        Examples
        --------
        api.ParticipantAFQ(
            dwi_data_file, bval_file, bvec_file, t1_file, output_dir,
            csd_sh_order_max=4)
        api.ParticipantAFQ(
            dwi_data_file, bval_file, bvec_file, t1_file, output_dir,
            reg_template_spec="mni_t2", reg_subject_spec="b0")

        Notes
        -----
        In tracking_params, parameters with the suffix mask which are also
        an image from AFQ.definitions.image will be handled automatically by
        the api.
        """
        if not isinstance(output_dir, str):
            raise TypeError("output_dir must be a str")
        if not isinstance(dwi_data_file, str):
            raise TypeError("dwi_data_file must be a str")
        if not isinstance(bval_file, str):
            raise TypeError("bval_file must be a str")
        if not isinstance(bvec_file, str):
            raise TypeError("bvec_file must be a str")
        if not op.exists(output_dir):
            raise ValueError(f"output_dir does not exist: {output_dir}")
        if "tractography_params" in kwargs:
            raise ValueError(
                (
                    "unrecognized parameter tractography_params, "
                    "did you mean tracking_params ?"
                )
            )

        self.logger = logging.getLogger("AFQ")

        # This is remembered to warn users
        # if their inputs are unused
        self.og_kwargs = kwargs.copy()

        self.kwargs = dict(
            dwi_data_file=dwi_data_file,
            bval_file=bval_file,
            bvec_file=bvec_file,
            t1_file=t1_file,
            output_dir=output_dir,
            base_fname=get_base_fname(output_dir, dwi_data_file),
            **kwargs,
        )
        self.make_workflow()

    def make_workflow(self):
        # construct immlib plans
        if "mapping_definition" in self.kwargs and isinstance(
            self.kwargs["mapping_definition"], SlrMap
        ):
            plans = {  # if using SLR map, do tractography first
                "structural": get_structural_plan(self.kwargs),
                "data": get_data_plan(self.kwargs),
                "tissue": get_tissue_plan(self.kwargs),
                "tractography": get_tractography_plan(self.kwargs),
                "mapping": get_mapping_plan(self.kwargs, use_sls=True),
                "segmentation": get_segmentation_plan(self.kwargs),
                "viz": get_viz_plan(self.kwargs),
            }
        else:
            plans = {  # Otherwise, do mapping first
                "structural": get_structural_plan(self.kwargs),
                "data": get_data_plan(self.kwargs),
                "tissue": get_tissue_plan(self.kwargs),
                "mapping": get_mapping_plan(self.kwargs),
                "tractography": get_tractography_plan(self.kwargs),
                "segmentation": get_segmentation_plan(self.kwargs),
                "viz": get_viz_plan(self.kwargs),
            }

        # Fill in defaults not already set
        for _, kwargs_in_section in kwargs_descriptors.items():
            for key, value in kwargs_in_section.items():
                if key not in self.kwargs:
                    self.kwargs[key] = value.get("default", None)
                self.kwargs[key] = pyafq_str_to_val(self.kwargs[key])

        # chain together a complete plan from individual plans
        used_kwargs = dict.fromkeys(self.og_kwargs, 0)
        previous_plans = {}
        for name, plan in plans.items():
            plan_kwargs = {}
            for key in plan.inputs:
                # Mark kwarg was used
                if key in used_kwargs:
                    used_kwargs[key] = 1

                # Construct kwargs for plan
                if key in self.kwargs:
                    plan_kwargs[key] = self.kwargs[key]
                elif key in previous_plans:
                    plan_kwargs[key] = previous_plans[key]
                elif name not in ["data", "structural"] and key == "dwi_affine":
                    # simplifies syntax to access commonly used dwi_affine
                    plan_kwargs[key] = previous_plans["data_imap"][key]
                else:
                    raise NotImplementedError(
                        f"Missing required parameter {key} for {name} plan"
                    )

            previous_plans[f"{name}_imap"] = plan(**plan_kwargs)

        for key, val in used_kwargs.items():
            if val == 0 and key not in used_kwargs_exceptions:
                self.logger.warning(
                    f"Parameter {key} was not used in any plan. "
                    "This may be a mistake, please check your parameters."
                )

        self.plans_dict = previous_plans

    def export(self, attr_name="help"):
        """
        Export a specific output. To print a list of available outputs,
        call export without arguments.

        Parameters
        ----------
        attr_name : str
            Name of the output to export. Default: "help"

        Returns
        -------
        output : any
            The specific output, or None if called without arguments.
        """
        section = check_attribute(attr_name)
        if not section:
            return None

        if section is None:
            return self.plans_dict[attr_name]
        return self.plans_dict[section][attr_name]

    def export_up_to(self, attr_name="help"):
        f"""
        Export all derivatives up to, but not including, the specified output.
        To print a list of available outputs,
        call export_up_to without arguments.
        {valid_exports_string}

        Parameters
        ----------
        attr_name : str
            Name of the output to export up to. Default: "help"
        """
        section = check_attribute(attr_name)
        if not section or section is None:
            return None

        calcdata = self.plans_dict[section].plan.calcdata
        idx = calcdata.sources[attr_name]
        if isinstance(idx, tuple):
            idx = idx[0]
        for inputs in calcdata.calcs[idx].inputs:
            self.export(inputs)

    def export_all(self, viz=True, xforms=True, indiv=True):
        f""" Exports all the possible outputs
        {valid_exports_string}

        Parameters
        ----------
        viz : bool
            Whether to output visualizations. This includes tract profile
            plots, a figure containing all bundles, and, if using the AFQ
            segmentation algorithm, individual bundle figures.
            Default: True
        xforms : bool
            Whether to output the reg_template image in subject space and,
            depending on if it is possible based on the mapping used, to
            output the b0 in template space.
            Default: True
        indiv : bool
            Whether to output individual bundles in their own files, in
            addition to the one file containing all bundles. If using
            the AFQ segmentation algorithm, individual ROIs are also
            output.
            Default: True
        """
        start_time = time()
        export_all_helper(self, xforms, indiv, viz)
        self.logger.info(f"Time taken for export all: {time() - start_time}")

    def participant_montage(self, images_per_row=2):
        """
        Generate montage of all bundles for a given subject.

        Parameters
        ----------
        images_per_row : int
            Number of bundle images per row in output file.
            Default: 2

        Returns
        -------
        filename of montage images
        """
        tdir = tempfile.gettempdir()

        all_fnames = []
        bundle_dict = self.export("bundle_dict")
        self.logger.info("Generating Montage...")
        viz_backend = self.export("viz_backend")
        best_scalar = self.export(self.export("best_scalar"))
        t1 = nib.load(self.export("t1_masked"))
        size = (images_per_row, math.ceil(len(bundle_dict) / images_per_row))
        for ii, bundle_name in enumerate(tqdm(bundle_dict)):
            flip_axes = [False, False, False]
            for i in range(3):
                flip_axes[i] = self.export("dwi_affine")[i, i] < 0

            figure = viz_backend.visualize_volume(
                t1, flip_axes=flip_axes, interact=False, inline=False
            )
            figure = viz_backend.visualize_bundles(
                self.export("bundles"),
                affine=t1.affine,
                shade_by_volume=best_scalar,
                color_by_direction=True,
                flip_axes=flip_axes,
                bundle=bundle_name,
                figure=figure,
                interact=False,
                inline=False,
            )

            view, direc = BEST_BUNDLE_ORIENTATIONS.get(bundle_name, ("Axial", "Top"))
            eye = get_eye(view, direc)

            this_fname = tdir + f"/t{ii}.png"
            if "plotly" in viz_backend.backend:
                figure.update_layout(
                    scene_camera=dict(
                        projection=dict(type="orthographic"),
                        up={"x": 0, "y": 0, "z": 1},
                        eye=eye,
                        center=dict(x=0, y=0, z=0),
                    ),
                    showlegend=False,
                )
                figure.write_image(this_fname, scale=4)

                # temporary fix for memory leak
                import plotly.io as pio

                pio.kaleido.scope._shutdown_kaleido()
            else:
                from dipy.viz import window

                direc = np.fromiter(eye.values(), dtype=int)
                data_shape = np.asarray(nib.load(self.export("b0")).get_fdata().shape)
                figure.set_camera(
                    position=direc * data_shape,
                    focal_point=data_shape // 2,
                    view_up=(0, 0, 1),
                )
                figure.zoom(0.5)
                window.snapshot(figure, fname=this_fname, size=(600, 600))

        def _save_file(curr_img):
            save_path = op.abspath(
                op.join(self.kwargs["output_dir"], "bundle_montage.png")
            )
            curr_img.save(save_path)
            all_fnames.append(save_path)

        this_img_trimmed = {}
        max_height = 0
        max_width = 0
        for ii, bundle_name in enumerate(bundle_dict):
            this_img = Image.open(tdir + f"/t{ii}.png")
            try:
                this_img_trimmed[ii] = trim(this_img)
            except IndexError:  # this_img is a picture of nothing
                this_img_trimmed[ii] = this_img

            text_sz = 70
            width, height = this_img_trimmed[ii].size
            height = height + text_sz
            result = Image.new(
                this_img_trimmed[ii].mode, (width, height), color=(255, 255, 255)
            )
            result.paste(this_img_trimmed[ii], (0, text_sz))
            this_img_trimmed[ii] = result

            draw = ImageDraw.Draw(this_img_trimmed[ii])
            draw.text(
                (0, 0),
                bundle_name,
                (0, 0, 0),
                font=ImageFont.truetype("Arial", text_sz),
            )

            if this_img_trimmed[ii].size[0] > max_width:
                max_width = this_img_trimmed[ii].size[0]
            if this_img_trimmed[ii].size[1] > max_height:
                max_height = this_img_trimmed[ii].size[1]

        curr_img = Image.new(
            "RGB", (max_width * size[0], max_height * size[1]), color="white"
        )

        for ii in range(len(bundle_dict)):
            x_pos = ii % size[0]
            _ii = ii // size[0]
            y_pos = _ii % size[1]
            _ii = _ii // size[1]
            this_img = this_img_trimmed[ii].resize((max_width, max_height))
            curr_img.paste(this_img, (x_pos * max_width, y_pos * max_height))

        _save_file(curr_img)
        return all_fnames

    def cmd_outputs(
        self, cmd="rm", dependent_on=None, up_to=None, exceptions=None, suffix=""
    ):
        """
        Perform some command some or all outputs of pyafq.
        This is useful if you change a parameter and need
        to recalculate derivatives that depend on it.
        Some examples: cp, mv, rm .
        -r will be automatically added when necessary.

        Parameters
        ----------
        cmd : str
            Command to run on outputs. Default: 'rm'
        dependent_on : str or None
            Which derivatives to perform command on .
            If None, perform on all.
            If "track", perform on all derivatives that depend on the
            tractography.
            If "recog", perform on all derivatives that depend on the
            bundle recognition.
            If "prof", perform on all derivatives that depend on the
            bundle profiling.
            Default: None
        up_to : str or None
            If None, will perform on all derivatives.
            If "track", will perform on all derivatives up to
            (but not including) tractography.
            If "recog", will perform on all derivatives up to
            (but not including) bundle recognition.
            If "prof", will perform on all derivatives up to
            (but not including) bundle profiling.
            Default: None
        exceptions : list of str
            Name outputs that the command should not be applied to.
            Default: []
        suffix : str
            Parts of command that are used after the filename.
            Default: ""
        """
        if exceptions is None:
            exceptions = []
        exception_file_names = []
        for exception in exceptions:
            file_name = self.export(exception)
            if isinstance(file_name, str):
                exception_file_names.append(file_name)
            else:
                self.logger.warning(
                    (
                        f"The exception '{exception}' does not correspond"
                        " to a filename and will be ignored."
                    )
                )

        apply_cmd_to_afq_derivs(
            self.kwargs["output_dir"],
            self.export("base_fname"),
            cmd=cmd,
            exception_file_names=exception_file_names,
            suffix=suffix,
            dependent_on=dependent_on,
            up_to=up_to,
        )

        # do not assume previous calculations are still valid
        # after file operations
        self.make_workflow()

    clobber = cmd_outputs  # alias for default of cmd_outputs
