import subprocess
import tempfile
import os

import numpy as np
import nibabel as nib

from brainchop.utils import get_model
from brainchop.niimath import (
    conform,
    bwlabel)

from tinygrad import Tensor
from tinygrad.helpers import Context

import logging


logger = logging.getLogger('AFQ')


def run_brainchop(t1_img, model):
    """
    Run the Brainchop command line interface with the provided arguments.
    """
    model = get_model(model)
    output_dtype = "char"

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_t1_file = f"{temp_dir}/t1.nii.gz"
        tmp_out_file = f"{temp_dir}/output.nii.gz"

        t1_data = t1_img.get_fdata()
        t1_data = np.clip(t1_data, 0, t1_data.max())
        nib.save(nib.Nifti1Image(t1_data, t1_img.affine), tmp_t1_file)

        volume, header = conform(tmp_t1_file)

        image = Tensor(volume.transpose((2, 1, 0)).astype(np.float32)).rearrange(
            "... -> 1 1 ..."
        )

        use_cpu = (os.getenv("CC") is not None) or \
            (os.getenv("clang") is not None)

        with Context(CPU=use_cpu):
            output_channels = model(image)

        output = (
            output_channels.argmax(axis=1)
            .rearrange("1 x y z -> z y x")
            .numpy()
            .astype(np.uint8)
        )

        labels, new_header = bwlabel(header, output)
        full_input = new_header + labels.tobytes()

        cmd = [
            "niimath", "-", "-reslice_nn", tmp_t1_file,
            "-gz", "1", tmp_out_file, "-odt", output_dtype]

        subprocess.run(cmd, input=full_input, check=True)

        output_img = nib.load(tmp_out_file)
        output_img = nib.Nifti1Image(
            output_img.get_fdata().copy(),
            output_img.affine.copy())  # force into memory

    return output_img
