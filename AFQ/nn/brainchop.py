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


def _brainchop_reslice(tmp_t1_file, tmp_out_file, output_dtype, full_input):
    cmd = [
        "niimath", "-", "-reslice_nn", tmp_t1_file,
        "-gz", "1", tmp_out_file, "-odt", output_dtype]

    subprocess.run(cmd, input=full_input, check=True)


def _run_brainchop_command(func, args):
    """
    Run a Brainchop command line interface
    with the provided arguments, but with error handling.
    """
    try:
        return func(*args)
    except subprocess.CalledProcessError as e:
        logger.error("Command failed: %s", e.cmd)
        logger.error("Return code: %s", e.returncode)
        if e.stdout:
            if isinstance(e.stdout, bytes):
                logger.error("STDOUT:\n%s", e.stdout.decode())
            else:
                logger.error("STDOUT:\n%s", e.stdout)
        if e.stderr:
            if isinstance(e.stderr, bytes):
                logger.error("STDERR:\n%s", e.stderr.decode())
            else:
                logger.error("STDERR:\n%s", e.stderr)
        raise


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

        volume, header = _run_brainchop_command(conform, [tmp_t1_file])

        image = Tensor(volume.transpose((2, 1, 0)).astype(np.float32)).rearrange(
            "... -> 1 1 ..."
        )

        try:
            output_channels = _run_brainchop_command(model, [image])
        except Exception as e:
            if "clang" in str(e).lower():
                with Context(PYTHON=1):
                    output_channels = model(image)
            else:
                raise

        output = (
            output_channels.argmax(axis=1)
            .rearrange("1 x y z -> z y x")
            .numpy()
            .astype(np.uint8)
        )

        labels, new_header = _run_brainchop_command(
            bwlabel,
            [header, output])
        full_input = new_header + labels.tobytes()

        _run_brainchop_command(
            _brainchop_reslice,
            [
                tmp_t1_file,
                tmp_out_file,
                output_dtype,
                full_input
            ]
        )

        output_img = nib.load(tmp_out_file)
        # This line below forces the data into memory to avoid issues with
        # temporary files being deleted too early.
        # Otherwise, nibabel lazy-loads the data and the file gets deleted
        # before the data is accessed, because the file is in a temporary
        # Directory.
        output_img = nib.Nifti1Image(
            output_img.get_fdata().copy(),
            output_img.affine.copy())

    return output_img
