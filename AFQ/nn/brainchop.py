import subprocess

import numpy as np
import nibabel as nib
import nibabel.processing as nbp

from skimage.measure import label

import logging

from brainchop.utils import get_model

from tinygrad import Tensor


logger = logging.getLogger('AFQ')


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


def run_brainchop(t1_img, model_name):
    """
    Run the Brainchop command line interface with the provided arguments.
    """
    model = get_model(model_name)

    t1_img_conformed = nbp.conform(
        t1_img,
        out_shape=(256, 256, 256),
        voxel_size=(1.0, 1.0, 1.0),
        orientation="LIA")

    t1_data = t1_img_conformed.get_fdata()
    p02 = np.nanpercentile(t1_data, 2)
    p98 = np.nanpercentile(t1_data, 98)
    t1_data = np.clip(t1_data, p02, p98)
    t1_data = (t1_data - p02) / (p98 - p02)
    t1_data = (t1_data * 255.0)
    
    image = Tensor(t1_data.astype(np.float32)).rearrange(
        "... -> 1 1 ..."
    )

    logger.info(f"Running {model_name}...")
    output_channels = _run_brainchop_command(model, [image])

    output = (
        output_channels.argmax(axis=1)
        .squeeze(0)
        .numpy()
        .astype(np.uint8)
    )
    
    output = label(output, background=0)

    output_img = nbp.resample_from_to(
        nib.Nifti1Image(
            output.astype(np.float32),
            t1_img_conformed.affine),
        t1_img)

    return output_img
