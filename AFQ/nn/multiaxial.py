import onnxruntime as ort
from tqdm import tqdm
import numpy as np
import nibabel as nib
import nibabel.processing as nbp
from dipy.align import resample

from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, binary_dilation

from AFQ.data.fetch import fetch_multiaxial_models, afq_home

import logging
import os.path as op

logger = logging.getLogger('AFQ')


__all__ = ["run_multiaxial", "extract_brain_mask", "multiaxial"]


def multiaxial(img,
               model_sagittal, model_axial, model_coronal,
               consensus_model):
    """
    Perform multiaxial segmentation using three ONNX models
    and a consensus model [1].

    Parameters
    ----------
    img : ndarray
        3D T1 image to segment.
    model_sagittal : str
        Path to sagittal ONNX model.
    model_axial : str
        Path to axial ONNX model.
    model_coronal : str
        Path to coronal ONNX model.
    consensus_model : str
        Path to consensus ONNX model.
    
    Returns
    -------
    pred : ndarray
        Segmentation labels for each coordinate.

    References
    ----------
    [1] Birnbaum, Andrew M., et al. "Full-head segmentation of MRI
    with abnormal brain anatomy: model and data release." Journal of
    Medical Imaging 12.5 (2025): 054001-054001.
    """
    img = img.astype(np.float32)
    coords = _create_coord_grid().astype(np.float32)
    pbar = tqdm(total=4)

    input_ = img[..., None]
    sagittal_results = _run_onnx_model(model_sagittal, input_, coords)
    pbar.update(1)

    input_ = np.swapaxes(img, 0, 1)[..., None]
    coronal_results = np.swapaxes(
        _run_onnx_model(model_coronal, input_, coords),
        0, 1)
    pbar.update(1)

    input_ = np.transpose(img, (2, 0, 1))[..., None]
    axial_results = np.transpose(
        _run_onnx_model(model_axial, input_, coords),
        (1, 2, 0, 3))
    pbar.update(1)

    X = np.concatenate([
        img[..., None], sagittal_results,
        coronal_results, axial_results],-1)
    sess = ort.InferenceSession(
        consensus_model,
        providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    yhat = sess.run([output_name], {input_name: X[None, ...]})[0]
    pbar.update(1)
    pbar.close()
    pred = np.argmax(yhat[0],-1)

    return pred

def _run_onnx_model(model, input_, coords):
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    coord_name = sess.get_inputs()[1].name
    output_name = sess.get_outputs()[0].name
    results = np.zeros((256, 256, 256, 7), np.float32)
    for ii in tqdm(range(input_.shape[0]), leave=False):
        results[ii] = sess.run([output_name], {
            input_name: input_[ii:ii+1],
            coord_name: coords[ii:ii+1]})[0]
    return results

def _create_coord_grid():
    x, y, z = (256, 256, 256)
    ac = (128, 128, 128)  # assume anterior commissure
    meshgrid = np.meshgrid(
        np.linspace(0, x - 1, x),
        np.linspace(0, y - 1, y),
        np.linspace(0, z - 1, z), indexing='ij')
    coordinates = np.stack(meshgrid, axis=-1) - np.array(ac)
    coords = np.concatenate([
        coordinates,
        np.ones((
            coordinates.shape[0],
            coordinates.shape[1],
            coordinates.shape[2], 1))], axis=-1)

    coords = coords[:,:,:,:3]
    coords = coords/256.
    return coords.astype(np.int16)

def _get_multiaxial_model():
    model_dict = {}
    for model_name in [
            "sagittal_model",
            "axial_model",
            "coronal_model",
            "consensus_model"]:
        model_path = op.join(
            afq_home,
            'multiaxial_models_onnx',
            model_name + ".onnx")
        if not op.exists(model_path):
            fetch_multiaxial_models()
        model_dict[model_name] = model_path
    return model_dict


def run_multiaxial(t1_img):
    """
    Run the multiaxial model.
    """
    model_dict = _get_multiaxial_model()

    t1_img_conformed = nbp.conform(
        t1_img,
        out_shape=(256, 256, 256),
        voxel_size=(1.0, 1.0, 1.0),
        orientation="RAS")

    t1_data = t1_img_conformed.get_fdata()
    p02 = np.nanpercentile(t1_data, 2)
    p98 = np.nanpercentile(t1_data, 98)
    t1_data = np.clip(t1_data, p02, p98)
    t1_data = (t1_data - p02) / (p98 - p02)
    
    logger.info("Running multiaxial T1w segmentation...")
    output = multiaxial(
        t1_data,
        model_dict["sagittal_model"],
        model_dict["axial_model"],
        model_dict["coronal_model"],
        model_dict["consensus_model"])

    output_img = nbp.resample_from_to(
        nib.Nifti1Image(
            output.astype(np.uint8),
            t1_img_conformed.affine),
        t1_img)

    return output_img


def extract_brain_mask(predictions):
    """
    Extract brain mask from multiaxial predictions.

    Parameters
    ----------
    predictions : Nifti1Image
        Multiaxial segmentation predictions.

    Returns
    -------
    bm_data : ndarray
        Brain mask data.
    """
    gm = predictions.get_fdata() == 2
    wm = predictions.get_fdata() == 3
    csf = predictions.get_fdata() == 4

    bm_data = (wm | gm | csf)

    return bm_data


def extract_pve(prediction):
    """
    Extract PVE maps from multiaxial predictions.

    Parameters
    ----------
    prediction : Nifti1Image
        Multiaxial segmentation predictions.

    Returns
    -------
    pve_img : Nifti1Image
        PVE image with CSF, GM, and WM segmentations.
    """
    gm = prediction.get_fdata() == 2
    wm = prediction.get_fdata() == 3
    csf = prediction.get_fdata() == 4

    pve_data = np.zeros(
        prediction.get_fdata().shape + (3,),
        dtype=np.float32)
    pve_data[..., 0] = csf.astype(np.float32)
    pve_data[..., 1] = gm.astype(np.float32)
    pve_data[..., 2] = wm.astype(np.float32)

    pve_img = nib.Nifti1Image(
        pve_data.astype(np.float32),
        prediction.affine)

    return pve_img