########################
 What's new in PyAFQ 3.0
########################

Up to now, pyAFQ has focused on the core white matter. This is because the 
SNR of diffusion MRI is the highest in the core white matter, and tracking
is more robust to different methods in the core white matter. Thus, it is
easier to find reliable, robust results in the core white matter
when using tractometry.

PyAFQ 3.0 shifts the focus to improving AFQ in the superficial white matter.
The superificial white matter is the layer of white matter just below the
cortex. This is important to relate to functional or structural measures
and the gray matter.

A number of changes have been done to support this shift. Principally, pyAFQ
now requires a T1-weighted image. This is used to make a brain mask with
brainchop. Additionally, it is reccomended to provide partial volume
estimates (PVEs) from other pipelines such as FSLfast or Freesurfer.
If these are not provided, pyAFQ will generate them using SynthSeg
and the T1.

These PVEs are used for tractography, which has been radically changed in
pyAFQ. First, we now seed from the white matter gray matter interface to
reduce biases in streamline termination points. Second, we propogate using
CSD FODs filtered using a unified filtering method to be asymmetric, which
is necessary to track in the superificial white matter. Third, we use 
particle filtering tractography (PFT) instead of local tracking, again to
better track in the superficial white matter. Fourth, we use
anatomically constrained tractography (ACT) to ensure streamlines terminate
in the gray matter instead of the CSF. Finally, we only consider the middle
60% of identified bundles for cleaning, to allow fanning near the cortex.

There are also a few minor changes, such as the addition of different fitting
methods for MSMT, increased parallelization using Ray and Numba, changes to
vertical occipital fasciculus segmentation, and the addition of using
IsolationForest to do density-based cleaning for bundles that are not tube-like.
