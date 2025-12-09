"""
======================================
Running pyAFQ 2.x defauls in pyAFQ 3.x
======================================
"""
from AFQ.api.group import GroupAFQ
import AFQ.data.fetch as afd
import AFQ.definitions.image as afm
import AFQ.api.bundle_dict as abd

import os.path as op

afd.organize_stanford_data()

################################################################
# Tractography parameters in the old way
# --------------------------------------------------------------
# In pyAFQ 2.x, we used CSD with no asymmetric filtering,
# and seeded streamlines throughout the white matter instead of
# on the interface.

tracking_params = dict(
    odf_model="csd",
    n_seeds=1,
    random_seeds=False,
    minlen=50,
    tracker="local",
    seed_mask=afm.ScalarImage("dti_fa"),
    seed_threshold=0.2
)

################################################################
# Partial Volume Estimate in the old way
# --------------------------------------------------------------
# In pyAFQ 2.x, we did not use PVE and instead thresholded
# on fractional anisotropy (FA) maps to create
# seed and stopping masks. Here, we recreate
# the PVE images using the FA maps.
# Note there the CSF map is not used in this case.

pve = afm.PVEImages(
    afm.ThresholdedScalarImage(
        "dti_fa",
        upper_bound=0.0),
    afm.ThresholdedScalarImage(
        "dti_fa",
        upper_bound=0.2),
    afm.ThresholdedScalarImage(
        "dti_fa",
        lower_bound=0.2))

################################################################
# VOF / pAF / CST in the old way
# --------------------------------------------------------------
# In pyAFQ 2.x, the vertical occipital fasciculus (VOF)
# and posterior arcuate fasciculus (pAF) were definded differently.
# The pAF in 3.0 has an increased restriction that it cannot overlap with
# the arcuate by more than 30%. The VOF has several changes:
# 1. one endpoint ROI instead of both, but there is a minimum length
#    requirement to of 25mm to compensate;
# 2. The allowed overlap with the pAF has been reduced;
# 3. it must be lateral to the inferior fronto-occipital fasciculus
#    instead of the inferior longitudinal fasciculus;
# 4. cleaning has been changed: there is now mahalanobis cleaning on
#    orientation, and isolation forest cleaning instead of mahalanobis for 
#    distance.
# Additionally, in the new version, the inferior endpoints of the
# corticospinal tracts (CST) were removed.

templates = afd.read_templates(as_img=False)
old_vof_paf_cst_definitions = abd.BundleDict({
        'Left Corticospinal': {
            'cross_midline': False,
            'include': [templates['CST_roi2_L'],
                        templates['CST_roi1_L']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['CST_L_prob_map'],
            'end': templates['CST_L_start'],
            'start': templates['CST_L_end']},
        'Right Corticospinal': {
            'cross_midline': False,
            'include': [templates['CST_roi2_R'],
                        templates['CST_roi1_R']],
            'exclude': [],
            'space': 'template',
            'prob_map': templates['CST_R_prob_map'],
            'end': templates['CST_R_start'],
            'start': templates['CST_R_end']},
        'Left Posterior Arcuate': {'cross_midline': False,
                                   'include': [templates['SLFt_roi2_L']],
                                   'exclude': [templates['SLF_roi1_L']],
                                   'space': 'template',
                                   'start': templates['pARC_L_start'],
                                   'primary_axis': 'I/S',
                                   'primary_axis_percentage': 40},
        'Right Posterior Arcuate': {'cross_midline': False,
                                    'include': [templates['SLFt_roi2_R']],
                                    'exclude': [templates['SLF_roi1_R']],
                                    'space': 'template',
                                    'start': templates['pARC_R_start'],
                                    'primary_axis': 'I/S',
                                    'primary_axis_percentage': 40},
        'Left Vertical Occipital': {'cross_midline': False,
                                    'space': 'template',
                                    'start': templates['VOF_L_start'],
                                    'end': templates['VOF_L_end'],
                                    'inc_addtol': [4, 0],
                                    'Left Arcuate': {
                                        'node_thresh': 20},
                                    'Left Posterior Arcuate': {
                                        'node_thresh': 1,
                                        'core': 'Anterior'},
                                    'Left Inferior Longitudinal': {
                                        'core': 'Right'},
                                    'primary_axis': 'I/S',
                                    'primary_axis_percentage': 40},
        'Right Vertical Occipital': {'cross_midline': False,
                                     'space': 'template',
                                     'start': templates['VOF_R_start'],
                                     'end': templates['VOF_R_end'],
                                     'inc_addtol': [4, 0],
                                     'Right Arcuate': {
                                         'node_thresh': 20},
                                     'Right Posterior Arcuate': {
                                         'node_thresh': 1,
                                         'core': 'Anterior'},
                                     'Right Inferior Longitudinal': {
                                         'core': 'Left'},
                                     'primary_axis': 'I/S',
                                     'primary_axis_percentage': 40}})

##################################################################
# Callosal bundles in the old way
# ----------------------------------------------------------------
# In pyAFQ 2.x, the callosal bundles were cleaned using mahalnobis
# instead of isolation forest.

callosal_templates =\
    afd.read_callosum_templates(as_img=False)
callosal_bd = abd.BundleDict({
    'Callosum Anterior Frontal': {
        'cross_midline': True,
        'include': [callosal_templates['R_AntFrontal'],
                    callosal_templates['Callosum_midsag'],
                    callosal_templates['L_AntFrontal']],
        'exclude': [],
        'space': 'template'},
    'Callosum Motor': {
        'cross_midline': True,
        'include': [callosal_templates['R_Motor'],
                    callosal_templates['Callosum_midsag'],
                    callosal_templates['L_Motor']],
        'exclude': [],
        'space': 'template'},
    'Callosum Occipital': {
        'cross_midline': True,
        'include': [callosal_templates['R_Occipital'],
                    callosal_templates['Callosum_midsag'],
                    callosal_templates['L_Occipital']],
        'exclude': [],
        'space': 'template'},
    'Callosum Orbital': {
        'cross_midline': True,
        'include': [callosal_templates['R_Orbital'],
                    callosal_templates['Callosum_midsag'],
                    callosal_templates['L_Orbital']],
        'exclude': [],
        'space': 'template'},
    'Callosum Posterior Parietal': {
        'cross_midline': True,
        'include': [callosal_templates['R_PostParietal'],
                    callosal_templates['Callosum_midsag'],
                    callosal_templates['L_PostParietal']],
        'exclude': [],
        'space': 'template'},
    'Callosum Superior Frontal': {
        'cross_midline': True,
        'include': [callosal_templates['R_SupFrontal'],
                    callosal_templates['Callosum_midsag'],
                    callosal_templates['L_SupFrontal']],
        'exclude': [],
        'space': 'template'},
    'Callosum Superior Parietal': {
        'cross_midline': True,
        'include': [callosal_templates['R_SupParietal'],
                    callosal_templates['Callosum_midsag'],
                    callosal_templates['L_SupParietal']],
        'exclude': [],
        'space': 'template'},
    'Callosum Temporal': {
        'cross_midline': True,
        'include': [callosal_templates['R_Temporal'],
                    callosal_templates['Callosum_midsag'],
                    callosal_templates['L_Temporal']],
        'exclude': [],
        'space': 'template'}})


bundle_info = abd.default18_bd() + \
    old_vof_paf_cst_definitions + \
    callosal_bd

################################################################
# Run GroupAFQ with these parameters
# --------------------------------------------------------------
# Finally, we can run GroupAFQ with the 2.0 parameters.
# In sum, we changed:
# Tractography parameters to use CSD and seed throughout
# the white matter;
# PVE images to use FA thresholding;
# Bundle definitions for VOF, pAF, and CST to use the old definitions;
# Callosal bundles to use mahalanobis cleaning.

myafq = GroupAFQ(
    bids_path=op.join(afd.afq_home, 'stanford_hardi'),
    dwi_preproc_pipeline='vistasoft',
    t1_preproc_pipeline='freesurfer',
    tracking_params=tracking_params,
    pve=pve,
    bundle_info=bundle_info)

myafq.export_all()
