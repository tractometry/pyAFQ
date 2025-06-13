==========================
Cleaning Parameters
==========================

This page documents the configuration options for controlling
bundle cleaning in pyAFQ. These parameters can be set in your
configuration file or passed as arguments when using the API.
Note that this goes inside of segmentation_params.

Example Usage
=============

.. code-block:: python

    from AFQ.api.group import GroupAFQ
    import AFQ.data.fetch as afd
    import os.path as op

    afd.organize_stanford_data()

    myafq = GroupAFQ(
        bids_path=op.join(afd.afq_home, 'stanford_hardi'),
        preproc_pipeline='vistasoft',
        segmentation_params=dict(
            cleaning_params=dict(
                distance_threshold=5,
                length_threshold=5,  # More lenient cleaning
            ))
        )

Cleaning Parameter Reference
================================

.. autofunction:: AFQ.recognition.cleaning.clean_bundle
   :noindex:
