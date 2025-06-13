==========================
Tractography Parameters
==========================

This page documents the configuration options for controlling
tractography in pyAFQ. These parameters can be set in your configuration file
or passed as arguments when using the API.

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
        tracking_params=dict(
            n_seeds=25000,
            random_seeds=True,
            rng_seed=2025,
            trx=True)
        )

Tractography Parameter Reference
================================

.. autofunction:: AFQ.tractography.tractography.track
   :noindex:
