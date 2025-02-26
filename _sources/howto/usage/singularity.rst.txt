The pyAFQ singularity image
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming the singularity image ``pyafq_latest.sif`` exists, trigger the
entrypoint ``pyafq`` to run the workflow with::

    apptainer run \
        --bind bids_dir:bids_dir \
        pyafq_latest.sif bids_dir/config.toml

.. note::

    It may be necessary to set the global variable TEMPLATEFLOW_HOME to
    reference a pull of `TemplateFlow <https://www.templateflow.org>`_:

    ``export TEMPLATEFLOW_HOME=/path/to/templateflow``
