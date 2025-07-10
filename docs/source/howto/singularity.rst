The pyAFQ singularity image
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can build a singularity image for pyAFQ from the Docker image using this command::

    apptainer build pyafq_latest.sif docker://ghcr.io/nrdg/pyafq:latest

We have a singularity definition file for building pyAFQ for the GPU specifically,
which can be found `here <https://github.com/tractometry/pyAFQ/blob/main/gpu_docker/cuda_track_template.def>`_.

Assuming the singularity image ``pyafq_latest.sif`` exists, trigger the
entrypoint ``pyafq`` to run the workflow with::

    apptainer run \
        --bind bids_dir:bids_dir \
        pyafq_latest.sif bids_dir/config.toml

This is using the config file which you can read about
in `The pyAFQ configuration file <../reference/config.html>`_.
You can also launch python inside the container and use the normal pyAFQ.

.. note::

    It may be necessary to set the global variable TEMPLATEFLOW_HOME to
    reference a pull of `TemplateFlow <https://www.templateflow.org>`_:

    ``export TEMPLATEFLOW_HOME=/path/to/templateflow``
