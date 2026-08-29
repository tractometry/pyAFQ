The pyAFQ docker image
~~~~~~~~~~~~~~~~~~~~~~

Every time a new commit is made to master the
`pyAFQ github <https://github.com/tractometry/pyAFQ>`_,
a new image is pushed to the
`NRDG github <https://github.com/orgs/nrdg/packages/container/package/pyafq>`_.
This image contains an installation of the latest version of
pyAFQ with fslpy. This image also contains an entrypoint, and can be
run with::
    docker run -v bids_dir:/bids_dir:rw ghcr.io/nrdg/pyafq path/to/dwi path/to/t1 path/to/output_folder

This is using the CLI which you can
read about in `The pyAFQ CLI <../reference/cli.html>`_.
You can also launch python inside the container and use the normal pyAFQ.
