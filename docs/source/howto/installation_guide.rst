.. _installation_guide:

##########################
 How to install ``pyAFQ``
##########################

The ``pyAFQ`` software works (at least) on Python 3.9.

************************************
 How to install the release version
************************************

The released version of the software is the one that is officially supported,
and if you are getting started with ``pyAFQ``, this is probably where you should
get started

To install it, in a shell or command line, issue the following:

.. code::

   pip install pyAFQ

****************************************
 How to install the development version
****************************************

The development version is probably less stable, but might include new features
and fixes. There are two ways to install this version. The first uses ``pip``:

.. code::

   pip install git+https://github.com/tractometry/pyAFQ.git

The other requires that you clone the source code to your machine:

.. code::

   git clone https://github.com/tractometry/pyAFQ.git

With both installation methods, you can include extensions to the base
installation in square brackets. Change your working directory into the
top-level directory of this repo and issue:

.. code::

   pip install -e .[dev,fury,afqbrowser,plot]

On some platforms, you may need to add quotes around the ``.[]`` part:

.. code::

   pip install -e .'[dev,fury,afqbrowser,plot]'

.. note::

   The project follows the standard GitHub fork and pull request workflow. So if
   you plan on contributing to pyAFQ it is recommended that you fork the
   repository and issue pull requests. See :ref:`contributing`

.. note::

   It is also recommended that you utilize python virtual environment and
   package mangagement tools (e.g., conda) and begin with a clean environment.

.. note::

   Some of the examples in the documentation require additional dependencies. To
   install these, you can run `pip install pyAFQ[plot]`, which will include
   visualization tools that are required in these examples. For examples
   involving the cloudknot distributed computing library, you will also need to
   set up an [AWS account]([Create Account -
   aws.amazon.com](https://aws.amazon.com/resources/create-account/)) and have
   [docker](https://www.docker.com/) installed.

*****************************
 How to install using Docker
*****************************

pyAFQ automatically builds and pushes a Docker image with pyAFQ installed for
every commit. The images can be found `here
<https://github.com/orgs/nrdg/packages/container/package/pyafq>`_ To pull an
image, you can either pull the latest:

.. code::

   docker pull ghcr.io/nrdg/pyafq:latest

or specify the commit using its hash:

.. code::

   docker pull ghcr.io/nrdg/pyafq:41c03ce18fa2fd872ece9df72165e7d8d8f58baf

pyAFQ also automatically builds and pushes a Docker image with pyAFQ and
`QSIprep <https://qsiprep.readthedocs.io/en/latest/>`_ installed for every
commit to master. This image may be useful if you want an all-in-one image for
pre-processing and tractometry. You can pull the latest of this image or use a
specific commit or tag as well:

.. code::

   docker pull ghcr.io/nrdg/afqsi:latest

***********************************************
 How to build an Apptainer (Singularity) image
***********************************************

If the user intends to execute pyAFQ as a program from the command line
(``$pyAFQ /path/to/config.toml``) in an administered environment where root
access is not available (e.g., High Performance Computing cluster) then one
solution is to build an Apptainer (also known as Singularity) image from a local
pull of the pyAFQ docker container.

Start by running a docker registry:

.. code::

   docker run -d -p 5000:5000 --restart=always --name registry registry:2

Next, create a local tag which allows a push of the pyAFQ container to the local
registry:

.. code::

   docker tag ghcr.io/nrdg/pyafq:latest localhost:5000/pyafq
   docker push localhost:5000/pyafq:latest

Finally build the singularity file referencing the local registry:

.. code::

   APPTAINER_NOHTTPS=1 apptainer build pyafq_latest.sif docker://localhost:5000/pyafq:latest
