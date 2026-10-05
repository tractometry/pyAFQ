#######################
 The pyAFQ Mapping API
#######################

pyAFQ has a system for users to specify how to register an image from each
subject to a given template, called the mapping. The mapping API is similar to
our Mask API.

In the :mod:`AFQ.definitions.mapping` module, there are several mapping classes one
can use to specify the mapping. As a user, one should initialize mapping classes
and pass them to the AFQ.api objects, or write out the initialization as a
string inside to pass into the CLI.

-  :class:`AFQ.definitions.mapping.SynMap`: The default mapping class is to use
   Symmetric Diffeomorphic Image Registration (SyN). This is done with an
   optional linear pre-alignment by default. The parameters of the pre-alginment
   can be specified when initializing the SynMap.

-  :class:`AFQ.definitions.mapping.SlrMap`: Use this class to tell pyAFQ to use
   Streamline-based Linear Registration (SLR) for registration. Note that the
   reg_template and reg_subject parameters passed to
   :class:`AFQ.api.group.GroupAFQ` should be streamlines when using this
   registration.

-  :class:`AFQ.definitions.mapping.AffMap`: This will only perform a linear
   alignment for registration.

-  :class:`AFQ.definitions.mapping.FnirtMap`: If you have an existing mapping
   calculated using Fnirt, you can pass bids filters to
   :class:`AFQ.definitions.mapping.FnirtMap` and pyAFQ will find and use that
   mapping.

-  :class:`AFQ.definitions.mapping.AntsMap`: If you have existing ANTs
   transforms, such as the ``from-ACPC_to-MNI152NLin2009cAsym`` and
   ``from-MNI152NLin2009cAsym_to-ACPC`` ``.h5`` files from QSIPrep,
   :class:`AFQ.definitions.mapping.AntsMap` will find them with BIDS and
   use them. Requires ``pip install antspyx`` (or ``pip install pyAFQ[ants]``).
