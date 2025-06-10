.. _home:


Automated Fiber Quantification in Python (pyAFQ)
-------------------------------------------------

pyAFQ is an open-source software tool for the analysis of brain white matter in
diffusion MRI measurements. It implements a complete and automated data
processing pipeline for tractometry, from raw DTI data to white matter tract
identification, as well as quantification of tissue properties along the length
of the major long-range brain white matter connections.

- To get started, please refer to the `getting started <tutorials/index.html>`_ page. In particular, these two examples are very useful:

  - `Getting started with pyAFQ - GroupAFQ <tutorials/tutorial_examples/plot_001_group_afq_api.html>`_
  - `Getting started with pyAFQ - ParticipantAFQ <tutorials/tutorial_examples/plot_002_participant_afq_api.html>`_
- What is the difference between tractography and tractometry? See in the `explanations <explanations/index.html>`_ page.
- For more detailed information on the variety of uses of pyAFQ, see the `how to <howto/index.html>`_ page. In particular, this one example is useful for understanding tractometry:

  - `Understanding the different stages of tractometry with videos <howto/howto_examples/plot_stages_of_tractometry.html>`_
- For a detailed description of the methods and objects used in pyAFQ, see the `reference documentation <reference/index.html>`_ page.

Here are some useful reference pages:

- For a list of the major fiber tracts supported by pyAFQ, see the `Major Fiber Tracts <reference/fibertracts.html>`_ page.
- For a list of the supported tissue properties, see the `Tissue Properties <reference/tissue_properties.html>`_ page.
- For a grand list of all pyAFQ outputs, see `The pyAFQ API methods <reference/methods.html>`_.
- For a grand list of all pyAFQ arguments, see `The pyAFQ API optional arguments <reference/kwargs.html>`_.  

Citing
~~~~~~~
If you use *pyAFQ* in a scientific publication, please cite our paper:

Kruper, J., Yeatman, J. D., Richie-Halford, A., Bloom, D., Grotheer, M., Caffarra, S., Kiar, G., Karipidis, I. I., Roy, E., Chandio, B. Q., Garyfallidis, E., & Rokem, A.
`Evaluating the Reliability of Human Brain White Matter Tractometry <https://doi.org/10.52294/e6198273-b8e3-4b63-babb-6e6b0da10669>`_.
DOI:10.52294/e6198273-b8e3-4b63-babb-6e6b0da10669

.. code-block:: bibtex

  @article {Kruper2021-xb,
    title     = "Evaluating the reliability of human brain white matter
                 tractometry",
    author    = "Kruper, John and Yeatman, Jason D and Richie-Halford, Adam and
                 Bloom, David and Grotheer, Mareike and Caffarra, Sendy and Kiar,
                 Gregory and Karipidis, Iliana I and Roy, Ethan and Chandio,
                 Bramsh Q and Garyfallidis, Eleftherios and Rokem, Ariel",
    journal   = "Apert Neuro",
    publisher = "Organization for Human Brain Mapping",
    volume    =  1,
    number    =  1,
    month     =  nov,
    year      =  2021,
    doi       =  10.52294/e6198273-b8e3-4b63-babb-6e6b0da10669,
  }

Guide Layout
~~~~~~~~~~~~

.. grid:: 2

    .. grid-item-card::
        :link: tutorials/index.html

        :octicon:`book;3em;sd-text-center`

        Tutorials
        ^^^^^^^^^

        Beginner's guide to pyAFQ. This guide introduces pyAFQ'S
        basic concepts and walks through fundamentals of using the software.

        +++

    .. grid-item-card::
        :link: howto/index.html

        :octicon:`rocket;3em;sd-text-center`

        How To
        ^^^^^^

        User's guide to pyAFQ. This guide assumes you know
        the basics and walks through some other commonly used functionality.

        +++

    .. grid-item-card::
        :link: explanations/index.html

        :octicon:`comment-discussion;3em;sd-text-center`

        Explanations
        ^^^^^^^^^^^^

        This guide contains in depth explanations of the various pyAFQ methods.

        +++

    .. grid-item-card::
        :link: reference/index.html

        :octicon:`search;3em;sd-text-center`

        API Reference
        ^^^^^^^^^^^^^

        The API Reference contains technical descriptions of methods
        and objects used in pyAFQ. It also contains descriptions
        of how methods work and the parameters used for each method.

        +++


Acknowledgements
~~~~~~~~~~~~~~~~

Work on this software was supported through grant `1RF1MH121868-01 <https://projectreporter.nih.gov/project_info_details.cfm?aid=9886761&icde=46874320&ddparam=&ddvalue=&ddsub=&cr=2&csb=default&cs=ASC&pball=>`_ from the `National Institutes for Mental Health <https://www.nimh.nih.gov/index.shtml>`_ / `The BRAIN Initiative <https://braininitiative.nih.gov>`_
and by a grant from the
`Gordon & Betty Moore Foundation <https://www.moore.org/>`_,  and from the
`Alfred P. Sloan Foundation <http://www.sloan.org/>`_ to the
`University of Washington eScience Institute <http://escience.washington.edu/>`_, by grant `R01EB027585 <https://reporter.nih.gov/search/jnnzzQ8Rj0CLD3R3l92GPg/project-details/10735068>`_ to Eleftherios Garyfallidis (PI) and Ariel Rokem, grant `R01HD095861 <https://reporter.nih.gov/search/j2JXd89oR0i4cCnIDo7fFA/project-details/10669103>`_ to Jason Yeatman, `R21HD092771 <https://reporter.nih.gov/search/j2JXd89oR0i4cCnIDo7fFA/project-details/9735358>`_  to Jason Yeatman and Pat Kuhl, by NSF grants `1551330 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1551330>`_ to Jason Yeatman and `1934292 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=1934292>`_ to Magda Balazinska (PI) and Ariel Rokem (co-PI). John Kruper's work on pyAFQ has been supported through the NSF Graduate Research Fellowship program (DGE-2140004).


.. figure:: _static/eScience_Logo_HR.png
   :align: center
   :figclass: align-center
   :target: http://escience.washington.edu

.. figure:: _static/BDE_Banner_revised20160211-01.jpg
   :align: center
   :figclass: align-center
   :target: http://brainandeducation.com

.. toctree::
    guides_index
