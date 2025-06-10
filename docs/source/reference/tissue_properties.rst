Tissue Properties in pyAFQ
==========================

This document lists some of the more common tissue property parameters
available in pyAFQ, though not all of them.
They are organized by the diffusion model they are derived from.
Custom tissue properties can be imported, see
`How to add custom tissue properties from another pipeline <howto/usage/scalars.html>`_.
If there is a tissue property in DIPY not in pyAFQ, you can add it in
`AFQ/tasks/data.py`, or post an issue.

DTI Model Parameters
--------------------

.. list-table:: DTI Model Parameters
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - dti_fa (Fractional Anisotropy)
     - Measures the directional dependence of water diffusion (0 = isotropic, 1 = fully anisotropic)
   * - dti_md (Mean Diffusivity)
     - Average of the three eigenvalues, representing overall diffusion magnitude
   * - dti_ad (Axial Diffusivity)
     - Largest eigenvalue (λ₁), representing diffusion parallel to fibers
   * - dti_rd (Radial Diffusivity)
     - Average of the second and third eigenvalues (λ₂+λ₃)/2, representing diffusion perpendicular to fibers
   * - dti_ga (Geodesic Anisotropy)
     - Similar to FA but based on Riemannian geometry
   * - dti_cfa (Color FA)
     - FA map where colors represent principal diffusion direction (RGB = XYZ)
   * - dti_pdd (Principal Diffusion Direction)
     - The dominant direction of diffusion (first eigenvector)
   * - dti_lt0 (Dxx)
     - First element of diffusion tensor (left-right diffusion)
   * - dti_lt1 (Dyy)
     - Second element of diffusion tensor (posterior-anterior diffusion)
   * - dti_lt2 (Dzz)
     - Third element of diffusion tensor (inferior-superior diffusion)
   * - dti_lt3 (Dxy)
     - Fourth element of diffusion tensor (xy plane relationship)
   * - dti_lt4 (Dxz)
     - Fifth element of diffusion tensor (xz plane relationship)
   * - dti_lt5 (Dyz)
     - Sixth element of diffusion tensor (yz plane relationship)

Free-water DTI (FWDTI) Parameters
--------------------------------

.. list-table:: FWDTI Model Parameters
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - fwdti_fa (Fractional Anisotropy)
     - FA of the tissue compartment (excluding free water)
   * - fwdti_md (Mean Diffusivity)
     - MD of the tissue compartment (excluding free water)
   * - fwdti_fwf (Free Water Fraction)
     - Fraction of the signal attributed to free water diffusion

DKI Model Parameters
-------------------

.. list-table:: DKI Model Parameters
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - dki_fa (Fractional Anisotropy)
     - FA derived from DKI tensor
   * - dki_md (Mean Diffusivity)
     - MD derived from DKI tensor
   * - dki_mk (Mean Kurtosis)
     - Average kurtosis over all directions
   * - dki_kfa (Kurtosis FA)
     - Fractional anisotropy of the kurtosis tensor
   * - dki_ga (Geodesic Anisotropy)
     - Geodesic anisotropy derived from DKI
   * - dki_rd (Radial Diffusivity)
     - Radial diffusivity derived from DKI
   * - dki_ad (Axial Diffusivity)
     - Axial diffusivity derived from DKI
   * - dki_rk (Radial Kurtosis)
     - Kurtosis perpendicular to the main fiber direction
   * - dki_ak (Axial Kurtosis)
     - Kurtosis parallel to the main fiber direction
   * - dki_awf (Axonal Water Fraction)
     - Estimated fraction of water in the axonal compartment
   * - dki_kt0-dki_kt14
     - 15 independent elements of the kurtosis tensor
   * - dki_lt0-dki_lt5
     - 6 independent elements of the diffusion tensor

Mean Signal DKI (MSDKI) Parameters
---------------------------------

.. list-table:: MSDKI Model Parameters
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - msdki_msd (Mean Signal Diffusivity)
     - Diffusivity derived from mean signal approach
   * - msdki_msk (Mean Signal Kurtosis)
     - Kurtosis derived from mean signal approach

CSD Model Parameters
-------------------

.. list-table:: CSD Model Parameters
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - csd_apm (Anisotropic Power Map)
     - Map of fiber orientation distribution function (fODF) power
   * - csd_ai (Anisotropic Index)
     - Index of anisotropy derived from fODF

GQ (Generalized Q-Sampling) Parameters
-------------------------------------

.. list-table:: GQ Model Parameters
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - gq_apm (Anisotropic Power Map)
     - Power map from GQ sampling
   * - gq_ai (Anisotropic Index)
     - Anisotropy index from GQ sampling
   * - gq_aso (Anisotropic Component)
     - Anisotropic part of the ODF
   * - gq_iso (Isotropic Component)
     - Isotropic part of the ODF

OPDT (Orientation Probability Density Transform) Parameters
---------------------------------------------------------

.. list-table:: OPDT Model Parameters
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - opdt_apm (Anisotropic Power Map)
     - Power map from OPDT
   * - opdt_ai (Anisotropic Index)
     - Anisotropy index from OPDT
   * - opdt_gfa (Generalized FA)
     - Generalized fractional anisotropy

CSA (Constant Solid Angle) Parameters
-----------------------------------

.. list-table:: CSA Model Parameters
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - csa_apm (Anisotropic Power Map)
     - Power map from CSA
   * - csa_ai (Anisotropic Index)
     - Anisotropy index from CSA
   * - csa_gfa (Generalized FA)
     - Generalized fractional anisotropy

RUMBA Model Parameters
----------------------

.. list-table:: RUMBA Model Parameters
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - rumba_f_wm (White Matter Fraction)
     - Estimated white matter volume fraction
   * - rumba_f_gm (Gray Matter Fraction)
     - Estimated gray matter volume fraction
   * - rumba_f_csf (CSF Fraction)
     - Estimated cerebrospinal fluid volume fraction
