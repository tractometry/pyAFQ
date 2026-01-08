import pytest

import AFQ.api.bundle_dict as abd
import AFQ.data.fetch as afd
from AFQ.api.group import GroupAFQ
from AFQ.tests.test_api import create_dummy_bids_path


def test_AFQ_custom_bundle_dict():
    bids_path = create_dummy_bids_path(3, 1)
    bundle_dict = abd.default_bd()
    GroupAFQ(bids_path, dwi_preproc_pipeline="synthetic", bundle_info=bundle_dict)


def test_BundleDict():
    """
    Tests bundle dict
    """

    # test defaults
    afq_bundles = abd.default_bd()

    assert len(afq_bundles) == 20

    # Arcuate Fasciculus
    afq_bundles = abd.default_bd()["Left Arcuate", "Right Arcuate"]

    assert len(afq_bundles) == 2

    del afq_bundles["Left Arcuate"]
    assert len(afq_bundles) == 1

    # Forceps Minor and Major
    afq_bundles = abd.callosal_bd()["Callosum Occipital", "Callosum Anterior Frontal"]

    assert len(afq_bundles) == 2

    # Test "custom" bundle
    afq_templates = afd.read_templates()
    afq_bundles = abd.BundleDict(
        {
            "custom_bundle": {
                "include": [afq_templates["FA_L"], afq_templates["FP_R"]],
                "cross_midline": False,
            }
        }
    )
    afq_bundles.get("custom_bundle")

    assert len(afq_bundles) == 1

    # misspelled bundle that does not exist in afq templates
    with pytest.raises(ValueError, match=" is not in this BundleDict"):
        abd.default_bd()[
            "Left Vertical Occipital Quinticulus",
            "Right Vertical Occipital Quinticulus",
        ]

    afq_bundles = abd.reco_bd(80)["VOF_L", "VOF_R"]
    assert len(afq_bundles) == 2
