import numpy as np
import pytest
from skillcomp.frozen_inference import repository_probabilities,validate_records


def test_unknown_repository_does_not_use_query_distribution_or_targets():
    bundle={'prior_mean':np.array([.2,.8]),'prior_strength':20,'repository_counts':{'known':(10,np.array([5,7]))}}
    p=repository_probabilities(bundle,['known','unknown'])
    np.testing.assert_allclose(p[0],[(5+4)/30,(7+16)/30]);np.testing.assert_allclose(p[1],[.2,.8])


def test_inference_schema_excludes_target_outputs():
    row={'issue text':'fix','issue description':'stack trace','Repo Name':'repo'};validate_records([row])
    with pytest.raises(ValueError):validate_records([row|{'target_count':3}])
