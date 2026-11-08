import numpy as np
from skillcomp.availability_calibration import fit_availability_intercepts,calibrate_availability


def test_constant_fit_domains_and_unknown_fallback_need_no_query_availability():
    probability=np.array([.2,.4,.6,.8]);known=np.array([0,0,1,1]);repos=np.array(['none','none','all','all']);model=fit_availability_intercepts(probability,known,repos,20);np.testing.assert_array_equal(calibrate_availability(model,probability,repos),known);query=calibrate_availability(model,[.3,.7],['new','none']);np.testing.assert_allclose(query,[.3,0],atol=1e-14);np.testing.assert_array_equal(query[:1],calibrate_availability(model,[.3],['new']))


def test_nonconstant_fit_probability_mean_matches_smoothed_training_rate():
    p=np.array([.05,.1,.2,.3]);known=np.array([1,0,1,0]);repos=np.array(['a']*4);model=fit_availability_intercepts(p,known,repos,20);q=calibrate_availability(model,p,repos);np.testing.assert_allclose(q.mean(),.5,atol=1e-14);assert (np.diff(q)>0).all()
