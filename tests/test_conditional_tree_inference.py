import numpy as np
from skillcomp.conditional_tree_inference import conditional_tree_mapping


class Probabilities:
    def __init__(self,value):self.value=np.asarray(value)
    def predict_proba(self,x,repositories):return self.value.copy()


class PeerPrior:
    def predict(self,repositories):return np.array([[.2,.4]]),np.array([[3,2]]),np.array([4])


def test_tree_channel_and_conditional_union_applied_once():
    channel={'global_rates':np.array([[.1,.2],[.9,.8]]),'repository_rates':{},'mean':np.array([.5,.5])};component={'neighbors':Probabilities([[.5,.25]]),'tree':Probabilities([[.8,.3]]),'conditional_prior':PeerPrior(),'channel':channel};out=conditional_tree_mapping(component,np.ones((1,3)),['x'],np.array([.6]));conditional=np.array([[.2+.8*.6*.5,.4+.6*.6*.25]]);tree=np.array([[.1+.8*.8,.2+.6*.3]]);np.testing.assert_allclose(out,.5*tree+.5*conditional,atol=1e-15,rtol=0)
