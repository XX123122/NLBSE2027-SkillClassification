import numpy as np
from skillcomp.conditional_peer_prior import ConditionalPeerPrior


def test_conditional_counts_match_literal_component_deletion_and_unknown_repo():
    rng=np.random.default_rng(44);f=rng.integers(0,2,(30,4));s=rng.integers(0,2,(30,4));pub=np.maximum(f,s);groups=np.repeat(np.arange(15),2);repos=np.array(['rare']*2+['a']*14+['b']*14);model=ConditionalPeerPrior().fit(f,s,repos,groups,pub);score,counts,n=model.fit_centre(repos,groups)
    for i in range(len(f)):
        keep=groups!=groups[i];literal=ConditionalPeerPrior().fit(f[keep],s[keep],repos[keep],groups[keep],pub[keep]);p,c,nn=literal.predict(repos[i:i+1]);np.testing.assert_array_equal(score[i],p[0]);np.testing.assert_array_equal(counts[i],c[0]);assert n[i]==nn[0]
    zero=s==0;rate=((f*zero).sum(0)+.5)/(zero.sum(0)+1);np.testing.assert_array_equal(model.predict(['unseen'])[0][0],rate)


def test_f_given_S_zero_differs_from_unconditional_when_correlated():
    f=np.array([[1],[1],[0],[0]]);s=f.copy();pub=np.maximum(f,s);model=ConditionalPeerPrior().fit(f,s,['a']*4,np.arange(4),pub);p=model.predict(['unseen'])[0];np.testing.assert_array_equal(p,[[.5/3]]);assert p[0,0]!=f.mean()
