function [winstSec, winendSec] = getFixedWinsFine(tendTrial, D, d)

st = 0:d:tendTrial;
en = st+D;
inds = find(en <= tendTrial);
winstSec=st(inds);
winendSec=en(inds);
