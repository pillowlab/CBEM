%% Requires 2 variables
% X = T x (N x M) vector - the raw stimulus (time by space)
% Y = T x 1 vector - the corresponding spike train 



%%

nTemporalRFs = 10;     %number of stimulus basis functions
numShortSHFilts = 5; %each is 0.4ms long - used for refractory period

% nPixels     = [size(X,2) size(X,3)]; 
% nSpatialRFs = [size(X,2) size(X,3)];%this will set the spatial basis to be the pixel basis while the temporal basis is the raised cosine
% 
% stimFilterRank = 2; %stimulus filter is the sum of a set of spatiotemporally separable filters. This rank says how many to use. If negative, uses full-rank filter


nSHfilters = 7+numShortSHFilts;%7+

nInh = 1;
nExc = 1;

useLeakCondExc = false;
usePrior = true;

CBEM = setupCBEMspatiotemporal(nTemporalRFs,nSpatialRFs,nPixels,stimFilterRank,nSHfilters,dt,nInh,nExc,numShortSHFilts,useLeakCondExc,usePrior);


%% Build design matrix: Convolves stimulus & spike history with filter basis functions

TT = length(X);

SpikeStim = zeros(TT,CBEM.stimNumBasisVectors_spatial *CBEM.stimNumBasisVectors_temporal);
X2 = reshape(X,TT,prod(nPixels))*CBEM.stimBasisVectors_spatial;

for ii = 1:size(X2,2)
    cc = conv2(X2(:,ii),CBEM.stimBasisVectors_temporal);
    SpikeStim(:,(1:CBEM.stimNumBasisVectors_temporal) + (ii-1)*CBEM.stimNumBasisVectors_temporal) = cc(1:TT,:);
end
spkHist = conv2(Y,CBEM.spkHistBasisVectors);
spkHist = [zeros(1,size(spkHist,2)); spkHist(1:TT-1,:)];
