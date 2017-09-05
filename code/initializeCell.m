%% Requires 2 variables
% X = T x 1 vector - the raw stimulus
% Y = T x 1 vector - the corresponding spike train 



%%
dt = 1e-4;

nLinearRFs = 10;     %number of stimulus basis functions
numShortSHFilts = 5; %each is 0.4ms long - used for refractory period
nSHfilters = 7+numShortSHFilts;%7+

nInh = 1;
nExc = 1;

useLeakCondExc = true;
usePrior = true;

CBEM = setupCBEMsimple(nLinearRFs,nSHfilters,dt,nInh,nExc,numShortSHFilts,useLeakCondExc,usePrior);


%% Convolves stimulus & spike history with filter basis functions

TT = length(X);

SpikeStim = conv2(X,CBEM.stimBasisVectors);
SpikeStim = SpikeStim(1:TT,:);
spkHist = conv2(Y,CBEM.spkHistBasisVectors);
spkHist = [zeros(1,size(spkHist,2)); spkHist(1:TT-1,:)];
