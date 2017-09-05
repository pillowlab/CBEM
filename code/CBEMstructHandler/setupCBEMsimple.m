%% initialize a simple CBEM structure for a one dimensional stimulus
%  inputs:
%     nTemporalRFs = number of stimulus RF basis functions
%                    180ms long
%     nSHfilters   = number of basis functions for the spike history filter
%                    90ms long
%     delta_t      = time width of bin in milliseconds
%
%                    first 6 spike history basis functions will be blocks
%                    0.3ms longs
%
function [CBEM] = setupCBEMsimple(nTemporalRFs,nSHfilters,delta_t,nInh,nExc,numShortBlocks,useLeakCondExc,usePrior)

CBEM.dt = delta_t;

if(nargin < 4)
    nInh = 1;
end
if(nargin < 5)
    nExc = 1;
end
if(nargin < 6)
    numShortBlocks = 5;
end
if(nargin < 7)
    
    useLeakCondExc = fase;
end
if(nargin < 8)
    
    usePrior = true;
end

%% stimulus basis
CBEM.stimBasisParams.RFstart = 2e-3;%0?
CBEM.stimBasisParams.RFend   = 150e-3;%150e-3 or 180e-3;
CBEM.stimBasisParams.b = 0.02;%0.02;
[~,CBEM.stimBasisVectors,CBEM.stimBasisVectors_0] = makeRaisedCosBasis(nTemporalRFs,delta_t,[CBEM.stimBasisParams.RFstart CBEM.stimBasisParams.RFend],CBEM.stimBasisParams.b);%0.1
CBEM.stimNumBasisVectors = size(CBEM.stimBasisVectors,2);

%% spike history basis
%numShortBlocks = 7; %num blocks for refractory period
refractoryBlockLength_ms = 0.4e-3;
refractoryBlockLength    = round(refractoryBlockLength_ms/delta_t); %width of refractory period blocks
refractoryBlockLength_ms = refractoryBlockLength*delta_t;

CBEM.spkHistBasisParams.b = 1e-4;

if(nSHfilters > numShortBlocks)
    SHend = (90e-3)-(refractoryBlockLength*numShortBlocks*delta_t);
    SHstart = refractoryBlockLength_ms*numShortBlocks;
    [~,spikeHistoryBasis2,CBEM.spikeHistoryBasis_0] = makeRaisedCosBasis(nSHfilters-numShortBlocks,delta_t,[SHstart SHend],CBEM.spkHistBasisParams.b);
    spikeHistoryBasis2(1:(numShortBlocks*refractoryBlockLength),:) = 0;
    filterLength = size(spikeHistoryBasis2,1);
else
    spikeHistoryBasis2 = [];
    filterLength = numShortBlocks*refractoryBlockLength;
    SHend = -1;
    SHstart = -1;
end
spikeHistoryBasis1 = zeros(filterLength,min(numShortBlocks,nSHfilters));
for ii = 1:size(spikeHistoryBasis1,2)
    spikeHistoryBasis1((1:refractoryBlockLength) + refractoryBlockLength*(ii-1),ii) = 1/refractoryBlockLength;
end

CBEM.spkHistBasisParams.refractoryBlockLength_ms = refractoryBlockLength_ms;
CBEM.spkHistBasisParams.refractoryBlockLength    = refractoryBlockLength;
CBEM.spkHistBasisParams.numShortBlocks    = numShortBlocks;
CBEM.spkHistBasisParams.SHend = SHend;
CBEM.spkHistBasisParams.SHstart = SHstart;

CBEM.spkHistBasisVectors = [spikeHistoryBasis1 spikeHistoryBasis2];
CBEM.spkHistNumBasisVectors = size(CBEM.spkHistBasisVectors,2);

%% some default parameters
nTotalConds = nExc + nInh;
InhConds    = nExc+(1:nInh);
ExcConds    = 1:nExc;

CBEM.E_s = zeros(nTotalConds,1); %1 = exc, 2 = inh, 3 = ahp, 4 = true Exc, 5 = true Inh
CBEM.f_s = cell(nTotalConds,1);
CBEM.k_s = cell(nTotalConds,1);
CBEM.k_s_section = cell(nTotalConds,1);
CBEM.g_s_bar = zeros(nTotalConds,1);
CBEM.condType = zeros(nTotalConds,1); 

CBEM.E_l = -80;
CBEM.E_s(ExcConds) = 0; 
CBEM.E_s(InhConds) = -80;
CBEM.condType(ExcConds) = 2;
CBEM.condType(InhConds) = 2;
%CBEM.condType(AhpConds) = 1;

CBEM.g_l = 1000/4; %4 ms time constant
CBEM.log_g_l = log(CBEM.g_l);

for cc = 1:nExc
    CBEM.f_s{ExcConds(cc)} = @(x,g_e) logOnePlusExpX(x,g_e);
    CBEM.k_s_section{ExcConds(cc)} = [];
end
for cc = 1:nInh
    CBEM.f_s{InhConds(cc)} = @(x,g_i) logOnePlusExpX(x,g_i);
    CBEM.k_s_section{InhConds(cc)} = [];
end
CBEM.g_s_bar(:) = 80;

CBEM.fixedThreshold = true;
%fixed threshold -> spike rate at time t is exp(c*(V(t)+b))
%otherwuse rate is exp(c*V(t)+b)
%the fixed threshold formulation makes b seem like a voltage threshold,
%mathematically it's all the same
CBEM.spikeNonlinearity.c = 0.3;

baseFR = 20;

baseV  = -65; %-60

CBEM.h_spk = zeros(CBEM.spkHistNumBasisVectors,1);
CBEM.h_spk(1:2) = -50;
CBEM.h_spk(3) = -2;

for cc = 1:nExc
    CBEM.k_s{ExcConds(cc)} = zeros(nTemporalRFs+1,1);
end
for cc = 1:nInh
    CBEM.k_s{InhConds(cc)} = zeros(nTemporalRFs+1,1);
end

initV = baseV-3;
inh_m = 1;
offsetInit = (-CBEM.g_l*(initV-CBEM.E_l))/( (initV-CBEM.E_s(ExcConds(1))) + inh_m*(initV-CBEM.E_s(InhConds(1))));
offsetInit = max(offsetInit,10);

for cc = 1:nExc
    CBEM.k_s{ExcConds(cc)}(end) = offsetInit./nExc;
end
for cc = 1:nInh
    CBEM.k_s{InhConds(cc)}(end) = inh_m*offsetInit./nInh;
end

%save the initial values of the variables that will be fit
CBEM.g_l_init = CBEM.g_l;
CBEM.k_s_init = CBEM.k_s;
CBEM.h_spk_init = CBEM.h_spk;
CBEM.fitOrder = cell(0,1);


%%

%% initialize a bunch of CBEM parameters to some blank default values
ExcConds = find(CBEM.E_s >  -40 & CBEM.condType == 2 );
InhConds = find(CBEM.E_s <= -40 & CBEM.condType == 2 );
StimConds = find(CBEM.condType == 2);
LeakConds = find(CBEM.condType == 3); %#ok<NASGU>

%%
priorWeight_e = 1;%0.1
priorWeight_e_c = 1e-2; %1e-2
priorWeight_i = 1;% 0.1
priorWeight_i_c = 1e-2; %1e-2

priorWeight_l = 1e-3;%1e-3

priorWeight_d = 0;%2 or 0
priorWeight_m = 5e-2;%5e-2 or 0


%% setup E/I conductances
unitMult = 1;


CBEM.E_l    = -80*unitMult;
CBEM.E_s(InhConds) = -80*unitMult;
CBEM.E_s(ExcConds) =  0*unitMult;
CBEM.g_l = 1000/40;% = 4ms time constant from leak alone, right now this is constant
CBEM.log_g_l = log(CBEM.g_l); %keep the log around for optimization

CBEM.f_s_mex = cell(length(ExcConds) + length(InhConds),2);

for cc = 1:length(ExcConds)
    CBEM.f_s{ExcConds(cc)} = @(x,g_e) logOnePlusExpX(x,g_e);
end
for cc = 1:length(InhConds)
    CBEM.f_s{InhConds(cc)} = @(x,g_i) logOnePlusExpX(x,g_i);
end

    
CBEM.condType = CBEM.condType(StimConds);
CBEM.k_s = CBEM.k_s(StimConds);
CBEM.E_s = CBEM.E_s(StimConds);
CBEM.f_s = CBEM.f_s(StimConds);
CBEM.f_s_mex = CBEM.f_s_mex(StimConds,:);
CBEM.g_s_bar = CBEM.g_s_bar(StimConds);

CBEM.g_s_bar(StimConds) = 30;

%% setup leak conductances
MaxLeakReversalPotential = -50;
LeakCond = length(CBEM.f_s)+1;
if(useLeakCondExc)
    LeakCond = [LeakCond LeakCond+1];
end
total_leak = zeros(length(LeakCond)+1,2);
total_leak(end,1) = CBEM.g_l;
total_leak(end,2) = CBEM.E_l;

for cc = 1:length(LeakCond)
    CBEM.k_s{LeakCond(cc)} = 6;
    CBEM.k_s_init{LeakCond(cc)} = CBEM.k_s{LeakCond(cc)} ;
    CBEM.f_s{LeakCond(cc)} = @(x,g_h) expTransfer(x,g_h);
    CBEM.condType(LeakCond(cc)) = 3;
    CBEM.k_s{LeakCond(1)} = 10;
    if(cc == 1)
        CBEM.E_s(LeakCond(cc)) = CBEM.E_s(InhConds(1));
    else
        CBEM.E_s(LeakCond(cc)) = MaxLeakReversalPotential;
    end
    CBEM.k_s_section{LeakCond(cc)} = [];
    CBEM.g_s_bar(LeakCond(cc)) = 40;
    total_leak(cc,1) = CBEM.f_s{LeakCond(cc)}(CBEM.k_s{LeakCond(cc)},CBEM.g_s_bar(LeakCond(cc)) );
    total_leak(cc,2) = CBEM.E_s(LeakCond(cc)) ;
end


inh_m = 1;

%%
totalG = 300-exp(CBEM.log_g_l);

if(numel(CBEM.E_s) > 3)
    initGl = [(CBEM.E_s(3) - initV) (CBEM.E_s(4) - initV); 1 1]\[-exp(CBEM.log_g_l)*(CBEM.E_l - initV);totalG];
    CBEM.k_s{3} = log(initGl(1));
    CBEM.k_s{4} = log(initGl(2));
else
    CBEM.E_s(3) = initV;
    CBEM.k_s{3} = log(totalG);
end


%% setup filters
CBEM.h_spk = zeros(size(CBEM.spkHistBasisVectors,2),1);
CBEM.h_spk(1:3) = -50;
CBEM.h_spk(4)   = -2;

for cc = 1:length(ExcConds)
    if(mod(cc,2) == 0)
        CBEM.k_s{ExcConds(cc)} =  -CBEM.k_s{ExcConds(cc-1)}*0 + randn(size(CBEM.stimBasisVectors,2),1)*0.5;
    else
        CBEM.k_s{ExcConds(cc)} =  [CBEM.stimBasisVectors'*randn(size(CBEM.stimBasisVectors,1),1); randn]*2;%*2
    end
end
for cc = 1:length(InhConds)
    if(mod(cc,2) == 0)
        CBEM.k_s{InhConds(cc)} = -CBEM.k_s{InhConds(cc-1)}*0 + randn(size(CBEM.stimBasisVectors,2),1)*0.5;
    else
        CBEM.k_s{InhConds(cc)} =  CBEM.k_s{ExcConds(cc)}*0 + [CBEM.stimBasisVectors'*randn(size(CBEM.stimBasisVectors,1),1); randn]*2;%*2
    end
end


total_leak_sum = 0;
for ii = 1:size(total_leak,1)
    total_leak_sum = total_leak_sum + total_leak(ii,1)*(total_leak(ii,2) - initV);
end
offsetInit = -total_leak_sum/( (CBEM.E_s(ExcConds(1)) - initV) + inh_m*(CBEM.E_s(InhConds(1)) - initV));
offsetInit = max(offsetInit,10);

for cc = 1:length(ExcConds)
    CBEM.k_s{ExcConds(cc)}(end) = offsetInit./length(ExcConds);
end
for cc = 1:length(InhConds)
    CBEM.k_s{InhConds(cc)}(end) = inh_m*offsetInit./length(InhConds);
end




ExcConds = find(CBEM.E_s >  -40 & (CBEM.condType == 2 | CBEM.condType == 6));
InhConds = find(CBEM.E_s <= -40 & (CBEM.condType == 2 | CBEM.condType == 6));
%StimConds = find(CBEM.condType == 2);
LeakConds = find(CBEM.condType == 3);


%% setup priors
if(usePrior)
    %%
    refract    = 1:4;
    nonRefract = 5:length(length(CBEM.h_spk));
    CBEM.prior.h_spk_sig = zeros(length(CBEM.h_spk),length(CBEM.h_spk));
    CBEM.prior.h_spk_sig(refract,refract) = 0;
    CBEM.prior.h_spk_sig(nonRefract,nonRefract) = 0.0*CBEM.spkHistBasisVectors(:,nonRefract)'*CBEM.spkHistBasisVectors(:,nonRefract);
    
    CBEM.prior.k_s_sig = cell(length(InhConds) + length(ExcConds),1);
    
    for cc = 1:length(ExcConds)
        M = priorWeight_e*(CBEM.stimBasisVectors'*CBEM.stimBasisVectors);
        CBEM.prior.k_s_sig{ExcConds(cc)} = [[M zeros(size(M,1),1)];[zeros(1,size(M,2)) priorWeight_e_c]];
    end
    
    for cc = 1:length(InhConds)
        M = priorWeight_i*(CBEM.stimBasisVectors'*CBEM.stimBasisVectors);
        CBEM.prior.k_s_sig{InhConds(cc)} = [[M zeros(size(M,1),1)];[zeros(1,size(M,2)) priorWeight_i_c]];
    end

    for cc = 1:length(LeakConds)
        CBEM.prior.k_s_sig{LeakConds(cc)}  = priorWeight_l;
    end
    
    
    CBEM.prior.comp.d = priorWeight_d;
    CBEM.prior.comp.m = priorWeight_m;
    
elseif(isfield(CBEM,'prior'))
    CBEM = rmfield(CBEM,'prior');
end

initV  = -60;
CBEM.initV = initV;
CBEM.spikeNonlinearity.c     = 90;
CBEM.spikeNonlinearity.log_c = log(CBEM.spikeNonlinearity.c); 
CBEM.spikeNonlinearity.alpha = 0.45;
CBEM.spikeNonlinearity.b     = 53*CBEM.spikeNonlinearity.alpha;
CBEM.spikeNonlinearity.name = 'softrec';
CBEM.spikeNonlinearity.f = @(V,c,b,sph,a)c*log(1+exp(min(80,max(-80,a*V+b+sph))));

%% save init variables
CBEM.spikeNonlinearity.b_init     = CBEM.spikeNonlinearity.b;
CBEM.spikeNonlinearity.c_init     = CBEM.spikeNonlinearity.c;
CBEM.g_l_init   = CBEM.g_l;
CBEM.k_s_init   = CBEM.k_s;
CBEM.h_spk_init = CBEM.h_spk;
CBEM.fitOrder = cell(0,1);

end