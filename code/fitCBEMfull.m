function [CBEMp,nll,nlpost] = fitCBEMfull(CBEM,SpikeStim,spkHist,Y,addOnesColumnToStim,doFitForGL,noFit)

% noFit = if true: quick rapper to get NLL and log posterior instead of
% fitting

%% setup CBEM_lin
CBEMp = CBEM;



spikeTimes   = find(Y == 1);
TT = size(SpikeStim,1);

if(addOnesColumnToStim)
    SpikeStim = [SpikeStim ones(TT,1)];
end

if(nargin < 7)
    noFit = false;
end

if(noFit)

    CBEMtoOptimize = buildDefaultOptSettings(false,CBEMp);
    CBEMtoOptimize.h_spk  = true;
    CBEMtoOptimize.k_s(1:2) = true;
    CBEMtoOptimize.k_s(3:end) = doFitForGL;

    nllFunc = @(Xvec) optimizationFunction(Xvec,CBEMp,CBEMtoOptimize,SpikeStim,spkHist,spikeTimes,ones(TT,1),{[],[]},{[],[]});
    initPoint = cbemStructToVector(CBEMp,CBEMtoOptimize); 
    nlpost = nllFunc(initPoint);
    
    CBEMtoOptimize = buildDefaultOptSettings(false,CBEMp);
    CBEMtoOptimize.h_spk  = true;
    if(isfield(CBEMp,'prior'))
        CBEMp = rmfield(CBEMp,'prior');
    end
    CBEMtoOptimize = buildDefaultOptSettings(false,CBEMp);
    CBEMtoOptimize.h_spk  = true;

    nllFunc = @(Xvec) optimizationFunction(Xvec,CBEMp,CBEMtoOptimize,SpikeStim,spkHist,spikeTimes,ones(TT,1),{[],[]},{[],[]});
    initPoint = cbemStructToVector(CBEMp,CBEMtoOptimize); 
    nll = nllFunc(initPoint);
    
else
%% optimize

    %% optimize
    CBEMtoOptimize = buildDefaultOptSettings(false,CBEMp);
    CBEMtoOptimize.h_spk  = true;
    CBEMtoOptimize.k_s(1:2) = true;
    CBEMtoOptimize.k_s(3:end) = doFitForGL;

    nllFunc = @(Xvec) optimizationFunction(Xvec,CBEMp,CBEMtoOptimize,SpikeStim,spkHist,spikeTimes,ones(TT,1),{[],[]},{[],[]});

    opts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'Hessian','on','MaxIter',500,'Display','iter','FunctionTolerance',1e-8,'StepTolerance',1e-8);

    initPoint = cbemStructToVector(CBEMp,CBEMtoOptimize);

    %%
    [finalPoint,nll] = fminunc(nllFunc,initPoint,opts);

    CBEMp = cbemVectorToStruct(finalPoint,CBEMp,CBEMtoOptimize);
end

