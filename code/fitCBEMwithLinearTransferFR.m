function [CBEMp, CBEM_lin,nll] = fitCBEMwithLinearTransferFR(CBEM,SpikeStim,spkHist,Y,noFit)
if(nargin < 5)
    noFit = false;
end
%%fits CBEM with a single linear conductance. Fits only the conductance
%%filter and the spike history (g_l is fixed)


%% setup CBEM_lin
CBEMp = CBEM;
CBEM_lin = CBEM;

CBEM_lin.condType = CBEM.condType([1,3:end]);
CBEM_lin.k_s      = CBEM.k_s([1,3:end]);
CBEM_lin.k_s{1}   = randn(size(CBEM_lin.k_s{1}(1:end-1)))*0.0;
CBEM_lin.E_s      = CBEM.E_s([1,3:end]);
CBEM_lin.g_s_bar  = CBEM.g_s_bar([1,3:end]);

if(isfield(CBEM_lin,'prior'))
    CBEM_lin.prior.k_s_sig = CBEM.prior.k_s_sig([1, 3:end]);
    CBEM_lin.prior.k_s_sig{1} = CBEM_lin.prior.k_s_sig{1}(1:end-1,1:end-1);
end

CBEM_lin.f_s     = CBEM.f_s([1,3:end]);
CBEM_lin.f_s_mex = CBEM.f_s_mex([1,3:end],:);

CBEM_lin.f_s{1} = @(x,g_e) linTransfer(x,g_e);

%% load things onto GPU
spikeTimes   = find(Y == 1);
TT = size(SpikeStim,1);

if(noFit)
    if(isfield(CBEM_lin,'prior'))
        CBEM_lin = rmfield(CBEM_lin,'prior');
    end
    CBEM_lin.k_s{1} = CBEM.k_s{1}(1:end-1);
    CBEMtoOptimize = buildDefaultOptSettings(false,CBEM_lin);
    CBEMtoOptimize.h_spk  = true;

    nllFunc = @(Xvec) optimizationFunction(Xvec,CBEM_lin,CBEMtoOptimize,SpikeStim,spkHist,spikeTimes,ones(TT,1),{[],[]},{[],[]});
    initPoint = cbemStructToVector(CBEM_lin,CBEMtoOptimize); 
    nll = nllFunc(initPoint);
else
%% optimize
    CBEMtoOptimize = buildDefaultOptSettings(false,CBEM_lin);
    CBEMtoOptimize.h_spk  = true;
    CBEMtoOptimize.k_s(1) = true;

    nllFunc = @(Xvec) optimizationFunction(Xvec,CBEM_lin,CBEMtoOptimize,SpikeStim,spkHist,spikeTimes,ones(TT,1),{[],[]},{[],[]});

    opts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'Hessian','on','MaxIter',200,'Display','iter');

    initPoint = cbemStructToVector(CBEM_lin,CBEMtoOptimize);

    %%

    [finalPoint,nll] = fminunc(nllFunc,initPoint,opts);

    CBEM_lin_final = cbemVectorToStruct(finalPoint,CBEM_lin,CBEMtoOptimize);
    
    %% setup CBEM
    CBEMp.k_s{1} = [ CBEM_lin_final.k_s{1};1];
    CBEMp.k_s{2} = [-CBEM_lin_final.k_s{1};1];
    CBEMp.h_spk  = CBEM_lin_final.h_spk;

    CBEMp.k_s{3} = CBEM_lin_final.k_s{2};
    if(numel(CBEMp.E_s) > 3)
        CBEMp.k_s{4} = CBEM_lin_final.k_s{3};
    end
end




