function [CBEMp, CBEM_lin,nll] = fitCBEMwithLinearTransferFR_spatiotemporal(CBEM,SpikeStim,spkHist,Y,noFit)
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
CBEM_lin.k_spatial  = CBEM.k_spatial([1,3:end]);
CBEM_lin.k_temporal = CBEM.k_temporal([1,3:end]);
CBEM_lin.k_s{1}   = randn(size(CBEM_lin.k_s{1}(1:end-1)))*0.0;
CBEM_lin.E_s      = CBEM.E_s([1,3:end]);
CBEM_lin.g_s_bar  = CBEM.g_s_bar([1,3:end]);

CBEM_lin.stimFilterRank = CBEM.stimFilterRank;

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
    CBEM_lin.k_s{1} = CBEM_lin.k_temporal{1}*CBEM_lin.k_spatial{1}';
    CBEM_lin.k_s{1} = CBEM_lin.k_s{1}(:);
    
    CBEMtoOptimize = buildDefaultOptSettings(false,CBEM_lin);
    CBEMtoOptimize.h_spk  = true;

    nllFunc = @(Xvec) optimizationFunction(Xvec,CBEM_lin,CBEMtoOptimize,SpikeStim,spkHist,spikeTimes,ones(TT,1),{[],[]},{[],[]});
    initPoint = cbemStructToVector(CBEM_lin,CBEMtoOptimize); 
    nll = nllFunc(initPoint);
else
%% optimize
    %alternate between fitting spatial and temporal filters
    nll_prev = inf;
    maxIters = 100;
    for iter = 1:maxIters
        fprintf('Fitting iteration %d.\n',iter);
        for ff = 1:2
            if(ff == 1)
                fprintf('  fitting spatial filters\n');
                CBEM_lin.k_s{1} = CBEM_lin.k_spatial{1}(:);
                if(isfield(CBEM_lin,'prior'))
                    CBEM_lin.prior.k_s_sig{1} = CBEM_lin.prior.k_s_lambda{1}*kron(CBEM_lin.k_temporal{1}'*CBEM_lin.k_temporal{1},eye(size(CBEM_lin.k_spatial{1},1)));
                end
                B_s = zeros(size(SpikeStim,2),CBEM.stimNumBasisVectors_temporal*CBEM_lin.stimFilterRank);
                for ii = 1:CBEM_lin.stimFilterRank
                    idx = (ii-1)*CBEM.stimNumBasisVectors_spatial + (1:CBEM.stimNumBasisVectors_spatial);
                    B_s(:,idx) = kron(eye(size(CBEM_lin.k_spatial{1},1)),CBEM_lin.k_temporal{1}(:,ii));
                end
                SpikeStim_c = SpikeStim*B_s;
                
            elseif(ff == 2)
                fprintf('  fitting temporal filters\n');
                CBEM_lin.k_s{1} = CBEM_lin.k_temporal{1}(:);
                if(isfield(CBEM_lin,'prior'))
                    CBEM_lin.prior.k_s_sig{1} = CBEM_lin.prior.k_s_lambda{1}*kron(CBEM_lin.k_spatial{1}'*CBEM_lin.k_spatial{1},eye(size(CBEM_lin.k_temporal{1},1)));
                end
                
                B_t = kron(CBEM_lin.k_spatial{1},eye(size(CBEM_lin.k_temporal{1},1)));
                
                SpikeStim_c = SpikeStim*B_t;
            end
            
            CBEMtoOptimize = buildDefaultOptSettings(false,CBEM_lin);
            CBEMtoOptimize.h_spk  = true;
            CBEMtoOptimize.k_s(1) = true;

            nllFunc = @(Xvec) optimizationFunction(Xvec,CBEM_lin,CBEMtoOptimize,SpikeStim_c,spkHist,spikeTimes,ones(TT,1),{[],[]},{[],[]});

            opts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'Hessian','on','MaxIter',200,'Display','iter');

            initPoint = cbemStructToVector(CBEM_lin,CBEMtoOptimize);

            %%

            [finalPoint,nll] = fminunc(nllFunc,initPoint,opts);

            CBEM_lin2 = cbemVectorToStruct(finalPoint,CBEM_lin,CBEMtoOptimize);
            
            %%
            CBEM_lin.h_spk = CBEM_lin2.h_spk;
            CBEM_lin.k_s{2:end} = CBEM_lin2.k_s{2:end};
            if(ff == 1)
                %renormalizes for identifiability
                spatial_rf  = reshape(CBEM_lin2.k_s{1},[],CBEM_lin.stimFilterRank);
                sn =  sqrt(sum(spatial_rf.^2,1));
                CBEM_lin.k_temporal{1} = CBEM_lin.k_temporal{1}.*sn;
                CBEM_lin.k_spatial{1}  = spatial_rf./sn;
            elseif(ff == 2)
                CBEM_lin.k_temporal{1} = reshape(CBEM_lin2.k_s{1},[],CBEM_lin.stimFilterRank);
            end
        end
        if(nll >= nll_prev - 1e-3)
            fprintf('Local minimum found.\n');
            break;
        elseif(iter == maxIters)
            fprintf('Maximum fitting iterations acheived. Terminating early.\n');
            break;
        else
            nll_prev = nll;
        end
    end
    CBEM_lin_final = CBEM_lin;
    
    
    %% setup CBEM
    CBEMp.k_spatial{1} = CBEM_lin_final.k_spatial{1};
    CBEMp.k_temporal{1} = CBEM_lin_final.k_temporal{1};
    CBEMp.k_spatial{2} = CBEM_lin_final.k_spatial{1};
    CBEMp.k_temporal{2} = -CBEM_lin_final.k_temporal{1};
    CBEMp.h_spk  = CBEM_lin_final.h_spk;

    CBEMp.k_s{3} = CBEM_lin_final.k_s{2};
    if(numel(CBEMp.E_s) > 3)
        CBEMp.k_s{4} = CBEM_lin_final.k_s{3};
    end
end




