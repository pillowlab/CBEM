function [CBEMp,nll,nlpost] = fitCBEMfull_spatiotemporal(CBEM,SpikeStim,spkHist,Y,addOnesColumnToStim,doFitForGL,noFit)

% noFit = if true: quick rapper to get NLL and log posterior instead of
% fitting

%% setup CBEMp
CBEMp = CBEM;



spikeTimes   = find(Y == 1);
TT = size(SpikeStim,1);



if(nargin < 7)
    noFit = false;
end

if(noFit)

    for ii = 1:2
        f = CBEMp.k_temporal{ii}*CBEMp.k_spatial{ii}';
        CBEMp.k_s{ii} = [f(:); CBEMp.k_baseline{ii}];
    end    
        
    
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
    %alternate between fitting spatial and temporal filters
    nll_prev = inf;
    maxIters = 100;
    for iter = 1:maxIters
        fprintf('Fitting iteration %d.\n',iter);
        for ff = 1:2
            if(ff == 1)
                fprintf('  fitting spatial filters\n');
                SpikeStim_c = cell(2,1);
                for ii = 1:2
                    CBEMp.k_s{ii} = [CBEMp.k_spatial{ii}(:); CBEMp.k_baseline{ii}];
                    if(isfield(CBEMp,'prior'))
                        CBEMp.prior.k_s_sig{ii} = blkdiag(CBEMp.prior.k_s_lambda{ii}*kron(CBEMp.k_temporal{ii}'*CBEMp.k_temporal{ii},eye(size(CBEMp.k_spatial{ii},1))),0);
                    end
                    B_s = zeros(size(SpikeStim,2),CBEMp.stimNumBasisVectors_temporal*CBEMp.stimFilterRank);
                    for jj = 1:CBEMp.stimFilterRank
                        idx = (jj-1)*CBEMp.stimNumBasisVectors_spatial + (1:CBEMp.stimNumBasisVectors_spatial);
                        B_s(:,idx) = kron(eye(size(CBEMp.k_spatial{ii},1)),CBEMp.k_temporal{ii}(:,jj));
                    end
                    SpikeStim_c{ii} = [SpikeStim*B_s ones(TT,1)];
                end
            elseif(ff == 2)
                fprintf('  fitting temporal filters\n');
                SpikeStim_c = cell(2,1);
                for ii = 1:2
                    CBEMp.k_s{ii} = [CBEMp.k_temporal{ii}(:); CBEMp.k_baseline{ii}];
                    if(isfield(CBEMp,'prior'))
                        CBEMp.prior.k_s_sig{ii} = blkdiag(CBEMp.prior.k_s_lambda{ii}*kron(CBEMp.k_spatial{ii}'*CBEMp.k_spatial{ii},eye(size(CBEMp.k_temporal{ii},1))),0);
                    end

                    B_t = kron(CBEMp.k_spatial{ii},eye(size(CBEMp.k_temporal{ii},1)));

                    SpikeStim_c{ii} = [SpikeStim*B_t ones(TT,1)];
                end
            end
            CBEMtoOptimize = buildDefaultOptSettings(false,CBEMp);
            CBEMtoOptimize.h_spk  = true;
            CBEMtoOptimize.k_s(1:2) = true;
            CBEMtoOptimize.k_s(3:end) = doFitForGL;
            
            CBEMp.condType(2) = 4;

            nllFunc = @(Xvec) optimizationFunction(Xvec,CBEMp,CBEMtoOptimize,SpikeStim_c{1},spkHist,spikeTimes,ones(TT,1),SpikeStim_c{2},{[],[]});

            opts_grad = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'Hessian','off','MaxIter',500,'Display','iter','FunctionTolerance',1e-6,'StepTolerance',1e-6);
            opts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'Hessian','on','MaxIter',500,'Display','iter','FunctionTolerance',1e-8,'StepTolerance',1e-8);

            initPoint = cbemStructToVector(CBEMp,CBEMtoOptimize);

            %
            fprintf('Fitting with gradient only...\n');
            initPoint = fminunc(nllFunc,initPoint,opts_grad);
            fprintf('Continuing fit with Hessian...\n');
            [finalPoint,nll] = fminunc(nllFunc,initPoint,opts);

            CBEMp = cbemVectorToStruct(finalPoint,CBEMp,CBEMtoOptimize);
            
            %%
            for ii = 1:2
                CBEMp.k_baseline{ii} = CBEMp.k_s{ii}(end);
                
                if(ff == 1)
                    %renormalizes for identifiability
                    spatial_rf  = reshape(CBEMp.k_s{ii}(1:end-1),[],CBEMp.stimFilterRank);
                    temporal_rf = CBEMp.k_temporal{ii};
                    try 
                        d = chol(spatial_rf'*spatial_rf);
                        temporal_rf2 = temporal_rf*d';
                        spatial_rf2  = spatial_rf/d;
                    catch
                        sn =  max(1e-4,sqrt(sum(spatial_rf.^2,1)));
                        temporal_rf2 = temporal_rf.*sn;
                        spatial_rf2 = spatial_rf./sn;
                    end
                    CBEMp.k_temporal{ii} = temporal_rf2;
                    CBEMp.k_spatial{ii}  = spatial_rf2;
                elseif(ff == 2)
                    CBEMp.k_temporal{ii} = reshape(CBEMp.k_s{ii}(1:end-1),[],CBEMp.stimFilterRank);
                end
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
    
    
    CBEMp.condType(2) = 2;
end

