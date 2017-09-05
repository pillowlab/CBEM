%Turns the CBEM datastructure into a vector for optimization routines
% the second argument allows to turn on/off elements for optimization


function [X] = cbemStructToVector(CBEM,CBEMtoOptimize)

if(~isfield(CBEM,'log_g_l'))
    CBEM.log_g_l = log(CBEM.g_l);
end
if(~isfield(CBEM,'log_c'))
    CBEM.spikeNonlinearity.log_c = log(CBEM.spikeNonlinearity.c);
end

if(nargin < 2)

    k_s = [];
    nConds = length(CBEM.k_s);
    for ii = 1:nConds
        k_s = [k_s;CBEM.k_s{ii}];  %#ok<*AGROW>
    end
    
    X = [k_s;
         CBEM.h_spk;
         CBEM.spikeNonlinearity.c;
         CBEM.spikeNonlinearity.b;
         CBEM.E_s;
         CBEM.E_l];
 
else
    
    X = [];
    
    nConds = length(CBEM.k_s);
    for ii = 1:nConds
        if(CBEMtoOptimize.k_s(ii))
            X = [X;CBEM.k_s{ii}]; 
        end
    end
    
    if(CBEMtoOptimize.h_spk)
        X = [X;CBEM.h_spk];
    end
    
    if(CBEMtoOptimize.c)
        X = [X;CBEM.spikeNonlinearity.log_c];
    end
    if(CBEMtoOptimize.b)
        X = [X;CBEM.spikeNonlinearity.b];
    end
    
    for ii = 1:nConds
        if(CBEMtoOptimize.E_s(ii))
            X = [X;CBEM.E_s(ii)];
        end
    end
    
    if(CBEMtoOptimize.E_l)
        X = [X;CBEM.E_l];
    end
end
     