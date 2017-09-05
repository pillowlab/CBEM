%takes a vectorized CBEM and converts it back into a structure
% the vector is assumed to contain only certain values as defined in the
% CBEMtoOptimize input. Everything else is taken from the input CBEM.
function [CBEM] = cbemVectorToStruct(X,CBEM,CBEMtoOptimize)


allOn = false;
if(nargin < 3)
    allOn = true;
end

% X = [CBEM.k_s;
%      CBEM.h_spk;
%      CBEM.spikeNonlinearity.c
%      CBEM.spikeNonlinearity.b
%      CBEM.E_s;
%      CBEM.E_l;];

ce = 1;

nConds = length(CBEM.k_s);
for ii = 1:nConds
    if(CBEMtoOptimize.k_s(ii) || allOn)
        CBEM.k_s{ii} = X(ce:ce+length(CBEM.k_s{ii})-1);
        ce = ce + length(CBEM.k_s{ii});
    end
end

if(CBEMtoOptimize.h_spk || allOn)
    CBEM.h_spk = X(ce:ce+length(CBEM.h_spk)-1);
    ce = ce + length(CBEM.h_spk);
end



if(~isfield(CBEM,'log_g_l'))
    CBEM.log_g_l = log(CBEM.g_l);
end
if(~isfield(CBEM,'log_c'))
    CBEM.spikeNonlinearity.log_c = log(CBEM.spikeNonlinearity.c);
end


if(CBEMtoOptimize.c || allOn)
    CBEM.spikeNonlinearity.log_c = X(ce:ce+length(CBEM.spikeNonlinearity.c)-1);
    CBEM.spikeNonlinearity.c = exp(CBEM.spikeNonlinearity.log_c);
    ce = ce + length(CBEM.spikeNonlinearity.c); 
end
if(CBEMtoOptimize.b || allOn)
    CBEM.spikeNonlinearity.b = X(ce:ce+length(CBEM.spikeNonlinearity.b)-1);
    ce = ce + length(CBEM.spikeNonlinearity.b); 
end

for ii = 1:nConds
    if(CBEMtoOptimize.E_s(ii) || allOn)
        CBEM.E_s(ii) = X(ce:ce+length(CBEM.E_s(ii))-1);
        ce = ce + length(CBEM.E_s(ii));
    end
end
if(CBEMtoOptimize.E_l || allOn)
    CBEM.E_l = X(ce:ce+length(CBEM.E_l)-1);
    ce = ce + length(CBEM.E_l);%#ok<NASGU>
end