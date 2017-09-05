function CBEMtoOptimize = buildDefaultOptSettings(allOn,CBEM)

if(nargin < 1)
    allOn = false;
end
CBEMtoOptimize.k_s = false(length(CBEM.k_s),1);
CBEMtoOptimize.E_s = false(length(CBEM.E_s),1);

CBEMtoOptimize.k_s(:)  = allOn;
CBEMtoOptimize.h_spk   = allOn;
CBEMtoOptimize.g_l     = allOn;
CBEMtoOptimize.b       = allOn;
CBEMtoOptimize.c       = allOn;
CBEMtoOptimize.E_s(:)  = allOn;
CBEMtoOptimize.E_l     = allOn;