

%this script assumes a CBEM with 2 stimulus conductances (exc and inh)
%and 1 or 2 leak condutances for 

doFitForGL1    = false; %if true, fits the leak conductance values. Otherwise, keeps them fixed

initV  = -60; %-62

priorWeight_e1 = 1;%5^2;
priorWeight_e2 = 1;%5^2;
priorWeight_i = priorWeight_e2/5;%5^2;
CBEM.prior.comp.m = 0;
CBEM.prior.comp.d = 0;
CBEM.prior.k_s_lambda{1} = priorWeight_e1;
CBEM.prior.k_s_lambda{2} = priorWeight_i;

CBEM.g_l = 25;
CBEM.log_g_l = log(CBEM.g_l);
totalG = 300 - exp(CBEM.log_g_l);
if(numel(CBEM.E_s) > 3)
    %if using 2 leak conductances (to fit both E_l and g_l) - this is the
    %default setup
    initGl = [(CBEM.E_s(3) - initV) (CBEM.E_s(4) - initV); 1 1]\[-exp(CBEM.log_g_l)*(CBEM.E_l - initV);totalG];
    CBEM.k_s{3} = log(initGl(1));
    CBEM.k_s{4} = log(initGl(2));
    
    CBEM.prior.k_s_sig{3} = 1/10;
    CBEM.prior.k_s_sig{4} = 1/10;
else
    %if using 1 leak conductance (to fit g_l with fixed E_l)
    CBEM.E_l      = initV;
    CBEM.E_s(3) = initV;
    CBEM.k_s{3} = log(totalG);
    
    CBEM.prior.k_s_sig{3} = 0/10;
end


fprintf('Fitting CBEM with single linear conductance...\n');
[CBEM_lin] = fitCBEMwithLinearTransferFR_spatiotemporal(CBEM,SpikeStim,spkHist,Y);

[~,~,CBEMlin_nll]      = fitCBEMwithLinearTransferFR_spatiotemporal(CBEM_lin,SpikeStim,spkHist,Y,true);




addOnesColumnToStim = true; %if true, function adds on a column of 1's to the end of SpikeStim


%%
CBEM_init = CBEM_lin;
% CBEM_init.k_s{1}(:) = 0;
% CBEM_init.k_s{2}(:) = 0;
CBEM_init.k_spatial{1}   =   CBEM_init.k_spatial{1};
CBEM_init.k_temporal{1}  =   CBEM_init.k_temporal{1};
CBEM_init.k_spatial{2}   =   CBEM_init.k_spatial{1};
CBEM_init.k_temporal{2}  =  -CBEM_init.k_temporal{1};
CBEM_init.k_baseline{1} = 20;
CBEM_init.k_baseline{2} = 20;
CBEM_init.prior.k_s_lambda{1} = priorWeight_e2;
CBEM_init.prior.k_s_lambda{2} = priorWeight_i;

fprintf('Fitting CBEM with LN excitatory and inhibitory conductances...\n');
[CBEM_fit] = fitCBEMfull_spatiotemporal(CBEM_init, SpikeStim,spkHist,Y,addOnesColumnToStim,doFitForGL1);


[~,CBEMnll]      = fitCBEMfull_spatiotemporal(CBEM_fit,SpikeStim,     spkHist,     Y,     addOnesColumnToStim,false,true);

fprintf('done.\n');
