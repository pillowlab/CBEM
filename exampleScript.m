addpath code/
addpath code/CBEMstructHandler
addpath code/fitting
addpath code/utils


%% load Stimulus X and spike train Y (Y is full vector of 1s and 0s)
% example data provided is from a simulated CBEM (CBEM_true)

load Data/example

%% Initialize CBEM for fitting
initializeCell; % the initialization uses the same param values as CBEM_true

%% run fitting routine

doCBEMfit;

%% plot filters 
%  The example data given is very short (6 seconds of actual stimulus) -
%  fitted filters won't be exact

figure(1);
clf;
subplot(1,3,1);
hold on
plot((1:size(CBEM_true.stimBasisVectors,1))*CBEM_true.dt*1e3,CBEM_true.stimBasisVectors*CBEM_true.k_s{1}(1:CBEM_true.stimNumBasisVectors));
plot((1:size(CBEM_fit.stimBasisVectors,1))*CBEM_fit.dt*1e3  ,CBEM_fit.stimBasisVectors*CBEM_fit.k_s{1}(1:CBEM_fit.stimNumBasisVectors));
%CBEM_true.k_s{1}(1:10) holds the filter weights, CBEM_true.k_s{1}(11) is a baseline weight 

ylabel('filter weight');
xlabel('time (ms)');
title('excitatory filter');
legend({'true filter','estimated filter'});
hold off


subplot(1,3,2);
hold on
plot((1:size(CBEM_true.stimBasisVectors,1))*CBEM_true.dt*1e3,CBEM_true.stimBasisVectors*CBEM_true.k_s{2}(1:CBEM_true.stimNumBasisVectors));
plot((1:size(CBEM_fit.stimBasisVectors,1))*CBEM_fit.dt*1e3  ,CBEM_fit.stimBasisVectors*CBEM_fit.k_s{2}(1:CBEM_fit.stimNumBasisVectors));

ylabel('filter weight');
xlabel('time (ms)');
title('inhibitory filter');
legend({'true filter','estimated filter'});
hold off


subplot(1,3,3);
hold on
plot((1:size(CBEM_true.spkHistBasisVectors,1))*CBEM_true.dt*1e3,CBEM_true.spkHistBasisVectors*CBEM_true.h_spk(1:CBEM_true.spkHistNumBasisVectors));
plot((1:size(CBEM_fit.spkHistBasisVectors,1))*CBEM_fit.dt*1e3  ,CBEM_fit.spkHistBasisVectors*CBEM_fit.h_spk(1:CBEM_fit.spkHistNumBasisVectors));

ylabel('filter weight');
xlabel('time (ms)');
title('spike history filter');
legend({'true filter','estimated filter'});
hold off

%% Simulating from model fit

[Mtsp,spks,V_fit,g_s_fit,l_s_fit] = simulateCBEM(CBEM_fit,SpikeStim,10);