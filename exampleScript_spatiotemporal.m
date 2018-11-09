addpath code/
addpath code/CBEMstructHandler
addpath code/fitting
addpath code/utils


%% load Stimulus X and spike train Y (Y is full vector of 1s and 0s)
% example data provided is from a simulated CBEM (CBEM_true)

dt = 1e-4; %important variable! bin size in seconds.

load Data/example_spatiotemporal.mat


nPixels     = [size(X,2) size(X,3)]; 
nSpatialRFs = [size(X,2) size(X,3)];%this will set the spatial basis to be the pixel basis while the temporal basis is the raised cosine

stimFilterRank = 2; %stimulus filter is matrix of this rank. The RF matrix is (w_time x w_spatial') where the w's have stimFilterRank columns. If negative, uses full-rank filter


%% Initialize CBEM for fitting
initializeCell_spatiotemporal; % the initialization uses the same param values as CBEM_true

%this code is only designed to work with one excitatory and one inhibitory
%filter! Modifications would be needed to include more.

%% run fitting routine

doCBEMfit_spatiotemporal;

%% plot filters 
%  The example data given is very short (6 seconds of actual stimulus) -
%  fitted filters won't be exact


k_e_true = reshape((CBEM_true.stimBasisVectors_temporal*CBEM_true.k_temporal{1})*(CBEM_true.stimBasisVectors_spatial*CBEM_true.k_spatial{1})',[],size(X,2),size(X,3));
k_i_true = reshape((CBEM_true.stimBasisVectors_temporal*CBEM_true.k_temporal{2})*(CBEM_true.stimBasisVectors_spatial*CBEM_true.k_spatial{2})',[],size(X,2),size(X,3));
k_e_fit  = reshape((CBEM_fit.stimBasisVectors_temporal*CBEM_fit.k_temporal{1})*(CBEM_fit.stimBasisVectors_spatial*CBEM_fit.k_spatial{1})',[],size(X,2),size(X,3));
k_i_fit  = reshape((CBEM_fit.stimBasisVectors_temporal*CBEM_fit.k_temporal{2})*(CBEM_fit.stimBasisVectors_spatial*CBEM_fit.k_spatial{2})',[],size(X,2),size(X,3));

tts = (1:size(k_e_true,1))*CBEM_true.dt*1e3;

midpoint = [3 3]; %middle point for plotting filter temporal dynamics

[~,k_e_fit_maxT] = max(abs(k_e_fit(:,midpoint(1),midpoint(2))));
[~,k_i_fit_maxT] = max(abs(k_i_fit(:,midpoint(1),midpoint(2))));


figure(1);
clf;
subplot(3,3,1);
hold on
plot(tts,k_e_true(:,midpoint(1),midpoint(2)));
plot(tts,k_i_true(:,midpoint(1),midpoint(2)));

ylabel('filter weight');
xlabel('time (ms)');
title('excitatory filter');
legend({'true filter','estimated filter'});
hold off

subplot(3,3,4)
imagesc(squeeze(k_e_true(k_e_fit_maxT,:,:)));
title('excitatory spatial profile');
subplot(3,3,7)
title('inhibitory spatial profile');
imagesc(squeeze(k_i_true(k_i_fit_maxT,:,:)));


subplot(3,3,5)
imagesc(squeeze(k_e_fit(k_e_fit_maxT,:,:)));
subplot(3,3,8)
imagesc(squeeze(k_i_fit(k_i_fit_maxT,:,:)));


subplot(3,3,2);
hold on
plot(tts,k_e_fit(:,midpoint(1),midpoint(2)));
plot(tts,k_i_fit(:,midpoint(1),midpoint(2)));


ylabel('filter weight');
xlabel('time (ms)');
title('inhibitory filter');
legend({'true filter','estimated filter'});
hold off


subplot(3,3,3);
hold on
plot((1:size(CBEM_true.spkHistBasisVectors,1))*CBEM_true.dt*1e3,CBEM_true.spkHistBasisVectors*CBEM_true.h_spk(1:CBEM_true.spkHistNumBasisVectors));
plot((1:size(CBEM_fit.spkHistBasisVectors,1))*CBEM_fit.dt*1e3  ,CBEM_fit.spkHistBasisVectors*CBEM_fit.h_spk(1:CBEM_fit.spkHistNumBasisVectors));

ylabel('filter weight');
xlabel('time (ms)');
title('spike history filter');
legend({'true filter','estimated filter'});
hold off

figure(2)
subplot(2,2,1)
imagesc(reshape(k_e_true,[],prod(nPixels),1)');
xlabel('time')
ylabel('space')
title('flattened true exc');
subplot(2,2,2)
imagesc(reshape(k_i_true,[],prod(nPixels),1)');
title('flattened true inh');
subplot(2,2,3)
imagesc(reshape(k_e_fit,[],prod(nPixels),1)');
title('flattened estimated exc');
subplot(2,2,4)
imagesc(reshape(k_i_fit,[],prod(nPixels),1)');
title('flattened estimated inh');

%% Simulating from model fit

[Mtsp,spks,V_fit,g_s_fit,l_s_fit] = simulateCBEM(CBEM_fit,SpikeStim,10);