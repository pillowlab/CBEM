function [Mtsp,spks,V_fit,g_s_fit,l_s_fit] = simulateCBEM(CBEM,Stim,nReps)
%% function [Mtsp,spks,V_fit,g_s_fit,l_s_fit] = simulateCBEM(CBEM,Stim,nReps)
% Input:
%  CBEM = the CBEM struct to simulate
%  Stim = matrix: stimulus to simulate convolved with the CBEM's stim basis
%  nReps = number of trials to simulate
%
% Output:
%  Mtsp = cell array, each cell is vector of spike times for each repeat
%  spks = sparse matrix. 1's are the spike times given in Mtsp
%  V_fit = Vector of the voltage trace for the CBEM given the Stim
%  g_s_fit = matrix giving CBEM's conductances
%  l_s_fit = matrix of the linear stage of the conductances (pass through
%            the conductance nonlinearities to get g_s_fit)


Mtsp = cell(nReps,1);

TT = size(Stim,1);
nConds = length(CBEM.k_s);
g_s_fit = zeros(TT,nConds);
l_s_fit = zeros(TT,nConds);


for cc = 1:nConds
    if(CBEM.condType(cc) == 2)
        %first, check if Stim needs and extra column of 1's
        if(size(Stim,2) + 1 == length(CBEM.k_s{cc}))
            Stim = [Stim ones(TT,1)];
        end
        l_s_fit(:,cc) = Stim*CBEM.k_s{cc};
        g_s_fit(:,cc) = CBEM.f_s{cc}(l_s_fit(:,cc),   CBEM.g_s_bar(cc));
    elseif(CBEM.condType(cc) == 1)
        error('Does not function with AHP currents yet');
    elseif(CBEM.condType(cc) == 3)
        l_s_fit(:,cc) = ones(size(Stim,1),1)*CBEM.k_s{cc};
        g_s_fit(:,cc) = CBEM.f_s{cc}(l_s_fit(:,cc),CBEM.g_s_bar(cc));
    end
end

c_cond = sum(g_s_fit,2)      + CBEM.g_l;
b_cond = g_s_fit * CBEM.E_s + CBEM.g_l*CBEM.E_l;
exc = exp(c_cond*CBEM.dt);
I = b_cond./c_cond.*(exc-1);
I(1) = I(1) + CBEM.E_l;
M = spdiags([-1*ones(TT,1) exc], [-1 0],TT,TT) ;
V_fit  = M\(I);

spks = sparse(zeros(nReps,TT));

h_spk = CBEM.spkHistBasisVectors*CBEM.h_spk;
h_spk = h_spk(end:-1:1);
HT = length(h_spk);

fprintf('Simulating CBEM...\n');

% rn = rand(nReps,TT);
for tt = HT+1:TT
    spkHist = spks(:,(tt-HT):tt-1)*h_spk;
%     spkRate = exp(CBEM.spikeNonlinearity.c*(CBEM.spikeNonlinearity.b + V_fit(tt)) + spkHist + log(CBEM.dt));
    spkRate = CBEM.dt*CBEM.spikeNonlinearity.f(V_fit(tt),CBEM.spikeNonlinearity.c,CBEM.spikeNonlinearity.b,spkHist,CBEM.spikeNonlinearity.alpha);
    ps = 1 - exp(-spkRate);
    
    spks(:,tt) = sparse(binornd(1,ps)); %#ok<SPRIX>
%     spks(ps < rn(:,tt),tt) = 1; %#ok<SPRIX>
    if(mod(tt,10000) == 0)
        sr = full(mean(sum(spks(:,1:tt),2)));
        sr = (sr/tt)/CBEM.dt;
        fprintf(' tt = %d / %d ( spike rate = %2.2f )\n',tt,TT,sr);
    end
end


fprintf('  summarizing spikes...\n');
for ii = 1:nReps
    Mtsp{ii} = find(spks(ii,:) == 1)' * CBEM.dt;
end
fprintf('done\n');