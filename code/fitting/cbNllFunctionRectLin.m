function [LL,dLL,d2LL] = cbNllFunctionRectLin(spkTimes,CBEM,CBEMtoOptimize, fitLength, varargin) 

%for a rectified linear spike rate nonlinearity
% spkTimes = spk times
% k_s      = basis vector weights for conductances
% h_spk    = spike history filter
% g_l      = constant leak conductance
% E_l      = leak reversal potential
% E_m      = reversal potentials for other conductances - must be a column vector 

% condType = 2+ - stims, 1 - spk 
%           index into varargin for what variables the conductance cares about
%           spike history always uses index 1
% varargin = the inputs, the spike history term is expected to be in
%            varargin 1 (this could hold any term you'd like to make linear
%            in the model, but spike history is the sensible one for
%            refractory periods and all that)
%calculates conductances and their derivative+Hessian (if needed)
%see the LaTeX file for the derivation

%constants for numerical stability (divide by zero and stuff)
EXP_MAX       = 500;  %exp(x) = exp(min(EXP_MAX,x))
LOG_MIN       = 1e-300; % log(x) = log(max(LOG_MIN,x))
POS_DENOM_MIN = 1e-15; %1/x = 1/max(POS_DENOM_MIN,x) where x>0 should hold


T = size(varargin{1},1); %num timesteps
nConds = length(CBEM.k_s); %number of conductances in CBEM
dt = CBEM.dt; %timebin length

l_e = zeros(T,nConds); 
g_s    = zeros(T,nConds);
if(nargout > 1)
    d_g_s  = zeros(T,nConds);
    if(nargout > 2)
        d2_g_s = zeros(T,nConds);
    end
end

%display('here');

for cc = 1:nConds
    %if(~isempty(CBEM.k_s_section{cc}))
    l_e(:,cc) = varargin{CBEM.condType(cc)}*CBEM.k_s{cc};
        
    if(nargout > 1 && CBEMtoOptimize.k_s(cc))
        if(nargout > 2)
            if(~isfield(CBEM,'f_s_mex') || size(CBEM.f_s_mex,1) < cc || size(CBEM.f_s_mex,2) < 3 || isempty(CBEM.f_s_mex{cc,3}))
                [g_s(:,cc), d_g_s(:,cc), d2_g_s(:,cc)] = CBEM.f_s{cc}(l_e(:,cc),CBEM.g_s_bar(cc)); %f_e should be logOnePlusExpX
            else
                CBEM.f_s_mex{cc,3}(l_e(:,cc),CBEM.g_s_bar(cc),cc,g_s, d_g_s, d2_g_s); 
            end
        else
            if(~isfield(CBEM,'f_s_mex') || size(CBEM.f_s_mex,1) < cc || size(CBEM.f_s_mex,2) < 2  || isempty(CBEM.f_s_mex{cc,2}))
                [g_s(:,cc), d_g_s(:,cc)] = CBEM.f_s{cc}(l_e(:,cc),CBEM.g_s_bar(cc));
            else
                CBEM.f_s_mex{cc,2}(l_e(:,cc),CBEM.g_s_bar(cc),cc,g_s, d_g_s);
            end
        end
    else
        if(~isfield(CBEM,'f_s_mex') || size(CBEM.f_s_mex,1) < cc || size(CBEM.f_s_mex,2) < 1 || isempty(CBEM.f_s_mex{cc,1}))
            g_s(:,cc) = CBEM.f_s{cc}(l_e(:,cc),CBEM.g_s_bar(cc));
        else
            CBEM.f_s_mex{cc,1}(l_e(:,cc),CBEM.g_s_bar(cc),cc,g_s);
        end
    end
end
%for optimization purposes, this function accepts log of g_l instead of g_l
%(to keep the contraint g_l>0)
g_l = exp(CBEM.log_g_l);
c = g_l + sum(g_s,2);


b = g_l*CBEM.E_l + g_s*CBEM.E_s;
bc = b./c;

exc = exp(min(EXP_MAX,c*dt));
I = bc.*(exc-1);


if(isfield(CBEMtoOptimize,'outputVoltage') &&  CBEMtoOptimize.outputVoltage && isfield(CBEMtoOptimize,'initVoltage'))
    I(1) = I(1) + CBEM.E_l;
else
    I(1) = I(1) + CBEMtoOptimize.initVoltage;
end

% M = spdiags([-ones(T,1) exc], [-1 0],T,T) ;
% M = spdiags(exc, 0,T,T) -spdiags(ones(T,1), -1,T,T);


M = spdiags([-ones(T,1) exc], [-1 0],T,T) ;
V = M\I;

%fixed threshold -> spike rate at time t is exp(c*(V(t)+b))
%otherwuse rate is exp(c*V(t)+b)
alpha = CBEM.spikeNonlinearity.alpha;

CBEM.spikeNonlinearity.c = exp(CBEM.spikeNonlinearity.log_c); 
l = alpha*V + varargin{1}*CBEM.h_spk + CBEM.spikeNonlinearity.b;%,1e4),-1e4)
expl  = exp(min(EXP_MAX, l));
nexpl = exp(min(EXP_MAX,-l));

r_h = CBEM.spikeNonlinearity.c*log(1+expl); 
r = dt*r_h; %spike rate at each time
r_nexp = exp(min(EXP_MAX,-r));%r_nexp = max(1e-16,min(exp(-r),1e30));
r_exp  = exp(min(EXP_MAX,r));%r_exp  = max(1e-16,min(exp(r), 1e30));
sum_r  = sum(r);

LL = sum(log(max(LOG_MIN, 1-r_nexp(spkTimes)))) + sum(r(spkTimes)) - sum_r; 

if( isinf(LL))
    fprintf('f is inf! (cbNllFunctionRectLin)\n');
    LL = -1e30;
end
if( isnan(LL))
    fprintf('f is nan! (cbNllFunctionRectLin)\n');
    LL = -1e30;
end

if(g_l > 1000) %no time constant faster than 1ms 
    LL = -1e30;
end
if(g_l < 5) %no time constant slower than 200ms 
    LL = -1e30;
end


if(CBEM.spikeNonlinearity.c <= 0)
    LL = -1e30;
end

if(isfield(CBEM,'prior'))
    usePrior = isfield(CBEM.prior,'k_s_sig') & isfield(CBEM.prior,'k_s_sig');
    
    if(usePrior)
        for cc = 1:nConds
            if(length(CBEM.prior.k_s_sig) >= cc && CBEMtoOptimize.k_s(cc))
                
                LL = LL - 1/2*(CBEM.k_s{cc}'*(CBEM.prior.k_s_sig{cc}*CBEM.k_s{cc}));
            end
        end
        
        if(CBEMtoOptimize.h_spk)
            LL = LL - 1/2*(CBEM.h_spk'*(CBEM.prior.h_spk_sig*CBEM.h_spk));
        end
        
        if(CBEMtoOptimize.b)
            LL = LL - 1/2*(CBEM.prior.b_sig*(CBEM.spikeNonlinearity.b-CBEM.prior.b_mu)^2);
        end
        
    end
else
    usePrior = false;
end

%% derivatives
if(nargout > 1)
    if(nargin >=7)
        dLL = zeros(fitLength,1);
        if(nargout > 2)
            d2LL = zeros(fitLength,fitLength);
        else
            d2LL = 0;
        end
    else
        if(nargout > 2)
            d2LL = zeros(1,1);
        else
            d2LL = 0;
        end
        dLL = 0;
    end
    
    %% derivatives for each conductance
    conductancesToOptimize = find(CBEMtoOptimize.k_s);
    numCondToOptimize = length(conductancesToOptimize);
    
    hessIndices = cell(numCondToOptimize,1);
    m_s = zeros(T,numCondToOptimize);
    p_s = zeros(T,numCondToOptimize);
    %V_s = zeros(T,numCondToOptimize);
    gamma_s = zeros(T,numCondToOptimize);
    
    
    for cc = 1:numCondToOptimize
        cNum = conductancesToOptimize(cc);
        if(cc == 1)
            hessIndices{cc} = 1:length(CBEM.k_s{cNum});
            c2 = c.^2; %this might be calculated previously
            q = -ones(T,1);
            q1 = 1./max(POS_DENOM_MIN,(r_exp(spkTimes) - 1));
            q2 = r_nexp(spkTimes)./max(POS_DENOM_MIN,(1-r_nexp(spkTimes)));
            q(spkTimes) = min(q1,q2);
            
            a = 1./(1+nexpl); 
            qa = q./(1+nexpl);%q.*a;
            
%             n_M = spdiags([zeros(T,1) qa -1*ones(T,1)],[-1 0 1],T,T)\exc;
            n_M = spdiags([zeros(T,1) exc -1*ones(T,1)],[-1 0 1],T,T)\qa;
        else
            hessIndices{cc} = (1:length(CBEM.k_s{cNum})) + hessIndices{cc-1}(end);
        end
        
        eb_c2 = CBEM.E_s(cNum)./c - bc./c;
        p_s(:,cc) = d_g_s(:,cNum).*eb_c2;
        m_s(:,cc) = dt*d_g_s(:,cNum).*exc;
        r_s = bc.*m_s(:,cc) + p_s(:,cc).*(exc-1);
        gamma_s(:,cc) = m_s(:,cc).*V - r_s;
        
        %g_k_s2 = -(CBEM.spikeNonlinearity.c*dt*alpha)*kcMtimesVector(Stim_gpu,n_M.*gamma_s(:,cc));
        g_k_s = -(CBEM.spikeNonlinearity.c*dt*alpha)* (varargin{CBEM.condType(cNum)}'*(n_M.*gamma_s(:,cc)));
%         g_k_s = -(CBEM.spikeNonlinearity.c*dt*alpha)* ((varargin{CBEM.condType(cNum)}.*gamma_s(:,cc))'*n_M);
                
        
        if(usePrior)
            if(length(CBEM.prior.k_s_sig) >= cNum)
                
                g_k_s = g_k_s - (CBEM.prior.k_s_sig{cNum}*CBEM.k_s{cNum});
            end
        end
        
        dLL(hessIndices{cc}) = g_k_s;

        if(nargout > 2)
            
            if(cc == 1)
                %c3 = c.^3;
                u = spalloc(T,1,length(spkTimes));
                u1 = -dt*r_nexp(spkTimes)./max(POS_DENOM_MIN,(r_nexp(spkTimes).*r_nexp(spkTimes) + 1 - 2*r_nexp(spkTimes)));%#ok<*SPRIX>
                u2 = -dt*r_exp(spkTimes)./max(POS_DENOM_MIN,(r_exp(spkTimes ).*r_exp(spkTimes ) + 1 - 2*r_exp(spkTimes )));
                u(spkTimes) = max(u1,u2);
                
                d = a.*(1-a);%1./(2+expl+nexpl);
                
                %sq  = (q.*d + CBEM.spikeNonlinearity.c*(u.*a.^2));
                sq = (q.*d + CBEM.spikeNonlinearity.c*(u.*a.^2));
                sq2 = alpha*sq;
                
            end

            
            ds = d_g_s(:,cNum).^2;
            
            
            q_c  = (d2_g_s(:,cNum)./c - 2*(d_g_s(:,cNum).^2./c.^2)).*(CBEM.E_s(cNum) - bc);
            m_cc = dt.*(d2_g_s(:,cNum) + dt*ds).*exc;
            
            r_cc = 2*p_s(:,cc).*m_s(:,cc) + bc.*m_cc + q_c.*(exc-1);
            
            gamma_cc = m_cc.*V - r_cc;


%             M_d = exc.^2./maxabs(gamma_s(:,cc).^2.*sq2,POS_DENOM_MIN);
%             M_d(2:end) = M_d(2:end) + 1./maxabs(gamma_s(2:end,cc).*gamma_s(2:end,cc).*sq2(1:end-1),POS_DENOM_MIN);
%             M_u2 = [-exc(1:end-1)./(gamma_s(2:end,cc).*gamma_s(1:end-1,cc).*sq2(1:end-1));0];
%             M_l2 = [0;-exc(1:end-1)./(gamma_s(1:end-1,cc).*gamma_s(2:end,cc).*sq2(1:end-1))];
%             M_u = [0;-exc(1:end-1)./maxabs(gamma_s(2:end,cc).*gamma_s(1:end-1,cc).*sq2(1:end-1),POS_DENOM_MIN)];
%             M_l = [-exc(1:end-1)./maxabs(gamma_s(1:end-1,cc).*gamma_s(2:end,cc).*sq2(1:end-1),POS_DENOM_MIN);0];
            
%             h_k_s = kcCBGLM_hess2(Stim_gpu,Stim_gpu, space_gpu, a_gpu, b_gpu, c_gpu, M_d, M_l2, M_u2, gamma_cc.*n_M, n_M.*m_s(:,cc),n_M.*m_s(:,cc),gamma_s(:,cc),gamma_s(:,cc),exc );
            
            %h_k_s = CBEM_hess2(varargin{CBEM.condType(cNum)},varargin{CBEM.condType(cNum)}, M_d, M_l, M_u, gamma_cc.*n_M, n_M.*m_s(:,cc),n_M.*m_s(:,cc),gamma_s(:,cc),gamma_s(:,cc),exc );
            h_k_s = CBEM_hess4(varargin{CBEM.condType(cNum)},varargin{CBEM.condType(cNum)}, sq2, gamma_cc.*n_M, n_M.*m_s(:,cc),n_M.*m_s(:,cc),gamma_s(:,cc),gamma_s(:,cc),M );
                
            
            h_k_s = (CBEM.spikeNonlinearity.c*alpha*dt)*h_k_s;

            if(usePrior)
                if(length(CBEM.prior.k_s_sig) >= cNum)
                    sig = CBEM.prior.k_s_sig{cNum};
                    kp = sig*CBEM.k_s{cc};
                    nn  = CBEM.k_s{cc}'*(kp);
                    pen = (sig*sqrt(nn) - 1/(2*sqrt(nn))*(kp*kp'))./nn;
                    if(nn <= 1e-15)
                        pen = 0;
                    end
                    h_k_s = h_k_s  - pen;
                end
            end
            d2LL(hessIndices{cc}, hessIndices{cc}) =  h_k_s;

            %% mixed with previous currents
            for ccp = 1:(cc-1)
                cNump = conductancesToOptimize(ccp);
                %eb_c2 = CBEM.E_s(cNum)./c - b./c2;
                %p_s(:,cc) = d_g_s(:,cNum).*eb_c2;
                
                q_cp = (-(CBEM.E_s(cNum)+CBEM.E_s(cNump)) + 2*bc).*d_g_s(:,cNum).*d_g_s(:,cNump)./c2;
                
                m_cp = dt^2*(d_g_s(:,cNump).*d_g_s(:,cNum).*exc);
                r_cp = bc.*m_cp+p_s(:,ccp).*m_s(:,cc)+q_cp.*(exc-1) + p_s(:,cc).*m_s(:,ccp);
                gamma_cp = m_cp.*V - r_cp;% + m_s(:,cc).*V_s(:,ccp)


%                 M_d = exc.^2./maxabs(gamma_s(:,cc).*gamma_s(:,ccp).*sq2,POS_DENOM_MIN) ;
%                 M_d(2:end) = M_d(2:end) + 1./maxabs(gamma_s(2:end,cc).*gamma_s(1:end-1,ccp).*sq2(1:end-1),POS_DENOM_MIN);
%                 M_u = [0;-exc(1:end-1)./maxabs(gamma_s(2:end,ccp).*gamma_s(1:end-1,cc).*sq2(1:end-1),POS_DENOM_MIN)];
%                 M_l = [-exc(1:end-1)./maxabs(gamma_s(1:end-1,cc).*gamma_s(2:end,ccp).*sq2(1:end-1),POS_DENOM_MIN);0];
%                 h_ks2 = CBEM_hess2(varargin{CBEM.condType(cNum)},varargin{CBEM.condType(cNump)}, M_d, M_l, M_u, gamma_cp.*n_M, n_M.*m_s(:,cc),n_M.*m_s(:,ccp),gamma_s(:,cc),gamma_s(:,ccp),exc );
                h_ks2 = CBEM_hess4(varargin{CBEM.condType(cNum)},varargin{CBEM.condType(cNump)}, sq2, gamma_cp.*n_M, n_M.*m_s(:,cc),n_M.*m_s(:,ccp),gamma_s(:,cc),gamma_s(:,ccp),M );
                
                h_ks2 = (CBEM.spikeNonlinearity.c*alpha*dt)*h_ks2;
                
                d2LL(hessIndices{cc},  hessIndices{ccp}) = h_ks2';
                d2LL(hessIndices{ccp}, hessIndices{cc})  = h_ks2;
            end
        end

    end


    %% h_spk
    if(CBEMtoOptimize.h_spk)
        if(isempty(conductancesToOptimize))
            %c2 = c.^2;
            q = -ones(T,1);
            q1 = 1./max(POS_DENOM_MIN,(r_exp(spkTimes) - 1));
            q2 = r_nexp(spkTimes)./max(POS_DENOM_MIN,(1-r_nexp(spkTimes)));
            q(spkTimes) = min(q1,q2);
            
            
            a = 1./(1+nexpl); 
            qa = q.*a;
            h_spkIndices = 1:length(CBEM.h_spk);
        else
            h_spkIndices = (1:length(CBEM.h_spk)) + hessIndices{end}(end);
        end
        
        g_h_spk = (dt*CBEM.spikeNonlinearity.c)*(varargin{1}'*qa);
        
        if(usePrior)
        	g_h_spk = g_h_spk - CBEM.prior.h_spk_sig*CBEM.h_spk;
        end

        dLL(h_spkIndices) = g_h_spk;



        if(nargout > 2)
            %% h_spk Hessian 
            if(isempty(conductancesToOptimize))
                u = spalloc(T,1,length(spkTimes));
                %u(spkTimes) = -1./max(POS_DENOM_MIN,(r_exp(spkTimes)./dt + r_nexp(spkTimes)./dt - 2/dt)); %#ok<*SPRIX>
                u1 = -dt*r_nexp(spkTimes)./max(POS_DENOM_MIN,(r_nexp(spkTimes).*r_nexp(spkTimes) + 1 - 2*r_nexp(spkTimes)));%#ok<*SPRIX>
                u2 = -dt*r_exp(spkTimes)./max(POS_DENOM_MIN,(r_exp(spkTimes ).*r_exp(spkTimes ) + 1 - 2*r_exp(spkTimes )));
                u(spkTimes) = max(u1,u2);
                
                d = a.*(1-a);%1./(2+expl+nexpl);
                sq = (q.*d + CBEM.spikeNonlinearity.c*(u.*a.^2)); %u.*a*CBEM_c is derivate of q, d is derivative of a
                
            end
            
            h_h_spk = (CBEM.spikeNonlinearity.c*dt)*((varargin{1}.*sq)'*varargin{1});
            if(usePrior)
                h_h_spk = h_h_spk - CBEM.prior.h_spk_sig;
            end
            d2LL(h_spkIndices, h_spkIndices) = h_h_spk;

            %% h_spk & currents
            for cc = 1:numCondToOptimize
                cNum = conductancesToOptimize(cc);
                h_h_spk_k = (CBEM.spikeNonlinearity.c*alpha*dt)*CBEM_hess3(varargin{CBEM.condType(cNum)}, varargin{1}, sq ,gamma_s(:,cc),M);

                d2LL(h_spkIndices, hessIndices{cc}) = h_h_spk_k';
                d2LL(hessIndices{cc}, h_spkIndices) = h_h_spk_k;
            end
        end
    end
    
    
else
    dLL = 0;
    d2LL = 0;
end

if(sum(isnan(dLL) | isinf(dLL)) > 0 || sum(sum(isnan(r) | isinf(r))) > 0 || isnan(LL) || isinf(LL))
    fprintf('Unknown error in cbNllFunctionRectLin working.\n')
    LL = -1e50;
    dLL(:) = 0;
    d2LL(:,:) = 0;
end

LL = -LL;
dLL = -dLL;
d2LL = -d2LL;

if(isfield(CBEMtoOptimize,'outputVoltage') &&  CBEMtoOptimize.outputVoltage)
    LL = [LL;V(end)];
end


function h = CBEM_hess3(X,H,n_e,g_e,M)

h = -(n_e.*(M\(g_e.*X)))'*H;

% function h = CBEM_hess2(X,Z,M_d,M_l,M_u, nM_g_ei,nM_m_e,nM_m_i,g_e,g_i,M)
% 
% T = length(M_d);
% S = spdiags([M_d M_l M_u],[0 -1 1],T,T);
% H_0 = X'*(S\Z);
% 
% 
% H_1 = X'*(Z.*nM_g_ei);
% 
% S2 = spdiags([M -1*ones(T,1)],[0 -1], T,T);
% H_2 = X'*(nM_m_e.*(S2\(g_i.*Z)));
% 
% 
% S3 = spdiags([M -1*ones(T,1)],[0 -1], T,T);
% H_3 = (nM_m_i.*(S3\(g_e.*X)))'*Z;
% 
% h = H_0 - H_1 + H_2 + H_3;


function h = CBEM_hess4(X,Z,sq, nM_g_ei,nM_m_e,nM_m_i,g_e,g_i,M)

X2 = M\(X.*g_e);
Z2 = M\(Z.*g_i);
H_0 = X2'*(sq.*Z2);

H_1 = X'*(Z.*nM_g_ei);

H_2 = X'*(nM_m_e.*Z2);

H_3 = (nM_m_i.*X2)'*Z;

h = H_0 - H_1 + H_2 + H_3;




