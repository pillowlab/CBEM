%optimization function for a CBEM structure
%
% CBEM is the main structure
% Xvec contains certain values for the CBEM parameters in vector form
%  -CBEMtoOptimize says which values these are
function [f, g, h] = optimizationFunction(Xvec,CBEM,CBEMtoOptimize,Stim,spkHist,spikeTimes,StimG,StimCE,StimCI)

CBEM = cbemVectorToStruct(Xvec,CBEM,CBEMtoOptimize);

if(isempty(Xvec))
    display('Nothing selected to optimize (input Xvec is empty)')
    error('Empty optimization problem');
end


CBEMtoOptimize.outputVoltage = true;
CBEMtoOptimize.initVoltage   = CBEM.E_l;



if(nargout > 2)
    [f, g, h] = cbNllFunctionRectLin(spikeTimes,CBEM,CBEMtoOptimize, length(Xvec),spkHist,Stim,StimG,StimCE,StimCI);
elseif(nargout > 1)
    [f, g] = cbNllFunctionRectLin(spikeTimes,CBEM,CBEMtoOptimize, length(Xvec),spkHist,Stim,StimG,StimCE,StimCI);
    h = 0;
else
    [f] = cbNllFunctionRectLin(spikeTimes,CBEM,CBEMtoOptimize, length(Xvec),spkHist,Stim,StimG,StimCE,StimCI);
    g = 0;
    h = 0;
end
f = f(1);

    


eandi = sum(CBEM.condType(1:end) == 2);
e1 = (CBEM.condType == 2 & CBEM.E_s > -40);
i1 = (CBEM.condType == 2 & CBEM.E_s < -40);

if(sum(e1) > 0)
    e_on = max(CBEMtoOptimize.k_s(e1)); %#ok<*FNDSB>
else
    e_on = false;
end

if(sum(i1) > 0)
    i_on = max(CBEMtoOptimize.k_s(i1));
else
    i_on = false;
end

if(isfield(CBEM,'prior') && (e_on || i_on) && eandi >= 2)
    
    if(mod(sum(CBEM.condType == 2),2) == 0) 
        
        condType = CBEM.condType;
        pairs = reshape(1:sum(condType == 2),2,[])';

        for pp = 1:size(pairs,1)
            e_num = pairs(pp,1);
            i_num = pairs(pp,2);

            e_on = CBEMtoOptimize.k_s(e_num);
            i_on = CBEMtoOptimize.k_s(i_num);

            k_e = CBEM.k_s{e_num}(1:end-1);
            k_i = CBEM.k_s{i_num}(1:end-1);
            B = CBEM.stimBasisVectors;
            B2 = B'*B;

            d = (k_e+k_i)'*(B2)*(k_e+k_i);
            mt = k_e'*B2*k_e - k_i'*B2*k_i;
            m = ( mt)^2;

            f = f + CBEM.prior.comp.d*d;
            f = f + CBEM.prior.comp.m*m;

            e_idx = (1:length(k_e))  + size(Stim,2)*(e_num-1);
            i_idx = (1:length(k_i))  + size(Stim,2)*(i_num-1);

            if(e_on && nargout > 1)

                dde = 1/2*(B2*(k_e+k_i));
                dme = 4*(mt)*B2*k_e;

                g(e_idx) = g(e_idx) + CBEM.prior.comp.m*dme;
                g(e_idx) = g(e_idx) + CBEM.prior.comp.d*dde;

                if(nargout > 2)
                    d2de = 1/2*B2;
                    d2me = 4*(2*(B2*k_e)*(B2*k_e)' + mt*B2 );

                    h(e_idx,e_idx) = h(e_idx,e_idx) + CBEM.prior.comp.m*d2me;
                    h(e_idx,e_idx) = h(e_idx,e_idx) + CBEM.prior.comp.d*d2de;
                end
            end

            if(i_on && nargout > 1)

                ddi = 1/2*(B2*(k_e+k_i));
                dmi = -4*(mt)*B2*k_i;

                g(i_idx) = g(i_idx) + CBEM.prior.comp.m*dmi;
                g(i_idx) = g(i_idx) + CBEM.prior.comp.d*ddi;

                if(nargout > 2)
                    d2di = 1/2*B2;
                    d2mi = -4*(-2*(B2*k_i)*(B2*k_i)' + mt*B2 );

                    h(i_idx,i_idx) = h(i_idx,i_idx) + CBEM.prior.comp.m*d2mi;
                    h(i_idx,i_idx) = h(i_idx,i_idx) + CBEM.prior.comp.d*d2di;
                end
            end

            if(e_on && i_on && nargout > 2)
                d2db = 0;
                d2mb = -4*(-2*(B2*k_i)*(B2*k_e)' );

                h(i_idx,e_idx) = h(i_idx,e_idx) + CBEM.prior.comp.m*d2mb;
                h(i_idx,e_idx) = h(i_idx,e_idx) + CBEM.prior.comp.d*d2db;

                h(e_idx,i_idx) = h(e_idx,i_idx) + CBEM.prior.comp.m*d2mb;
                h(e_idx,i_idx) = h(e_idx,i_idx) + CBEM.prior.comp.d*d2db;
            end
        end
    end
end