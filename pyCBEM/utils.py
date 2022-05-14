"""
Utility functions for working with the Conductance-based encoding model for retinal ganglion cells in:
    K. W. Latimer,  F. Rieke, & J. W. Pillow (2019). Inferring synaptic inputs from spikes with a conductance-based neural encoding model.eLife 8 (2019): e47012.

Requires Jax for computations.

Copyright (c) 2022 Kenneth Latimer
"""

import numpy as np
import jax.numpy as jnp
import jax
from jax import custom_vjp


## The following utility functions are for a tridiagonal solving step in the voltage function that required manually adding autodiff
@custom_vjp
def solveVoltage_tridiag(M, C):
    T = C.size
    return jax.lax.linalg.tridiagonal_solve(jnp.concatenate((jnp.zeros(1), M[:-1])), jnp.ones(T), jnp.zeros(T), C.reshape((T,1))).reshape((T));
    
def solveVoltage_tridiag_fwd(M, C):
    V = solveVoltage_tridiag(M, C);
    return V, (V, M, C);
    
def solveVoltage_tridiag_bwd(res, g):
    V, M, C = res # Gets residuals computed in solveVoltage_tridiag_fwd
    V2 = jnp.roll(V,1)
    V2 = V2.at[0].set(0);

    T = C.size
    O = jnp.ones(T)
    Z = jnp.zeros(T);
    K = jax.lax.linalg.tridiagonal_solve(Z, O, jnp.concatenate((M[1:], jnp.zeros(1))), g.reshape(T,1)).reshape((T));
    return (-K * V2, K)

solveVoltage_tridiag.defvjp(solveVoltage_tridiag_fwd, solveVoltage_tridiag_bwd);

def getVoltage(gs : jnp.ndarray, E_s : jnp.ndarray, g_l : float, E_l : float, V_0 : float, binSize_ms : float) -> jnp.ndarray:
    """
    Computes the voltage given the synaptic and leak conductances.
   
    :param gs:          jnp.ndarray [T x C]. Each column is the conductance value over time.
    :param E_s:         list [C] Reversal potential for each conductance,
    :param g_l:         scalar. Leak conductance.
    :param E_l:         scalar. Leak reveral potential.
    :param V_0:         scalar. Initial voltage (mV)
    :param binSize_ms:  scalar. Time bin size in milliseconds.

    :return jnp.ndarray [T]. Vector of the solved voltages (in mV).
    """
    I_tot = E_l * g_l + gs @ E_s;
    g_tot = g_l + gs.sum(axis=1);

    exp_ndg = -jnp.exp(-(binSize_ms/1e3) * g_tot);

    C       = I_tot / g_tot * (1 + exp_ndg);
    C = C.at[0].add(V_0);  # initial voltage

    return solveVoltage_tridiag(exp_ndg, C);

### Basis functions
# the CBEM paper used a raised cosine basis. I'm most using the cardinal spline basis now, so I have not included it in the Python code.

def ModifiedCardinalSpline(window_end : float, c_pt : list[float], window_start : float = None, s : float = 0.5, binSize_ms : float = 1, zero_first : bool = False, zero_last : bool = False) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Modified Cardinal Spline basis functions proposed by
    Sarmashghi, M., Jadhav, S. P., & Eden, U. (2021). Efficient Spline Regression for Neural Spiking Data. bioRxiv.

    Modified by Kenneth Latimer from Repository:
        https://github.com/MehradSm/Modified-Spline-Regression
        by Mehrad Sarmashghi

    :param window_end:      scalar. last time point (in ms)
    :param c_pt:            list. Locations of the knots.
    :param window_start:    scalar (default : binSize_ms) first time point (in ms)
    :param s:               scalar (default : 0.5)  tension parameter
    :param binSize_ms:      scalar (default : 1)  time bin size in milliseconds
    :param zero_first:      bool (default : False) Whether to set the end point at c_pt[0 ] to 0's (removes basis)
    :param zero_last:      bool (default : False) Whether to set the end point at c_pt[-1] to 0's (removes basis)

    : return (basis, time) :
        basis: numpy.ndarray [BT_stim x P_stim] Each column is a basis vector
        time:  numpy.ndarray [BT_stim] Time (in milliseconds) of the rows of the basis
    """

    if window_start is None:
        window_start = binSize_ms;
        
    c_pt.sort();
    c_pt = np.array(c_pt);
    tts  = np.arange(window_start, window_end, binSize_ms);
    T    = tts.size;
    
    assert len(c_pt) >= 4, f"Need at least 4 knot points greater than 0 expected, got: {len(c_pt)}"
    assert T > 0, f"Basis window is empty"

    HistSpl = np.zeros((T,len(c_pt)));

    # for each 1 ms timepoint, calculate the corresponding row of the glm input matrix
    for tt_idx, tt in enumerate(tts):
        nearest_c_pt_index = np.where(c_pt < tt)[0];
        assert len(nearest_c_pt_index) > 0, "Cannot find knot index for all points in window"
        nearest_c_pt_index = nearest_c_pt_index[-1];
        nearest_c_pt_time  = c_pt[nearest_c_pt_index];
        next_c_pt_time     = c_pt[nearest_c_pt_index+1];
        
        # Compute the fractional distance between timepoint i and the nearest knot
        u  = (tt - nearest_c_pt_time)/(next_c_pt_time - nearest_c_pt_time);
        lb = (c_pt[ 2]   - c_pt[ 0])/(c_pt[1]   - c_pt[0]);
        le = (c_pt[-2] - c_pt[-3])/(c_pt[-1] - c_pt[-2]);
        
        # Beginning knot 
        if(nearest_c_pt_time == c_pt[0]):
            S = np.array([[2-(s/lb), -2, s/lb], \
                [(s/lb)-3,  3, -s/lb], \
                    [0,  0,  0], \
                        [1,  0,  0]]);
            bbs = range(nearest_c_pt_index, nearest_c_pt_index+3);
            
        # End knot
        elif(nearest_c_pt_time == c_pt[-2]):
            S = np.array([[-s/le,  2, -2+(s/le)], \
                [2*s/le, -3, 3-(2*s/le)], \
                    [-s/le, 0, s/le], \
                        [0, 1, 0]]);
            bbs = range(nearest_c_pt_index-1, nearest_c_pt_index+2);
            
        # Interior knots
        else:
            privious_c_pt = c_pt[nearest_c_pt_index-1];
            next2 = c_pt[nearest_c_pt_index+2];
            l1 = next_c_pt_time - privious_c_pt;
            l2 = next2 - nearest_c_pt_time;
            S = np.array([[ -s/l1, 2-(s/l2), (s/l1)-2, s/l2], \
                [2*s/l1, (s/l2)-3, 3-2*(s/l1), -s/l2], \
                    [-s/l1, 0, s/l1, 0], \
                        [0, 1, 0, 0]]);
            bbs = range(nearest_c_pt_index-1, nearest_c_pt_index+3);

        p = jnp.array([[u**3, u**2, u, 1]]) @ S;
        HistSpl[tt_idx, bbs] = p; 

    assert not np.isnan(HistSpl).any() and not np.isinf(HistSpl).any(), f"basis error: cannot contain infs or nans"

    if zero_first:
        HistSpl = HistSpl[:, 1:];
    if zero_last:
        HistSpl = HistSpl[:, :-1];

    return (HistSpl, tts);

    

# Utilities for simple convolutions with the basis functions
def convolveStimulusWithBasis(stimulus : np.ndarray, basis : np.ndarray, add_ones : bool = True, is_balanced : bool = False) -> jnp.ndarray:
    """
    Convolves a basis set with a set of input vectors.
    :param stimulus:    ndarray [T x N_pixels] Each column is treated as a separate pixel.
    :param basis:       ndarray [BT x P] Each column is a basis function.
    :param add_ones:    bool (default : True)
                        Adds an extra column of ones at the end of the basis convolution for an offset term.
    :param is_balanced:    bool (default : False)
                        If basis is balanced so that the convolution can be called with mode="same".
                        If false, the basis is assumed to be causal.

    :return X: numpy.ndarray [T x (N_pixels*P + add_ones)] X[:, 0:P] is X[:,0] convolved with the basis (repeats for each pixel)
            Last column is ones if add_ones == true
    """
    stimulus = jnp.array(stimulus);
    basis = jnp.array(basis);
    stimulus = stimulus.reshape((stimulus.shape[0],-1))
    N_pixels = stimulus.shape[1];

    N_basis_vectors = basis.shape[1];
    # add zeros to basis. This makes the filter causal and we can use mode="same" to the convolve function
    if(not is_balanced):
        basis = jnp.concatenate((jnp.zeros((basis.shape[0] + 1, N_basis_vectors)), basis), 0)

    # convolves each pixel with each basis function
    X = jnp.ones((stimulus.shape[0], N_pixels*N_basis_vectors + add_ones));
    for pp in range(N_pixels):
        for bb in range(N_basis_vectors):
            X = X.at[:, pp*N_basis_vectors + bb].set(jnp.convolve(stimulus[:,pp], basis[:,bb], mode="same"));
    return X;

def convolveSpksWithBasis(spkTimes_bins : list[int], basis : np.ndarray, T_max : int, add_ones : bool = False, is_balanced : bool = False) -> tuple[jnp.ndarray, np.ndarray]:
    """
    Convolves a basis set with a set of spike times.
    :param spk times:   list. Spike times in bins
    :param basis:       ndarray [BT x P] Each column is a basis function.
    :param add_ones:    bool (default : False)
                        Adds an extra column of ones at the end of the basis convolution for an offset term.
    :param is_balanced:    bool (default : False)
                        If basis is balanced so that the convolution can be called with mode="same".
                        If false, the basis is assumed to be causal.

    :return (X, Y):
        X: numpy.ndarray [T x (N_pixels*P + add_ones)] X[:, 0:P] is X[:,0] convolved with the basis (repeats for each pixel)
            Last column is ones if add_ones == true
        Y: numpy.ndarray [T] Vectorized spike time vector. Bins are 0's or 1's to indicate whether a spike happened or not in each bin.
    """
    # vectorizes spike times and convolves with basis
    X = np.zeros(T_max);
    spkTimes_bins = np.array(spkTimes_bins);
    spkTimes_bins = spkTimes_bins[np.logical_and(spkTimes_bins >= 0, spkTimes_bins < T_max)];
    X[spkTimes_bins] = 1;
    return (convolveStimulusWithBasis(X, basis, add_ones, is_balanced), X);