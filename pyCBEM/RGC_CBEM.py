"""
Simple class to solve the Conductance-based encoding model for retinal ganglion cells in:
    K. W. Latimer,  F. Rieke, & J. W. Pillow (2019). Inferring synaptic inputs from spikes with a conductance-based neural encoding model.eLife 8 (2019): e47012.

Requires Jax for computations.

Copyright (c) 2022 Kenneth Latimer
"""

import numpy as np
import scipy
import jax.numpy as jnp
import jax
from jax import random
import math
from pyCBEM import utils



def getSimpleStimulusBasis(binSize_ms : float) -> tuple[np.ndarray, np.ndarray]:
    """
    A simple default basis to use for stimulus-dependent conductance filters made with a modified cardinal spline.

    Args:
      binSize_ms: time bin size in milliseconds

    Returns:
      A tuple (basis, time)

        basis: [BT_stim x P_stim] - Each column is a basis vector

        time:  [BT_stim] - Time (in milliseconds) of the rows of the basis
    """
    return utils.ModifiedCardinalSpline(190, [0, 8, 16, 24, 32, 40, 48, 64, 96, 128, 160, 192], binSize_ms=binSize_ms, zero_last=False, zero_first=True);
    
def getSimpleSpkHistBasis(binSize_ms : float) -> tuple[np.ndarray, np.ndarray]:
    """
    A simple default basis to use for spike history filters made with a modified cardinal spline.

    Args:
      binSize_ms: time bin size in milliseconds

    Returns:
      A tuple (basis, time)

        basis: [BT_hspk x P_hspk] - Each column is a basis vector

        time:  [BT_stim] - Time (in milliseconds) of the rows of the basis
    """
    return utils.ModifiedCardinalSpline(190, [0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 160, 192], binSize_ms=binSize_ms, zero_last=False, zero_first=False);
    
class CBEM_basic:
    """
    A basic implementation of the Conductance-based encoding model.
    The model has three inputs: excitatory and inhibitory conductance (soft-rectified, affine function in the stimulus)
                                a linear spike history (not conductance-based)
    The 'setObservations' sets up the data: current stimulus and spike train.
    """
    def __init__(self,  binSize_ms : float, basis_conductance : np.ndarray = None, basis_hspk : np.ndarray = None):
        """
        Initialize simple CBEM with one excitatory and one inhibitory conductance which share input.
        
        Note: bases are orthogonalized before fitting. Filter parameters will be in terms of the orthogonalized basis.

        Args:
          binSize_ms:           time bin size in milliseconds
          basis_conductance:    [BT_stim x P_stim], if None then getSimpleStimulusBasis(binSize_ms) -
                                Each column is a basis function for the conductances (conductances share a basis).
                                The basis is assumed to be causal: basis_conductance[0,:] are the basis weights for the previous time bin.
          basis_hspk:           [BT_hspk x P_hspk], if None then getSimpleSpkHistBasis(binSize_ms) -
                                Each column is a basis function for the spike history.
                                The basis is assumed to be causal: basis_conductance[0,:] are the basis weights for the previous time bin.
        """

        # discretization size
        self.binSize_ms    = binSize_ms;
        assert self.binSize_ms > 0, "bin size must be positive"

        # setup bases for filters
        if(basis_conductance is None):
            basis_conductance,tts = getSimpleStimulusBasis(binSize_ms);
            
        if(basis_hspk is None):
            basis_hspk,tts = getSimpleSpkHistBasis(binSize_ms);

        assert basis_conductance.ndim <= 2, "Basis cannot be more than 2 dimensions"
        assert basis_hspk.ndim <= 2, "Basis cannot be more than 2 dimensions"

        self.basis_conductance_0 = basis_conductance.reshape((basis_conductance.shape[0],-1));
        self.basis_hspk_0      = basis_hspk.reshape((basis_hspk.shape[0],-1));
        self.basis_conductance   = jnp.array(scipy.linalg.orth(self.basis_conductance_0));
        self.basis_hspk        = jnp.array(scipy.linalg.orth(self.basis_hspk_0 ));

        # setup LIF parameters 
        self.E_input = jnp.array([0, -80]); # mV
        self.E_l = -60; # mV
        self.g_l = 200; 
        self.V_0 = self.E_l; # initial voltage
        self.frNonlinearity = {"alpha" : 90, "mu" : -53, "beta" : 1.67}; # "beta" : 1.67 or could be 1/0.45?

        # no stimulus set yet - won't run model
        self.X_cond = None;
        self.X_lin  = None;
        self.spkTimes_bins  = None;

        self._B_cond = None;
        self._B_hspk  = None;
        self.N_pixels = 0;

    # Properties for the model parameters
    @property
    def B_cond(self):
        """jnp.ndarray: [(BT_stim + 1) x 2] - The conductance filters and baseline parameters."""
        return self._B_cond;

    @B_cond.setter
    def B_cond(self, value : jnp.ndarray):
        """Note: cannot change filter size"""
        assert not (self._B_cond is None), "model filters not initialized"
        assert value.shape == self._B_cond.shape, "Conductance filter is incorrect shape"
        self._B_cond = value;

    @property
    def B_hspk(self):
        """jnp.ndarray: [BT_hspk x 1] - The spike history filter parameters."""
        return self._B_hspk;

    @B_hspk.setter
    def B_hspk(self, value : jnp.ndarray):
        """Note: cannot change filter size"""
        assert not (self._B_hspk is None), "model filters not initialized"
        assert value.shape == self._B_hspk.shape, "Spike history filter is incorrect shape"
        self._B_hspk = value;


    def setObservations(self, Stimulus : np.ndarray, spkTimes_bins : list[int], window : range) -> None:
        """
        Initializes the stimulus and spike train observation for this neuron.
        This assumes only one spike per bin is possible.

        Args:
          Stimulus:         [T_stim x number of pixels] - Pixels are treated independently
          spkTimes_bins:    List of the spike times in bins.
          window:           The window of stimulus and spikes to fit. This input allows you to cut out the first 
                            part of the given spike train to avoid edge effects for spike history or stimulus.
        """
        window = jnp.array(window);
        # get the conductance desgin matrix(same for both conductances)
        self.X_cond   = jnp.array(utils.convolveStimulusWithBasis(Stimulus, self.basis_conductance)[window,:]);
        self.N_pixels = math.prod(Stimulus.shape[1:]);

        # get spike history
        X_l, Y_0      = utils.convolveSpksWithBasis(spkTimes_bins, self.basis_hspk, Stimulus.shape[0]);
        self.X_lin    = jnp.array(X_l[window,:]);

        # get spike times within the current window
        self.Y              = jnp.array(Y_0[window]); # vectorized spike times
        self.spkTimes_bins  = jnp.where(jnp.in1d(window, spkTimes_bins))[0]; 
        
        # setup empty parameters
        self._B_cond = jnp.zeros((self.basis_conductance.shape[1] * self.N_pixels + 1, 2));
        self._B_hspk  = jnp.zeros((self.basis_hspk.shape[1]));

    ### functions for handling the voltage model
    def getConductances(self, B_cond : jnp.ndarray = None) -> jnp.ndarray:
        """
        Computes the conductances given the conductance filter parameters for the current stimulus.
        The conductance nonlinearity is a soft-rectifier.

        Args:
          B_cond:   [(P_stim + 1) x 2], if None then self.get_B_cond() -
                        Parameters of the two conductance filters as weights on the basis functions. First column is excitatory, second inhibitory.
                        The last entry is the basline term.
        
        Returns:
          gs  [T_stim x 2] - first column is excitatory conductance, second inhibitory
        """
        if B_cond is None:
            B_cond = self._B_cond
        assert not (self.X_cond is None), "Stimulus not initialized!"
        assert jnp.shape(B_cond) == jnp.shape(self._B_cond), "Conductance parameters not correct shape"

        return jax.nn.softplus(self.X_cond @ B_cond);

    def getVoltage(self, B_cond : jnp.ndarray = None) -> jnp.ndarray:
        """
        Computes the voltage given the conductance parameters for the set stimulus.
        
        Args:
          B_cond:  [(P_stim + 1) x 2], if None then  self.get_B_cond() -
                        Parameters of the two conductance filters as weights on the basis function. First column is excitatory, second inhibitory.
                        The last entry is the basline term.
                        
        Returns:
          V [T_stim] - the solved voltage in millivolts
        """
        if B_cond is None:
            B_cond = self._B_cond

        gs = self.getConductances(B_cond);
        return jax.jit(utils.getVoltage)(gs, self.E_input, self.g_l, self.E_l, self.V_0, self.binSize_ms);

    def firingRateNonlinearity(self, V_tot : jnp.ndarray) -> jnp.ndarray:
        """
        Computes the firing rate nonlinearity on the given total voltage.
        The nonlinearity is alpha * softplus((V_tot - mu)/beta)

        - self.frNonlinearity holds the parameters alpha, beta, and mu

        Args:
          V_tot:    [T] - The total voltage terms.
                    Total voltage is the membrane potential from the membrane potential equation plus spike history terms.

        Returns:              
          spikeRate [T] - the firing rate for each term in V_tot
        """
        return self.frNonlinearity["alpha"] * jax.nn.softplus((V_tot - self.frNonlinearity["mu"])/ self.frNonlinearity["beta"]);

    def getSpikeHistory(self, B_hspk : jnp.ndarray = None) -> jnp.ndarray:
        """
        Computes the spike history given the parameters for the set spike train. (A simple linear function)

        Args:
          B_hspk:   [P_hspk x 1], if None then self.get_B_hspk() - Parameters of the spike history filter as weights on the basis functions.
                        
        Returns:
          hspk  [T_stim] - the linear spike history
        """
        if B_hspk is None:
            B_hspk = self._B_hspk
        assert jnp.shape(B_hspk) == jnp.shape(self._B_hspk), "Linear parameters not correct shape"

        return self.X_lin @ B_hspk;

    def getSpikeRate(self, B_cond  : jnp.ndarray = None, B_hspk  : jnp.ndarray = None) -> jnp.ndarray:
        """
        Computes the spike rate at each time given the conductance and spike history parameters for the set stimulus & spike history.

        Args:
          B_cond:   [(P_stim + 1) x 2], if None then  self.get_B_cond() - Parameters of the two conductance filters as weights on the basis function. First column is excitatory, second inhibitory.
          B_hspk:   [P_hspk x 1], if None then  self.get_B_hspk() - Parameters of the spike history filter as weights on the basis functions.

        Returns            
          spikeRate [T_stim] - the firing rate in each bin in units of spikes/sec
        """
        if B_cond is None:
            B_cond = self._B_cond
        if B_hspk is None:
            B_hspk = self._B_hspk

        V_tot  =  self.getVoltage(B_cond) + self.getSpikeHistory(B_hspk); # add membrane voltage and spike history 
        return self.firingRateNonlinearity(V_tot); # pass through transfer function

    ### log likelihood of spikes
    def getLogLike(self, B_cond : jnp.ndarray = None, B_hspk : jnp.ndarray  = None) -> jnp.ndarray:
        """
        Computes the log likelihood at each time bin given the conductance and spike history parameters for the set stimulus & spike history.
        This function uses a truncated Poisson likelihood. That is, Poisson(0 | rate) for bins without a spike and
                                                                 (1-Poisson(0 | rate)) for bins with a spike.

        Args:
          B_cond:   [(P_stim + 1) x 2], if None then self.get_B_cond() - Parameters of the two conductance filters as weights on the basis function. First column is excitatory, second inhibitory.
          B_hspk:   [P_hspk x 1], if None then  self.get_B_hspk() - Parameters of the spike history filter as weights on the basis functions.
                        
        Returns:
          ll_bins [T_stim] - the log likelihood of the observation (a spike or no spike) at each bin.
        """
        if B_cond is None:
            B_cond = self._B_cond
        if B_hspk is None:
            B_hspk = self._B_hspk

        ll_bins = -(self.binSize_ms/1e3) * self.getSpikeRate(B_cond, B_hspk);
        ll_bins = ll_bins.at[self.spkTimes_bins].set(jnp.log(1 - jnp.exp(ll_bins[self.spkTimes_bins])));
        return ll_bins;

    ### functions for handling parameters
    def randomizeParameters(self, offset_term_mean : float = 10, std_cond : float = 1, std_lin : float = 1) -> None:
        """
        Sets the parameters of the model using i.i.d. normal draws.
        All filter weights are drawn with mean 0.

        Args:
          offset_term_mean: Mean of the baseline term of the conductances. This probably should be positive.
          std_cond:         Standard deviation of the conductance parameters.
          std_lin:          Standard deviation of the spike history parameters.
                        
        Returns:
          A tuple (B_cond, B_hspk)

            B_cond: [(P_stim + 1) x 2] - Parameters of the two conductance filters as weights on the basis function. First column is excitatory, second inhibitory.

            B_hspk:  [P_hspk x 1] - Parameters of the spike history filter as weights on the basis functions.
        """
        assert not (self._B_cond is None), "model filters not initialized"
        self._B_hspk  = jnp.array(np.random.normal(size=self._B_hspk.shape)*std_cond);
        self._B_cond = jnp.array(np.random.normal(size=self._B_cond.shape)*std_lin);
        self._B_cond = self._B_cond.at[-1,:].set(self._B_cond[-1,:] + offset_term_mean);
        return (self._B_cond, self._B_hspk);


    def vectorizeParameters(self, B_cond : jnp.ndarray = None, B_hspk : jnp.ndarray  = None) -> jnp.ndarray:
        """
        Flatens the parameters into a vector for optimization.

        Args:
          B_cond:   [(P_stim + 1) x 2], if None then self.B_cond -
                    Parameters of the two conductance filters as weights on the basis function. First column is excitatory, second inhibitory.
          B_hspk:   [P_hspk x 1], if None then self.B_hspk -
                    Parameters of the spike history filter as weights on the basis functions.
                    The last entry is the basline term.
                        
        Returns:
          B [(P_stim + 1) * 2 + P_hspk] - Flattened B_cond and B_hspk
        """
        assert not (self._B_cond is None), "model filters not initialized"
        if B_cond is None:
            B_cond = self._B_cond
        if B_hspk is None:
            B_hspk = self._B_hspk
        return np.concatenate((B_cond.flatten(), B_hspk.flatten()));

    def devectorizeParameters(self, B : jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Takens in a vector of the parameters and returns them in more convenient separate, matrix forms.

        Args:
          B:  [(P_stim + 1) * 2 + P_hspk] - Flattened B_cond and B_hspk.
                This should correspond to concatenate((B_cond.flatten(), B_hspk.flatten))
        
        Returns:
          A tuple (B_cond, B_hspk)

            B_cond: [(P_stim + 1) x 2] - 
                Parameters of the two conductance filters as weights on the basis function. First column is excitatory, second inhibitory.

            B_hspk:  [P_hspk x 1] - 
                Parameters of the spike history filter as weights on the basis functions.
                The last entry is the basline term.
        """
        assert not (self._B_cond is None), "model filters not initialized"
        assert jnp.ndim(B) == 1 and B.size == self._B_cond.size + self._B_hspk.size, "B is incorrect size: must be a vector containing both conductance and linear terms"
        return (B[0:self._B_cond.size].reshape(self._B_cond.shape),  B[self._B_cond.size:].reshape(self._B_hspk.shape));

    def setParametersFromVector(self, B : jnp.ndarray) -> None:
        """
        Sets the current parameters from a vectorized form.

        Args:
          B:    [(P_stim + 1) * 2 + P_hspk] - Flattened B_cond and B_hspk.
                This should correspond to concatenate((B_cond.flatten(), B_hspk.flatten))
        """
        self._B_cond, self._B_hspk = self.devectorizeParameters(B)



    # Vectorized negative log likelihood functions for optimization
    def vectorizedNegLogLike(self, B : jnp.ndarray) -> jnp.ndarray:
        """
        Computes the negative log likelihood for the set stimulus and spike train.
        Takes in the parameters in a vectorized form for use with optimizers.

        Args:
          B:    [(P_stim + 1) * 2 + P_hspk] - Flattened B_cond and B_hspk.
                This should correspond to concatenate((B_cond.flatten(), B_hspk.flatten))

        Returns:
          nll [1] - The total negative log likelihood for the setup stimulus.
        """
        (B_cond, B_hspk) = self.devectorizeParameters(B);
        return -(self.getLogLike(B_cond, B_hspk).sum());
        
    def vectorizedPenalizedNegLogLike(self, B : jnp.ndarray, conductance_penalty : list[float] = [1, 0.2]) -> jnp.ndarray:
        """
        Computes the penalized negative log likelihood for the set stimulus and spike train.
        Takes in the parameters in a vectorized form for use with optimizers.
        The penalty terms are the weighted squared norms of the conductance filters.
        
        Args:
          B: [(P_stim + 1) * 2 + P_hspk] - Flattened B_cond and B_hspk.
                    This should correspond to concatenate((B_cond.flatten(), B_hspk.flatten))
          conductance_penalty:  [2] -
                    The weights of the penalty on the squared norms of the filter weights.
                    The first term is for the excitatory and the second for the inhibitory.
                    The baseline conductance terms do not contribute to this penalty.

        Returns:
          nll [1] - The total negative log likelihood for the setup stimulus plus the penalties.
        """
        assert not (self._B_cond is None), "model filters not initialized"
        (B_cond, B_hspk) = self.devectorizeParameters(B);
        nll = -(self.getLogLike(B_cond, B_hspk).sum());
        for ii in range(self._B_cond.shape[1]):
            nll += conductance_penalty[ii] * jnp.sum(B_cond[0:-1, ii] ** 2);
        return nll;

    def getConductanceFilter(self, cc : int) -> jnp.ndarray:
        """
        Gets the full conductance filter (basis times weights) for one of the conductance filters.

        Args:
          cc: Index of the conductance filter to return. 0 is excitatory, 1 is inhibitory.

        Returns:
          k_stim [BT_stim x N_pixels] - The filter for each pixel.
        """

        assert not (self._B_cond is None), "model filters not initialized"
        return self.basis_conductance @ jnp.reshape(self._B_cond[0:(self.N_pixels * self.basis_conductance.shape[1]),cc], (self.basis_conductance.shape[1], self.N_pixels));

    def getSpikeHistoryFilter(self) -> jnp.ndarray:
        """
        Gets the full spike history filter (basis times weights).

        Returns:
          h_spk [BT_hspk x 1] - The full spike history filter.
        """
        assert not (self._B_cond is None), "model filters not initialized"
        return self.basis_hspk @ self._B_hspk;

    # simulate a set of spike trains given the currently set stimulus
    def simulateSpikeTrains(self, Y_init : np.ndarray, N : int = None) -> jnp.ndarray:
        """
        Simulates a set of spike trains with the currently set stimulus. This can be really slow!
        This function requires an initial set of spikes to avoid dealing with edge effects of the spike history filter.

        Args:
          Y_init:  [T_0 x (N or 1)] - A matrix of ones and zerosThe initial spike trains. T_0 should be less than T_stim
          N:       (positive integer) - The number of simulations: only use this if Y_init.shape[1] == 1 and you want more than one simulation.
                                             If N > 1 and Y_init.shape[1] == 1, it uses the same initial spike train for all simulations.
                                                                 
        Returns:
          sps [T_stim x N] - A matrix of spikes for each of the N simulations. Spikes are either 1 or 0.
        """
        assert not (self._B_cond is None), "model filters not initialized"
        Y_init = Y_init.reshape((Y_init.shape[0], -1));
        if N is None:
            N = Y_init.shape[1];
        assert (N == Y_init.shape[1] or Y_init.shape[1] == 1), "Invalid arguments for number of simulations"

        T = self.X_cond.shape[0]; # total simulation length
        T_0 = Y_init.shape[0]; # initial seed length
        P_lin = self.basis_hspk.shape[0];
        assert T_0 > P_lin and T_0 > self.basis_conductance.shape[0], "initial spike train too short for bases"
        assert N > 0, "no simulations requested"

        # sets up spikes
        sps = jnp.zeros((T, N));
        sps = sps.at[:T_0, :].set(Y_init);

        # gets voltage (in this model, it's determined by the stimulus. Would need a more complex function for spike-dependent conductances)
        V = self.getVoltage();
        h_spk = jnp.flip(self.getSpikeHistoryFilter()).reshape((1, -1));

        # random values for spike generation
        key = random.PRNGKey(0);
        rs = random.uniform(key, (T, N));

        print("Running " + str(N) + " simulations... (WARNING: This function can be painfully slow!)");
        print("T_0 = " + str(T_0));
        for tt in range(T_0, T):
            if tt % 500 == 0:
                print("tt = " + str(tt) + " / " + str(T));
            # gets spike history & adds to voltage
            V_c = h_spk @ sps[(tt-P_lin):tt,:];
            V_c += V[tt];

            # gets spike probability
            fr = (self.binSize_ms/1e3) * self.firingRateNonlinearity(V_c);
            p_1 = 1 - (jnp.exp(-fr));
            spiked = (rs[tt,:] < p_1).flatten();

            # generates spike
            sps = sps.at[tt, :].set(spiked);
        print("Done.");
        return sps;


