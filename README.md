CBEM
=========================

Code to fit spike trains with the Conductance-based encoding model.

Contains Python and Matlab implementations of the analyses found in [Latimer et al. 2014](http://pillowlab.princeton.edu/pubs/Latimer_conductancePointProc_NIPS14.pdf).


Downloading the repository
------------

- **From command line:**

     ```git clone git@github.com:pillowlab/CBEM.git```

- **In browser:**   click to
  [Download ZIP](https://github.com/pillowlab/CBEM/archive/master.zip)
  and then unzip archive


Example Script
-
Open ``exampleScript.m`` to see it in action using a small simulated dataset

Python code
------------
The pyCBEM code uses a Python + Jax implementation of the CBEM for retinal ganglion cells.
The notebook "exampleCBEMfitting.ipynb" shows how to use this.

## References

- K. W. Latimer,  F. Rieke, & J. W. Pillow (2019)
[Inferring synaptic inputs from spikes with a conductance-based neural encoding model](https://elifesciences.org/articles/47012) eLife 8 (2019): e47012.

- K. W. Latimer, E. J. Chichilnisky, F. Rieke, & J. W. Pillow
 (2014).
 [Inferring synaptic conductances from spike trains with a biophysically inspired point process model](http://pillowlab.princeton.edu/pubs/Latimer_conductancePointProc_NIPS14.pdf) Advances in Neural Information Processings Systems 27, 954-962. 
