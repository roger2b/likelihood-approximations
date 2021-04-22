# likelihood-approximations

Code used in de Belsunce et al. (2021) to compute likelihood approximations to recover the optical depth to reionization. The signal is heavily noise and systematics dominated and, hence, a robust and accurate likelihood is required. 

We compared this method to two likelihood-approximation schemes:
1. Simulation-based likelihood (SimBaL)
2. Density-estimation likelihood-free inference (DELFI) which is a generalisation of SimBaL, instead of using a fixed functional form to describe the distribution of spectra, we use a neural network-based approach to fit the distribution of spectra using an ensemble of functions, here of Gaussians. 

## code
this code can compute the following: 
- quadratic maximum likelihood (QML) power spectrum estimates of given input CMB maps
- accurate noise and signal covariance, fisher and error matrices for the QML estimator
- a pixel-based temperature-polarisation likelihood 
- a likelihood-approximation scheme using the QML estimates and the noise covariance matrix
- this code has been integrated into Cobaya (https://github.com/CobayaSampler/cobaya) and the module is available as MyLikelihood.py

This code is under active development and, hence, many things are still changing. Please use with care. Do not hesitate to send me an e-mail with questions. 

## citation 
For details regarding implementation please have a look at out paper: https://arxiv.org/pdf/2103.14378.pdf and cite if you use any parts of this code. 

