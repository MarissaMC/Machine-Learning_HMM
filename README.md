Hidden Markov Model with Baum-Welch algorithm
============================================
The code is written with Matlab.

Input
--------
data: N*T matrix, N data samples of length T

A_guess: K*K matrix, where K is the number hidden states [initial guess for the transition matrix]

E_guess: K*E matrix, where E is the number of emissions [initial guess for the emission matrix]

Output
--------
A_estimate: estimate for the transition matrix after N_iter iterations of expectation-maximization

E_estimate: estimate for the emission matrix after N_iter iterations of expectation-maximization

Usage
-----------
   load('hmm_data.mat');
   A = [0.7,0.3;0.3,0.7];
   E = [0.25,0.25,0.25,0.25;0.25,0.25,0.25,0.25];

   [A_estimate, E_estimate] = baumwelch(data, A, E, 500)
