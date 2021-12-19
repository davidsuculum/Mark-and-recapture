import numpy as np
import pymc3 as pm

# Datos:
K = 100 #marked animals in first round
n = 100 #captured animals in second round
obs = 94 #observed marked animals in second round (the recapture)

with pm.Model() as my_model: 
    # Prior:
    N = pm.DiscreteUniform("N", lower=K, upper=100000)
    
    # Likelihood: an hypergeometric PDF
    likelihood = pm.HyperGeometric('likelihood', N=N, k=K, n=n, observed=obs) 

with my_model:
    step = pm.Metropolis()
    start = {"N": n}
    trace = pm.sample(10000, step, start)
    
    print(pm.summary(trace))
    print(trace['N'])
    
    
import matplotlib.pyplot as plt
%matplotlib inline
import arviz as az
az.style.use('arviz-darkgrid')

with my_model:
    # Plotting exluding the typical burn-in period
    az.plot_posterior(trace[2000:], var_names=['N'], ref_val=np.mean(trace['N'][2000:]), hdi_prob=0.95)
    az.summary(trace[2000:])
    # HPD (Highest Posterior Density) makes reference to a "credible interval" (Bayesian paradigm)

    # Check the evolution of the sampling:
    az.plot_trace(trace[1000:], var_names=['N'])
    plt.show()

    # We can calculate other statistics, for instance:
    print(np.median(trace['N'][2000:]))
