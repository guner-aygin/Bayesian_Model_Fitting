''' 
Bayesian Linear Regression Fitter

This code was created to perform a Bayesian linear regression on simple data which are assumed to be linearly dependant.

Before performing our Bayesian analysis, to improve computation time we perform a **simple least squares** regression to help inform our decision on our priors. 
'''

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog
import random
import pandas as pd
from scipy import stats


def linear_least_squares(X,Y):
    if len(X) == len(Y):
        
        N = len(X)
        sum_xy = np.sum(X*Y)
        sum_x = np.sum(X)
        sum_y = np.sum(Y)
        sum_x2 = np.sum(X**2)

        m_ls = (N * sum_xy - sum_x * sum_y)/(N * sum_x2 - (sum_x)**2)
        b_ls = (sum_y - m_ls * sum_x)/N

        predicted_y = m_ls * X + b_ls

        residuals = Y - predicted_y
        sig_ls = np.std(residuals)
        
        return m_ls, b_ls, sig_ls
    
    else:
        return print('Error: Dimension mismatch between X and Y')



def calculate_sample_size(len_X, base_size=1000, base_len=70):
    """
    Calculate the number of samples for MCMC based on the length of the dataset.
    :param len_X: Length of the dataset.
    :param base_size: Base number of samples for the base length of the dataset.
    :param base_len: The dataset length for which the base size is defined.
    :return: Calculated number of samples.
    """
    if len_X >= base_len:
        return base_size
    else:
        # Increase the number of samples for smaller datasets
        return int(base_size * (base_len / len_X))



def bayesian_linear_regression(X, Y, m_ls, b_ls, sig_ls): # --> X,Y are the data, m,b are the priors  
    
    num_samples = calculate_sample_size(len(X))

    with pm.Model() as linear_model:
        # Priors for unknown model parameters
        m = pm.Normal('$m$', mu=m_ls, sigma=20)
        b = pm.Normal('$b$', mu=b_ls, sigma=20)
        sigma = pm.HalfNormal("$\sigma$", sd=sig_ls)

        # Expected value of outcome (linear model)
        mu = m * X + b

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

        trace = pm.sample(num_samples)

        ppc = pm.sample_posterior_predictive(trace, var_names=['Y_obs'])

        # Calculate the mean prediction and the interval for each X
        pred_means = np.mean(ppc['Y_obs'], axis=0)
        pred_std = np.std(ppc['Y_obs'], axis=0)

    return trace, pred_means, pred_std


def plot_trace(X, Y, trace, save=False):
    pm.plot_trace(trace)
    
    m_mean = np.mean(trace['$m$'])
    b_mean = np.mean(trace['$b$'])
    sigma_mean = np.mean(trace['$\sigma$'])

    m_error = trace['$m$'].std()
    b_error = trace['$b$'].std()
    sigma_error = trace['$\sigma$'].std()
    
    if save:
        plt.savefig('trace_plot.png')
    plt.show()


def create_parameter_dataframe(trace):
    data = {
        'Parameter': ['$m$', '$b$', '$\sigma$'],
        'Mean': [np.mean(trace['$m$']), np.mean(trace['$b$']), np.mean(trace['$\sigma$'])],
        'Standard Deviation': [trace['$m$'].std(), trace['$b$'].std(), trace['$\sigma$'].std()]
    }

    df = pd.DataFrame(data)
    return df


def plot_results(X, Y, yerr, trace, pred_means, pred_std, save=False):
    # Sort X and rearrange Y, pred_means, and pred_intervals accordingly
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices]
    Y_sorted = Y[sorted_indices]
    pred_means_sorted = pred_means[sorted_indices]

    plt.figure(figsize=(8, 5))
    plt.errorbar(X_sorted, Y_sorted, yerr, fmt='.b', label='Data', zorder=99)
    plt.plot(X_sorted, pred_means_sorted, c='red', label='Mean Prediction')
    plt.fill_between(X_sorted, pred_means_sorted - pred_std, pred_means_sorted + pred_std, color='red', alpha=0.3, label='$\pm 2\sigma$')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bayesian Linear Regression with Uncertainty')
    plt.legend()

    if save:
        plt.savefig('regression_plot.png')
    plt.show()


def lin_reg_fit(X,Y, yerr=None):
    m_ls, b_ls, sig_ls = linear_least_squares(X,Y)
    trace, pred_means, pred_intervals = bayesian_linear_regression(X,Y,m_ls,b_ls,sig_ls)
    fig1 = plot_trace(X,Y,trace)
    df = create_parameter_dataframe(trace)
    fig2 = plot_results(X, Y, yerr, trace, pred_means, pred_intervals)
    return display(df.style.hide_index().set_properties(**{'text-align': 'center'}))

