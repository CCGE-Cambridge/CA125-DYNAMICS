import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm
from random import choices
import scipy

def sample_truncated_kde(kde, n=100):
    rng = np.random.default_rng(0)

    out = kde.resample(int(n) * 2)   # shape
    out = out.ravel()  # make it 1D
    lo, hi = np.quantile(out, [0.025, 0.975])
    out = out[(out>lo) & (out<hi)]
    out_ = rng.choice(out, size=n, replace=False)

    return np.array(out_)

'''
Functions
'''
def student_t_rng(nu, loc, scale, size=1):
    return loc + scale * np.random.standard_t(df=nu, size=size)

# Gompertz function
def Gompertz_(t, K, beta):
    '''

    Parameters
    ----------
    t : time
    K : carrying capacity parameter, K = ln(Vmax/V0)
    beta : decay rate

    Returns
    -------
    V = V0 * exp(K * (1 - exp(-beta * t))
    '''
    V0 = 1e-9
    return V0 * np.exp(K * (1 - np.exp(-beta * t)))


def get_time_to_vol_gompertz(V, beta, K, V0=1e-9):
    '''
    Assuming a Gompertz model, we estimate the time to reach a given volume, V

    Parameters
    ----------
    V: The volume for which we want the time. (cm3)
    beta: decay rate
    V0: init vol (cm3)
    K: log(Vmax/V0)

    Returns
    t : time to reach the given volume --> t = -1/beta * log[ 1 - 1/K log[V/V0]]
    -------
    '''

    t = -1 / beta * np.log(1 - 1 / K * np.log(V / V0))

    return t

def sample_truncated_normal(mu, sigma, n=100, batch=1000, alpha=0.05):
    rng = np.random.default_rng()

    # 90% central interval of the *original* normal
    lo, hi = norm.ppf([alpha/2, 1 - alpha/2], loc=mu, scale=sigma)

    out = []
    while len(out) < n:
        x = rng.normal(mu, sigma, size=batch)
        x = x[(x >= lo) & (x <= hi)]
        out.extend(x.tolist())

    return out[:n]


def get_ca125_at_vol(vol, inter_ca, inter_vol, slope_ca, slope_vol, resid_nu, resid_sd, res=False):
    '''

    Parameters
    ----------
    vol : the 'measured' volume at which we desired the CA125
    inter_ca : log_ca125 intercept at time =0 (ln_CA)
    inter_vol : log_vol interecpty at time = 0 (ln_V0)
    slope_ca : the ca125 growth rate (mu_ca)
    slope_vol : the vol growth rate (mu_vol)
    params: residual parameters for skew-t distribution

    ln_V = ln_V0 + mu_vol * dt + eps
    ln_CA = ln_CA0 + mu_ca * dt + eps

    Returns
    -------
    ca125 val
    '''

    if res:
        # dt = (np.log(vol) - inter_vol - student_t_rng(resid_nu, 0, resid_sd, size=1)) / slope_vol
        dt = (np.log(vol) - inter_vol) / slope_vol
        ln_ca125 = inter_ca + slope_ca * dt + student_t_rng(resid_nu, 0, resid_sd, size=1)
        ln_ca125 = ln_ca125[0]
    else:
        dt = (np.log(vol) - inter_vol) / slope_vol
        ln_ca125 = inter_ca + slope_ca * dt

    return np.exp(ln_ca125)

def get_vol_at_ca125(ca125, inter_ca, inter_vol, slope_ca, slope_vol, resid_nu, resid_sd, res=False):
    '''

    Parameters
    ----------
    vol : the 'measured' volume at which we desired the CA125
    inter_ca : log_ca125 intercept at time =0 (ln_CA)
    inter_vol : log_vol interecpty at time = 0 (ln_V0)
    slope_ca : the ca125 growth rate (mu_ca)
    slope_vol : the vol growth rate (mu_vol)
    params: residual parameters for skew-t distribution

    ln_V = ln_V0 + mu_vol * dt + eps
    ln_CA = ln_CA0 + mu_ca * dt + eps

    Returns
    -------
    ca125 val
    '''

    if res:
        dt = (np.log(ca125) - inter_ca - student_t_rng(resid_nu, 0, resid_sd, size=1)) / slope_ca
        ln_vol = inter_vol + slope_vol * dt + student_t_rng(resid_nu, 0, resid_sd, size=1)
    else:
        dt = (np.log(ca125) - inter_ca) / slope_ca
        ln_vol = inter_vol + slope_vol * dt

    return np.exp(ln_vol)[0]


# Logistic growth function
def logistic_(t, Vmax, r):
    '''

    Parameters
    ----------
    t : time
    Vmax : maximum volume
    r : starting growth rate

    Returns
    -------
    V = Vmax  /   (1 + (Vmax-V0) * exp (-r*t) / Vmax)
    '''
    V0 = 1e-9
    num = Vmax
    den = 1 + (Vmax - V0) / V0 * np.exp(-r * t)
    return (num / den)


def get_time_to_vol_logistic(V, r, Vmax, V0=1e-9):
    '''
    Assuming a logistic model, we estimate the time to reach a given volume, V

    Parameters
    ----------
    V: The volume for which we want the time. (cm3)
    r: initial growth rate
    V0: init vol (cm3)
    Vmax : max volume

    Returns
    t : time to reach the given volume --> t = -1/r * ln ((Vmax/V - 1) / (1 - V0/Vmax))
    -------
    '''

    t = -1/r * np.log((Vmax/V - 1) / (Vmax/V0 - 1))

    return t

