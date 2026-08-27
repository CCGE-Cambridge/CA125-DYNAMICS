'''
This is the same simulations as the normal forward sims but with uncertainty through draws etc..
In this I do a joint sampling
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm
from random import choices
import scipy
from utils import *
from itertools import product


# ------------ Load data -------------
sims = pd.read_csv('./../../data/simulated_ttms_anonymised.csv', index_col=0)
draws_vol_ca125 = pd.read_csv('./../../output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/site_specific_draws_original.csv')
draws_healthy = pd.read_csv('./../../data/CA125-in-healthy-individuals/draws_healthy_white.csv', index_col=0)

# Only use shedding parameters with sp and sm >0 - this should be there automatically
draws_vol_ca125 = draws_vol_ca125[(draws_vol_ca125.sp > 0) & (draws_vol_ca125.sm > 0)]


# ------ Functions for CUSUM ------

def CUSUM_numpy(z, k):
    s = np.zeros(len(z))
    for i in range(1, len(z)):
        s[i] = max(0, s[i-1] + z[i] - k)
        
    return s

def get_CUSUM(subdf, k):
    subdf = subdf.sort_values(by='dt').copy()
    
    subdf['S_t'] = CUSUM_numpy(subdf.Z_t.values, k)
    subdf['S_h'] = CUSUM_numpy(subdf.Z_h.values, k)
        
    return subdf


# ----------- Hyper-parameters ------------
N = 50_000

# Growth rates
K_OV = np.log(5000 / 1e-9)
K_OM = np.log(3000 / 1e-9)


df_results = []
N_iters = 20
for j in range(N_iters):


    # --------------------------------------------------------
    # --------- Baseline parameter sampling ------------------
    # --------------------------------------------------------

    # ----------- Samples ---------------
    # sample indices
    indices = choices(sims.index, k=N)

    # growth rates
    mus = pd.DataFrame({'ix': range(N),
                        'beta_ov': sims.iloc[indices]['beta_ov'].values,
                        'beta_om': sims.iloc[indices]['beta_om'].values,
                        'V_met': sims.iloc[indices]['size_pt_at_met'].values})

    # Baseline CA125
    indices = choices(draws_healthy.index, k= N)
    X_healthy = draws_healthy.loc[indices]
    X_healthy['ix'] = range(N) # Add indices to healthy parameter sets

    # CA125 shedding by location
    # draws_vol_ca125_ = draws_vol_ca125[draws_vol_ca125.c_i.gt(0)] # use only those with a positive CA125 intercept

    indices = choices(draws_vol_ca125.index, k= N)
    X = draws_vol_ca125.loc[indices]
    X['ix'] = range(N) # Add indices to healthy parameter sets

    # ------- Create vector of IDs and timepoints for sampling ----

    # Get a vector of times since first measurement for all the individuals
    delta_t = np.arange(0, 60, 0.5) / 12 # regular sampling every 0.5 months for 5 years (in years)
    T_vec = np.tile(delta_t, N) # Vector of sampling times

    # Get indices for each measurement and the ages at entry (Age 50)
    ix_vec = np.repeat(range(N), len(delta_t)) 

    # ------ Create dataframe of hyperparameters and data -------
    df_raw = pd.DataFrame({'ix': ix_vec, 'dt': T_vec})

    df_raw = pd.merge(df_raw, X_healthy, on='ix', how='left')
    df_raw = pd.merge(df_raw, mus, on='ix', how='left')
    df_raw = pd.merge(df_raw, X, on='ix', how='left')


    # Get the CA-125 at each of the times - vecotirse for speed
    df_raw['inter_healthy'] = df_raw['mu'] + (df_raw['dt'] / 10) * df_raw['dt_d'] 
    n = len(df_raw)
    loc = df_raw['inter_healthy'].to_numpy()
    scale = np.exp(df_raw['logsig'].to_numpy())
    df_raw['ca125_healthy'] = np.exp(loc + scale * np.random.standard_t(df=5, size=n))

    df_raw['ca125_bl'] = np.exp(df_raw['inter_healthy'])

    # ---------- Model tumour growth -----------
    # Doing it manually to speed it up
    t = df_raw['dt'].to_numpy() * 365 # time vector separately
    beta_ov = df_raw['beta_ov'].to_numpy()
    
    V_met = df_raw['V_met'].to_numpy()
    V0 = 1e-9
    
    # Direct Gompertz formula
    df_raw['vol_ov'] = V0 * np.exp(K_OV * (1 - np.exp(-beta_ov * t)))
    
    # Direct time to volume Gompertz
    df_raw['t_met'] = -1 / beta_ov * np.log(1 - 1/K_OV * np.log(V_met / V0))
    
    # Omental volumes with direct Gompertz formula
    mask = df_raw.vol_ov.to_numpy() > df_raw.V_met.to_numpy() # which ones are metastatic
    t_om = df_raw.loc[mask, 'dt'].to_numpy() * 365 - df_raw.loc[mask, 't_met'].to_numpy()
    beta_om = df_raw.loc[mask, 'beta_om'].to_numpy()
    df_raw['vol_om'] = 0
    df_raw.loc[mask, 'vol_om'] = V0 * np.exp(K_OM * (1 - np.exp(-beta_om * t_om)))

    df_raw['vol_tot'] = df_raw['vol_ov'] + df_raw['vol_om']

    # ----------- CA-125 shedding -----------
    df_raw['ca125_tum'] = df_raw.vol_ov * df_raw.sp + df_raw.vol_om * df_raw.sm

    df_raw['ca125_tot'] = df_raw['ca125_healthy'] + df_raw['ca125_tum']
    
    
    # -----------------------------------------------------------
    # ------------- CUSUM based detection!!! --------------------
    # -----------------------------------------------------------
    # ----------- Function for a given h,k pairing ---------------

    df = df_raw.copy()
    df['Z_t'] = (np.log(df.ca125_tot) -  df.inter_healthy) / np.exp(df.logsig) # total contribution
    df['Z_h'] = (np.log(df.ca125_healthy) - df.inter_healthy) / np.exp(df.logsig) # healthy only (no tumour)

    # ----------- k=0.5, check the number before mets ---------
    k = 0.5
    df_out = (df.sort_values(by=['ix', 'dt']).groupby('ix', group_keys=False).apply(lambda x: get_CUSUM(x, k=k)))

    # We can now get the h limit from the 95% percentile
    h_lim = df_out.groupby('ix').S_h.max().quantile(0.95)

    # Get the times now
    df_det = df_out[df_out.S_t > h_lim].groupby('ix').head(1)

    # How many have metastasised? - okay 53.59 would metastasise
    df_det['met'] = 0
    df_det.loc[df_det.vol_tot > df_det.vol_ov, 'met'] = 1

    # ------- group by CA125 bands and see -------
    df_det['ca125_gp'] = 0
    df_out['ca125_gp'] = 0
    grp_lims = [10, 15, 20, 25, 30]

    for i, g in enumerate(grp_lims):
        df_det.loc[np.exp(df_det.mu) > g, 'ca125_gp'] = i + 1
        df_out.loc[np.exp(df_out.mu) > g, 'ca125_gp'] = i + 1

    # Get proportion of mets by group
    counts = df_det.groupby('ca125_gp').met.value_counts().rename('counts').reset_index()
    counts = counts.pivot(index='ca125_gp', columns='met', values='counts').reset_index()
    counts['prop'] = counts[0] / (counts[0] + counts[1])
    x = [0]
    x.extend(grp_lims)
    labs = ['{}-{}'.format(x[i], x[i+1]) for i in range(len(x) - 1)]
    labs.append('{}+'.format(x[-1]))
    counts['labels'] = labs
    counts['run'] = j

    # ---------- Store information -----------
    df_results.append(counts)

df_results = pd.concat(df_results)
df_results.to_csv('./../../output/simulations/results_white_w_uncertainty_generated.csv')