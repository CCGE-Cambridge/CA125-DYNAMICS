'''
In this script, I switch the model to a logistic one and run the whole simulation
- I use the following relationship between Gompertz and logistic growth rates, based on equal maximal growth rates
r = 4*beta / e
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm
from random import choices
import scipy
from utils import *


# ------------ Load data -------------
sims = pd.read_csv('./../../data/simulated_ttms_anonymised.csv', index_col=0)
draws_vol_ca125 = pd.read_csv('./../../output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/site_specific_draws_original.csv')
draws_healthy = pd.read_csv('./../../data/CA125-in-healthy-individuals/draws_healthy_white.csv', index_col=0)

# -------------------------------------------------------
# ------------ Convert vols at met to logistic ----------
# -------------------------------------------------------
Vmax_OV = 5000
Vmax_OM = 3000
V0 = 1e-9
sims_log = sims[['ix', 'vol_ov', 'vol_om', 'beta_ov', 'beta_om']]

# Get the r values
sims_log['r_ov'] = sims_log['beta_ov'] * 4 / np.exp(1)
sims_log['r_om'] = sims_log['beta_om'] * 4 / np.exp(1)

# Get time to volume using logistic
sims_log['t1_ov'] = -1/sims_log.r_ov.to_numpy() * np.log((Vmax_OV / sims_log.vol_ov.to_numpy() - 1) / (Vmax_OV / V0 - 1))
sims_log['t1_om'] = -1/sims_log.r_om.to_numpy() * np.log((Vmax_OM / sims_log.vol_om.to_numpy() - 1) / (Vmax_OM / V0 - 1))

sims_log['ttm'] = sims_log.t1_ov - sims_log.t1_om

sims_log['size_pt_at_met'] = 0

# Do ovarian first
mask_ov = sims_log.ttm.to_numpy() > 0 # ovarian first
sims_log.loc[mask_ov, 'size_pt_at_met'] = Vmax_OV / (1 + (Vmax_OV - V0) / V0 * np.exp(-sims_log.loc[mask_ov, 'r_ov'] * sims_log.loc[mask_ov, 'ttm'].to_numpy()))

# Do omental - note the minus sign in fromt of the ttm
mask_om = sims_log.ttm.to_numpy() < 0 # omental originates first
sims_log.loc[mask_om, 'size_pt_at_met'] = Vmax_OM / (1 + (Vmax_OM - V0) / V0 * np.exp(-sims_log.loc[mask_om, 'r_om'] * -sims_log.loc[mask_om, 'ttm'].to_numpy()))

# ----------- Hyper-parameters ------------
N = 50_000

# PT size at met 
sizes = sims_log.size_pt_at_met.values

results = []
N_iters = 20
for j in range(N_iters):

    # ----------- Samples ---------------
    # sample indices
    indices = choices(sims_log.index, k=N)

    # growth rates and volumes of primary tumour at metastasis
    mus = pd.DataFrame({'ix': range(N),
                        'beta_ov': sims_log.iloc[indices]['beta_ov'].values,
                        'beta_om': sims_log.iloc[indices]['beta_om'].values,
                        'V_met': sims_log.iloc[indices]['size_pt_at_met'].values})

    # Baseline CA125
    indices = choices(draws_healthy.index, k= N)
    X_healthy = draws_healthy.loc[indices]
    X_healthy['ix'] = range(N) # Add indices to healthy parameter sets

    # CA125 shedding by location
    draws_vol_ca125_ = draws_vol_ca125[draws_vol_ca125.c_i.gt(0)] # use only those with a positive CA125 intercept

    indices = choices(draws_vol_ca125_.index, k= N)
    X = draws_vol_ca125.loc[indices]
    X['ix'] = range(N) # Add indices to healthy parameter sets

    # ------- Create vector of IDs and timepoints for sampling ----

    # Get a vector of times since first measurement for all the individuals
    delta_t = np.arange(0, 60, 0.5) / 12 # regular sampling every 0.5 months for 5 years (in years)
    T_vec = np.tile(delta_t, N) # Vector of sampling times

    # Get indices for each measurement and the ages at entry (Age 50)
    ix_vec = np.repeat(range(N), len(delta_t))

    # ------ Create dataframe of hyperparameters and data -------
    df_raw = pd.DataFrame({'ix': ix_vec,
                           'dt': T_vec})

    df_raw = pd.merge(df_raw, X_healthy, on='ix', how='left')
    df_raw = pd.merge(df_raw, mus, on='ix', how='left')
    df_raw = pd.merge(df_raw, X, on='ix', how='left')

    # Get the CA-125 at each of the times - vectorise for efficiency
    df_raw['inter_healthy'] = df_raw['mu'] + (df_raw['dt'] / 10) * df_raw['dt_d']
    n = len(df_raw)
    loc = df_raw['inter_healthy'].to_numpy()
    scale = np.exp(df_raw['logsig'].to_numpy())
    df_raw['ca125_healthy'] = np.exp(loc + scale * np.random.standard_t(df=5, size=n))

    df_raw['ca125_bl'] = np.exp(df_raw['inter_healthy'])

    # ---------- Model tumour growth -----------
    # Vectorise for efficiency
    t = df_raw['dt'].to_numpy() * 365           # time vector separately
    beta_ov = df_raw['beta_ov'].to_numpy()      # Gompertz growth rates
    r_ov = beta_ov * 4 / np.exp(1)              # Logistic growth rates
    V_met = df_raw['V_met'].to_numpy()          # Volumes at metastasis
    V0 = 1e-9                                   # Volume at 1 cell (V0)

    # Direct Logistic formula
    num = Vmax_OV
    den = 1 + (Vmax_OV - V0) / V0 * np.exp(-r_ov * t)
    df_raw['vol_ov'] = num / den

    # Direct time to volume Logistic
    df_raw['t_met'] = -1/r_ov * np.log((Vmax_OV / V_met - 1) / (Vmax_OV / V0 - 1))

    # Direct omental grwoth calculation
    mask = df_raw.vol_ov.to_numpy() > df_raw.V_met.to_numpy() # which ones are metastatic
    t_om = df_raw.loc[mask, 'dt'].to_numpy() * 365 - df_raw.loc[mask, 't_met'].to_numpy()
    beta_om = df_raw.loc[mask, 'beta_om'].to_numpy()
    r_om = beta_om * 4 / np.exp(1)
    df_raw['vol_om'] = 0
    num = Vmax_OM
    den = 1 + (Vmax_OM - V0) / V0 * np.exp(-r_om * t_om)
    df_raw.loc[mask, 'vol_om'] = num / den

    df_raw['vol_tot'] = df_raw['vol_ov'] + df_raw['vol_om']

    # ----------- CA-125 shedding -----------
    df_raw['ca125_tum'] = df_raw.vol_ov * df_raw.sp + df_raw.vol_om * df_raw.sm

    df_raw['ca125_tot'] = df_raw['ca125_healthy'] + df_raw['ca125_tum']

    # -----------------------------------------------------------
    # ------------- CUSUM based detection!!! --------------------
    # -----------------------------------------------------------
    # ----------- Function for a given h,k pairing ---------------
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

    # How many have metastasised?
    df_det['met'] = 0
    df_det.loc[df_det.vol_tot > df_det.vol_ov, 'met'] = 1
    results.append({'run': j,
                    'prop_met':df_det.met.value_counts(normalize=True).loc[1]})

# 74.9% [74.1, 75.4]
df_results = pd.DataFrame(results)
df_results.to_csv('./../../output/simulations/results_logistic_white.csv')
# ----------- Save model --------------- IF REQUIRED
# df_raw.to_csv('./../../output/simulations/sims_white_logistic_joint_20260726.csv')
