## Sensitivity analyses

library(tidyverse)
library(performance)
library(DHARMa)
library(reshape2)
library(brms)
library(ggplot2)


setwd("C:/Users/bn312/OneDrive - University of Cambridge/OV04/CA125-DYNAMICS/")
output_folder <- './output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/'

# Load df
df <- read_csv('./data/CA125-in-presence-of-HGSOC/site_specific_shedding.csv')

# Get CA125 in log scale
df$log_ca125 <- log(df$ca125)

# --------------------------------------------------
# -------- Fitting site-specific shedding model ----
# --------------------------------------------------
load(paste0(output_folder, "brms_fit_site_specific.RData"))
fit_orig <- fit

# --------- Orig posterior summary -----------------
orig_draws <- as_draws_df(fit_orig)

orig_summary <- orig_draws %>%
  summarise(
    sp_med = median(b_sp_Intercept),
    sp_l95 = quantile(b_sp_Intercept, 0.025),
    sp_u95 = quantile(b_sp_Intercept, 0.975),
    
    sm_med = median(b_sm_Intercept),
    sm_l95 = quantile(b_sm_Intercept, 0.025),
    sm_u95 = quantile(b_sm_Intercept, 0.975)
  )

print(orig_summary)

# --------------------------------------------------
# ------------------ Sensitivity -------------------
# --------------------------------------------------
# multiplicative Gaussian perturbation
cv <- 0.3 # coefficient of variation - 50% of mean
n_sens <- 20 # NUmber of runs
sens_results <- vector("list", n_sens)
library(truncnorm)

for (i in seq_len(n_sens)) {
  
  message("Running sensitivity fit ", i, " of ", n_sens)
  
  # Only use those perturbations that do not result in 0 volumes...
  df_pert <- df %>%
    mutate(
      mult_pod = rtruncnorm(
        n = n(),
        a = 0,
        b = Inf,
        mean = 1,
        sd = cv
      ),
      mult_met = rtruncnorm(
        n = n(),
        a = 0,
        b = Inf,
        mean = 1,
        sd = cv
      ),
      
      vol_pod = vol_pod * mult_pod,
      vol_met_om = vol_met_om * mult_met
    )
  fit_i <- update(
    fit_orig,
    newdata = df_pert,
    recompile = FALSE,
    refresh = 0,
    seed = 1000 + i
  )
  
  draws_i <- as_draws_df(fit_i)
  
  sens_results[[i]] <- draws_i %>%
    summarise(
      iter = i,
      
      sp_mean = mean(b_sp_Intercept),
      sp_l95 = quantile(b_sp_Intercept, 0.025),
      sp_u95 = quantile(b_sp_Intercept, 0.975),
      
      sm_mean = mean(b_sm_Intercept),
      sm_l95 = quantile(b_sm_Intercept, 0.025),
      sm_u95 = quantile(b_sm_Intercept, 0.975),
      
      # Add info about min, max volumes and noise
      vol_pod_l95 = quantile(df_pert$vol_pod, 0.025),
      vol_pod_u95 = quantile(df_pert$vol_pod, 0.975),
      vol_met_l95 = quantile(df_pert$vol_met_om, 0.025),
      vol_met_u95 = quantile(df_pert$vol_met_om, 0.975),
      
      
      prob_sm_gt_sp = mean(b_sm_Intercept > b_sp_Intercept)
    )
}

sens_results_df <- bind_rows(sens_results)

# Get the ratio
sens_results_df$ratio_sm_sp <- sens_results_df$sm_mean / sens_results_df$sp_mean
save(sens_results_df, file = paste0(output_folder,"volume_sensitivity.RData") )

# Mean sp and sm quantiles differences
quantile(abs(sens_results_df$sp_mean - 0.3) / 0.3, c(0.025, 0.975)) * 100 # 0.4%, 23.6%
mean(abs(sens_results_df$sp_mean - 0.3) / 0.3) # 9%
quantile(abs(sens_results_df$sm_mean - 3.0) / 3.0, c(0.025, 0.975)) * 100 # 1.7%, 31.4%
mean(abs(sens_results_df$sm_mean - 3.0) / 3.0) * 100 # 12.0%

quantile(abs(sens_results_df$ratio_sm_sp - 10.0) / 10.0, c(0.025, 0.975)) * 100 # 1.4%, 26.6%
mean(abs(sens_results_df$ratio_sm_sp - 10.0) / 10.0) * 100 # 13.2%


# -----------------------------------------------------------------
# ------------------ Leave-one-out sensitivity  -------------------
# -----------------------------------------------------------------
unique_ids <- unique(df$pat_id)
loo_results <- vector("list", length(unique_ids))

for (i in seq_along(unique_ids)){
  
  # Remove this id
  id_ <- unique_ids[i]
  df_ <- df[df$pat_id != id_, ]
  
  fit_i <- update(
    fit_orig,
    newdata = df_,
    recompile = FALSE,
    refresh = 0,
  )

  draws_i <- as_draws_df(fit_i)
  
  loo_results[[i]] <- draws_i %>%
    summarise(
      id_left_out = id_,
      n_obs = dim(df_)[1],

      sp_mean = mean(b_sp_Intercept),
      sp_l95 = quantile(b_sp_Intercept, 0.025),
      sp_u95 = quantile(b_sp_Intercept, 0.975),
      
      sm_mean = mean(b_sm_Intercept),
      sm_l95 = quantile(b_sm_Intercept, 0.025),
      sm_u95 = quantile(b_sm_Intercept, 0.975),
      
      prob_sm_gt_sp = mean(b_sm_Intercept > b_sp_Intercept)
    )
  
}

loo_results_df <- bind_rows(loo_results)

# Get the ratio
loo_results_df$ratio_sm_sp <- loo_results_df$sm_mean / loo_results_df$sp_mean

# Add on mean volumes per individual and met frac
df$vol_tot <- df$vol_pod + df$vol_met_om
df$met_frac <- df$vol_met_om / df$vol_tot
loo_addon <- df %>% group_by(pat_id) %>% summarise(mean_vol_tot = mean(vol_tot), mean_met_frac = mean(met_frac), mean_ca125 = mean(ca125))
colnames(loo_addon)[1] <- 'id_left_out'
loo_results_df <- merge(loo_results_df, loo_addon, by='id_left_out')

write.csv(loo_results_df, paste0(output_folder, "sensitivity_loo.csv", row.names = FALSE))
save(loo_results_df, file = paste0(output_folder, "loo_sensitivity.RData") )

# Mean sp and sm quantiles differences
quantile(abs(loo_results_df$sp_mean - 0.3) / 0.3, c(0.025, 0.975)) * 100 # 0.1%, 19.0%
mean(abs(loo_results_df$sp_mean - 0.3) / 0.3) * 100 # 3.7%
quantile(abs(loo_results_df$sm_mean - 3.0) / 3.0, c(0.025, 0.975)) * 100 # 0.4%, 13.3%
mean(abs(loo_results_df$sm_mean - 3.0) / 3.0) * 100 # 4.2%

quantile(abs(loo_results_df$ratio_sm_sp - 10.0) / 10.0, c(0.025, 0.975)) * 100 # 0.3%, 22.3%
mean(abs(loo_results_df$ratio_sm_sp - 10.0) / 10.0) * 100 # 5.8%

# -----------------------------------------------------------------
# ------------------ Prior sensitivity  -------------------
# -----------------------------------------------------------------

# I test three priors - one wider, one narrower and one that is systematically lower

# 1. Narrow prior
prior_narrow <- c(
  # Population median baseline CA125 around 12.5 U/mL
  prior(normal(log(12.4), 1.0), nlpar = "logc"),
  
  # Between-patient SD in log baseline CA125
  # chosen so that 95% interval is roughly 5.4 to 28.9 from the healthy CA125 data
  # 4 sigma ~= log(28.9) - log(5.4)
  prior(normal(0.42, 0.2), class = "sd", nlpar = "logc", lb = 0),
  
  prior(lognormal(log(1), 0.5), nlpar = "sp", lb = 0),
  prior(lognormal(log(1), 0.5), nlpar = "sm", lb = 0)
  
)

fit_narrow <- update(
  fit_orig,
  prior = prior_narrow,
  recompile = TRUE,
  seed = 123
)
# save(fit_narrow, file = "./output/analyses/site-specific/prior_sens_narrow_20260620.RData")
# 2. Wide prior
prior_wide <- c(
  # Population median baseline CA125 around 12.5 U/mL
  prior(normal(log(12.4), 1.0), nlpar = "logc"),
  
  # Between-patient SD in log baseline CA125
  # chosen so that 95% interval is roughly 5.4 to 28.9 from the healthy CA125 data
  # 4 sigma ~= log(28.9) - log(5.4)
  prior(normal(0.42, 0.2), class = "sd", nlpar = "logc", lb = 0),
  
  prior(lognormal(log(1), 1.5), nlpar = "sp", lb = 0),
  prior(lognormal(log(1), 1.5), nlpar = "sm", lb = 0)
  
)

fit_wide <- update(
  fit_orig,
  prior = prior_wide,
  recompile = TRUE,
  seed = 123,
  control = list(adapt_delta = 0.95, max_treedepth = 12)
)
# save(fit_wide, file = "./output/analyses/site-specific/prior_sens_wide_20260620.RData")

# 3. Shifted prior
prior_shift <- c(
  # Population median baseline CA125 around 12.5 U/mL
  prior(normal(log(12.4), 1.0), nlpar = "logc"),
  
  # Between-patient SD in log baseline CA125
  # chosen so that 95% interval is roughly 5.4 to 28.9 from the healthy CA125 data
  # 4 sigma ~= log(28.9) - log(5.4)
  prior(normal(0.42, 0.2), class = "sd", nlpar = "logc", lb = 0),
  
  prior(lognormal(log(0.5), 1), nlpar = "sp", lb = 0),
  prior(lognormal(log(0.5), 1), nlpar = "sm", lb = 0)
  
)

fit_shift <- update(
  fit_orig,
  prior = prior_shift,
  recompile = TRUE,
  seed = 123
)
# save(fit_shift, file = "./output/analyses/site-specific/prior_sens_shift_20260620.RData")
