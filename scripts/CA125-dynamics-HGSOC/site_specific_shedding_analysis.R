## Analysis of shedding rates by primary (pod) and metastatic (met_om) sites
library(tidyverse)
library(performance)
library(DHARMa)
library(reshape2)
library(brms)
library(ggplot2)
library(bayesplot)
library(posterior)
library(MASS)


setwd("C:/Users/bn312/OneDrive - University of Cambridge/OV04/CA125-DYNAMICS/")

# Load df
df <- read_csv('./data/CA125-in-presence-of-HGSOC/site_specific_shedding.csv')

# Get CA125 in log scale
df$log_ca125 <- log(df$ca125)

# -------- Fit Bayesian hierarchical model -----
# Model log-transformed CA125
fit <- brm(
  bf(
    log_ca125 ~ log(exp(logc) + sp * vol_pod + sm * vol_met_om),
    logc ~ 1 + (1 | pat_id), # log CA125 intercept by subject
    sp ~ 1,                  # association with primary burden
    sm ~ 1,                  # association with metastatic burden
    nl = TRUE
  ),
  data = df,
  prior = c(
    # Population median baseline CA125 around 12.5 U/mL from UKCTOCS healthy fitting
    prior(normal(log(12.4), 1.0), nlpar = "logc"),
    
    # Between-patient SD in log baseline CA125
    # chosen so that 95% interval is roughly 5.4 to 28.9 from the healthy CA125 data
    # 4 sigma ~= log(28.9) - log(5.4)
    prior(normal(0.42, 0.2), class = "sd", nlpar = "logc", lb = 0),
    
    prior(lognormal(log(1), 1), nlpar = "sp", lb = 0),
    prior(lognormal(log(1), 1), nlpar = "sm", lb = 0)
    
  ),
  chains = 3,
  iter = 6000,
  warmup = 4000
)

# This is to see how much of the model is explained by the volume and intercept structure...
# R2 is 0.51 [0.42, 0.59]
bayes_R2(fit)
save(fit, file = "./output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/brms_fit_site_specific.RData")

# --------------------------------------------------------
# -------------------- Model checks ----------------------
# --------------------------------------------------------
output_folder <- "./output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/plots/"

# ------------------ Diagnostic plots for MCMC chains ------------
plots <- plot(fit, nvariables=1) 

for (i in seq_along(plots)) {
  ggsave(
    filename = paste0(output_folder, "mcmc_plot_chain_", i, ".png"),
    plot = plots[[i]],
    width = 8, height = 4, dpi = 300
  )
}

# ----------------- Density overlay and ECDF ---------------

## Log scale
# Observed
y <- df$log_ca125

# log tranformed posterior predictive draws
yrep <- posterior_predict(fit, ndraws = 10)

p <- ppc_dens_overlay(
  y = y,
  yrep = yrep
) +
  labs(
    x = "log(CA125)",
    y = "Density",
    # title = "Posterior predictive check on log CA125 scale"
  ) +
  theme_bw()

ggsave(
  filename = paste0(output_folder, "ppc_density.png"),
  plot = p + theme(legend.position = 'none'),
  width = 8, height = 8, dpi = 300, units='cm'
)

# Again!
yrep <- posterior_predict(fit, ndraws = 10)
p <- ppc_dens_overlay(
  y = y,
  yrep = yrep
) +
  labs(
    x = "log(CA125)",
    y = "Density",
    # title = "Posterior predictive check on log CA125 scale"
  ) +
  theme_bw()

ggsave(
  filename = paste0(output_folder, "ppc_density_1.png"),
  plot = p + theme(legend.position = 'none'),
  width = 8, height = 8, dpi = 300, units='cm'
)

# ---- Residuals vs fitted values using posterior means----

p <- ppc_scatter_avg(
  y = y,
  yrep = yrep
) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  theme_bw() +
  labs(
    x = "Average predicted log(CA125)",
    y = "Observed log(CA125)",
    title = "Posterior predictive scatter check on log scale"
  )

ggsave(
  filename = paste0(output_folder, "scatter_log.png"),
  plot = p + theme_bw(),
  # width = 8, height = 4, dpi = 300
)


# -------------- LOO-PIT plots ----------------

fit <- add_criterion(
  fit,
  "loo",
  reloo     = TRUE,
  overwrite = TRUE
)
p <- brms::pp_check(
  fit,
  type = "loo_pit_ecdf",
  ndraws = 1000,
  loo = fit$criteria$loo
)
ggsave(
  filename = paste0(output_folder, "loo_ecdf.png"),
  plot = p + theme_bw(),
  # width = 8, height = 4, dpi = 300
)

p <- brms::pp_check(
  fit,
  type = "loo_pit_qq",
  ndraws = 3000,
)
ggsave(
  filename = paste0(output_folder, "loo_qq.png"),
  plot = p + theme_bw(),
)

p <- brms::pp_check(
  fit,
  type = "loo_pit_overlay",
  ndraws = 5000,
)
ggsave(
  filename = paste0(output_folder, "loo_overlay.png"),
  plot = p + theme_bw(),
)

# -------------------------------------------------------------
# ------------------- Posterior Draws -------------------------
# -------------------------------------------------------------

# Extract posterior draws
D <- as_draws_df(fit)

keep <- c(
  "b_logc_Intercept",
  "b_sp_Intercept",
  "b_sm_Intercept",
  "sd_pat_id__logc_Intercept",
  "sigma"
)
D <- D[, keep]
S <- nrow(D)

# 2) Simulate new individuals for one posterior draw
simulate_one_draw_sp <- function(s, n_per_draw) {
  
  c_pop  <- D$b_logc_Intercept[s] 
  sp_pop <- D$b_sp_Intercept[s]
  sm_pop <- D$b_sm_Intercept[s]
  tau_c  <- D$sd_pat_id__logc_Intercept[s]
  sigma  <- D$sigma[s]
  
  # Random effects for intercepts  
  u_c <- rnorm(n_per_draw, mean = 0, sd = tau_c)

  c_i <- exp(c_pop + u_c) # transform to exponential

  out <- data.frame(
    draw = s,
    c_i = c_i,
    sp = sp_pop,
    sm = sm_pop,
    sigma = sigma
  )
  
  out
}

# 3) Simulate full population over posterior draws
simulate_population_sp <- function(n_per_draw = 500) {
  out <- vector("list", S)
  
  for (s in seq_len(S)) {
    out[[s]] <- simulate_one_draw_sp(s, n_per_draw)
  }
  
  do.call(rbind, out)
}

theta_new <- simulate_population_sp(n_per_draw = 500)
write.csv(theta_new, "./output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/site_specific_draws_generated.csv", row.names = FALSE)

# ---------------------------------------------------
# -------- Pre- and post- fitting separately --------
# ---------------------------------------------------
fit_pre <- brm(
  bf(
    log_ca125 ~ log(exp(logc) + sp * vol_pod + sm * vol_met_om),
    logc ~ 1,
    sp ~ 1,
    sm ~ 1,
    nl = TRUE
  ),
  data = df1,
  prior = c(
    # Population median baseline CA125 around 12.5 U/mL
    prior(normal(log(12.4), 1.0), nlpar = "logc"),
    
    prior(lognormal(log(1), 1.0), nlpar = "sp", lb = 0), # These priors give 95% CI from 0.1 to 7.1
    prior(lognormal(log(1), 1.0), nlpar = "sm", lb = 0)
    
  ),
  # control = list(adapt_delta = 0.95, max_treedepth = 12),
  chains = 3,
  iter = 6000,
  warmup = 4000
)
# save(fit_pre, file = "./output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/brms_fit_site_specific_pre.RData")

fit_post <- brm(
  bf(
    log_ca125 ~ log(exp(logc) + sp * vol_pod + sm * vol_met_om),
    logc ~ 1 + (1 | pat_id),
    sp ~ 1,
    sm ~ 1,
    nl = TRUE
  ),
  data = df2,
  prior = c(
    # Population median baseline CA125 around 12.5 U/mL
    prior(normal(log(12.4), 1.0), nlpar = "logc"),
    
    # Between-patient SD in log baseline CA125
    # chosen so that 95% interval is roughly 5.4 to 28.9 from the healthy CA125 data
    # 4 sigma ~= log(28.9) - log(5.4)
    prior(normal(0.42, 0.2), class = "sd", nlpar = "logc", lb = 0),
    
    prior(lognormal(log(1), 1), nlpar = "sp", lb = 0),
    prior(lognormal(log(1), 1), nlpar = "sm", lb = 0)
    
  ),
  # control = list(adapt_delta = 0.95, max_treedepth = 12),
  chains = 3,
  iter = 6000,
  warmup = 4000
)
# save(fit_post, file = "../output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/brms_fit_site_specific_post.RData")
