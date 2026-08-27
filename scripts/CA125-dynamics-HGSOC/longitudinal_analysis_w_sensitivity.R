# Multivariate Bayesian Hierarchical modelling of volume and CA125 growth
# We use 37 sequences of volumes and CA125 measurements from 31 women in the CTCR-OV04 case cohort study
library(tidyverse)
library(performance)
library(DHARMa)
library(reshape2)
library(brms)


setwd("C:/Users/bn312/OneDrive - University of Cambridge/OV04/CA125-DYNAMICS/")
output_folder = './output/CA125-in-the-presence-of-HGSOC/longitudinal-modelling/'

# Load df
df <- read_csv('./data/CA125-in-presence-of-HGSOC/longitudinal_modelling.csv')

# Plot raw data if you like
ggplot(df, aes(x = dt, y = y, color = measure_type)) +
  geom_line(na.rm = TRUE) +
  geom_point(size = 1.1, alpha = 0.9, na.rm = TRUE) +
  facet_wrap(~ combined_id, scales = "fixed", ncol = 10) +  
  labs(x = "Time", y = "y", color = "Measure") +
  theme_bw() +
  theme(panel.grid.minor = element_blank(),
        legend.position = "bottom")


# ---------   BRMS modelling ------------

## Formulae:
# 1. Separate intercept and slope by measurement type
f <- bf(
  y ~ 0 + measure_type + dt_m:measure_type +
    (0 + measure_type + dt_m:measure_type | combined_id),
  nu=5
)

# 2. Same as 1. but different residual scaling factors by measurement type
# This is to check if the different residuals makes a difference
f_measure_spec_resids <- bf(
  y ~ 0 + measure_type + dt_m:measure_type +
    (0 + measure_type + dt_m:measure_type | combined_id),
  sigma ~ 0 + measure_type,
  nu=5
)

## Fitting

# 1. With lkj 1 prior
priors <- c(
  prior(normal(0,5), class='b'),
  prior(student_t(3, 0, 2.5), class='sigma'),
  prior(student_t(3, 0, 2.5), class='sd'),
  prior(lkj(1), class='cor')
)

fit_brm_lkj1 <- brm(
  formula = f,
  data = df,
  family = student(),
  prior = priors,
  chains=3, cores=3, iter=5000
)
save(fit_brm_lkj1, file = paste0(output_folder,"brms_fit_lkj1.RData"))


# 2. with lkj4 prior
priors <- c(
  prior(normal(0,5), class='b'),
  prior(student_t(3, 0, 2.5), class='sigma'),
  prior(student_t(3, 0, 2.5), class='sd'),
  prior(lkj(4), class='cor')
)

fit_brm_lkj4 <- brm(
  formula = f,
  data = df,
  family = student(),
  prior = priors,
  chains=3, cores=3, iter=5000
)
save(fit_brm_lkj4, file = paste0(output_folder, "brms_fit_lkj4.RData"))

# 3. separate residuals
priors <- c(
  prior(normal(0,5), class='b'),
  prior(student_t(3, 0, 2.5), class='sd'),
  prior(lkj(1), class='cor')
)

fit_brm_lkj1_resids <- brm(
  formula = f_measure_spec_resids,
  data = df,
  family = student(),
  prior = priors,
  chains=3, cores=3, iter=5000
)
save(fit_brm_lkj1_resids, file = paste0(output_folder, "brms_fit_lkj1_separate_resids.RData"))

## Compare separate residuals with just single residual
fit_brm_lkj1  <- add_criterion(fit_brm_lkj1,  "loo", reloo = TRUE, overwrite = TRUE)
fit_brm_resids <- add_criterion(fit_brm_lkj1_resids, "loo", reloo = TRUE, overwrite = TRUE)
loo_compare(fit_brm_lkj1, fit_brm_resids)

# *********************************************************
## ------------ Model analysis and checks ---------------
# *********************************************************

# We are continuing with the LKJ 1 prior with a single residual variance for volumes and CA125
fit_brm <- fit_brm_lkj1

# See if difference in means is significant
post <- as_draws_df(fit_brm)
delta <- post$'b_measure_typeca125:dt_m' - post$'b_measure_typevol:dt_m'
quantile(delta, c(0.025, 0.5, 0.975))

# See if difference in mean TVDTs is significant - no
delta <- (log(2)/post$'b_measure_typeca125:dt_m') - (log(2) / post$'b_measure_typevol:dt_m')
quantile(delta, c(0.025, 0.5, 0.975))

output_folder_plots <- paste0(output_folder, "plots/")

# ------------------ Diagnostic plots for MCMC chains ------------
plots <- plot(fit_brm, nvariables=1) 

for (i in seq_along(plots)) {
  ggsave(
    filename = paste0(output_folder_plots, "mcmc_plot_", i, ".png"),
    plot = plots[[i]],
    width = 8, height = 4, dpi = 300
  )
}

# ----------------- Density overlay and ECDF ---------------
p <- pp_check(fit_brm, ndraws=200, main='default density')      
ggsave(
  filename = paste0(output_folder_plots, "pp_density.png"),
  plot = p + theme_bw(),
)
p <- pp_check(fit_brm, type="ecdf_overlay", ndraws=2000)
ggsave(
  filename = paste0(output_folder_plots, "pp_ecdf.png"),
  plot = p + theme_bw(),
)

# Analyse fit by measure_type
p <- pp_check(fit_brm, type = "scatter_avg_grouped", group = "measure_type")
ggsave(
  filename = paste0(output_folder_plots, "pp_scatter.png"),
  plot = p + theme_bw(),
)

# ---- Residuals vs fitted values using posterior means----

res_df <- df %>%
  mutate(
    residual   = residuals(fit_brm, summary = TRUE)[, "Estimate"],
    fitted     = fitted(fit_brm, summary = TRUE)[, "Estimate"],
    abs_resid  = abs(residual)
  )

# Residuals vs fitted, coloured by measure_type
p <- ggplot(res_df, aes(x = fitted, y = residual, colour = measure_type)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_point(alpha = 0.6) +
  
  geom_smooth(method = "loess", se = FALSE, linewidth = 1) +
  
  geom_smooth(method = "lm", se = FALSE, linetype = "dotted") +
  labs(x = "Fitted value", y = "Residual",
       title = "Residuals vs fitted values (nonlinearity check)") +
  theme_bw()

ggsave(
  filename = paste0(output_folder_plots, "linearity.png"),
  plot = p + theme_bw(),
)

# Heteroscedasticity 
p <- ggplot(res_df, aes(x = fitted, y = abs_resid, colour = measure_type)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 1) +
  # facet_wrap(~ measure_type) +
  labs(x = "Fitted value", y = "|Residual|",
       title = "Absolute residuals vs fitted (heteroscedasticity)") +
  theme_bw()

ggsave(
  filename = paste0(output_folder_plots, "heteroscedasticity.png"),
  plot = p + theme_bw(),
  # width = 8, height = 4, dpi = 300
)


# -------------- LOO-PIT plots ----------------

fit_brm <- add_criterion(
  fit_brm,
  "loo",
  reloo     = TRUE,
  overwrite = TRUE
)
p <- brms::pp_check(
    fit_brm,
    type = "loo_pit_ecdf",
    ndraws = 1000,
    loo = fit_brm$criteria$loo
  )
ggsave(
  filename = paste0(output_folder_plots, "loo_ecdf.png"),
  plot = p + theme_bw(),
  # width = 8, height = 4, dpi = 300
)

p <- brms::pp_check(
  fit_brm,
  type = "loo_pit_qq",
  ndraws = 1000,
  loo = fit_brm$criteria$loo
)
ggsave(
  filename = paste0(output_folder_plots, "loo_qq.png"),
  plot = p + theme_bw(),
)

p <- brms::pp_check(
  fit_brm,
  type = "loo_pit_overlay",
  ndraws = 100,
  loo = fit_brm$criteria$loo
)
ggsave(
  filename = paste0(output_folder_plots, "loo_overlay.png"),
  plot = p + theme_bw(),
)

# ---------- Individual plots ----------

# Get fitted values including the random effects
fitted_indiv <- fitted(
  fit_brm,
  re_formula = NULL,     # include random effects
  summary = TRUE
)
fitted_pop <- fitted(
  fit_brm,
  re_formula = NA,
  summary = TRUE
)

# Convert to dataframe with the 
fitted_df <- df %>%
  mutate(
    fit_mean  = fitted_indiv[, "Estimate"],
    fit_lower = fitted_indiv[, "Q2.5"],
    fit_upper = fitted_indiv[, "Q97.5"],
    fit_pop = fitted_pop[, "Estimate"]
  )

for (id in unique(df$combined_id)) {
  
  df_sub <- fitted_df %>% filter(combined_id == id)
  
  p <- ggplot(df_sub, aes(x = dt_m, y = y, colour = measure_type)) +
    geom_point(alpha = 0.6) +
    geom_line(aes(y = fit_mean)) +
    geom_ribbon(aes(ymin = fit_lower, ymax = fit_upper, fill = measure_type),
                alpha = 0.2, colour = NA) +
    geom_line(aes(y=fit_pop), linetype='dashed') + 
    scale_fill_discrete(guide = "none") +
    scale_colour_discrete(guide = "none") +
    theme_bw(base_size = 14) +
    theme(
      panel.grid = element_blank(),        # remove grid
      panel.border = element_blank(),   # removes full border
      axis.line = element_line(
        arrow = arrow(type = "closed", length = unit(0.25, "cm"))
      )
    ) + 
    labs(
      y = expression(y^c*"/"*y^v),
      x = expression(Delta*"t (months)")
    )
  
  ggsave(
    filename = paste0(output_folder_plots, 'indiv_plot_', id, ".png"),
    plot = p,
    width = 4, height = 3.5, dpi = 300
  )
}


