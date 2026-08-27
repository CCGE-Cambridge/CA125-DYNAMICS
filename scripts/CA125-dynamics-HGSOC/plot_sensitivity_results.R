library(dplyr)
library(ggplot2)
library(patchwork)
library(brms)


# Load data
setwd("C:/Users/bn312/OneDrive - University of Cambridge/OV04/CA125-DYNAMICS/")
load("./output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/volume_sensitivity.RData")
load('./output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/loo_sensitivity.RData')
load("./output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/brms_fit_site_specific.RData")


fit_orig <- fit

# ----------------- Original estimates ---------------------
orig <- fixef(fit_orig, probs = c(0.025, 0.975))

sp_mean_orig <- orig["sp_Intercept", "Estimate"]
sp_l95_orig  <- orig["sp_Intercept", "Q2.5"]
sp_u95_orig  <- orig["sp_Intercept", "Q97.5"]

sm_mean_orig <- orig["sm_Intercept", "Estimate"]
sm_l95_orig  <- orig["sm_Intercept", "Q2.5"]
sm_u95_orig  <- orig["sm_Intercept", "Q97.5"]


# Function definition for plotting
plot_sensitivity_runs <- function(data,
                                  med_col,
                                  l95_col,
                                  u95_col,
                                  orig_med,
                                  orig_l95,
                                  orig_u95,
                                  xlab,
                                  centre_on_original = TRUE) {
  
  plot_df <- data %>%
    mutate(run = seq_len(n())) %>%
    arrange(.data[[med_col]]) %>%
    mutate(
      run_order = seq_len(n()),
      run_f = factor(run_order, levels = rev(run_order))
    )
  
  # Symmetric x-axis around original median
  if (centre_on_original) {
    all_x <- c(
      plot_df[[l95_col]],
      plot_df[[u95_col]],
      orig_l95,
      orig_u95
    )
    
    half_range <- max(abs(all_x - orig_med), na.rm = TRUE) * 1.05
    x_limits <- c(0, orig_med + half_range)
    
  } else {
    x_limits <- range(
      c(
        plot_df[[l95_col]],
        plot_df[[u95_col]],
        orig_l95,
        orig_u95
      ),
      na.rm = TRUE
    )
  }
  
  ggplot(plot_df, aes(y = run_f)) +
    
    # Run-specific 95% posterior interval
    geom_segment(
      aes(
        x = .data[[l95_col]],
        xend = .data[[u95_col]],
        yend = run_f
      ),
      linewidth = 0.45
    ) +
    
    # Run-specific posterior median
    geom_point(
      aes(x = .data[[med_col]]),
      shape = 23,
      size = 2.4,
      fill = "white",
      stroke = 0.7
    ) +
    
    # Original posterior median
    geom_vline(
      xintercept = orig_med,
      linewidth = 0.8
    ) +
    
    # Original 95% posterior interval
    geom_vline(
      xintercept = c(orig_l95, orig_u95),
      linetype = "dashed",
      linewidth = 0.4
    ) +
    
    coord_cartesian(xlim = x_limits) +
    
    labs(
      x = xlab,
      y = NULL
    ) +
    
    theme_classic(base_size = 12) +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      plot.margin = margin(5.5, 10, 5.5, 5.5)
    )
}

# --------------- Plot volumetric sensitivity ------------

# sp plot
p_sp <- plot_sensitivity_runs(
  data = sens_results_df,
  med_col = "sp_mean",
  l95_col = "sp_l95",
  u95_col = "sp_u95",
  orig_med = sp_mean_orig,
  orig_l95 = sp_l95_orig,
  orig_u95 = sp_u95_orig,
  xlab = expression(s[p]),
  centre_on_original = TRUE
)

# sm plot
p_sm <- plot_sensitivity_runs(
  data = sens_results_df,
  med_col = "sm_mean",
  l95_col = "sm_l95",
  u95_col = "sm_u95",
  orig_med = sm_mean_orig,
  orig_l95 = sm_l95_orig,
  orig_u95 = sm_u95_orig,
  xlab = expression(s[m]),
  centre_on_original = TRUE
)

# Combine plots
p_sensitivity <- p_sp + p_sm +
  plot_annotation(
    title = "Sensitivity to 30% multiplicative volume perturbation"
  )

# Save plots
ggsave(
  filename = paste0('./output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/plots/vol_sensitivity.png'),
  plot = p_sensitivity,
  width = 15, height = 8, dpi = 300, units='cm'
)

# --------------- LOO sensitivity ------------

# sp plot
p_sp <- plot_sensitivity_runs(
  data = loo_results_df,
  med_col = "sp_mean",
  l95_col = "sp_l95",
  u95_col = "sp_u95",
  orig_med = sp_mean_orig,
  orig_l95 = sp_l95_orig,
  orig_u95 = sp_u95_orig,
  xlab = expression(s[p]),
  centre_on_original = TRUE
)

# sm plot
p_sm <- plot_sensitivity_runs(
  data = loo_results_df,
  med_col = "sm_mean",
  l95_col = "sm_l95",
  u95_col = "sm_u95",
  orig_med = sm_mean_orig,
  orig_l95 = sm_l95_orig,
  orig_u95 = sm_u95_orig,
  xlab = expression(s[m]),
  centre_on_original = TRUE
)

# Combine plots
p_loo <- p_sp + p_sm +
  plot_annotation(
    title = "Sensitivity to 30% multiplicative volume perturbation"
  )

ggsave(
  filename = paste0('./output/CA125-in-the-presence-of-HGSOC/site-specific-shedding/plots/loo_sensitivity.png'),
  plot = p_loo,
  width = 15, height = 8, dpi = 300, units='cm'
)


# ------------ Some numbers ----------------
# Volumetric sensitivity
quantile(abs(sens_results_df$sp_mean - sp_mean_orig) * 100 / sp_mean_orig, c(0.025, 0.975))
mean(abs(sens_results_df$sp_mean - sp_mean_orig) * 100 / sp_mean_orig)

quantile(abs(sens_results_df$sm_mean - sm_mean_orig) * 100 / sm_mean_orig, c(0.025, 0.975))
mean(abs(sens_results_df$sm_mean - sm_mean_orig) * 100 / sm_mean_orig)


# LOO sensitivity
quantile(abs(loo_results_df$sp_mean - sp_mean_orig) * 100 / sp_mean_orig, c(0.025, 0.975))
mean(abs(loo_results_df$sp_mean - sp_mean_orig) * 100 / sp_mean_orig)

quantile(abs(loo_results_df$sm_mean - sm_mean_orig) * 100 / sm_mean_orig, c(0.025, 0.975))
mean(abs(loo_results_df$sm_mean - sm_mean_orig) * 100 / sm_mean_orig)
