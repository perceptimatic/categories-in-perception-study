TOP <- Sys.getenv("CPTOP")
INTERACTIVE <- as.logical(Sys.getenv("CPINT"))
CORES <- as.numeric(Sys.getenv("CPCORES"))
GPU <- Sys.getenv("CPGPU")
SCRIPTS <- paste0(TOP, "/analysis")
PLOTS <- paste0(TOP, "/analysis")
MODELS <- paste0(TOP, "/analysis")

Sys.setlocale(locale = "en_US.UTF-8")
library(tidyverse)
library(brms)
library(marginaleffects)
library(foreach)
library(patchwork)

options(mc.cores = CORES)

source(paste0(SCRIPTS, "/pathnames.R"))
source(paste0(SCRIPTS, "/aggregation.R"))
source(paste0(SCRIPTS, "/cleanup.R"))
source(paste0(SCRIPTS, "/identification.R"))
source(paste0(SCRIPTS, "/discrimination.R"))
source(paste0(SCRIPTS, "/plotting.R"))
source(paste0(SCRIPTS, "/regression.R"))

discr_by_contrast_pam_overlap <- left_join(
  discriminability_by_contrast,
  pam_overlap,
  by = c(
    "Listener Group",
    "Phone Language (Long)",
    "Phone Language (Code)",
    "Phone Contrast (Language)"
  )
)

certaccuracy_by_skld_plot <- ggplot(discr_by_contrast_pam_overlap,
                                    aes(x = Overlap, y = `Accuracy and Certainty`)) +
  geom_point(stroke = 0.8, shape = 21) +
  facet_grid(~ `Listener Group`, scales = "free_x") +
  cp_theme() +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.margin = margin(t = 0, b = 0),
    legend.spacing.y = unit(0, "in")
  )  +
  scale_size_area() +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 3))

certaccuracy_by_skld_pam_plot <- ggplot(
  discr_by_contrast_pam_overlap,
  aes(
    x = Overlap,
    y = `Accuracy and Certainty`,
    fill = `Same Top Choice`,
    size = `Maximum Categorization Threshold`
  )
) +
  geom_point(stroke = 0.8, shape = 21) +
  facet_grid(~ `Listener Group`, scales = "free_x") +
  cp_theme() +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.margin = margin(t = 0, b = 0),
    legend.spacing.y = unit(0, "in")
  )  +
  scale_fill_manual(values = c(No = "#dddddd88", Yes = "#33333388")) +
  scale_size(range = c(0.1, 12)) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 3))

overlap_pam_stats <- discr_by_contrast_pam_overlap %>%
  summarize(
    `Cor overlap` = cor(`Accuracy and Certainty`, Overlap),
    `Median overlap` = median(Overlap),
    `Cor haskins` = cor(`Accuracy and Certainty`, Haskins),
    .by = c("Listener Group")
  )

overlap_pam_tc_sc_stats <- discr_by_contrast_pam_overlap %>%
  summarize(
    `Median threshold` = median(`Maximum Categorization Threshold`),
    `Q3 threshold` = quantile(`Maximum Categorization Threshold`, 0.75),
    .by = c("Same Top Choice", "Listener Group")
  )

discr_pam_overlap <- left_join(
  discrimination,
  pam_overlap,
  by = c(
    "Listener Group",
    "Phone Language (Long)",
    "Phone Language (Code)",
    "Phone Contrast (Language)"
  )
)

goodness_difference_mean <- mean(discr_pam_overlap$`Goodness Difference`)
goodness_difference_sd <- sd(discr_pam_overlap$`Goodness Difference`)

model_specs <- list(
  ordinal_null = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Listener.Group*Trial.Number +
                    (1|Participant) + (1 + Listener.Group|filename)"
    ),
    subset = TRUE,
    dvmode = "ordered"
  ),
  ordinal_overlap = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Overlap*Listener.Group +
                    Listener.Group*Trial.Number +
                    (1 + Overlap|Participant) + (1 + Listener.Group|filename)"
    ),
    subset = TRUE,
    dvmode = "ordered"
  ),
  ordinal_haskins = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Haskins*Listener.Group +
                    Listener.Group*Trial.Number +
                    (1 + Haskins|Participant) + (1 + Listener.Group|filename)"
    ),
    subset = TRUE,
    dvmode = "ordered"
  ),
  sigmoid_1c_null = list(
    formula = brmsformula(
      "Accuracy.and.Certainty ~
                         Listener.Group +
                         Listener.Group*Trial.Number +
                         (1|Participant) + (1|filename)"
    ),
    subset = discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode = "binarized"
  ),
  sigmoid_1c_gd_overlap = list(
    formula = brmsformula(
      "Accuracy.and.Certainty ~
                         Overlap*Goodness.Difference*Listener.Group +
                         Listener.Group*Trial.Number +
                         (1|Participant) + (1|filename)"
    ),
    subset = discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode = "binarized"
  ),
  sigmoid_1c_mct_gd_overlap = list(
    formula = brmsformula(
      "Accuracy.and.Certainty ~
                         Overlap*Maximum.Categorization.Threshold*Goodness.Difference*Listener.Group -
                         Overlap:Maximum.Categorization.Threshold:Goodness.Difference:Listener.Group +
                         Listener.Group*Trial.Number +
                         (1|Participant) + (1|filename)"
    ),
    subset = discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode = "binarized"
  ),
  sigmoid_2c_null = list(
    formula = brmsformula(
      "Accuracy.and.Certainty ~
                         Listener.Group +
                         Listener.Group*Trial.Number +
                         (1|Participant) + (1|filename)"
    ),
    subset = discr_pam_overlap$`Same Top Choice` == "No",
    dvmode = "binarized"
  ),
  sigmoid_2c_mct_overlap = list(
    formula = brmsformula(
      "Accuracy.and.Certainty ~
                         Overlap*Maximum.Categorization.Threshold*Listener.Group +
                         Listener.Group*Trial.Number +
                         (1|Participant) + (1|filename)"
    ),
    subset = discr_pam_overlap$`Same Top Choice` == "No",
    dvmode = "binarized"
  ),
  sigmoid_2c_overlap = list(
    formula = brmsformula(
      "Accuracy.and.Certainty ~
                         Overlap*Listener.Group +
                         Listener.Group*Trial.Number +
                         (1|Participant) + (1|filename)"
    ),
    subset = discr_pam_overlap$`Same Top Choice` == "No",
    dvmode = "binarized"
  )
)

models <- foreach(
  m = names(model_specs),
  .final = function(x)
    setNames(x, names(model_specs))
) %do% {
  run_brms_model(model_specs[[m]][["formula"]],
                 discr_pam_overlap[model_specs[[m]][["subset"]], ],
                 get_filename(m),
                 GPU,
                 model_specs[[m]][["dvmode"]],
                 goodness_difference_mean,
                 goodness_difference_sd)
}

models <- foreach(
  m = names(model_specs),
  .final = function(x)
    setNames(x, names(model_specs))
) %do% {
  add_criterion(models[[m]], "loo", file = get_filename(m))
}


ll_by_contrast <- repeated_average(
  mutate(
    discrimination,
    `Log Lik. Overlap minus Haskins` =
      log_lik(models$ordinal_overlap, ndraws = 100) %>% colMeans() -
      log_lik(models$ordinal_haskins, ndraws = 100) %>% colMeans()
  ),
  c(
    "filename",
    "Context",
    "Phone Contrast Asymmetrical (Language)",
    "Phone Contrast (Language)"
  ),
  c(
    "Listener Group",
    "Phone Language (Code)",
    "Phone Language (Long)"
  ),
  c("Log Lik. Overlap minus Haskins")
) %>% left_join(
  pam_overlap,
  by = c(
    "Listener Group",
    "Phone Language (Long)",
    "Phone Language (Code)",
    "Phone Contrast (Language)"
  )
) %>% mutate(`Positive difference` = ifelse(`Log Lik. Overlap minus Haskins` >= 0, "Yes", "No"))

overlap_vs_haskins_plot <- ggplot(
  ll_by_contrast,
  aes(
    y = Haskins,
    x = Overlap,
    size = `Log Lik. Overlap minus Haskins`,
    fill = `Positive difference`
  )
) +
  geom_point(pch = 21, stroke = 0.8) +
  facet_grid(~ `Listener Group`) +
  cp_theme() +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.margin = margin(t = 0, b = 0),
    legend.spacing.y = unit(0, "in")
  )  +
  scale_size_area() +
  scale_fill_manual(values = c(Yes = "black", No = "white"))

mct_plot <- foreach(m_info = list(
  c("sigmoid_2c_mct_overlap", "Different Top Category"),
  c("sigmoid_1c_mct_gd_overlap", "Same Top Category")
)) %do% {
  m_name <- m_info[1]
  m_title <- m_info[2]
  m <- models[[m_name]]
  dg <- list(
    sigmoid_2c_mct_overlap = datagrid(
      model = models$sigmoid_2c_mct_overlap,
      Maximum.Categorization.Threshold = seq(-2, 5, 1),
      Listener.Group = c(-0.5, 0.5),
      Overlap = mean(models$sigmoid_2c_mct_overlap$data$Overlap),
      Participant = NA,
      filename = NA
    ),
    sigmoid_1c_mct_gd_overlap = datagrid(
      model = models$sigmoid_1c_mct_gd_overlap,
      Maximum.Categorization.Threshold = seq(-2, 5, 1),
      Listener.Group = c(-0.5, 0.5),
      Overlap = mean(models$sigmoid_1c_mct_gd_overlap$data$Overlap),
      Goodness.Difference = mean(models$sigmoid_1c_mct_gd_overlap$data$Goodness.Difference),
      Participant = NA,
      filename = NA
    )
  )[[m_name]]
  predictions(m,
              newdata = dg,
              type = "response") %>%
    rename(
      `Maximum Categorization Threshold` = Maximum.Categorization.Threshold,
      `Listener Group` = Listener.Group,
      `Predicted Accuracy and Certainty` = estimate
    ) %>%
    mutate(
      `Listener Group` = ifelse(`Listener Group` == -0.5, "English", "French"),
      `Maximum Categorization Threshold` =
        `Maximum Categorization Threshold` * 0.1 + 0.5,
      `Predicted Accuracy and Certainty` =
        (`Predicted Accuracy and Certainty` - 0.5) * 6,
      conf.low = (conf.low - 0.5) * 6,
      conf.high = (conf.high - 0.5) * 6
    ) %>%
    ggplot(
      aes(
        x = `Maximum Categorization Threshold`,
        y = `Predicted Accuracy and Certainty`,
        ymin = conf.low,
        ymax = conf.high
      )
    ) +
    geom_line() +
    geom_ribbon(alpha = 0.2) +
    facet_grid(`Listener Group` ~ .) +
    ggtitle(m_title) +
    coord_cartesian(ylim = c(-3, 3)) +
    cp_theme() +
    theme(plot.title = element_text(hjust = 0.5))
} %>%
  wrap_plots(nrow = 1) + plot_layout(axis_titles = "collect")

mct_overlap_gd_plot <- foreach(pred=list(c("Overlap"), 
                                         c("Goodness Difference")
                                         )) %do% {
  dg <- list(
    Overlap = datagrid(
      model = models$sigmoid_1c_mct_gd_overlap,
      Maximum.Categorization.Threshold = c(-1.5, 0, 1.5, 3),
      Listener.Group = c(-0.5, 0.5),
      Overlap = seq(0.75, 1, 0.01),
      Goodness.Difference = mean(
        models$sigmoid_1c_mct_gd_overlap$data$Goodness.Difference
      ),
      Participant = NA,
      filename = NA
    ),
    `Goodness Difference` = datagrid(
      model = models$sigmoid_1c_mct_gd_overlap,
      Maximum.Categorization.Threshold = c(-1.5, 0, 1.5, 3),
      Listener.Group = c(-0.5, 0.5),
      Overlap = mean(models$sigmoid_1c_mct_gd_overlap$data$Overlap),
      Goodness.Difference = seq(-2, 3, 0.1),
      Participant = NA,
      filename = NA
    )
  )[[pred]]
  predictions(
    models$sigmoid_1c_mct_gd_overlap,
    newdata = dg,
    type = "response"
  ) %>%
    rename(
      `Maximum Categorization Threshold` = Maximum.Categorization.Threshold,
      `Listener Group` = Listener.Group,
      `Goodness Difference` = Goodness.Difference,
      `Predicted Accuracy and Certainty` = estimate
    ) %>%
    mutate(
      `Listener Group` = ifelse(`Listener Group` == -0.5, "English", "French"),
      `Maximum Categorization Threshold` =
        factor(`Maximum Categorization Threshold` * 0.1 + 0.5),
      `Predicted Accuracy and Certainty` = (`Predicted Accuracy and Certainty` -
                                              0.5) * 6,
      `Goodness Difference` = `Goodness Difference` * goodness_difference_sd +
        goodness_difference_mean,
      conf.low = (conf.low - 0.5) * 6,
      conf.high = (conf.high - 0.5) * 6
    ) %>%
    ggplot(
      aes(
        x = .data[[pred]],
        y = `Predicted Accuracy and Certainty`,
        linetype = `Maximum Categorization Threshold`,
        ymin = conf.low,
        ymax = conf.high
      )
    ) +
    geom_line() +
    geom_ribbon(alpha = 0.2) +
    facet_grid(`Listener Group` ~ .) +
    scale_linetype_manual(values = c(
      `0.8` = 1,
      `0.65` = 2,
      `0.5` = 4,
      `0.35` = 3
    )) +
    cp_theme() +
    coord_cartesian(ylim = c(-3, 3))
} %>%
  wrap_plots(nrow = 1) + plot_layout(guides = "collect") &
  theme(legend.position="bottom")


print(
  cor.test(
    discr_by_contrast_pam_overlap$`Goodness Difference`,
    1 - discr_by_contrast_pam_overlap$`Overlap`,
    method = "spearman",
    exact = FALSE
  )
)

if (INTERACTIVE) {
  print(certaccuracy_by_skld_plot)
  print(certaccuracy_by_skld_pam_plot)
  print(overlap_vs_haskins_plot)
  print(mct_plot)
  print(mct_overlap_gd_plot)
  View(overlap_pam_stats)
  View(overlap_pam_tc_sc_stats)
} else {
  ggsave(
    paste0(PLOTS, "/certaccuracy_by_skld_pam_plot_600.png"),
    plot = certaccuracy_by_skld_pam_plot,
    width = 6.52,
    height = 4.5,
    units = "in",
    dpi = 600
  )
  ggsave(
    paste0(PLOTS, "/overlap_vs_haskins_plot_600.png"),
    plot = overlap_vs_haskins_plot,
    width = 6.52,
    height = 4.5,
    units = "in",
    dpi = 600
  )
  ggsave(
    paste0(PLOTS, "/mct_plot_600.png"),
    plot = mct_plot,
    width = 6.52,
    height = 3,
    units = "in",
    dpi = 600
  )
  ggsave(
    paste0(PLOTS, "/mct_overlap_gd_plot_600.png"),
    plot = mct_overlap_gd_plot,
    width = 6.52,
    height = 3,
    units = "in",
    dpi = 600
  )
  print(overlap_pam_stats)
  print(overlap_pam_tc_sc_stats)
}
print(models)
print(loo(models[["ordinal_null"]], models[["ordinal_overlap"]]))
print(loo(models[["ordinal_null"]], models[["ordinal_haskins"]]))
print(loo(models[["ordinal_overlap"]], models[["ordinal_haskins"]]))
print(loo(models[["sigmoid_1c_mct_gd_overlap"]], models[["sigmoid_1c_gd_overlap"]]))
print(loo(models[["sigmoid_2c_mct_overlap"]], models[["sigmoid_2c_overlap"]]))
print(loo(models[["sigmoid_1c_mct_gd_overlap"]], models[["sigmoid_1c_null"]]))
