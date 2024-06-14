TOP <- Sys.getenv("CPTOP")
INTERACTIVE <- as.logical(Sys.getenv("CPINT"))
CORES <- as.numeric(Sys.getenv("CPCORES"))
GPU <- Sys.getenv("CPGPU")
SCRIPTS <- paste0(TOP, "/analysis")
PLOTS <- paste0(TOP, "/analysis")
MODELS <- paste0(TOP, "/analysis")

Sys.setlocale(locale = "en_US.UTF-8")
library(conflicted)
library(tidyverse)
library(brms)
library(marginaleffects)
library(foreach)
library(patchwork)

conflicts_prefer(dplyr::filter, dplyr::select)

options(mc.cores = CORES)

source(paste0(SCRIPTS, "/pathnames.R"))
source(paste0(SCRIPTS, "/aggregation.R"))
source(paste0(SCRIPTS, "/cleanup.R"))
source(paste0(SCRIPTS, "/discrimination.R"))
source(paste0(SCRIPTS, "/identification.R"))
source(paste0(SCRIPTS, "/plotting.R"))
source(paste0(SCRIPTS, "/regression.R"))

calculate_all_deltas <- function(d) {
  for (col_tgt in names(d)[grepl("distance_tgt$", names(d))]) {
    col_oth <- sub("_tgt$", "_oth", col_tgt)
    col_delta <- sub("distance_tgt$", "delta", col_tgt)
    d[[col_delta]] <- (d[[col_oth]] - d[[col_tgt]])
  }
  return(d)
}

distances <- readr::read_csv("discrimination_exp/triplet_data_study2.csv",
                             col_types=cols(TGT_first = col_logical(),
                                            TGT_first_code = col_number(),
                                            language_stimuli_code = col_number(),
                                            .default = col_guess())) %>%
  calculate_all_deltas() %>%
  clean_discrimination_items() %>%
  mutate(`Δ DTW Mel Filterbank`=fb_dtw_cosine_delta)
  
distances_by_contrast <- repeated_average(
  distances,
  c(
    "filename",
    "Context",
    "Phone Contrast Asymmetrical (Language)",
    "Phone Contrast (Language)"
  ),
  c("Phone Language (Code)", "Phone Language (Long)"),
  names(distances)[grepl("_delta$", names(distances))]
) %>% rename(`Δ DTW Mel Filterbank`=fb_dtw_cosine_delta)

discr_by_contrast_distances <- left_join(
  discriminability_by_contrast,
  distances_by_contrast,
  by = c(
    "Phone Language (Long)",
    "Phone Language (Code)",
    "Phone Contrast (Language)"
  )
) %>% left_join(
  pam_overlap,
  by = c(
    "Phone Language (Long)",
    "Phone Language (Code)",
    "Phone Contrast (Language)",
    "Listener Group")) %>%
  mutate(`Δ Overlap`=1-Overlap)


certaccuracy_by_fb_plot <- ggplot(
  discr_by_contrast_distances,
  aes(
    x = `Δ DTW Mel Filterbank`,
    y = `Accuracy and Certainty`
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
  coord_cartesian(ylim = c(0, 3))

overlap_by_fb_plot <- ggplot(
  discr_by_contrast_distances,
  aes(
    x = `Δ DTW Mel Filterbank`,
    y = `Δ Overlap`
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
  coord_cartesian(ylim = c(0, 1))

discr_distances <- left_join(
  discrimination,
  distances
) %>% left_join(
  rename(discr_by_contrast_distances,
         `Δ DTW Mel Filterbank (Phone Contrast)`=`Δ DTW Mel Filterbank`) %>%
    select(`Δ DTW Mel Filterbank (Phone Contrast)`, `Δ Overlap`,
           `Phone Language (Long)`, `Phone Language (Code)`,
           `Phone Contrast (Language)`, `Listener Group`),
  by = c(
    "Phone Language (Long)",
    "Phone Language (Code)",
    "Phone Contrast (Language)",
    "Listener Group")) 


if (INTERACTIVE) {
  print(certaccuracy_by_fb_plot)
  print(overlap_by_fb_plot)
} else {
  ggsave(
    paste0(PLOTS, "/certaccuracy_by_fb_plot_600.png"),
    plot = certaccuracy_by_fb_plot,
    width = 6.52,
    height = 3,
    units = "in",
    dpi = 600
  )  
  ggsave(
    paste0(PLOTS, "/overlap_by_fb_plot_600.png"),
    plot = overlap_by_fb_plot,
    width = 6.52,
    height = 3,
    units = "in",
    dpi = 600
  )    
}

print(with(filter(discr_by_contrast_distances, `Δ DTW Mel Filterbank` <= 0.025),
    cor(`Δ DTW Mel Filterbank`, `Accuracy and Certainty`)))
print(summary(lm(`Accuracy and Certainty` ~ `Δ DTW Mel Filterbank`,
  data=filter(discr_by_contrast_distances, `Δ DTW Mel Filterbank` <= 0.025) %>%
    mutate(`Δ DTW Mel Filterbank`=`Δ DTW Mel Filterbank`/0.05))))
print(with(filter(discr_by_contrast_distances, `Δ DTW Mel Filterbank` > 0.025),
    cor(`Δ DTW Mel Filterbank`, `Accuracy and Certainty`)))
print(summary(lm(`Accuracy and Certainty` ~ `Δ DTW Mel Filterbank`,
         data=filter(discr_by_contrast_distances, `Δ DTW Mel Filterbank` > 0.025) %>%
           mutate(`Δ DTW Mel Filterbank`=`Δ DTW Mel Filterbank`/0.05))))

model_specs <- list(
  ordinal_null = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Listener.Group*Trial.Number +
                    (1|Participant) + (1 + Listener.Group|filename)"
    ),
    dvmode = "ordered"
  ),
  ordinal_doverlap = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Δ.Overlap*Listener.Group + Listener.Group*Trial.Number +
                    (1|Participant) + (1 + Listener.Group|filename)"
    ),
    dvmode = "ordered"
  ),
  ordinal_dfb = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Δ.DTW.Mel.Filterbank*Listener.Group +Listener.Group*Trial.Number +
                    (1|Participant) + (1 + Listener.Group|filename)"
    ),
    dvmode = "ordered"
  ),  
  ordinal_doverlap_dfb = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Δ.Overlap*Δ.DTW.Mel.Filterbank*Listener.Group +Listener.Group*Trial.Number +
                    (1|Participant) +
                    (1 + Listener.Group|filename)"
    ),
    dvmode = "ordered"
  ),    
  ordinal_doverlap_dfb_nodfb = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Δ.Overlap*Δ.DTW.Mel.Filterbank*Listener.Group +
                    Listener.Group*Trial.Number -
                    Δ.DTW.Mel.Filterbank -
                    Δ.DTW.Mel.Filterbank:Listener.Group +
                    (1|Participant) +
                    (1 + Listener.Group|filename)"
    ),
    dvmode = "ordered"
  ),      
  ordinal_dfbavg = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Δ.DTW.Mel.Filterbank..Phone.Contrast.*Listener.Group +Listener.Group*Trial.Number +
                    (1|Participant) +
                    (1 + Listener.Group|filename)"
    ),
    dvmode = "ordered"
  ),    
  ordinal_doverlap_dfbavg = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Δ.Overlap*Δ.DTW.Mel.Filterbank..Phone.Contrast.*Listener.Group +Listener.Group*Trial.Number +
                    (1|Participant) +
                    (1 + Listener.Group|filename)"
    ),
    dvmode = "ordered"
  ),
  ordinal_doverlap_dfbavg_nodfbavg = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Δ.Overlap*Δ.DTW.Mel.Filterbank..Phone.Contrast.*Listener.Group - 
                    Δ.DTW.Mel.Filterbank..Phone.Contrast. -
                    Δ.DTW.Mel.Filterbank..Phone.Contrast.:Listener.Group  +
                    Listener.Group*Trial.Number +
                    (1|Participant) +
                    (1 + Listener.Group|filename)"
    ),
    dvmode = "ordered"
  )  
)

models <- foreach(
  m = names(model_specs),
  .final = function(x)
    setNames(x, names(model_specs))
) %do% {
   run_brms_model(model_specs[[m]][["formula"]],
                 discr_distances,
                 get_filename(m),
                 GPU,
                 "ordered")
}

models <- foreach(
  m = names(model_specs),
  .final = function(x)
    setNames(x, names(model_specs))
) %do% {
  add_criterion(models[[m]], "loo", file = get_filename(m))
}


print(models)


dg <- datagrid(model=models$ordinal_doverlap_dfbavg_nodfbavg,
               Participant=NA, filename=NA,
               `Δ.Overlap`=seq(0, 1, 0.5),
               `Δ.DTW.Mel.Filterbank..Phone.Contrast.`=seq(-0.01, 0.06, 0.001)/0.05,
               Listener.Group=c(-.5, .5))
p_matrix <- rstantools::posterior_epred(models$ordinal_doverlap_dfbavg_nodfbavg, dg) %>%
  (function(x)
    matrix(
      rowSums(
        x*aperm(array(c(-3, -2, -1, 1, 2, 3), dim=c(6, nrow(dg), 4000)), c(3,2,1)),
        dims=2),
      4000, nrow(dg)))
ggplot() + 
  cp_theme() +
(mutate(
  dg,
  `Predicted Accuracy and Certainty` = colMeans(p_matrix),
  conf.low=apply(p_matrix, 2, function(x) quantile(x, 0.025)),
  conf.high=apply(p_matrix, 2, function(x) quantile(x, 0.975))
) %>%
rename(
  `Listener Group` = Listener.Group,
  `Δ DTW Mel Filterbank`=`Δ.DTW.Mel.Filterbank..Phone.Contrast.`,
  `Δ Overlap`=`Δ.Overlap`
) %>%
  mutate(
    `Δ DTW Mel Filterbank`=`Δ DTW Mel Filterbank`*0.05,
    `Δ Overlap`=as.character(`Δ Overlap`),
    `Listener Group` = ifelse(`Listener Group` == -0.5, "English", "French")
  ) %>%
  (function(d) {
  list(
  geom_point(data=discr_by_contrast_distances,
             aes(x=`Δ DTW Mel Filterbank`, y=`Accuracy and Certainty`,
                 alpha=`Δ Overlap`), pch=20),
    geom_line(data=d, aes(x=`Δ DTW Mel Filterbank`, y=`Predicted Accuracy and Certainty`,
                linetype=`Δ Overlap`), linewidth=0.6),
  geom_ribbon(data=d, aes(x=`Δ DTW Mel Filterbank`, ymin=conf.low, ymax=conf.high,
                            group=`Δ Overlap`), alpha=0.1),
  facet_wrap(~ `Listener Group`),
  scale_alpha_continuous(limits=c(0,1), range=c(0.1,1), label = as.character,
                         n.breaks=3),
  scale_linetype_manual(values = c(
        `0` = 5,
        `0.5` = 4,
        `1` = 1)),
  guides(alpha=guide_legend(order=1))
  )
    }))  

dg <- datagrid(model=models$ordinal_doverlap_dfbavg,
               Participant=NA, filename=NA,
               `Δ.Overlap`=seq(0, 1, 0.01),
               `Δ.DTW.Mel.Filterbank..Phone.Contrast.`=seq(0, 0.05, 0.025)/0.05,
               Listener.Group=c(-.5, .5))
p_matrix <- rstantools::posterior_epred(models$ordinal_doverlap_dfbavg, dg) %>%
  (function(x)
    matrix(
      rowSums(
        x*aperm(array(c(-3, -2, -1, 1, 2, 3), dim=c(6, nrow(dg), 4000)), c(3,2,1)),
        dims=2),
      4000, nrow(dg)))
ggplot() + 
  cp_theme() +
  (mutate(
    dg,
    `Predicted Accuracy and Certainty` = colMeans(p_matrix),
    conf.low=apply(p_matrix, 2, function(x) quantile(x, 0.025)),
    conf.high=apply(p_matrix, 2, function(x) quantile(x, 0.975))
  ) %>%
    rename(
      `Listener Group` = Listener.Group,
      `Δ DTW Mel Filterbank`=`Δ.DTW.Mel.Filterbank..Phone.Contrast.`,
      `Δ Overlap`=`Δ.Overlap`
    ) %>%
    mutate(
      `Δ DTW Mel Filterbank`=as.character(`Δ DTW Mel Filterbank`*0.05),
      `Listener Group` = ifelse(`Listener Group` == -0.5, "English", "French")
    ) %>%
    (function(d) {
      list(
        geom_point(data=discr_by_contrast_distances,
                   aes(alpha=`Δ DTW Mel Filterbank`, y=`Accuracy and Certainty`,
                       x=`Δ Overlap`), pch=20),
        geom_line(data=d, aes(linetype=`Δ DTW Mel Filterbank`, y=`Predicted Accuracy and Certainty`,
                              x=`Δ Overlap`), linewidth=0.6),
        geom_ribbon(data=d, aes(group=`Δ DTW Mel Filterbank`, ymin=conf.low, ymax=conf.high,
                                x=`Δ Overlap`), alpha=0.1),
        facet_wrap(~ `Listener Group`),
        scale_alpha_continuous(limits=c(0,0.05), range=c(0.1,1), label = as.character,
                               oob=scales::squish, n.breaks=3),
        scale_linetype_manual(values = c(
          `0` = 5,
          `0.025` = 4,
          `0.05` = 1)),
        guides(alpha=guide_legend(order=1))
        )
    }))  
