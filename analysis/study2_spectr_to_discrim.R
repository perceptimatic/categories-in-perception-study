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
source(paste0(SCRIPTS, "/delta.R"))

distances <- readr::read_csv("discrimination_exp/triplet_data_study2.csv",
                             col_types=cols(TGT_first = col_logical(),
                                            TGT_first_code = col_number(),
                                            language_stimuli_code = col_number(),
                                            .default = col_guess())) %>%
  calculate_all_deltas() %>%
  clean_discrimination_items() %>%
  mutate(`Δ DTW Mel Filterbank`=`fb_dtw_cosine_Δ`)
  
distances_by_contrast <- repeated_average(
  distances,
  c(
    "filename",
    "Context",
    "Phone Contrast Asymmetrical (Language)",
    "Phone Contrast (Language)"
  ),
  c("Phone Language (Code)", "Phone Language (Long)"),
  names(distances)[grepl("_Δ$", names(distances))]
) %>% rename(`Δ DTW Mel Filterbank`=`fb_dtw_cosine_Δ`)

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

discr_distances <- left_join(
  discrimination,
  distances
) %>% left_join(
  rename(discr_by_contrast_distances,
         `Δ DTW Mel Filterbank (Phone Contrast)`=`Δ DTW Mel Filterbank`) %>%
    select(`Δ DTW Mel Filterbank (Phone Contrast)`, `Δ Overlap`, Overlap,
           `Phone Language (Long)`, `Phone Language (Code)`,
           `Phone Contrast (Language)`, `Listener Group`),
  by = c(
    "Phone Language (Long)",
    "Phone Language (Code)",
    "Phone Contrast (Language)",
    "Listener Group")) 



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
                    Δ.Overlap +
                    Listener.Group*Trial.Number +
                    Δ.Overlap:Listener.Group +
                    Overlap:Δ.DTW.Mel.Filterbank +
                    Overlap:Δ.DTW.Mel.Filterbank:Listener.Group +      
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
                    Δ.Overlap +
                    Listener.Group*Trial.Number +
                    Δ.Overlap:Listener.Group +
                    Overlap:Δ.DTW.Mel.Filterbank..Phone.Contrast. +
                    Overlap:Δ.DTW.Mel.Filterbank..Phone.Contrast.:Listener.Group +
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


ndraws <- 4000
doverlap_dfb_plot <- foreach(pred=list("Δ Overlap", 
                                        "Δ DTW Mel Filterbank")) %do% {
  dg <- list(
    `Δ Overlap` = datagrid(
      model=models$ordinal_doverlap_dfbavg_nodfbavg,
      Participant=NA, filename=NA,
      `Δ.Overlap`=seq(0, 1, 0.01),
      `Δ.DTW.Mel.Filterbank..Phone.Contrast.`=seq(0, 0.05, 0.025)/0.05,
      Listener.Group=c(-.5, .5)),
    `Δ DTW Mel Filterbank` = datagrid(
      model=models$ordinal_doverlap_dfbavg_nodfbavg,
      Participant=NA, filename=NA,
      `Δ.Overlap`=seq(0, 1, 0.5),
      `Δ.DTW.Mel.Filterbank..Phone.Contrast.`=seq(-0.01, 0.06, 0.001)/0.05,
      Listener.Group=c(-.5, .5))
  )[[pred]]  %>%
    mutate(Overlap=1-`Δ.Overlap`)
  p_matrix <- rstantools::posterior_epred(models$ordinal_doverlap_dfbavg_nodfbavg, dg) %>%
    (function(x)
      matrix(
        rowSums(
          x*aperm(array(c(-3, -2, -1, 1, 2, 3),
                        dim=c(6, nrow(dg), ndraws)),
                  c(3,2,1)),
          dims=2),
        ndraws, nrow(dg)))
  ggplot() + cp_theme() +
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
       `Listener Group` = ifelse(`Listener Group` == -0.5, "English", "French"),
       `Δ DTW Mel Filterbank`=`Δ DTW Mel Filterbank`*0.05
     ) %>% (
       list(
         `Δ DTW Mel Filterbank`=
           function(d) mutate(d, `Δ Overlap`=as.character(`Δ Overlap`)),
         `Δ Overlap`=
           function(d) mutate(d, `Δ DTW Mel Filterbank`=as.character(`Δ DTW Mel Filterbank`))
       )[[pred]]
     ) %>% (
       function(d) {
        list(
           geom_point(data=discr_by_contrast_distances,
             aes(x=.data[[pred]],
                 y=`Accuracy and Certainty`,
                 alpha=.data[[c(
                   `Δ DTW Mel Filterbank`="Δ Overlap",
                   `Δ Overlap`="Δ DTW Mel Filterbank"
                 )[pred]]]), pch=20),
           geom_line(data=d, aes(x=.data[[pred]],
                                 y=`Predicted Accuracy and Certainty`,
                                 linetype=.data[[c(
                                   `Δ DTW Mel Filterbank`="Δ Overlap",
                                   `Δ Overlap`="Δ DTW Mel Filterbank"
                                   )[pred]]]),
                     linewidth=0.6),          
           geom_ribbon(data=d, aes(x=.data[[pred]],
                                   ymin=conf.low, ymax=conf.high,
                                   group=.data[[c(
                                     `Δ DTW Mel Filterbank`="Δ Overlap",
                                     `Δ Overlap`="Δ DTW Mel Filterbank"
                                   )[pred]]]), alpha=0.1),           
           scale_alpha_continuous(limits=list(
             `Δ DTW Mel Filterbank`=c(0,1),
             `Δ Overlap`=c(0,0.05)
             )[[pred]],
                                  range=c(0.1,1),
                                  label = as.character,
                                  oob=scales::squish,
                                  n.breaks=3),
           scale_linetype_manual(values =
            list(`Δ Overlap`=c(
                 `0` = 5,
                 `0.025` = 4,
                 `0.05` = 1),
                 `Δ DTW Mel Filterbank`=c(
                   `0` = 5,
                   `0.5` = 4,
                   `1` = 1))[[pred]], 
            labels=c("", "", "")
            ),
             facet_grid(`Listener Group` ~ .),
             guides(alpha=guide_legend(order=1,nrow=3),
                    linetype=guide_legend(nrow=3, title=NULL)),
             theme(legend.key.width=unit(1, "cm"),
                   legend.text.position = "left",
                   legend.spacing = unit(0, "cm"))) }
           ) 
         )
} %>% wrap_plots(nrow = 1)
print(doverlap_dfb_plot)

ggsave(
  paste0(PLOTS, "/doverlap_dfb_plot.png"),
  plot = doverlap_dfb_plot,
  width = 6.52,
  height = 4.5,
  units = "in",
  dpi = 600
)

loo_compare(models$ordinal_doverlap_dfb, models$ordinal_doverlap)
loo_compare(models$ordinal_doverlap_dfbavg, models$ordinal_doverlap)
loo_compare(models$ordinal_doverlap_dfbavg, models$ordinal_doverlap_dfb)
loo_compare(models$ordinal_doverlap_dfbavg, models$ordinal_dfbavg)
loo_compare(models$ordinal_doverlap_dfbavg, models$ordinal_doverlap_dfbavg_nodfbavg)
