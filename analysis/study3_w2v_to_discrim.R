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
library(lme4)

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

distances <- bind_rows(
  bind_rows(
    read_delta("discrimination_exp/triplet_data_w2v_french.csv", layers=TRUE),
    read_delta("discrimination_exp/triplet_data_hubert_french.csv", layers=TRUE),
    read_delta("discrimination_exp/triplet_data_study2.csv") %>%
      filter(Model == "fb_dtw_cosine")
  ) %>% mutate(`Listener Group`="French"),
  bind_rows(
    read_delta("discrimination_exp/triplet_data_w2v_english.csv", layers=TRUE),
    read_delta("discrimination_exp/triplet_data_hubert_english.csv", layers=TRUE),
    read_delta("discrimination_exp/triplet_data_study2.csv") %>%
      filter(Model == "fb_dtw_cosine")
  ) %>% mutate(`Listener Group`="English")
) %>% mutate(Model=ifelse(Model == "fb_dtw_cosine", "DTW Mel Filterbank", Model))


distances_by_contrast <- repeated_average(
  distances,
  c(
    "filename",
    "Context",
    "Phone Contrast Asymmetrical (Language)",
    "Phone Contrast (Language)"
  ),
  c("Listener Group", "Model", "Layer", "Phone Language (Code)", "Phone Language (Long)"),
  "Δ Model"
) 

discr_by_contrast_distances <- left_join(
  discriminability_by_contrast,
  distances_by_contrast,
  by = c(
    "Phone Language (Long)",
    "Phone Language (Code)",
    "Phone Contrast (Language)",
    "Listener Group"
  )
) %>% left_join(
  pam_overlap,
  by = c(
    "Phone Language (Long)",
    "Phone Language (Code)",
    "Phone Contrast (Language)",
    "Listener Group")) %>%
  mutate(`Δ Overlap`=1-Overlap)

certaccuracy_by_delta_plot <- ggplot(
  discr_by_contrast_distances,
  aes(
    x = `Δ Model`,
    y = `Accuracy and Certainty`,
    fill= Model
  )
) +
  geom_point(stroke = 0.8, shape = 21) +
  facet_wrap(~ `Listener Group` + str_pad(Layer, 2), ncol=13, scales = "free_x") +
  scale_fill_manual(values=c(`HuBERT (Transformer)`="black",
                              `wav2vec 2.0 (Transformer)`="white",
                              `DTW Mel Filterbank`="grey")) +
  cp_theme() +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.margin = margin(t = 0, b = 0),
    legend.spacing.y = unit(0, "in")
  )  +
  coord_cartesian(ylim = c(0, 3))

overlap_by_delta_plot <- ggplot(
  discr_by_contrast_distances,
  aes(
    x = `Δ Model`,
    y = `Δ Overlap`,
    fill= Model
  )
) +
  geom_point(stroke = 0.8, shape = 21) +
  facet_wrap(~ `Listener Group` + str_pad(Layer, 2), ncol=13, scales = "free_x") +
  scale_fill_manual(values=c(`HuBERT (Transformer)`="black",
                             `wav2vec 2.0 (Transformer)`="white",
                              `DTW Mel Filterbank`="grey")) +
  cp_theme() +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.margin = margin(t = 0, b = 0),
    legend.spacing.y = unit(0, "in")
  )  +
  coord_cartesian(ylim = c(0, 1))

print(certaccuracy_by_delta_plot)
print(overlap_by_delta_plot)




discr_by_contrast_distances_melf <- filter(discr_by_contrast_distances,
                                           Model == "DTW Mel Filterbank") %>%
  select(`Phone Contrast (Language)`, `Δ Model`, `Δ Overlap`,
         `Listener Group`) %>% 
  rename(`Δ DTW Mel Filterbank`=`Δ Model`)

scorr <- discr_by_contrast_distances %>%
  group_by(Layer, Model, `Listener Group`) %>%
  nest() %>%
  mutate(`Δ Model vs Discriminability`=
           map_dbl(data,
                   ~ with(.x,
                          cor(`Accuracy and Certainty`, `Δ Model`,
                          method="spearman"))),
         `Δ Model vs Δ Overlap`=
           map_dbl(data,
                   ~ with(.x,
                          cor(`Δ Overlap`, `Δ Model`,
                          method="spearman"))),
         `Δ Model vs Δ FB`=
           map_dbl(data,
                   ~ with(left_join(.x, discr_by_contrast_distances_melf,
                                    by=join_by(`Phone Contrast (Language)`,
                                               `Δ Overlap`)),
                          cor(`Δ DTW Mel Filterbank`, `Δ Model`,
                              method="spearman")))
         ) %>%
  select(-data) %>%
  pivot_longer(starts_with("Δ Model vs "),
               values_to="Spearman Correlation",
               names_to="Measure")

correlation_plot <- ggplot(
  scorr %>% mutate(Layer=ifelse(is.na(Layer), 12, Layer)),
  aes(
    x = Layer,
    y = `Spearman Correlation`,
    linetype = Measure,
    shape = Measure,
  )) +
  geom_point() +
  geom_line() +
  facet_grid(`Listener Group` ~ Model) + 
  cp_theme() + 
  scale_x_continuous(breaks = seq(1, 12))
  
if (INTERACTIVE) {
  print(correlation_plot)
}
  
# Conclusion: Focusing on last layer of HuBERT

scale_and_recenter <- function(x) {
  s <- scale(x)
  c(s) + attr(s, "scaled:center")
}

distances_h12 <- filter(distances, Model=="HuBERT (Transformer)", Layer==12) %>%
  rename(`Δ HuBERT`=`Δ Model`)
distances_by_contrast_h12 <- filter(distances_by_contrast,
                                    Model=="HuBERT (Transformer)", Layer==12) %>%
  rename(`Δ HuBERT`=`Δ Model`)
discr_by_contrast_distances_h12 <- filter(discr_by_contrast_distances,
                                          Model=="HuBERT (Transformer)",
                                          Layer==12) %>%
  rename(`Δ HuBERT`=`Δ Model`)

discr_by_contrast_distances_h12 <- filter(discr_by_contrast_distances,
         Model=="HuBERT (Transformer)", Layer==12) %>%
    rename(`Δ HuBERT`=`Δ Model`) %>%
    mutate(`Δ HuBERT (Scaled)`=scale_and_recenter(`Δ HuBERT`))


certaccuracy_by_h12_plot <- ggplot(
  discr_by_contrast_distances_h12,
  aes(
    x = `Δ HuBERT`,
    y = `Accuracy and Certainty`
  )
) +
  geom_point(stroke = 0.8, shape = 21) +
  facet_grid( ~ `Listener Group`, scales = "free_x") +
  cp_theme() +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.margin = margin(t = 0, b = 0),
    legend.spacing.y = unit(0, "in")
  ) + coord_cartesian(ylim = c(0, 3)) 

if (INTERACTIVE) {
 print(certaccuracy_by_h12_plot)
} else {
  ggsave(
    paste0(PLOTS, "/certaccuracy_by_hubert_plot_600.png"),
    plot = certaccuracy_by_h12_plot,
    width = 6.52,
    height = 3,
    units = "in",
    dpi = 600
  )  
}

print(with(discr_by_contrast_distances_h12,
            cor(`Δ Overlap`, `Δ HuBERT`)))
print(with(discr_by_contrast_distances_melf,
            cor(`Δ Overlap`, `Δ DTW Mel Filterbank`)))
print(
  with(
    left_join(
      discr_by_contrast_distances_h12,
      discr_by_contrast_distances_melf,
      by=c("Phone Contrast (Language)", "Listener Group")),
    cor(`Δ DTW Mel Filterbank`, `Δ HuBERT`)))
  

# HuBERT is better correlated with overlap, but still best correlated with MFB


 
discr_distances_h12 <- left_join(
  discrimination,
  distances_h12
) %>% left_join(
  rename(discr_by_contrast_distances_h12,
         `Δ HuBERT (Phone Contrast)`=`Δ HuBERT`) %>%
    select(`Δ HuBERT (Phone Contrast)`, `Δ Overlap`, Overlap,
           `Phone Language (Long)`, `Phone Language (Code)`,
           `Phone Contrast (Language)`, `Listener Group`),
  by = c(
    "Phone Language (Long)",
    "Phone Language (Code)",
    "Phone Contrast (Language)",
    "Listener Group")) 

discr_distances_h12_swapped <- discr_distances_h12 %>%
  select(`Listener Group`, `filename`, `Δ HuBERT`, `Δ HuBERT (Phone Contrast)`) %>%
  rename(`Δ HuBERT Other Language`=`Δ HuBERT`, `Δ HuBERT Other Language (Phone Contrast)`=`Δ HuBERT (Phone Contrast)`) %>%
  mutate(`Listener Group`=ifelse(`Listener Group` == "English", "French", "English")) %>%
  distinct()

discr_distances_h12 <- left_join(discr_distances_h12, discr_distances_h12_swapped) %>%
  mutate(`Δ HuBERT English`=ifelse(`Listener Group`=="English", `Δ HuBERT`, `Δ HuBERT Other Language`),
         `Δ HuBERT French`=ifelse(`Listener Group`=="English", `Δ HuBERT Other Language`, `Δ HuBERT`),
         `Δ HuBERT English (Phone Contrast)`=ifelse(`Listener Group`=="English", `Δ HuBERT (Phone Contrast)`, `Δ HuBERT Other Language (Phone Contrast)`),
         `Δ HuBERT French (Phone Contrast)`=ifelse(`Listener Group`=="English", `Δ HuBERT Other Language (Phone Contrast)`, `Δ HuBERT (Phone Contrast)`))

discr_distances_melf <-  left_join(
  discrimination,
  filter(distances,
         Model == "DTW Mel Filterbank") %>%
    rename(`Δ DTW Mel Filterbank`=`Δ Model`)) %>%
  left_join(
      rename(discr_by_contrast_distances_melf,
             `Δ DTW Mel Filterbank (Phone Contrast)`=`Δ DTW Mel Filterbank`) %>%
        select(`Δ DTW Mel Filterbank (Phone Contrast)`, `Δ Overlap`, 
               `Phone Contrast (Language)`, `Listener Group`),
      by = c("Phone Contrast (Language)", "Listener Group"))


m_null <- lme4::lmer(`Accuracy and Certainty` ~ `Listener Group`*`Trial Number` +
             (1|Participant) + (1 + `Listener Group`|filename),
           data=discr_distances_h12, REML=FALSE,
           control=lmerControl(optimizer="Nelder_Mead"))

m_overlap <- lme4::lmer(`Accuracy and Certainty` ~ `Δ Overlap`*`Listener Group` +
                          `Listener Group`*`Trial Number` +
             (1|Participant) + (1 + `Listener Group`|filename),
           data=discr_distances_h12, REML=FALSE,
           control=lmerControl(optimizer="Nelder_Mead"))

m_hubert <- lme4::lmer(`Accuracy and Certainty` ~ I(`Δ HuBERT`/0.25)*`Listener Group` +
                          `Listener Group`*`Trial Number` +
             (1|Participant) + (1 + `Listener Group`|filename),
           data=discr_distances_h12, REML=FALSE,
           control=lmerControl(optimizer="Nelder_Mead"))


m_hubert_bycon <- lme4::lmer(`Accuracy and Certainty` ~ I(`Δ HuBERT (Phone Contrast)`/0.25)*`Listener Group` +
                         `Listener Group`*`Trial Number` +
                         (1|Participant) + (1 + `Listener Group`|filename),
                       data=discr_distances_h12, REML=FALSE,
                       control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_hubert <- lme4::lmer(`Accuracy and Certainty` ~
                                 `Δ Overlap`*I(`Δ HuBERT`/0.25)*`Listener Group` +
                          `Listener Group`*`Trial Number` +
             (1|Participant) + (1 + `Listener Group`|filename),
           data=discr_distances_h12, REML=FALSE,
           control=lmerControl(optimizer="Nelder_Mead"))

m_melf <- lme4::lmer(`Accuracy and Certainty` ~
                                 I(`Δ DTW Mel Filterbank`/0.05)*`Listener Group` +
                                 `Listener Group`*`Trial Number` +
                                 (1|Participant) + (1 + `Listener Group`|filename),
                               data=discr_distances_melf, REML=FALSE,
           control=lmerControl(optimizer="Nelder_Mead"))

m_melf_bycon <- lme4::lmer(`Accuracy and Certainty` ~
                       I(`Δ DTW Mel Filterbank (Phone Contrast)`/0.05)*`Listener Group` +
                       `Listener Group`*`Trial Number` +
                       (1|Participant) + (1 + `Listener Group`|filename),
                     data=discr_distances_melf, REML=FALSE,
                     control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_melf <- lme4::lmer(`Accuracy and Certainty` ~
                                 `Δ Overlap`*I(`Δ DTW Mel Filterbank`/0.05)*`Listener Group` +
                                 `Listener Group`*`Trial Number` +
                                 (1|Participant) + (1 + `Listener Group`|filename),
                               data=discr_distances_melf, REML=FALSE,
           control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_melf_decomp <- lme4::lmer(`Accuracy and Certainty` ~
                               `Δ Overlap`*(I(`Δ DTW Mel Filterbank (Phone Contrast)`/0.05) + I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05))*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant) + (1 + `Listener Group`|filename),
                             data=discr_distances_melf, REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_hubert_bycon <- lme4::lmer(`Accuracy and Certainty` ~
                                 `Δ Overlap`*I(`Δ HuBERT (Phone Contrast)`/0.25)*`Listener Group` +
                                 `Listener Group`*`Trial Number` +
                                 (1|Participant) + (1 + `Listener Group`|filename),
                               data=discr_distances_h12, REML=FALSE,
                               control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_melf_bycon <- lme4::lmer(`Accuracy and Certainty` ~
                               `Δ Overlap`*I(`Δ DTW Mel Filterbank (Phone Contrast)`/0.05)*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant) + (1 + `Listener Group`|filename),
                             data=discr_distances_melf, REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))

m_hubert_decomp <- lme4::lmer(`Accuracy and Certainty` ~
                                 (I(`Δ HuBERT (Phone Contrast)`/0.25) + I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25))*`Listener Group` +
                                 `Listener Group`*`Trial Number` +
                                 (1|Participant) + (1 + `Listener Group`|filename),
                               data=discr_distances_h12, REML=FALSE,
                               control=lmerControl(optimizer="Nelder_Mead"))


m_huberts_decomp <- lme4::lmer(`Accuracy and Certainty` ~
                                (I(`Δ HuBERT Other Language (Phone Contrast)`/0.25) + I((`Δ HuBERT Other Language` - `Δ HuBERT Other Language (Phone Contrast)`)/0.25))*`Listener Group` +
                                `Listener Group`*`Trial Number` +
                                (1|Participant) + (1 + `Listener Group`|filename),
                              data=discr_distances_h12, REML=FALSE,
                              control=lmerControl(optimizer="Nelder_Mead"))


m_huberte_decomp <- lme4::lmer(`Accuracy and Certainty` ~
                                (I(`Δ HuBERT English (Phone Contrast)`/0.25) + I((`Δ HuBERT English` - `Δ HuBERT English (Phone Contrast)`)/0.25))*`Listener Group` +
                                `Listener Group`*`Trial Number` +
                                (1|Participant) + (1 + `Listener Group`|filename),
                              data=discr_distances_h12, REML=FALSE,
                              control=lmerControl(optimizer="Nelder_Mead"))

m_hubertf_decomp <- lme4::lmer(`Accuracy and Certainty` ~
                                (I(`Δ HuBERT French (Phone Contrast)`/0.25) + I((`Δ HuBERT French` - `Δ HuBERT French (Phone Contrast)`)/0.25))*`Listener Group` +
                                `Listener Group`*`Trial Number` +
                                (1|Participant) + (1 + `Listener Group`|filename),
                              data=discr_distances_h12, REML=FALSE,
                              control=lmerControl(optimizer="Nelder_Mead"))



m_melf_decomp <- lme4::lmer(`Accuracy and Certainty` ~
                               (I(`Δ DTW Mel Filterbank (Phone Contrast)`/0.05) + I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05))*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant) + (1 + `Listener Group`|filename),
                             data=discr_distances_melf, REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_hubert_resid <- lme4::lmer(`Accuracy and Certainty` ~
                                 `Δ Overlap`*I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25)*`Listener Group` +
                                 `Listener Group`*`Trial Number` +
                                 (1|Participant) + (1 + `Listener Group`|filename),
                               data=discr_distances_h12, REML=FALSE,
                               control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_melf_resid <- lme4::lmer(`Accuracy and Certainty` ~
                               `Δ Overlap`*I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05)*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant) + (1 + `Listener Group`|filename),
                             data=discr_distances_melf, REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_hubert_decomp <- lme4::lmer(`Accuracy and Certainty` ~
                                 `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25) + I((`Δ HuBERT (Phone Contrast)`)/0.25))*`Listener Group` +
                                 `Listener Group`*`Trial Number` +
                                 (1|Participant) + (1 + `Listener Group`|filename),
                               data=discr_distances_h12, REML=FALSE,
                               control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_melf_decomp_ro <- lme4::lmer(`Accuracy and Certainty` ~
                               `Δ Overlap`*(I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) - `Δ Overlap`)*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant) + (1 + `Listener Group`|filename),
                             data=discr_distances_melf, REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_hubert_decomp_ro <- lme4::lmer(`Accuracy and Certainty` ~
                                        `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25 - `Δ Overlap`) + I((`Δ HuBERT (Phone Contrast)`)/0.25 - `Δ Overlap`))*`Listener Group` +
                                        `Listener Group`*`Trial Number` +
                                        (1|Participant) + (1 + `Listener Group`|filename),
                                      data=discr_distances_h12, REML=FALSE,
                                      control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_huberts_decomp_ro <- lme4::lmer(`Accuracy and Certainty` ~
                                        `Δ Overlap`*(I((`Δ HuBERT Other Language` - `Δ HuBERT Other Language (Phone Contrast)`)/0.25 - `Δ Overlap`) + I((`Δ HuBERT Other Language (Phone Contrast)`)/0.25 - `Δ Overlap`))*`Listener Group` +
                                        `Listener Group`*`Trial Number` +
                                        (1|Participant) + (1 + `Listener Group`|filename),
                                      data=discr_distances_h12, REML=FALSE,
                                      control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_melf_decomp_ro <- lme4::lmer(`Accuracy and Certainty` ~
                                      `Δ Overlap`*(I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`))*`Listener Group` +
                                      `Listener Group`*`Trial Number` +
                                      (1|Participant) + (1 + `Listener Group`|filename),
                                    data=discr_distances_melf, REML=FALSE,
                                    control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_both <- lme4::lmer(`Accuracy and Certainty` ~
                                      `Δ Overlap`*(I((`Δ HuBERT`)/0.25) + I((`Δ DTW Mel Filterbank`)/0.05))*`Listener Group` +
                                      `Listener Group`*`Trial Number` +
                                      (1|Participant) + (1 + `Listener Group`|filename),
                                    data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                    REML=FALSE,
                                    control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_both_decomp <- lme4::lmer(`Accuracy and Certainty` ~
                               `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25) + I((`Δ HuBERT (Phone Contrast)`)/0.25)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05))*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant) + (1 + `Listener Group`|filename),
                             data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                             REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))
m_overlap_both_decomp_ro <- lme4::lmer(`Accuracy and Certainty` ~
                               `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25 - `Δ Overlap`) + I((`Δ HuBERT (Phone Contrast)`)/0.25 - `Δ Overlap`)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`))*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant) + (1 + `Listener Group`|filename),
                             data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                             REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_boths_decomp <- lme4::lmer(`Accuracy and Certainty` ~
                               `Δ Overlap`*(I((`Δ HuBERT Other Language` - `Δ HuBERT Other Language (Phone Contrast)`)/0.25) + I((`Δ HuBERT Other Language (Phone Contrast)`)/0.25)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05))*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant) + (1 + `Listener Group`|filename),
                             data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                             REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))
m_overlap_boths_decomp_ro <- lme4::lmer(`Accuracy and Certainty` ~
                               `Δ Overlap`*(I((`Δ HuBERT Other Language` - `Δ HuBERT Other Language (Phone Contrast)`)/0.25 - `Δ Overlap`) + I((`Δ HuBERT Other Language (Phone Contrast)` - `Δ Overlap`)/0.25)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`))*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant) + (1 + `Listener Group`|filename),
                             data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                             REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_bothe_decomp <- lme4::lmer(`Accuracy and Certainty` ~
                               `Δ Overlap`*(I((`Δ HuBERT English` - `Δ HuBERT English (Phone Contrast)`)/0.25) + I((`Δ HuBERT English (Phone Contrast)`)/0.25)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05))*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant) + (1 + `Listener Group`|filename),
                             data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                             REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))
m_overlap_bothe_decomp_ro <- lme4::lmer(`Accuracy and Certainty` ~
                               `Δ Overlap`*(I((`Δ HuBERT English` - `Δ HuBERT English (Phone Contrast)`)/0.25 - `Δ Overlap`) + I((`Δ HuBERT English (Phone Contrast)`)/0.25 - `Δ Overlap`)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`))*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant) + (1 + `Listener Group`|filename),
                             data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                             REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_bothf_decomp <- lme4::lmer(`Accuracy and Certainty` ~
                                       `Δ Overlap`*(I((`Δ HuBERT French` - `Δ HuBERT French (Phone Contrast)`)/0.25) + I((`Δ HuBERT French (Phone Contrast)`)/0.25)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05))*`Listener Group` +
                                       `Listener Group`*`Trial Number` +
                                       (1|Participant) + (1 + `Listener Group`|filename),
                                     data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                     REML=FALSE,
                                     control=lmerControl(optimizer="Nelder_Mead"))
m_overlap_bothf_decomp_ro <- lme4::lmer(`Accuracy and Certainty` ~
                                          `Δ Overlap`*(I((`Δ HuBERT French` - `Δ HuBERT French (Phone Contrast)`)/0.25 - `Δ Overlap`) + I((`Δ HuBERT French (Phone Contrast)`)/0.25 - `Δ Overlap`)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`))*`Listener Group` +
                                          `Listener Group`*`Trial Number` +
                                          (1|Participant) + (1 + `Listener Group`|filename),
                                        data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                        REML=FALSE,
                                        control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_both_decomp_ro_nomr <- lme4::lmer(`Accuracy and Certainty` ~
                                         `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25 - `Δ Overlap`) + I((`Δ HuBERT (Phone Contrast)`)/0.25 - `Δ Overlap`) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`))*`Listener Group` +
                                         `Listener Group`*`Trial Number` +
                                         (1|Participant) + (1 + `Listener Group`|filename),
                                       data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                       REML=FALSE,
                                       control=lmerControl(optimizer="Nelder_Mead"))
m_overlap_both_decomp_ro_nomm <- lme4::lmer(`Accuracy and Certainty` ~
                                         `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25 - `Δ Overlap`) + I((`Δ HuBERT (Phone Contrast)`)/0.25 - `Δ Overlap`)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`) )*`Listener Group` +
                                         `Listener Group`*`Trial Number` +
                                         (1|Participant) + (1 + `Listener Group`|filename),
                                       data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                       REML=FALSE,
                                       control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_both_decomp_ro_nohr <- lme4::lmer(`Accuracy and Certainty` ~
                                         `Δ Overlap`*(I((`Δ HuBERT (Phone Contrast)`)/0.25 - `Δ Overlap`)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`))*`Listener Group` +
                                         `Listener Group`*`Trial Number` +
                                         (1|Participant) + (1 + `Listener Group`|filename),
                                       data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                       REML=FALSE,
                                       control=lmerControl(optimizer="Nelder_Mead"))
m_overlap_both_decomp_ro_nohm <- lme4::lmer(`Accuracy and Certainty` ~
                                         `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25 - `Δ Overlap`) + I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`))*`Listener Group` +
                                         `Listener Group`*`Trial Number` +
                                         (1|Participant) + (1 + `Listener Group`|filename),
                                       data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                       REML=FALSE,
                                       control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_both_decomp_ro_nomr_nohm <- lme4::lmer(`Accuracy and Certainty` ~
                                              `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25 - `Δ Overlap`) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05 - `Δ Overlap`))*`Listener Group` +
                                              `Listener Group`*`Trial Number` +
                                              (1|Participant) + (1 + `Listener Group`|filename),
                                            data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                            REML=FALSE,
                                            control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_both_decomp_nohr <- lme4::lmer(`Accuracy and Certainty` ~
                               `Δ Overlap`*(I((`Δ HuBERT (Phone Contrast)`)/0.25)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05))*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant) + (1 + `Listener Group`|filename),
                             data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                             REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_both_decomp_nomr <- lme4::lmer(`Accuracy and Certainty` ~
                                      `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25) + I((`Δ HuBERT (Phone Contrast)`)/0.25)+ I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05))*`Listener Group` +
                                      `Listener Group`*`Trial Number` +
                                      (1|Participant) + (1 + `Listener Group`|filename),
                                    data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                    REML=FALSE,
                                    control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_both_decomp_nomm <- lme4::lmer(`Accuracy and Certainty` ~
                                      `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25) + I((`Δ HuBERT (Phone Contrast)`)/0.25)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05))*`Listener Group` +
                                      `Listener Group`*`Trial Number` +
                                      (1|Participant) + (1 + `Listener Group`|filename),
                                    data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                    REML=FALSE,
                                    control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_both_decomp_nohm <- lme4::lmer(`Accuracy and Certainty` ~
                                      `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25) + I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05))*`Listener Group` +
                                      `Listener Group`*`Trial Number` +
                                      (1|Participant) + (1 + `Listener Group`|filename),
                                    data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                    REML=FALSE,
                                    control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_both_decomp_nohm_nomr <- lme4::lmer(`Accuracy and Certainty` ~
                                           `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25)  + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05))*`Listener Group` +
                                           `Listener Group`*`Trial Number` +
                                           (1|Participant) + (1 + `Listener Group`|filename),
                                         data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                         REML=FALSE,
                                         control=lmerControl(optimizer="Nelder_Mead"))



m_overlap_both_decomp_nomm_nohr <- lme4::lmer(`Accuracy and Certainty` ~
                                      `Δ Overlap`*(I((`Δ HuBERT (Phone Contrast)`)/0.25)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05))*`Listener Group` +
                                      `Listener Group`*`Trial Number` +
                                      (1|Participant) + (1 + `Listener Group`|filename),
                                    data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)),
                                    REML=FALSE,
                                    control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_hubert_resido <- lme4::lmer(`Accuracy and Certainty` ~
                                       `Δ Overlap`*I(`Δ HuBERT`/0.25 - `Δ Overlap`)*`Listener Group` +
                                       `Listener Group`*`Trial Number` +
                                       (1|Participant) + (1 + `Listener Group`|filename),
                                     data=discr_distances_h12, REML=FALSE,
                                     control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_melf_resido <- lme4::lmer(`Accuracy and Certainty` ~
                                     `Δ Overlap`*I(`Δ DTW Mel Filterbank`/0.05 - `Δ Overlap`)*`Listener Group` +
                                     `Listener Group`*`Trial Number` +
                                     (1|Participant) + (1 + `Listener Group`|filename),
                                   data=discr_distances_melf, REML=FALSE,
                                   control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_hubert_residoc <- lme4::lmer(`Accuracy and Certainty` ~
                                        `Δ Overlap`*I(`Δ HuBERT (Phone Contrast)`/0.25 - `Δ Overlap`)*`Listener Group` +
                                        `Listener Group`*`Trial Number` +
                                        (1|Participant) + (1 + `Listener Group`|filename),
                                      data=discr_distances_h12, REML=FALSE,
                                      control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_melf_residoc <- lme4::lmer(`Accuracy and Certainty` ~
                                      `Δ Overlap`*I(`Δ DTW Mel Filterbank (Phone Contrast)`/0.05 - `Δ Overlap`)*`Listener Group` +
                                      `Listener Group`*`Trial Number` +
                                      (1|Participant) + (1 + `Listener Group`|filename),
                                    data=discr_distances_melf, REML=FALSE,
                                    control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_hubert_residonr <- lme4::lmer(`Accuracy and Certainty` ~
                                        `Δ Overlap`*I(`Δ HuBERT`/0.25 - `Δ Overlap`)*`Listener Group` +
                                        `Listener Group`*`Trial Number` +
                                        (1|Participant),
                                      data=discr_distances_h12, REML=FALSE,
                                      control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_melf_residonr <- lme4::lmer(`Accuracy and Certainty` ~
                                      `Δ Overlap`*I(`Δ DTW Mel Filterbank`/0.05 - `Δ Overlap`)*`Listener Group` +
                                      `Listener Group`*`Trial Number` +
                                      (1|Participant),
                                    data=discr_distances_melf, REML=FALSE,
                                    control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_hubert_nor <- lme4::lmer(`Accuracy and Certainty` ~
                                 `Δ Overlap`*I(`Δ HuBERT`/0.25)*`Listener Group` +
                                 `Listener Group`*`Trial Number` +
                                 (1|Participant),
                               data=discr_distances_h12, REML=FALSE,
                               control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_melf_nor <- lme4::lmer(`Accuracy and Certainty` ~
                               `Δ Overlap`*I(`Δ DTW Mel Filterbank`/0.05)*`Listener Group` +
                               `Listener Group`*`Trial Number` +
                               (1|Participant),
                             data=discr_distances_melf, REML=FALSE,
                             control=lmerControl(optimizer="Nelder_Mead"))



m_overlap_hubert_resid_nor <- lme4::lmer(`Accuracy and Certainty` ~
                                       `Δ Overlap`*I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25)*`Listener Group` +
                                       `Listener Group`*`Trial Number` +
                                       (1|Participant),
                                     data=discr_distances_h12, REML=FALSE,
                                     control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_melf_resid_nor <- lme4::lmer(`Accuracy and Certainty` ~
                                     `Δ Overlap`*I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05)*`Listener Group` +
                                     `Listener Group`*`Trial Number` +
                                     (1|Participant),
                                   data=discr_distances_melf, REML=FALSE,
                                   control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_nor <- lme4::lmer(`Accuracy and Certainty` ~ `Δ Overlap`*`Listener Group` +
                          `Listener Group`*`Trial Number` +
                          (1|Participant),
                        data=discr_distances_h12, REML=FALSE,
                        control=lmerControl(optimizer="Nelder_Mead"))



print(AIC(m_null, m_hubert, m_hubert_bycon, m_overlap, m_melf, m_melf_bycon, m_overlap_hubert, m_overlap_melf,
          m_hubert_decomp, m_melf_decomp,
    m_overlap_hubert_bycon, m_overlap_melf_bycon, m_overlap_hubert_resid, m_overlap_melf_resid,
    m_overlap_hubert_decomp, m_overlap_melf_decomp, m_overlap_both_decomp,
    m_huberts_decomp, 
    m_hubertf_decomp, 
    m_huberte_decomp, 
    m_overlap_boths_decomp,
    m_overlap_both,
    m_overlap_both_decomp_ro,
    m_overlap_boths_decomp_ro,
    m_overlap_bothe_decomp,
    m_overlap_bothe_decomp_ro,
    m_overlap_bothf_decomp,
    m_overlap_bothf_decomp_ro,
    m_overlap_huberts_decomp_ro,
    m_overlap_both_decomp_ro_nohr,
    m_overlap_both_decomp_ro_nohm,
    m_overlap_both_decomp_ro_nomr,
    m_overlap_both_decomp_ro_nomr_nohm,
    m_overlap_both_decomp_ro_nomm,
    m_overlap_both_decomp_nohr,
    m_overlap_both_decomp_nomr,
    m_overlap_both_decomp_nohm,
    m_overlap_both_decomp_nomm,
    m_overlap_both_decomp_nohm_nomr,
    m_overlap_both_decomp_nomm_nohr,
    m_overlap_hubert_decomp_ro,
    m_overlap_melf_decomp_ro,
    m_overlap_hubert_resido, m_overlap_melf_resido, m_overlap_hubert_residoc, m_overlap_melf_residoc,
    m_overlap_hubert_residonr, m_overlap_melf_residonr,
    m_overlap_hubert_resid_nor, m_overlap_melf_resid_nor,
    m_overlap_hubert_nor, m_overlap_melf_nor, m_overlap_nor) %>% arrange(AIC))




m_overlap_both_decomp_engon <- lme4::lmer(`Accuracy and Certainty` ~
                                      `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25) + I((`Δ HuBERT (Phone Contrast)`)/0.25)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05)) +
                                      `Trial Number` +
                                      (1|Participant) + (1|filename),
                                      
                                    data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "English"),
                                    REML=FALSE,
                                    control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_mfb_decomp_engon <- lme4::lmer(`Accuracy and Certainty` ~
                                            `Δ Overlap`*(I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05)) +
                                            `Trial Number` +
                                      (1|Participant) + (1|filename),
                                          data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "English"),
                                          REML=FALSE,
                                          control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_hubert_decomp_engon <- lme4::lmer(`Accuracy and Certainty` ~
                                           `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.05) + I((`Δ HuBERT (Phone Contrast)`)/0.05)) +
                                           `Trial Number` +
                                           (1|Participant) + (1|filename),
                                         data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "English"),
                                         REML=FALSE,
                                         control=lmerControl(optimizer="Nelder_Mead"))


m_mfb_decomp_engon <- lme4::lmer(`Accuracy and Certainty` ~
                                           (I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05)) +
                                           `Trial Number` +
                                           (1|Participant) + (1|filename),
                                         data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "English"),
                                         REML=FALSE,
                                         control=lmerControl(optimizer="Nelder_Mead"))

m_hubert_decomp_engon <- lme4::lmer(`Accuracy and Certainty` ~
                                              (I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.05) + I((`Δ HuBERT (Phone Contrast)`)/0.05)) +
                                              `Trial Number` +
                                              (1|Participant) + (1|filename),
                                            data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "English"),
                                            REML=FALSE,
                                            control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_engon <- lme4::lmer(`Accuracy and Certainty` ~
                                           `Δ Overlap` +
                                           `Trial Number` +
                                           (1|Participant) + (1|filename),
                                         data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "English"),
                                         REML=FALSE,
                                         control=lmerControl(optimizer="Nelder_Mead"))


m_null_engon <- lme4::lmer(`Accuracy and Certainty` ~
                                `Trial Number` +
                                      (1|Participant) + (1|filename),
                              data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "English"),
                              REML=FALSE,
                              control=lmerControl(optimizer="Nelder_Mead"))




m_overlap_both_decomp_freon <- lme4::lmer(`Accuracy and Certainty` ~
                                      `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.25) + I((`Δ HuBERT (Phone Contrast)`)/0.25)+ I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05)) +
                                      `Trial Number` +
                                      (1|Participant) + (1|filename),
                                    data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "French"),
                                    REML=FALSE,
                                    control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_mfb_decomp_freon <- lme4::lmer(`Accuracy and Certainty` ~
                                           `Δ Overlap`*(I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05)) +
                                           `Trial Number` +
                                      (1|Participant) + (1|filename),
                                         data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "French"),
                                         REML=FALSE,
                                         control=lmerControl(optimizer="Nelder_Mead"))

m_overlap_hubert_decomp_freon <- lme4::lmer(`Accuracy and Certainty` ~
                                              `Δ Overlap`*(I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.05) + I((`Δ HuBERT (Phone Contrast)`)/0.05)) +
                                              `Trial Number` +
                                      (1|Participant) + (1|filename),
                                            data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "French"),
                                            REML=FALSE,
                                            control=lmerControl(optimizer="Nelder_Mead"))


m_overlap_freon <- lme4::lmer(`Accuracy and Certainty` ~
                                              `Δ Overlap` +
                                              `Trial Number` +
                                      (1|Participant) + (1|filename),
                                            data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "French"),
                                            REML=FALSE,
                                            control=lmerControl(optimizer="Nelder_Mead"))

m_null_freon <- lme4::lmer(`Accuracy and Certainty` ~
                                `Trial Number` +
                                (1|Participant) + (1|filename),
                              data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "French"),
                              REML=FALSE,
                              control=lmerControl(optimizer="Nelder_Mead"))


m_mfb_decomp_freon <- lme4::lmer(`Accuracy and Certainty` ~
                                           (I((`Δ DTW Mel Filterbank` - `Δ DTW Mel Filterbank (Phone Contrast)`)/0.05) + I((`Δ DTW Mel Filterbank (Phone Contrast)`)/0.05)) +
                                           `Trial Number` +
                                           (1|Participant) + (1|filename),
                                         data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "French"),
                                         REML=FALSE,
                                         control=lmerControl(optimizer="Nelder_Mead"))

m_hubert_decomp_freon <- lme4::lmer(`Accuracy and Certainty` ~
                                              (I((`Δ HuBERT` - `Δ HuBERT (Phone Contrast)`)/0.05) + I((`Δ HuBERT (Phone Contrast)`)/0.05)) +
                                              `Trial Number` +
                                              (1|Participant) + (1|filename),
                                            data=left_join(select(discr_distances_melf, -distance_tgt, -distance_oth), select(discr_distances_h12, -distance_tgt, -distance_oth, -Model, -Layer)) %>% filter(`Listener Group` == "French"),
                                            REML=FALSE,
                                            control=lmerControl(optimizer="Nelder_Mead"))

print(arrange(AIC(m_mfb_decomp_engon, m_overlap_mfb_decomp_engon, m_hubert_decomp_engon, m_overlap_hubert_decomp_engon, m_null_engon, m_overlap_engon, m_overlap_both_decomp_engon), AIC))
print(arrange(AIC(m_mfb_decomp_freon, m_overlap_mfb_decomp_freon, m_hubert_decomp_freon, m_overlap_hubert_decomp_freon, m_null_freon, m_overlap_freon, m_overlap_both_decomp_freon), AIC))

# Sum up of temporary models
#   wav2vec 2.0
#   df      AIC
#   m_null                  9 115031.9
#   m_hubert               11 114695.1
#   m_hubert_bycon         11 114516.4
#   m_overlap              11 114203.2
#   m_melf                 11 114848.3
#   m_melf_bycon           11 114665.7
#   m_overlap_hubert       15 114029.0
#   m_overlap_melf         15 114080.2
#   m_overlap_hubert_bycon 15 114134.9
#   m_overlap_melf_bycon   15 114086.6
#   
#   HuBERT
#   df      AIC
#   m_null                  9 115031.9
#   m_hubert               11 114724.2
#   m_hubert_bycon         11 114531.0
#   m_overlap              11 114203.2
#   m_melf                 11 114848.3
#   m_melf_bycon           11 114665.7
#   m_overlap_hubert       15 114095.6
#   m_overlap_melf         15 114080.2
#   m_overlap_hubert_bycon 15 114145.1
#   m_overlap_melf_bycon   15 114086.6

# Temporary models: start here - switch to ordinal
# Also share models between study 2 and study 3 (null, mfb)

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
