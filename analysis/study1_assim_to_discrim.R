TOP <- Sys.getenv("CPTOP")
INTERACTIVE <- as.logical(Sys.getenv("CPINT"))
CORES <- as.numeric(Sys.getenv("CPCORES"))
SCRIPTS <- paste0(TOP, "/analysis")
PLOTS <- paste0(TOP, "/analysis")
MODELS <- paste0(TOP, "/analysis")

library(tidyverse)
library(brms)
options(mc.cores=CORES)

source(paste0(SCRIPTS, "/pathnames.R"))
source(paste0(SCRIPTS, "/aggregation.R"))
source(paste0(SCRIPTS, "/cleanup.R"))
source(paste0(SCRIPTS, "/identification.R"))
source(paste0(SCRIPTS, "/discrimination.R"))
source(paste0(SCRIPTS, "/plotting.R"))

makenamesize <- function(d) {
  names(d) <- make.names(colnames(d))
  return(d)
}
 
smooth <- function(x, eps) {
  new_max <- 1 - eps * sum(near(x, 0))
  return(ifelse(near(x, 0), eps, x * new_max))
}

max_skld <- function(x, eps) {
  n <- length(x)
  x1 <- c(rep(eps, n-1), 1-eps*(n-1))
  x2 <- c(1-eps*(n-1), rep(eps, n-1))
  k1 <- sum(x1 * log(x1 / x2))
  k2 <- sum(x2 * log(x2 / x1))
  return((k1+k2)/2.)
}

haskins_score <- function(ass_tgt, ass_oth, with_p50=TRUE) {
  tgt <- unlist(c(ass_tgt))
  oth <- unlist(c(ass_oth))
  p <- sum((tgt^2)*(1-oth))
  p50 <- 0
  if (with_p50) 
    p50 <- sum(tgt*(1-tgt)*(1-oth)) + sum((tgt^2)*oth)
  return(as.numeric(p + 0.5*p50))
}


assimilation_to_contrast <- function(phones, assimilation, phone_var) {
  assimilation_p1 <- assimilation %>% rename(`Phone 1` = {{phone_var}}) %>%
    rename_with( ~ paste0(.x, ":Phone 1"), .cols = !`Phone 1`)
  assimilation_p2 <- assimilation %>% rename(`Phone 2` = {{phone_var}}) %>%
    rename_with( ~ paste0(.x, ":Phone 2"), .cols = !`Phone 2`)
  contrasts <- crossing(A = phones, B = phones) %>% filter(A != B)  %>%
    transmute(
      `Phone 1` = get_phone1(A, B),
      `Phone 2` = get_phone2(A, B),
      `Phone Contrast` = contrast_label(`Phone 1`, `Phone 2`)
    ) %>%
    unique()
  contrasts_with_ass <- contrasts %>%
    left_join(assimilation_p1, by = "Phone 1") %>%
    left_join(assimilation_p2, by = "Phone 2")
  return(contrasts_with_ass)
}

assimilation_to_contrast_grouped <- function(assimilation,
                                             phone_var,
                                             value_vars,
                                             grouping_vars) {
  assimilation %>%
    reframe(assimilation_to_contrast(.data[[phone_var]], pick(all_of(
      c(value_vars, phone_var)
    )), phone_var),
    .by = all_of(grouping_vars))
}


skld_score <- function(ass_tgt, ass_oth, eps) {
  k <- max_skld(ass_tgt, eps)
  t <- smooth(ass_tgt, eps)
  o <- smooth(ass_oth, eps)
  kld_to <- sum(t * log(t / o))
  kld_ot <- sum(o * log(o / t))
  return(1-(kld_to + kld_ot)/(2.*k))
}

pam_overlap <- assimilation_vectors %>%
  pivot_wider(
    id_cols = c("Phone", "Phone Language (Long)", "Phone Language (Code)"),
    names_from = c("Listener Group", "Response"),
    values_from = c("Proportion of Responses", "Goodness")
  ) %>%
  nest(
    `Assimilation:English` = starts_with("Proportion of Responses_English_"),
    `Assimilation:French` = starts_with("Proportion of Responses_French_"),
    `Goodness:English` = starts_with("Goodness_English_"),
    `Goodness:French` = starts_with("Goodness_French_")
  ) %>%
  pivot_longer(
    starts_with("Assimilation") | starts_with("Goodness"),
    names_pattern = "(.*):(.*)",
    names_to = c(".value", "Listener Group")
  ) %>%
  mutate(
    `Top Percentage` = map_dbl(Assimilation, ~ .x[[max.col(.x)]]),
    `Top Goodness` = map2_dbl(Assimilation, Goodness, ~ .y[[max.col(.x)]]),
    `Top Choice` = map_chr(Assimilation, ~ str_split_i(colnames(.x)[max.col(.x)], "_", 3))
  ) %>%
  assimilation_to_contrast_grouped(
    "Phone",
    c("Assimilation", "Top Goodness", "Top Choice", "Top Percentage"),
    c(
      "Listener Group",
      "Phone Language (Long)",
      "Phone Language (Code)"
    )
  ) %>%
  mutate(`Phone Contrast (Language)` = paste0(`Phone Contrast`, " (", `Phone Language (Code)`, ")")) %>%
  mutate(
    `Goodness Difference` = abs(`Top Goodness:Phone 1` -
                                  `Top Goodness:Phone 2`),
    Overlap = map2_dbl(
      `Assimilation:Phone 1`,
      `Assimilation:Phone 2`,
      ~ skld_score(unlist(.x), unlist(.y), 0.001)
    ),
    Dot = map2_dbl(
      `Assimilation:Phone 1`,
      `Assimilation:Phone 2`,
      ~ sum(.x * .y)
    ),
    Haskins=map2_dbl(
      `Assimilation:Phone 1`,
      `Assimilation:Phone 2`,
      ~ 1- 0.5*(haskins_score(.x, .y) + haskins_score(.y, .x))
    ),    
     `Categorization Threshold` = map2_dbl(`Top Percentage:Phone 1`,
                                           `Top Percentage:Phone 2`, min),
    `Same Top Choice` = map2_chr(`Top Choice:Phone 1`, `Top Choice:Phone 2`,
                                 ~ c("No", "Yes")[(.x == .y) + 1])
  )

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

certaccuracy_by_skld_pam_plot <- ggplot(
  discr_by_contrast_pam_overlap,
  aes(
    x = Dot,
    y = `Accuracy and Certainty`,
    fill = `Same Top Choice`,
    size = `Categorization Threshold`
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
    `Median threshold` = median(`Categorization Threshold`),
    `Q3 threshold` = quantile(`Categorization Threshold`, 0.75),
    `Cor with threshold` = cor(`Categorization Threshold`,
                               `Accuracy and Certainty`, method = "spearman"),
    `GLM Threshold-only` = list(glm(
      `Accuracy and Certainty`/3 ~ rank(`Categorization Threshold`),
      family=gaussian(link="logit")
    )),
   `GLM Threshold + Overlap` = list(glm(
      `Accuracy and Certainty`/3 ~
        rank(`Categorization Threshold`) + `Overlap`,
        family=gaussian(link="logit")
    )),
    `Cor with GD` = cor(`Goodness Difference`,
      `Accuracy and Certainty`,
      method = "spearman"
    ),
    `GLM GD-only` = list(glm(
      `Accuracy and Certainty`/3 ~ rank(`Goodness Difference`),
      family=gaussian(link="logit")
    )),
    `GLM GD + Overlap` = list(glm(
      `Accuracy and Certainty`/3 ~ rank(`Goodness Difference`) + Overlap, 
      family=gaussian(link="logit")
    )),
    .by = c("Same Top Choice", "Listener Group")
  ) %>%
  mutate(`GLM Threshold-only >0 pval`=map_dbl(`GLM Threshold-only`,
                                          ~ pt(coef(summary(.x))[2,3],
                                               .x$df.residual, lower=FALSE)),
         `GLM Threshold-only <0 pval`=map_dbl(`GLM Threshold-only`,
                                          ~ pt(coef(summary(.x))[2,3],
                                               .x$df.residual, lower=TRUE)),
         `GLM Threshold + Overlap >0 pval`=map_dbl(`GLM Threshold + Overlap`,
                                          ~ pt(coef(summary(.x))[2,3],
                                               .x$df.residual, lower=FALSE)),
         `GLM Threshold + Overlap <0 pval`=map_dbl(`GLM Threshold + Overlap`,
                                          ~ pt(coef(summary(.x))[2,3],
                                               .x$df.residual, lower=TRUE)),
         `GLM Overlap + Threshold <0 pval`=map_dbl(`GLM Threshold + Overlap`,
                                          ~ pt(coef(summary(.x))[3,3],
                                               .x$df.residual, lower=TRUE)),
         `GLM GD-only >0 pval`=map_dbl(`GLM GD-only`,
                                          ~ pt(coef(summary(.x))[2,3],
                                               .x$df.residual, lower=FALSE)),
         `GLM GD-only <0 pval`=map_dbl(`GLM GD-only`,
                                          ~ pt(coef(summary(.x))[2,3],
                                               .x$df.residual, lower=TRUE)),
         `GLM GD + Overlap >0 pval`=map_dbl(`GLM GD + Overlap`,
                                          ~ pt(coef(summary(.x))[2,3],
                                               .x$df.residual, lower=FALSE)),
         `GLM GD + Overlap <0 pval`=map_dbl(`GLM GD + Overlap`,
                                          ~ pt(coef(summary(.x))[2,3],
                                               .x$df.residual, lower=TRUE))
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

stop()

regression_models <- tibble(
  `Listener Group`=c(rep("French", 2), rep("English", 2)),
  Predictor=rep(c("Overlap", "Haskins"), 2),
  Formula=map(Predictor, ~ formula(paste0(
    "Accuracy.and.Certainty + 4 ~ ", .x, " + (1+", .x, "|Participant) + (1|",
    .x, "|filename)")))
)

regression_formulas <- list(
  French=list(Overlap=formula(Accuracy.and.Certainty + 4 ~ Overlap +
                                 (1+Overlap|Participant) +
                                 (1+Overlap|filename)),
              Haskins=formula(Accuracy.and.Certainty + 4 ~ Haskins +
                                 (1+Haskins|Participant) +
                                 (1+Haskins|filename))),
          list(Overlap=formula(Accuracy.and.Certainty + 4 ~ Overlap +
                                 (1+Overlap|Participant) +
                                 (1+Overlap|filename)),
               Haskins=formula(Accuracy.and.Certainty + 4 ~ Haskins +
                                 (1+Haskins|Participant) +
                                 (1+Haskins|filename))))
regression_models <- list(French=list(Overlap=c(), Haskins=c()),
                          English=list(Overlap=c(), Haskins=c()))
regression_loo <- list(French=list(Overlap=c(), Haskins=c()),
                          English=list(Overlap=c(), Haskins=c()))

for (language in names(regression_formulas)) {
  formulas_lg <- regression_formulas[[language]]
  for (predictor in formulas_lg) {
    formula_lg_pr <- formulas_lg[[predictor]]
    d <- makenamesize(filter(discr_pam_overlap, `Listener Group` == language))
  }
}

model_overlap_french <- brm(Accuracy.and.Certainty + 4 ~ Overlap +
                              (1+Overlap|Participant) + (1+Overlap|filename),
                            data=filter(makenamesize(discr_pam_overlap),
                                        Listener.Group == "French"),
                            family="cumulative",
                            file="model_overlap_french",
                            save_model="model_overlap_french.stan")
model_haskins_french <- brm(Accuracy.and.Certainty + 4 ~ Haskins +
                              (1+Haskins|Participant) + (1+Haskins|filename),
                            data=filter(makenamesize(discr_pam_overlap),
                                        Listener.Group == "French"),
                            family="cumulative", cores=16,
                            file="model_haskins_french")
model_overlap_english <- brm(Accuracy.and.Certainty + 4 ~ Overlap +
                              (1+Overlap|Participant) + (1+Overlap|filename),
                            data=filter(makenamesize(discr_pam_overlap),
                                        Listener.Group == "English"),
                            family="cumulative", cores=16,
                            file="model_overlap_english")
model_haskins_english <- brm(Accuracy.and.Certainty + 4 ~ Haskins +
                              (1+Haskins|Participant) + (1+Haskins|filename),
                            data=filter(makenamesize(discr_pam_overlap),
                                        Listener.Group == "English"),
                            family="cumulative", cores=16,
                            file="model_haskins_english")

if (INTERACTIVE) {
  print(certaccuracy_by_skld_pam_plot)
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
  print(overlap_pam_stats)
  print(overlap_pam_tc_sc_stats)
}
