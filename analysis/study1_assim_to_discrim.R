TOP <- Sys.getenv("CPTOP")
INTERACTIVE <- as.logical(Sys.getenv("CPINT"))
CORES <- as.numeric(Sys.getenv("CPCORES"))
GPU <- Sys.getenv("CPGPU")
SCRIPTS <- paste0(TOP, "/analysis")
PLOTS <- paste0(TOP, "/analysis")
MODELS <- paste0(TOP, "/analysis")

Sys.setlocale(locale="en_US.UTF-8")
library(tidyverse)
library(brms)
library(foreach)

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
     `Maximum Categorization Threshold` = map2_dbl(`Top Percentage:Phone 1`,
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

certaccuracy_by_skld_plot <- ggplot(
  discr_by_contrast_pam_overlap,
  aes(
    x = Overlap,
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
  )  +
  scale_size(range = c(0.1, 12)) +
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
    `Median threshold` = median(`Maximum Categorization Threshold`),
    `Q3 threshold` = quantile(`Maximum Categorization Threshold`, 0.75),
    .by = c("Same Top Choice", "Listener Group")
  )
    
#    ,
#    `Cor with threshold` = cor(`Maximum Categorization Threshold`,
#                               `Accuracy and Certainty`, method = "spearman"),
#    `GLM Threshold-only` = list(glm(
#      `Accuracy and Certainty`/3 ~ rank(`Maximum Categorization Threshold`),
#      family=gaussian(link="logit")
#    )),
#   `GLM Threshold + Overlap` = list(glm(
#      `Accuracy and Certainty`/3 ~
#        rank(`Maximum Categorization Threshold`) + `Overlap`,
#        family=gaussian(link="logit")
#    )),
#    `Cor with GD` = cor(`Goodness Difference`,
#      `Accuracy and Certainty`,
#      method = "spearman"
#    ),
#    `GLM GD-only` = list(glm(
#      `Accuracy and Certainty`/3 ~ rank(`Goodness Difference`),
#      family=gaussian(link="logit")
#    )),
#    `GLM GD + Overlap` = list(glm(
#      `Accuracy and Certainty`/3 ~ rank(`Goodness Difference`) + Overlap, 
#      family=gaussian(link="logit")
#    )),
#    .by = c("Same Top Choice", "Listener Group")
#  ) %>%
#  mutate(`GLM Threshold-only >0 pval`=map_dbl(`GLM Threshold-only`,
#                                          ~ pt(coef(summary(.x))[2,3],
#                                               .x$df.residual, lower=FALSE)),
#         `GLM Threshold-only <0 pval`=map_dbl(`GLM Threshold-only`,
#                                          ~ pt(coef(summary(.x))[2,3],
#                                               .x$df.residual, lower=TRUE)),
#         `GLM Threshold + Overlap >0 pval`=map_dbl(`GLM Threshold + Overlap`,
#                                          ~ pt(coef(summary(.x))[2,3],
#                                               .x$df.residual, lower=FALSE)),
#         `GLM Threshold + Overlap <0 pval`=map_dbl(`GLM Threshold + Overlap`,
#                                          ~ pt(coef(summary(.x))[2,3],
#                                               .x$df.residual, lower=TRUE)),
#         `GLM Overlap + Threshold <0 pval`=map_dbl(`GLM Threshold + Overlap`,
#                                          ~ pt(coef(summary(.x))[3,3],
#                                               .x$df.residual, lower=TRUE)),
#         `GLM GD-only >0 pval`=map_dbl(`GLM GD-only`,
#                                          ~ pt(coef(summary(.x))[2,3],
#                                               .x$df.residual, lower=FALSE)),
#         `GLM GD-only <0 pval`=map_dbl(`GLM GD-only`,
#                                          ~ pt(coef(summary(.x))[2,3],
#                                               .x$df.residual, lower=TRUE)),
#         `GLM GD + Overlap >0 pval`=map_dbl(`GLM GD + Overlap`,
#                                          ~ pt(coef(summary(.x))[2,3],
#                                               .x$df.residual, lower=FALSE)),
#         `GLM GD + Overlap <0 pval`=map_dbl(`GLM GD + Overlap`,
#                                          ~ pt(coef(summary(.x))[2,3],
#                                               .x$df.residual, lower=TRUE))
#         )

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

#get_filename <- function(listener, predictor) {
#  return(paste0(MODELS, "/model_", listener, "_", predictor))
#}

get_filename <- function(model_name) {
  return(paste0(MODELS, "/model_", model_name))
}

run_brms_model <- function(f, d, filename, gpuid, dvmode) {
  if (dvmode == "ordered") {
    d <- mutate(d, `Accuracy and Certainty`=factor(`Accuracy and Certainty`,
                                                   ordered=TRUE))
  } else if (dvmode == "binarized") {
    d <- mutate(d, `Accuracy and Certainty`=`Accuracy and Certainty`/6 + 0.5)    
  }
  d <- makenamesize(mutate(d,
       `Listener Group`=ifelse(`Listener Group` == "English", -1/2, 1/2)))
  if (gpuid != "") {
    m <- brm(f, file=filename, data=d,
             save_pars = save_pars(all = TRUE),
	           backend="cmdstanr",
             opencl=eval(parse(text=gpuid)),
	           stan_model_args = list(stanc_options = list("O1")))
  } else {
    m <- brm(f, file=filename, data=d,
             save_pars = save_pars(all = TRUE),
	           backend="cmdstanr",
	           stan_model_args = list(stanc_options = list("O1")))    
  }
  return(m)
}

model_specs <- list(
  ordinal_null=list(
    formula=formula("Accuracy.and.Certainty ~
                    Listener.Group + 
                    (1|Participant) + (1 + Listener.Group|filename)",
                    family="cumulative"),
    subset=TRUE,
    dvmode="ordered"
  ),
  ordinal_overlap=list(
    formula=formula("Accuracy.and.Certainty ~
                    Overlap + Listener.Group + Overlap:Listener.Group +
                    (1 + Overlap|Participant) + (1 + Listener.Group|filename)",
                    family="cumulative"),
    subset=TRUE,
    dvmode="ordered"
  ),
  ordinal_haskins=list(
    formula=formula("Accuracy.and.Certainty ~
                    Haskins + Listener.Group + Haskins:Listener.Group +
                    (1 + Haskins|Participant) + (1 + Listener.Group|filename)",
                    family="cumulative"),
    subset=TRUE,
    dvmode="ordered"
  ),
  sigmoid_1c_null=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group + 
                         (1|Participant) + (1|filename)",
                        family=gaussian(link="logit")),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),
  sigmoid_2c_null=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group + 
                         (1|Participant) + (1 + Listener.Group|filename)",
                        family=gaussian(link="logit")),
    subset=discr_pam_overlap$`Same Top Choice` == "No",
    dvmode="binarized"
  ),
  sigmoid_1c_mct=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group + Maximum.Categorization.Threshold +
                         Listener.Group:Maximum.Categorization.Threshold +
                         (1|Participant) + (1|filename)",
                        family=gaussian(link="logit")),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),
  sigmoid_1c_gd=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group + Goodness.Difference +
                         Listener.Group:Goodness.Difference +
                         (1|Participant) + (1|filename)",
                        family=gaussian(link="logit")),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),  
  sigmoid_1c_mct_gd=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group +
                         Maximum.Categorization.Threshold + Goodness.Difference +
                         Maximum.Categorization.Threshold:Goodness.Difference +
                         Listener.Group:Goodness.Difference +
                         Maximum.Categorization.Threshold:Listener.Group +
                         Listener.Group:Maximum.Categorization.Threshold:Goodness.Difference +
                         (1|Participant) + (1|filename)",
                        family=gaussian(link="logit")),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  )
)


models <- foreach(m = names(model_specs),
                  .final = function(x) setNames(x, names(model_specs))) %do% {
  model <- run_brms_model(model_specs[[m]][["formula"]],
                 discr_pam_overlap[model_specs[[m]][["subset"]],],
                 get_filename(m),
                 GPU,
                 model_specs[[m]][["dvmode"]])
}

#regression_model_meta <- tibble(
#  `Listener Group`=c(rep("French", 2), rep("English", 2)),
#  Predictor=rep(c("Overlap", "Haskins"), 2),
#  Formula=map(Predictor, ~ formula(paste0(
#    "Accuracy.and.Certainty ~ ", .x, " + (1+", .x, "|Participant) + (1|filename)"))),
#  Filename=get_filename(`Listener Group`, Predictor)
#)

#f <- regression_model_meta$Formula[[model_to_run]]
#l <- regression_model_meta$`Listener Group`[[model_to_run]]
#fn <- regression_model_meta$Filename[[model_to_run]]
#run_brms_model(f, filter(discr_pam_overlap, `Listener Group` == l), fn)
#stop()

#regression_models <- regression_model_meta %>%
#  mutate(Model=pmap(list(Formula, `Listener Group`, Filename),
#                    \(f,l,fn) run_brms_model(
#                      f, filter(discr_pam_overlap, `Listener Group` == l), fn
#                    )))

#model_comparison <- pivot_wider(
#  regression_models, id_cols=`Listener Group`, names_from=Predictor,
#  values_from=Model) %>%
#  mutate(Loo=map2(Overlap, Haskins, loo_compare),
#         ELPD_Overlap=map_dbl(Loo, ~ .x[1,3]),
#         ELPD_Haskins=map_dbl(Loo, ~ .x[2,3]),
#         ELPD_Diff=map_dbl(Loo, ~ .x[2,1]),
#         ELPD_SEDiff=map_dbl(Loo, ~.x[2,2])) %>%
#  select(-Overlap, -Haskins)

haskins_overlap_relation <- lm(Haskins ~ Overlap - 1,
                               data=discr_by_contrast_pam_overlap)
haskins_lm <- lm(`Accuracy and Certainty` ~ Haskins,
                 data=discr_by_contrast_pam_overlap)
overlap_lm <- lm(`Accuracy and Certainty` ~ Overlap,
                 data=discr_by_contrast_pam_overlap)
diff_pred_ho <- predict(haskins_lm) - predict(overlap_lm)
wrongness_ho <- abs(resid(haskins_lm)) - abs(resid(overlap_lm))



print(cor.test(discr_by_contrast_pam_overlap$`Goodness Difference`,
               1-discr_by_contrast_pam_overlap$`Overlap`,
               method="spearman", exact=FALSE))

if (INTERACTIVE) {
  print(certaccuracy_by_skld_plot)
  print(certaccuracy_by_skld_pam_plot)
  View(overlap_pam_stats)
  View(overlap_pam_tc_sc_stats)
#  View(model_comparison)
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
#  print(model_comparison)
}
print(cor.test(discr_by_contrast_pam_overlap$`Maximum Categorization Threshold`,
               diff_pred_ho, method="spearman", exact=FALSE))
print(cor.test(discr_by_contrast_pam_overlap$`Maximum Categorization Threshold`,
               diff_pred_ho*wrongness_ho, method="spearman", exact=FALSE))  
print(models)