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

get_filename <- function(model_name) {
  return(paste0(MODELS, "/model_", model_name))
}

run_brms_model <- function(f, d, filename, gpuid, dvmode) {
  if (dvmode == "ordered") {
    d <- mutate(d, `Accuracy and Certainty`=factor(`Accuracy and Certainty`,
                                                   ordered=TRUE))
    family <- "cumulative"
  } else if (dvmode == "binarized") {
    d <- mutate(d, `Accuracy and Certainty`=`Accuracy and Certainty`/6 + 0.5)    
    family <- gaussian(link="logit")
  }
  d <- makenamesize(mutate(d,
       `Listener Group`=ifelse(`Listener Group` == "English", -1/2, 1/2),
       Overlap=scale(Overlap),
       `Maximum Categorization Threshold`=scale(`Maximum Categorization Threshold`),
       `Goodness Difference`=scale(`Goodness Difference`)))
  if (gpuid != "") {
    m <- brm(f, file=filename, data=d,
             family=family,
             save_pars = save_pars(all = TRUE),
	           backend="cmdstanr",
             opencl=eval(parse(text=gpuid)),
	           stan_model_args = list(stanc_options = list("O1")))
  } else {
    m <- brm(f, file=filename, data=d,
             family=family,
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
                    (1|Participant) + (1 + Listener.Group|filename)"),
    subset=TRUE,
    dvmode="ordered"
  ),
  ordinal_overlap=list(
    formula=formula("Accuracy.and.Certainty ~
                    Overlap + Listener.Group + Overlap:Listener.Group +
                    (1 + Overlap|Participant) + (1 + Listener.Group|filename)"),
    subset=TRUE,
    dvmode="ordered"
  ),
  ordinal_haskins=list(
    formula=formula("Accuracy.and.Certainty ~
                    Haskins + Listener.Group + Haskins:Listener.Group +
                    (1 + Haskins|Participant) + (1 + Listener.Group|filename)"),
    subset=TRUE,
    dvmode="ordered"
  ),
  sigmoid_1c_null=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group + 
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),
  sigmoid_1c_mct=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group + Maximum.Categorization.Threshold +
                         Listener.Group:Maximum.Categorization.Threshold +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),
  sigmoid_1c_mct_overlap=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group + Maximum.Categorization.Threshold +
                         Listener.Group:Maximum.Categorization.Threshold +
                         Overlap +
                         Overlap:Maximum.Categorization.Threshold +
                         Listener.Group:Overlap +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),
  sigmoid_1c_gd_overlap=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group + Maximum.Categorization.Threshold +
                         Listener.Group:Maximum.Categorization.Threshold +
                         Goodness.Difference +
                         Goodness.Difference:Maximum.Categorization.Threshold +
                         Listener.Group:Overlap +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),
  sigmoid_1c_gd=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group + Goodness.Difference +
                         Listener.Group:Goodness.Difference +
                         (1|Participant) + (1|filename)"),
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
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),  
  sigmoid_1c_mct_gd_overlap=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group +
                         Maximum.Categorization.Threshold + Goodness.Difference +
                         Maximum.Categorization.Threshold:Goodness.Difference +
                         Listener.Group:Goodness.Difference +
                         Overlap + Listener.Group:Overlap +
                         Maximum.Categorization.Threshold:Listener.Group +
                         Listener.Group:Maximum.Categorization.Threshold:Goodness.Difference +
                         Overlap:Maximum.Categorization.Threshold +
                         Overlap:Goodness.Difference +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),
  sigmoid_1c_overlap=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group + Overlap +
                         Listener.Group:Overlap +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),
  sigmoid_2c_null=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group + 
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "No",
    dvmode="binarized"
  ),
  sigmoid_2c_mct=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group +
                         Maximum.Categorization.Threshold + 
                         Maximum.Categorization.Threshold:Listener.Group +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "No",
    dvmode="binarized"
  ),
  sigmoid_2c_mct_overlap=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group +
                         Maximum.Categorization.Threshold + 
                         Maximum.Categorization.Threshold:Listener.Group +
                         Overlap + 
                         Overlap:Maximum.Categorization.Threshold +
                         Overlap:Listener.Group +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "No",
    dvmode="binarized"
  ),
  sigmoid_2c_overlap=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group +
                         Overlap + 
                         Overlap:Listener.Group +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "No",
    dvmode="binarized"
  )
)


models <- foreach(m = names(model_specs),
                  .final = function(x) setNames(x, names(model_specs))) %do% {
  run_brms_model(model_specs[[m]][["formula"]],
                 discr_pam_overlap[model_specs[[m]][["subset"]],],
                 get_filename(m),
                 GPU,
                 model_specs[[m]][["dvmode"]])
}

models <- foreach(m = names(model_specs),
                  .final = function(x) setNames(x, names(model_specs))) %do% {
  add_criterion(models[[m]], "loo", file=get_filename(m))
}

loo_overlap <- loo(models[["ordinal_null"]], models[["ordinal_overlap"]],
                   models[["ordinal_haskins"]])

loo_1c <- loo(models[["sigmoid_1c_null"]],
              models[["sigmoid_1c_mct"]],
              models[["sigmoid_1c_gd"]],
              models[["sigmoid_1c_mct_gd"]],
              models[["sigmoid_1c_mct_overlap"]],
              models[["sigmoid_1c_mct_gd_overlap"]],
              models[["sigmoid_1c_overlap"]])

loo_2c <- loo(models[["sigmoid_2c_null"]],
              models[["sigmoid_2c_mct"]],
              models[["sigmoid_2c_mct_overlap"]],
              models[["sigmoid_2c_overlap"]])


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
print(models)
print(loo_overlap)
print(loo_1c)
print(loo_2c)

print(cor.test(discr_by_contrast_pam_overlap$`Maximum Categorization Threshold`,
               diff_pred_ho, method="spearman", exact=FALSE))
print(cor.test(discr_by_contrast_pam_overlap$`Maximum Categorization Threshold`,
               diff_pred_ho*wrongness_ho, method="spearman", exact=FALSE))  
