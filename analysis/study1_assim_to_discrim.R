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
library(marginaleffects)
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
    `Goodness Difference` = scale(abs(`Top Goodness:Phone 1` -
                                  `Top Goodness:Phone 2`)),
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
      ~ 1-0.5*(haskins_score(.x, .y) + haskins_score(.y, .x))
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

clean_predictions <- function(predictions, d, dvmode, factorvars,
                              rescalevars) {
  for (var in unique(c(factorvars, rescalevars,
                       "Listener Group"))) {
    mnvar <- make.names(var)
    predictions[[var]] <- predictions[[mnvar]]
  }
  if (dvmode == "binarized") {
    predictions <- mutate(
      predictions,
      estimate=(estimate - 0.5)*6,
      conf.low=(conf.low - 0.5)*6,
      conf.high=(conf.high - 0.5)*6
    )
  } else if (dvmode == "collapse_ordinal") {
    predictions
  }
  for (var in factorvars) {
    predictions[[var]] <- as.numeric(as.character(predictions[[var]]))
  }
  predictions <- mutate(
    predictions,  
    `Listener Group`=ifelse(`Listener Group` == -0.5, "English", "French")
  )
  for (var in rescalevars) {
    s <- attr(scale(d[[var]]), "scaled:scale")
    m <- attr(scale(d[[var]]), "scaled:center")
    predictions[[var]] <- predictions[[var]]*s + m
  }
  for (var in factorvars) {
    predictions[[var]] <- factor(format(predictions[[var]], digits=2))
    #predictions[[var]] <- cut(round(predictions[[var]], digits=2), 3)
  }
  return(predictions)
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
       `Maximum Categorization Threshold`=`Maximum Categorization Threshold`-0.5,
       Haskins=2*Haskins))
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
                    Overlap*Listener.Group +
                    (1 + Overlap|Participant) + (1 + Listener.Group|filename)"),
    subset=TRUE,
    dvmode="ordered"
  ),
  ordinal_haskins=list(
    formula=formula("Accuracy.and.Certainty ~
                    Haskins*Listener.Group +
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
                         Listener.Group*Maximum.Categorization.Threshold +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),
  sigmoid_1c_mct_overlap=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Overlap*Maximum.Categorization.Threshold*Listener.Group +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),  
  sigmoid_1c_mct_gd=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group*Maximum.Categorization.Threshold*Goodness.Difference +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),  
  sigmoid_1c_mct_gd_overlap=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Overlap*Maximum.Categorization.Threshold*Goodness.Difference*Listener.Group -
                         Overlap:Maximum.Categorization.Threshold:Goodness.Difference:Listener.Group +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "Yes",
    dvmode="binarized"
  ),
  sigmoid_1c_overlap=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Listener.Group*Overlap +
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
                         Maximum.Categorization.Threshold*Listener.Group +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "No",
    dvmode="binarized"
  ),
  sigmoid_2c_mct_overlap=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Overlap*Maximum.Categorization.Threshold*Listener.Group +
                         (1|Participant) + (1|filename)"),
    subset=discr_pam_overlap$`Same Top Choice` == "No",
    dvmode="binarized"
  ),
  sigmoid_2c_overlap=list(
    formula=brmsformula("Accuracy.and.Certainty ~
                         Overlap*Listener.Group +
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

loo_overlap <- loo(models[["ordinal_null"]], models[["ordinal_overlap"]])
loo_haskins <- loo(models[["ordinal_null"]], models[["ordinal_haskins"]])
loo_haskins_overlap <- loo(models[["ordinal_overlap"]], models[["ordinal_haskins"]])


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


plot_pam_1c <- ggplot(
  clean_predictions(
    plot_predictions(
      models$sigmoid_1c_mct_gd_overlap,
      condition=c("Goodness.Difference", 
                  "Maximum.Categorization.Threshold",
                  "Listener.Group"),
      draw=FALSE),
    discr_pam_overlap[discr_pam_overlap$`Same Top Choice` == "Yes",],
    "binarized",  "Maximum Categorization Threshold",
    c("Goodness Difference", "Maximum Categorization Threshold")) %>%
    rename(`Estimated Accuracy and Certainty`=estimate),
  aes(x=`Goodness Difference`,
      y=`Estimated Accuracy and Certainty`,
      linetype=`Maximum Categorization Threshold`)) +
  geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1) +
  geom_line() + facet_grid(~ `Listener Group`) +
  scale_linetype_manual(
    values=c(`0.77`=1, `0.68`=5, `0.60`=2, `0.46`=4, `0.34`=3)) +
  cp_theme()

ggplot(
  clean_predictions(
    plot_predictions(
      models$sigmoid_1c_mct_overlap,
      condition=c("Overlap", 
                  "Maximum.Categorization.Threshold",
                  "Listener.Group"),
      draw=FALSE),
    discr_pam_overlap[discr_pam_overlap$`Same Top Choice` == "Yes",],
    "binarized",  "Maximum Categorization Threshold",
    c("Maximum Categorization Threshold", "Overlap")) %>%
    rename(`Estimated Accuracy and Certainty`=estimate),
  aes(x=`Overlap`,
      y=`Estimated Accuracy and Certainty`,
      linetype=`Maximum Categorization Threshold`)) +
  geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1) +
  geom_line() + facet_grid(~ `Listener Group`) +
  scale_linetype_manual(
    values=c(`0.62`=1, `0.61`=5, `0.60`=2, `0.58`=4, `0.57`=3)) +
  cp_theme()


ggplot(
  clean_predictions(
    plot_predictions(
      models$sigmoid_1c_mct_gd_overlap,
      condition=c("Overlap", 
                  "Maximum.Categorization.Threshold",
                  "Listener.Group"),
      draw=FALSE),
    discr_pam_overlap[discr_pam_overlap$`Same Top Choice` == "Yes",],
    "binarized",  "Maximum Categorization Threshold",
    c("Maximum Categorization Threshold", "Overlap")) %>%
    rename(`Estimated Accuracy and Certainty`=estimate),
  aes(x=`Overlap`,
      y=`Estimated Accuracy and Certainty`,
      linetype=`Maximum Categorization Threshold`)) +
  geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1) +
  geom_line() + facet_grid(~ `Listener Group`) +
  scale_linetype_manual(
    values=c(`0.62`=1, `0.61`=5, `0.60`=2, `0.58`=4, `0.57`=3)) +
  cp_theme()

plot_pam_overlap_1c <- ggplot(
  clean_predictions(
    plot_predictions(
      models$sigmoid_1c_mct_gd_overlap,
      condition=c("Overlap", 
                  "Maximum.Categorization.Threshold",
                  "Listener.Group"),
      draw=FALSE),
    discr_pam_overlap[discr_pam_overlap$`Same Top Choice` == "Yes",],
    "binarized",  "Maximum Categorization Threshold",
    c("Maximum Categorization Threshold", "Overlap")) %>%
    rename(`Estimated Accuracy and Certainty`=estimate),
  aes(x=`Overlap`,
      y=`Estimated Accuracy and Certainty`,
      linetype=`Maximum Categorization Threshold`)) +
  geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1) +
  geom_line() + facet_grid(~ `Listener Group`) +
  scale_linetype_manual(
    values=c(`0.77`=1, `0.68`=5, `0.60`=2, `0.46`=4, `0.34`=3)) +
  cp_theme()

plot_pam_overlap_2c <- ggplot(
  clean_predictions(
    plot_predictions(
      models$sigmoid_2c_mct_overlap,
      condition=c("Overlap", 
                  "Maximum.Categorization.Threshold",
                  "Listener.Group"),
      draw=FALSE),
    discr_pam_overlap[discr_pam_overlap$`Same Top Choice` == "No",],
    "binarized",  "Maximum Categorization Threshold",
    c("Maximum Categorization Threshold", "Overlap")) %>%
    rename(`Estimated Accuracy and Certainty`=estimate),
  aes(x=`Overlap`,
      y=`Estimated Accuracy and Certainty`,
      linetype=`Maximum Categorization Threshold`)) +
  geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1) +
  geom_line() + facet_grid(~ `Listener Group`) +
  scale_linetype_manual(
    values=c(`0.98`=1, `0.62`=5, `0.49`=2, `0.39`=4, `0.25`=3)) +
  cp_theme()



ll_by_contrast <- repeated_average(
  mutate(
    discrimination,
    `Log Lik. Overlap minus Haskins`=
      log_lik(models$ordinal_overlap, ndraws=100) %>% colMeans() -
      log_lik(models$ordinal_haskins, ndraws=100) %>% colMeans()
    ),
  c(
    "filename",
    "Context",
    "Phone Contrast Asymmetrical (Language)",
    "Phone Contrast (Language)"
  ),
  c("Listener Group", "Phone Language (Code)", "Phone Language (Long)"),
  c("Log Lik. Overlap minus Haskins")
) %>% left_join(
  pam_overlap,
  by = c(
    "Listener Group",
    "Phone Language (Long)",
    "Phone Language (Code)",
    "Phone Contrast (Language)"
  )
) %>% mutate(
  `Positive difference`=ifelse(`Log Lik. Overlap minus Haskins` >= 0,
                               "Yes", "No")
)

overlap_vs_haskins_plot <- ggplot(
  ll_by_contrast,
  aes(
    y =Haskins,
    x=Overlap,
    size=`Log Lik. Overlap minus Haskins`,
    fill=`Positive difference`
  )
) +
  geom_point(pch=21, stroke=0.8) +
  facet_grid( ~ `Listener Group`) +
  cp_theme() +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.margin = margin(t = 0, b = 0),
    legend.spacing.y = unit(0, "in")
  )  +
  scale_size_area() +
  scale_fill_manual(values=c(Yes="black", No="white"))


print(cor.test(discr_by_contrast_pam_overlap$`Goodness Difference`,
               1-discr_by_contrast_pam_overlap$`Overlap`,
               method="spearman", exact=FALSE))

if (INTERACTIVE) {
  print(certaccuracy_by_skld_plot)
  print(certaccuracy_by_skld_pam_plot)
  print(overlap_vs_haskins_plot)
  print(plot_pam_1c)
  print(plot_pam_overlap_1c)
  print(plot_pam_overlap_2c)
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
    paste0(PLOTS, "/pam_1c_plot_600.png"),
    plot = plot_pam_1c,
    width = 6.52,
    height = 4.5,
    units = "in",
    dpi = 600
  )
   
  ggsave(
    paste0(PLOTS, "/pam_overlap_1c_plot.png"),
    plot = plot_pam_overlap_1c,
    width = 6.52,
    height = 4.5,
    units = "in",
    dpi = 600
  )
  
  ggsave(
    paste0(PLOTS, "/pam_overlap_2c_plot.png"),
    plot = plot_pam_overlap_2c,
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
print(loo_haskins)
print(loo_haskins_overlap)
print(loo_1c)
print(loo_2c)

print(cor.test(discr_by_contrast_pam_overlap$`Maximum Categorization Threshold`,
               diff_pred_ho, method="spearman", exact=FALSE))
print(cor.test(discr_by_contrast_pam_overlap$`Maximum Categorization Threshold`,
               diff_pred_ho*wrongness_ho, method="spearman", exact=FALSE))  
