TOP <- Sys.getenv("CPTOP")
INTERACTIVE <- as.logical(Sys.getenv("CPINT"))
CORES <- as.numeric(Sys.getenv("CPCORES"))
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
source(paste0(SCRIPTS, "/discrimination.R"))
source(paste0(SCRIPTS, "/identification.R"))
source(paste0(SCRIPTS, "/plotting.R"))
source(paste0(SCRIPTS, "/regression.R"))

discr_by_contrast_overlap <- left_join(
  discriminability_by_contrast,
  overlap,
  by = c(
    "Listener Group",
    "Phone Language (Long)",
    "Phone Language (Code)",
    "Phone Contrast (Language)"
  )
)

model_specs <- list(
  ordinal_null = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Listener.Group +
                    (1|Participant) + (1 + Listener.Group|filename)"
    ),
    subset = TRUE,
    dvmode = "ordered"
  ),
  ordinal_overlap = list(
    formula = formula(
      "Accuracy.and.Certainty ~
                    Overlap*Listener.Group +
                    (1 + Overlap|Participant) + (1 + Listener.Group|filename)"
    ),
    subset = TRUE,
    dvmode = "ordered"
  )
)

models <- foreach(
  m = names(model_specs),
  .final = function(x)
    setNames(x, names(model_specs))
) %do% {
  run_brms_model(model_specs[[m]][["formula"]],
                 discr_overlap[model_specs[[m]][["subset"]], ],
                 get_filename(m),
                 "ordered")
}

models <- foreach(
  m = names(model_specs),
  .final = function(x)
    setNames(x, names(model_specs))
) %do% {
  add_criterion(models[[m]], "loo", file = get_filename(m))
}


if (INTERACTIVE) {
} else {
}
print(models)
