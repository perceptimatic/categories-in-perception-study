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

distances_english <- readr::read_csv("discrimination_exp/triplet_data_w2v_english_tr4.csv",
                             col_types=cols(TGT_first = col_logical(),
                                            TGT_first_code = col_number(),
                                            language_stimuli_code = col_number(),
                                            .default = col_guess())) %>%
  calculate_all_deltas() %>%
  clean_discrimination_items() %>%
  mutate(`Δ DTW W2V2_4`=w2vt4_eng_delta, `Model language`="English")

distances_french <- readr::read_csv("discrimination_exp/triplet_data_w2v_french_tr4.csv",
                                    col_types=cols(TGT_first = col_logical(),
                                                   TGT_first_code = col_number(),
                                                   language_stimuli_code = col_number(),
                                                   .default = col_guess())) %>%
  calculate_all_deltas() %>%
  clean_discrimination_items() %>%
  mutate(`Δ DTW W2V2_4`=w2vt4_fra_delta, `Model language`="French")

distances <- dplyr::bind_rows(distances_english, distances_french)

distances_by_contrast <- repeated_average(
  distances,
  c(
    "filename",
    "Context",
    "Phone Contrast Asymmetrical (Language)",
    "Phone Contrast (Language)"
  ),
  c("Model language", "Phone Language (Code)", "Phone Language (Long)"),
  names(distances)[grepl("_delta$", names(distances))]
) %>% rename(`Δ DTW W2V2_4`=fb_dtw_cosine_delta)

