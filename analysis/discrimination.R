discrimination <- read_csv(
  DISCR_DATA,
  col_types = cols(
    TGT_first = col_logical(),
    correct_answer = col_number(),
    binarized_answer = col_number(),
    nb_stimuli = col_number(),
    TGT_first_code = col_number(),
    language_indiv_code = col_number(),
    language_stimuli_code =
      col_number(),
    .default = col_character()
  ),
  name_repair = "unique_quiet"
) %>%
  select(-`...1`) %>%
  clean_discrimination_items() %>%
  clean_discrimination_responses() %>%
  select(-language_indiv_code,-language_stimuli_code)

discriminability_by_asymm_contrast <- repeated_average(
  discrimination,
  c(
    "filename",
    "Context",
    "Phone Contrast Asymmetrical (Language)"
  ),
  c("Listener Group", "Phone Language (Code)", "Phone Language (Long)"),
  c("Accuracy", "Accuracy and Certainty")
) 

discriminability_by_contrast <- repeated_average(
  discrimination,
  c(
    "filename",
    "Context",
    "Phone Contrast Asymmetrical (Language)",
    "Phone Contrast (Language)"
  ),
  c("Listener Group", "Phone Language (Code)", "Phone Language (Long)"),
  c("Accuracy", "Accuracy and Certainty")
) 


discriminability_by_contrast_wide <- discriminability_by_contrast %>%
  pivot_wider(names_from=`Listener Group`,
              values_from=c(`Accuracy`, `Accuracy and Certainty`),
              names_glue="{.value} {`Listener Group`}") %>%
  mutate(`Accuracy and Certainty Difference`=
           `Accuracy and Certainty French`-
           `Accuracy and Certainty English`,
         `Accuracy Difference`=`Accuracy French`-`Accuracy English`)
