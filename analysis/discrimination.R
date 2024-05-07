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
  rename(
    Participant = individual,
    `Accuracy and Certainty` =
      correct_answer,
    `Accuracy -1/1` =
      binarized_answer,
    `Trial Number` = nb_stimuli,
    `TGT was A (first)` = TGT_first_code,
    `Listener Group` = language_indiv,
    `Phone Language (Code)` = language_stimuli
  ) %>%
  mutate(
    `Listener Group` = str_to_title(`Listener Group`),
    Participant =
      paste0(
        "DscPart",
        ifelse(`Listener Group` == "English", "E", "F"),
        str_pad(
          Participant,
          width = 3,
          side = "left",
          pad = "0"
        )
      ),
    `Accuracy`=(`Accuracy -1/1`+1.)/2,
    TGT = sub(":", "ː", TGT),
    OTH = sub(":", "ː", OTH),
    Context = paste0(prev_phone, "–", next_phone),
    `TGT was A (first)` = 2 * (`TGT was A (first)` - 0.5),
    `Phone Language (Long)`=full_phone_languages(`Phone Language (Code)`),
    `Phone Language (Code)`=fix_phone_language_codes(`Phone Language (Code)`),
    `Phone Contrast Asymmetrical` = contrast_label(TGT, OTH),
    `Phone Contrast Asymmetrical (Language)` = paste0(
      `Phone Contrast Asymmetrical`, " (", `Phone Language (Code)`, ")"),
    `Phone 1` = get_phone1(TGT, OTH),
    `Phone 2` = get_phone2(TGT, OTH),
    `Phone Contrast` = contrast_label(`Phone 1`, `Phone 2`),
    `Phone Contrast (Language)`=paste0(`Phone Contrast`,
                              " (", `Phone Language (Code)`, ")"),
    `Triphone Contrast`=paste0(prev_phone, `Phone 1`, next_phone,
                               "–",
                               prev_phone, `Phone 2`, next_phone),
    `Triphone Contrast (Language)` = paste0(`Triphone Contrast`, " (",
                                         `Phone Language (Code)`, ")")
  ) %>%
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
