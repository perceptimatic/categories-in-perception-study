
    
  


fix_phone_language_codes <- function(x) {
  isoish_names <- c(FR="fr", BR="pt", EN="en", GR="de-m",
                    GL="de", ES="et", TU="tr")
  isoish_names[x]
}

full_phone_languages <- function(x) {
  language_names <- c(FR="Hexagonal French", BR="Brazilian Portuguese",
                      EN="American English", GR="Munich German",
                    GL="Standard German", ES="Estonian", TU="Turkish")
  language_names[x]
}

get_phone1 <- function(a, b) ifelse(a < b, a, b)
get_phone2 <- function(a, b) ifelse(a < b, b, a)
contrast_label <- function(p1, p2)  paste0(p1, "–", p2)

clean_items <- function(d) {
  rename(d, `Phone Language (Code)` = language_stimuli) %>%
    mutate(
      Context = paste0(prev_phone, "–", next_phone),
      `Phone Language (Long)`=full_phone_languages(`Phone Language (Code)`),
      `Phone Language (Code)`=fix_phone_language_codes(`Phone Language (Code)`))
}


clean_responses <- function(d, participant_prefix) {
  rename(d,
    Participant = individual,
    `Trial Number` = nb_stimuli,
    `Listener Group` = language_indiv,
  ) %>%
  mutate(
    `Listener Group` = str_to_title(`Listener Group`),
    Participant =
      paste0(
        participant_prefix,
        ifelse(`Listener Group` == "English", "E", "F"),
        str_pad(
          Participant,
          width = 3,
          side = "left",
          pad = "0"
        )))
}    

clean_discrimination_items <- function(d) {
  clean_items(d) %>%
  rename(`TGT was A (first)` = TGT_first_code) %>%
    mutate(
      TGT = sub(":", "ː", TGT),
      OTH = sub(":", "ː", OTH),      
      `TGT was A (first)` = 2 * (`TGT was A (first)` - 0.5),
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
                                          `Phone Language (Code)`, ")"))
}

clean_discrimination_responses <- function(d) {
  clean_responses(d, "DscPart") %>%
  rename(`Accuracy and Certainty` = correct_answer,
         `Accuracy -1/1` = binarized_answer) %>%
    mutate(`Accuracy`=(`Accuracy -1/1`+1.)/2)
}

clean_id_items <- function(d) {
  clean_items(d) %>%
  rename(Phone = `#phone`) %>%
    mutate(Phone = sub(":", "ː", Phone),
      `Phone (Language)`=paste0(Phone, " (", `Phone Language (Code)`, ")"),
      Triphone=paste0(prev_phone, Phone, next_phone),
      `Triphone (Language)` = paste0(Triphone, " (", `Phone Language (Code)`, ")"))
}

clean_id_responses <- function(d) {
  clean_responses(d, "IDPart") %>%
  rename(Goodness = grade, Response = assimilation)
}

