read_id <- function(id_filename) {
  read_csv(
    id_filename,
    col_types = cols(
      onset = col_number(),
      offset = col_number(),
      grade = col_number(),
      language_indiv_code = col_number(),
      language_stimuli_code = col_number(),
      code_assim = col_number(),
      .default = col_character()
    )
  ) %>%
    rename(
      Participant = individual,
      Goodness = grade,
      Response = assimilation,
      `Trial Number` = nb_stimuli,
      `Listener Group` = language_indiv,
      `Phone Language (Code)` = language_stimuli,
      Phone = `#phone`
    ) %>%
    mutate(
      `Listener Group`=str_to_title(`Listener Group`),
      Participant =
        paste0(
          "IDPart",
          ifelse(`Listener Group` == "English", "E", "F"),
          str_pad(
            Participant,
            width = 3,
            side = "left",
            pad = "0"
          )
        ),
      Phone = sub(":", "ː", Phone),
      Context = paste0(prev_phone, ":", next_phone),
      `Phone Language (Long)`=full_phone_languages(`Phone Language (Code)`),
      `Phone Language (Code)`=fix_phone_language_codes(`Phone Language (Code)`),
      `Phone (Language)`=paste0(Phone, " (", `Phone Language (Code)`, ")"),
      Triphone=paste0(prev_phone, Phone, next_phone),
      `Triphone (Language)` = paste0(Triphone, " (", `Phone Language (Code)`, ")")
    ) %>%
    select(-code_assim, -language_indiv_code, -language_stimuli_code)
}

get_response_percentages <- function(response_options, responses_given,
                                     response_counts) {
  all_response_counts <- rep(0, length(response_options))
  names(all_response_counts) <- response_options
  all_response_counts[responses_given] <- response_counts
  all_response_pcts <- all_response_counts/sum(all_response_counts)
  return(all_response_pcts)
}

get_assimilation_vectors <-
  function(id_data, grouping_vars, response_var, goodness_var) {
    response_options <- sort(unique(id_data[[response_var]]))
    response_counts <- summarize(id_data, `N Responses` = n(),
                                 .by = all_of(c(grouping_vars, response_var)))
    response_pcts <-
      reframe(
        response_counts,
        `Proportion of Responses` = get_response_percentages(response_options,
                                        pick(all_of(response_var))[[1]],
                                        `N Responses`),
        Response = response_options,
        .by = all_of(grouping_vars)
      )
    goodness_avgs <- id_data %>% summarize("{goodness_var}":=
                                             mean(.data[[goodness_var]]),
                                           .by=all_of(c(grouping_vars,
                                                        response_var)))
    assimilation_vectors <- left_join(response_pcts, goodness_avgs,
                                      by=c(grouping_vars, response_var))
    return(assimilation_vectors)
  }

assimilation_vectors <- 
  read_id(IDENT_DATA) %>%
  group_by(`Listener Group`) %>%
  do(get_assimilation_vectors(.,  c("Context", "Phone (Language)",
                                    "Phone", "Phone Language (Long)",
                                    "Phone Language (Code)"),
                              "Response", "Goodness") %>% 
       repeated_average(c("Context", "Phone (Language)"),
                        c("Phone", "Phone Language (Long)",
                          "Phone Language (Code)", "Response"),
                        c("Proportion of Responses", "Goodness"),
                        na.rm=TRUE)) %>%
  ungroup() %>%
  select(-`Phone (Language)`)