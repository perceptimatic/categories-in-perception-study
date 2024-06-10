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
    clean_id_items() %>%
    clean_id_responses()
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


smooth <- function(x, eps) {
  new_max <- 1 - eps * sum(near(x, 0))
  return(ifelse(near(x, 0), eps, x * new_max))
}

max_skld <- function(x, eps) {
  n <- length(x)
  x1 <- c(rep(eps, n - 1), 1 - eps * (n - 1))
  x2 <- c(1 - eps * (n - 1), rep(eps, n - 1))
  k1 <- sum(x1 * log(x1 / x2))
  k2 <- sum(x2 * log(x2 / x1))
  return((k1 + k2) / 2.)
}

haskins_score <- function(ass_tgt, ass_oth, with_p50 = TRUE) {
  tgt <- unlist(c(ass_tgt))
  oth <- unlist(c(ass_oth))
  p <- sum((tgt ^ 2) * (1 - oth))
  p50 <- 0
  if (with_p50)
    p50 <- sum(tgt * (1 - tgt) * (1 - oth)) + sum((tgt ^ 2) * oth)
  return(as.numeric(p + 0.5 * p50))
}

assimilation_to_contrast <- function(phones, assimilation, phone_var) {
  assimilation_p1 <- assimilation %>% rename(`Phone 1` = {{phone_var}}) %>%
    rename_with(~ paste0(.x, ":Phone 1"), .cols = !`Phone 1`)
  assimilation_p2 <- assimilation %>% rename(`Phone 2` = {{phone_var}}) %>%
    rename_with(~ paste0(.x, ":Phone 2"), .cols = !`Phone 2`)
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
  return(1 - (kld_to + kld_ot) / (2. * k))
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
    `Goodness Difference` = abs(`Top Goodness:Phone 1` - `Top Goodness:Phone 2`),
    Overlap = map2_dbl(
      `Assimilation:Phone 1`,
      `Assimilation:Phone 2`,
      ~ skld_score(unlist(.x), unlist(.y), 0.001)
    ),
    Dot = map2_dbl(`Assimilation:Phone 1`, `Assimilation:Phone 2`, ~ sum(.x * .y)),
    Haskins = map2_dbl(
      `Assimilation:Phone 1`,
      `Assimilation:Phone 2`,
      ~ 1 - 0.5 * (haskins_score(.x, .y) + haskins_score(.y, .x))
    ),
    `Maximum Categorization Threshold` = map2_dbl(`Top Percentage:Phone 1`, `Top Percentage:Phone 2`, min),
    `Same Top Choice` = map2_chr(`Top Choice:Phone 1`, `Top Choice:Phone 2`, ~ c("No", "Yes")[(.x == .y) + 1])
  )
