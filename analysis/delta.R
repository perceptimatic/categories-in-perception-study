calculate_all_deltas <- function(d) {
  for (col_tgt in names(d)[grepl("distance_tgt$", names(d))]) {
    col_oth <- sub("_tgt$", "_oth", col_tgt)
    col_delta <- sub("distance_tgt$", "Δ" , col_tgt)
    d[[col_delta]] <- (d[[col_oth]] - d[[col_tgt]])
  }
  return(d)
}

read_delta <- function(f, layers=FALSE) {
  readr::read_csv(f,
                  col_types=cols(TGT_first = col_logical(),
                                 TGT_first_code = col_number(),
                                 language_stimuli_code = col_number(),
                                 .default = col_guess())) %>%
    ({ if (layers) {
      . %>%
      pivot_longer(cols=contains("_distance_"),
                   names_to=c("Model", "Layer", "distance_type"),
                   names_pattern="(.*)_layer([0-9]*)_(distance_.*)") %>%
        mutate(Layer=as.numeric(Layer)) %>%
        pivot_wider(names_from=distance_type)        
    } else {
      . %>%
        pivot_longer(cols=contains("_distance_"),
                     names_to=c("Model", "distance_type"),
                     names_pattern="(.*)_(distance_.*)") %>%
        pivot_wider(names_from=distance_type)   
    } }) %>%
    calculate_all_deltas() %>%
    rename(`Δ Model`=`Δ`) %>%
    clean_discrimination_items()
}
