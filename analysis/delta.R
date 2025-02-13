calculate_all_deltas <- function(d) {
  for (col_tgt in names(d)[grepl("distance_tgt$", names(d))]) {
    col_oth <- sub("_tgt$", "_oth", col_tgt)
    col_delta <- sub("distance_tgt$", "Δ" , col_tgt)
    d[[col_delta]] <- (d[[col_oth]] - d[[col_tgt]])
  }
  return(d)
}

read_delta <- function(f) {
  raw <- readr::read_csv(f,
                  col_types=cols(TGT_first = col_logical(),
                                 TGT_first_code = col_number(),
                                 language_stimuli_code = col_number(),
                                 .default = col_guess()))
  raw_meta <- dplyr::select(raw, -contains("_distance_"))
  raw_dist_layer <- dplyr::select(raw, matches("_layer[0-9]*_distance_"))
  raw_dist_nolayer <- dplyr::select(raw, !matches("_layer[0-9]*_distance_") &
                                      matches("_distance_"))
  
  long_layer <- NULL
  long_nolayer <- NULL
  if (ncol(raw_dist_layer) > 0) {
    long_layer <- bind_cols(raw_meta, raw_dist_layer) %>%
      pivot_longer(cols=contains("_distance_"),
                     names_to=c("Model", "Layer", "distance_type"),
                     names_pattern="(.*)_layer([0-9]*)_(distance_.*)") %>%
          mutate(Layer=as.numeric(Layer)) %>%
          pivot_wider(names_from=distance_type)        
  }
  if (ncol(raw_dist_nolayer) > 0) {
    long_nolayer <- bind_cols(raw_meta, raw_dist_nolayer) %>%
          pivot_longer(cols=contains("_distance_"),
                       names_to=c("Model", "distance_type"),
                       names_pattern="(.*)_(distance_.*)") %>%
          pivot_wider(names_from=distance_type)       
  }

  long <- bind_rows(long_layer, long_nolayer)
    
  long %>%
    calculate_all_deltas() %>%
    rename(`Δ Model`=`Δ`) %>%
    clean_discrimination_items()
}
