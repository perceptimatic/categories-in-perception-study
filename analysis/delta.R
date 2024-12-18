
calculate_all_deltas <- function(d) {
  for (col_tgt in names(d)[grepl("distance_tgt$", names(d))]) {
    col_oth <- sub("_tgt$", "_oth", col_tgt)
    col_delta <- sub("distance_tgt$", "delta", col_tgt)
    d[[col_delta]] <- (d[[col_oth]] - d[[col_tgt]])
  }
  return(d)
}
