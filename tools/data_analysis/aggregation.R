grouped_mean <- function(d,
                         grouping_vars,
                         vars_to_average,
                         weights = NULL,
                         na.rm = FALSE) {
  d$`__WEIGHTS__` <- weights
  d_gr <- dplyr::group_by(d, dplyr::across(dplyr::all_of(
    grouping_vars
  )))
  if (is.null(weights)) {
    dplyr::summarize(d_gr,
                     dplyr::across(dplyr::all_of(vars_to_average),
                                   ~ mean(.x, na.rm=na.rm)),
                     .groups = "drop")
  } else {
    dplyr::summarize(d_gr,
                     dplyr::across(
                       dplyr::all_of(vars_to_average),
                       ~ weighted.mean(.x, `__WEIGHTS__`, na.rm=na.rm)
                     ),
                     .groups = "drop")
  }
}

repeated_average <- function(d,
                             aggregation_order,
                             grouping_vars_to_keep,
                             vars_to_average,
                             weights=NULL, na.rm=FALSE) {
  first_aggregation <- grouped_mean(d, c(aggregation_order,
                                        grouping_vars_to_keep),
                                           vars_to_average,
                                           weights, na.rm)
  if (length(aggregation_order) <= 1) {
    return(first_aggregation)
  }
  result <- first_aggregation
  for (i in 2:length(aggregation_order)) {
    groups_i <- aggregation_order[i:length(aggregation_order)]
    result <- grouped_mean(result,
                                  c(groups_i,
                                  grouping_vars_to_keep),
                                  vars_to_average,
                                  NULL, na.rm)
  }
  return(result)
}


replace_vars_with_aggregates <- function(d,
                                         weights,
                                         aggregation_order,
                                         vars_to_aggregate) {
  d_without_vars <-
    dplyr::select(d,-dplyr::any_of(vars_to_aggregate))
  dg <-
    repeated_average(d, aggregation_order, NULL, vars_to_aggregate,
                     weights)
  result <- dplyr::left_join(d_without_vars, dg)
  return(result)
}
