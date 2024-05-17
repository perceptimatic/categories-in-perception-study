pam18_rank <- function(pam18_type, goodness_diff, overlap,
                       goodness_overlap="GbO") {
  ranks <- rep(NA, length(pam18_type))
  ranks[pam18_type %in% c("SC", "Complete overlap")] <- 1
  rank_goodness <- rank(-goodness_diff[pam18_type == "CG"])
  rank_overlap <- rank(-overlap[pam18_type == "Partial overlap"])
  if (goodness_overlap == "GbO") {
    ranks[pam18_type == "Partial overlap"] <- rank_overlap + 1
    ranks[pam18_type == "CG"] <- rank_goodness + max(rank_overlap) + 1
  } else if (goodness_overlap == "ObG") {
    ranks[pam18_type == "CG"] <- rank_goodness + 1
    ranks[pam18_type == "Partial overlap"] <- rank_overlap +
      max(rank_goodness) + 1
  } else if (goodness_overlap == "mapping") {
    overlap_to_goodness <- lm(goodness_diff ~ overlap)
    pseudo_goodness <- goodness_diff
    pseudo_goodness[pam18_type == "Partial overlap"] <-
      predict(overlap_to_goodness)[pam18_type == "Partial overlap"]
    rank_pseudo_goodness <- rank(pseudo_goodness[pam18_type %in% c(
      "Partial overlap", "CG")])
    ranks[pam18_type %in% c("Partial overlap", "CG")] <- rank_pseudo_goodness+1
  }
  max_rank <- max(ranks, na.rm=TRUE)
  ranks[pam18_type %in% c("TC", "Non-overlap")] <- max_rank + 1
  return(ranks)
}

pam_overlap <- assimilation %>% 
  mutate(`Categorized 50`=map_lgl(Assimilation, ~ .x[[max.col(.x)]] >= 0.5),
         `Categorized 60`=map_lgl(Assimilation, ~ .x[[max.col(.x)]] >= 0.6),
         `Categorized 90`=map_lgl(Assimilation, ~ .x[[max.col(.x)]] >= 0.9),
         `Top Percentage`=map_dbl(Assimilation, ~ .x[[max.col(.x)]]),
         `Top Goodness`=map2_dbl(Assimilation, Goodness, ~ .y[[max.col(.x)]]),
         `Top Choice`=map_chr(Assimilation,
                                      ~ str_split_i(colnames(.x)[max.col(.x)],
                                                    "_", 3)),
         `Choice Set 0.1`=map_chr(Assimilation,
                                          ~ paste(str_split_i(
                                            colnames(.x)[as.matrix(.x) >= 0.1],
                                            "_", 3), collapse=" "))) %>% 
  assimilation_to_contrast_grouped("Phone",
                                   c("Categorized 50", "Categorized 60",
                                     "Categorized 90",
                                     "Top Goodness", "Top Choice",
                                     "Top Percentage",
                                     "Choice Set 0.1", "Assimilation",
                                     "Goodness"),
                                   c("Listener Group", "Phone Language (Long)",
                                     "Phone Language (Code)")) %>%
  mutate(`Phone Contrast (Language)`=paste0(`Phone Contrast`, " (",
                                            `Phone Language (Code)`, ")")) %>%
  mutate(`Goodness Difference`=abs(`Top Goodness:Phone 1` -
                                     `Top Goodness:Phone 2`),
    `PAM Type 90 E0.1`=ifelse(
    `Categorized 90:Phone 1` & `Categorized 90:Phone 2` & 
      (`Top Choice:Phone 1` != `Top Choice:Phone 2`), "TC",
    ifelse(
      (`Categorized 90:Phone 1` + `Categorized 90:Phone 2`) == 1 &
        (`Top Choice:Phone 1` != `Top Choice:Phone 2`), "UC",
      ifelse(
        !`Categorized 90:Phone 1` & !`Categorized 90:Phone 2`, "UU",
        ifelse(
          `Categorized 90:Phone 1` & `Categorized 90:Phone 2` &
            (`Top Choice:Phone 1` == `Top Choice:Phone 2`) &
            `Goodness Difference` <= 0.1, "SC",
          ifelse(`Categorized 90:Phone 1` & `Categorized 90:Phone 2` &
                   (`Top Choice:Phone 1` == `Top Choice:Phone 2`) &
                   `Goodness Difference` > 0.1, "CG", NA))))),
    `PAM Type 60 E0.1`=ifelse(
      `Categorized 60:Phone 1` & `Categorized 60:Phone 2` & 
        (`Top Choice:Phone 1` != `Top Choice:Phone 2`), "TC",
      ifelse(
        (`Categorized 60:Phone 1` + `Categorized 60:Phone 2`) == 1 &
          (`Top Choice:Phone 1` != `Top Choice:Phone 2`), "UC",
        ifelse(
          !`Categorized 60:Phone 1` & !`Categorized 60:Phone 2`, "UU",
          ifelse(
            `Categorized 60:Phone 1` & `Categorized 60:Phone 2` &
              (`Top Choice:Phone 1` == `Top Choice:Phone 2`) &
              abs(`Top Goodness:Phone 1` - `Top Goodness:Phone 2`) <= 0.1, "SC",
            ifelse(`Categorized 60:Phone 1` & `Categorized 60:Phone 2` &
                     (`Top Choice:Phone 1` == `Top Choice:Phone 2`) &
                     abs(`Top Goodness:Phone 1` - `Top Goodness:Phone 2`) > 0.1,
                   "CG", NA))))),
    `PAM Type 50 E0.1`=ifelse(
      `Categorized 50:Phone 1` & `Categorized 50:Phone 2` & 
        (`Top Choice:Phone 1` != `Top Choice:Phone 2`), "TC",
      ifelse(
        (`Categorized 50:Phone 1` + `Categorized 50:Phone 2`) == 1 &
          (`Top Choice:Phone 1` != `Top Choice:Phone 2`), "UC",
        ifelse(
          !`Categorized 50:Phone 1` & !`Categorized 50:Phone 2`, "UU",
          ifelse(
            `Categorized 50:Phone 1` & `Categorized 50:Phone 2` &
              (`Top Choice:Phone 1` == `Top Choice:Phone 2`) &
              abs(`Top Goodness:Phone 1` - `Top Goodness:Phone 2`) <= 0.1, "SC",
            ifelse(`Categorized 50:Phone 1` & `Categorized 50:Phone 2` &
                     (`Top Choice:Phone 1` == `Top Choice:Phone 2`) &
                     abs(`Top Goodness:Phone 1` - `Top Goodness:Phone 2`) > 0.1,
                   "CG", NA))))),
    `PAM Type 50 E0.01`=ifelse(
      `Categorized 50:Phone 1` & `Categorized 50:Phone 2` & 
        (`Top Choice:Phone 1` != `Top Choice:Phone 2`), "TC",
      ifelse(
        (`Categorized 50:Phone 1` + `Categorized 50:Phone 2`) == 1 &
        (`Categorized 50:Phone 1` + `Categorized 50:Phone 2`) == 1 &
          (`Top Choice:Phone 1` != `Top Choice:Phone 2`), "UC",
        ifelse(
          !`Categorized 50:Phone 1` & !`Categorized 50:Phone 2`, "UU",
          ifelse(
            `Categorized 50:Phone 1` & `Categorized 50:Phone 2` &
              (`Top Choice:Phone 1` == `Top Choice:Phone 2`) &
              abs(`Top Goodness:Phone 1` - `Top Goodness:Phone 2`) <= 0.01,"SC",
            ifelse(`Categorized 50:Phone 1` & `Categorized 50:Phone 2` &
                     (`Top Choice:Phone 1` == `Top Choice:Phone 2`) &
                     abs(`Top Goodness:Phone 1` - `Top Goodness:Phone 2`) >0.01,
                   "CG", NA))))),
    `Set Overlap Total`=`Choice Set 0.1:Phone 1` == `Choice Set 0.1:Phone 2`,
    `Set Overlap Empty`=map2_lgl(str_split(`Choice Set 0.1:Phone 1`, " "),
                                 str_split(`Choice Set 0.1:Phone 2`, " "),
                                 ~ length(intersect(.x, .y)) == 0),
    `PAM18 Type 90 E0.1`=ifelse(`PAM Type 90 E0.1` %in% c("TC", "CG", "SC"),
                                `PAM Type 90 E0.1`,
                                ifelse(`Set Overlap Total`, "Complete overlap",
                                       ifelse(`Set Overlap Empty`,
                                              "Non-overlap",
                                              "Partial overlap"))),
    `PAM18 Type 60 E0.1`=ifelse(`PAM Type 60 E0.1` %in% c("TC", "CG", "SC"),
                                `PAM Type 60 E0.1`,
                                ifelse(`Set Overlap Total`, "Complete overlap",
                                       ifelse(`Set Overlap Empty`,
                                              "Non-overlap",
                                              "Partial overlap"))),    
    `PAM18 Type 50 E0.1`=ifelse(`PAM Type 50 E0.1` %in% c("TC", "CG", "SC"),
                                `PAM Type 50 E0.1`,
                                ifelse(`Set Overlap Total`, "Complete overlap",
                                       ifelse(`Set Overlap Empty`,
                                              "Non-overlap",
                                              "Partial overlap"))),    
    `PAM18 Type 50 E0.01`=ifelse(`PAM Type 50 E0.01` %in% c("TC", "CG", "SC"),
                                `PAM Type 50 E0.01`,
                                ifelse(`Set Overlap Total`, "Complete overlap",
                                       ifelse(`Set Overlap Empty`,
                                              "Non-overlap",
                                              "Partial overlap"))),    
    `PAM Simplified Type 60 E0.1`=factor(ifelse(`PAM Type 60 E0.1` %in%
                                                   c("TC", "CG", "SC"),
                                          `PAM Type 60 E0.1`, "Other"),
                                          levels=c("SC",  "CG", "TC","Other")),      
    `PAM Simplified Type 50 E0.1`=factor(ifelse(`PAM Type 50 E0.1` %in%
                                                   c("TC", "CG", "SC"),
                                          `PAM Type 50 E0.1`, "Other"),
                                          levels=c("SC",  "CG", "TC","Other")),      
    `PAM Simplified Type 50 E0.01`=factor(ifelse(`PAM Type 50 E0.01` %in%
                                                   c("TC", "CG", "SC"),
                                          `PAM Type 50 E0.01`, "Other"),
                                          levels=c("SC",  "CG", "TC","Other")),
    CosOverlap=map2_dbl(`Assimilation:Phone 1`, `Assimilation:Phone 2`,
                        ~ sum(.x * .y)/(sqrt(sum(.x*.x)))*sqrt(sum(.y*.y))),
    `AssGood:Phone 1`=map2(`Assimilation:Phone 1`, `Goodness:Phone 1`,
                               ~ .x * .y),
    `AssGood:Phone 2`=map2(`Assimilation:Phone 2`, `Goodness:Phone 2`,
                               ~ .x * .y),
    SKLD=map2_dbl(`Assimilation:Phone 1`, `Assimilation:Phone 2`,
                 ~ skld_score(unlist(.x), unlist(.y), 0.001)),
    CosOverlapGood=map2_dbl(`AssGood:Phone 1`, `AssGood:Phone 2`,
                        ~ sum(.x * .y, na.rm=TRUE)/
                          (sqrt(sum(.x*.x, na.rm=TRUE)))*
                          sqrt(sum(.y*.y, na.rm=TRUE))),
    `Max Top Percentage`=map2_dbl(`Top Percentage:Phone 1`,
                                  `Top Percentage:Phone 2`, max),
    `Min Top Percentage`=map2_dbl(`Top Percentage:Phone 1`,
                                  `Top Percentage:Phone 2`, min),
    `Same Top`=map2_chr(`Top Choice:Phone 1`,
                        `Top Choice:Phone 2`,
                        ~ c("Different", "Same")[(.x == .y) + 1])    
    ) %>%
  group_by(`Listener Group`) %>%
  mutate(`PAM18 Ranking 90 E0.1`=pam18_rank(`PAM18 Type 90 E0.1`,
                                                  `Goodness Difference`,
                                                  CosOverlap),
         `PAM18 Ranking 50 E0.1 GbO`=pam18_rank(`PAM18 Type 50 E0.1`,
                                                `Goodness Difference`,
                                                CosOverlap, "GbO"),
         `PAM18 Ranking 50 E0.1 ObG`=pam18_rank(`PAM18 Type 50 E0.1`,
                                                `Goodness Difference`,
                                                CosOverlap, "ObG"),
         `PAM18 Ranking 50 E0.1 Mapping`=pam18_rank(`PAM18 Type 50 E0.1`,
                                                `Goodness Difference`,
                                                CosOverlap, "mapping"))

