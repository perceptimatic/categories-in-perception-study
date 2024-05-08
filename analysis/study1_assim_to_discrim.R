TOP <- Sys.getenv("CPTOP")
INTERACTIVE <- as.logical(Sys.getenv("CPINT"))
SCRIPTS <- paste0(TOP, "/analysis")
PLOTS <- paste0(TOP, "/analysis")

library(tidyverse)
source(paste0(SCRIPTS, "/pathnames.R"))
source(paste0(SCRIPTS, "/aggregation.R"))
source(paste0(SCRIPTS, "/cleanup.R"))
source(paste0(SCRIPTS, "/identification.R"))
source(paste0(SCRIPTS, "/discrimination.R"))

skld_score <- function(ass_tgt, ass_oth, eps) {
  k <- sum(c(eps,1)*log(c(eps,1)/c(1,eps)))
  t <- ifelse(near(ass_tgt, 0), eps, ass_tgt)
  o <- ifelse(near(ass_oth, 0), eps, ass_oth)
  kld_to <- sum(t*log(t/o))
  kld_ot <- sum(o*log(o/t))
  return(-(kld_to+kld_ot)/(2.*k))
}

haskins_score <- function(ass_tgt, ass_oth) {
  p <- 0
  for (i in 1:length(ass_tgt))
    for (j in 1:length(ass_oth))
      if (i != j)
        p <- p + ass_tgt[i]*ass_oth[j]*ass_tgt[i]
  p50 <- 0
  for (i in 1:length(ass_tgt))
    for (j in 1:length(ass_tgt))
      p50 <- p50 + ass_tgt[i]*ass_oth[i]*ass_tgt[j]
  for (i in 1:length(ass_tgt))
    for (j in 1:length(ass_oth))
      for (k in 1:length(ass_tgt))
        if ((i != j) & (i != k) & (j != k))
          p50 <- p50 + ass_tgt[i]*ass_oth[j]*ass_tgt[k]
  return(p + 0.5*p50)
}

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

assimilation_to_asymm_contrast <- function(phones, assimilation, phone_var) {
  assimilation_p1 <- assimilation %>% rename(TGT={{phone_var}}) %>%
    rename_with(~ paste0(.x, ":TGT"), .cols=!TGT)
  assimilation_p2 <- assimilation %>% rename(OTH={{phone_var}}) %>%
    rename_with(~ paste0(.x, ":OTH"), .cols=!OTH)
  contrasts <- crossing(TGT=phones, OTH=phones) %>% filter(TGT != OTH)  %>%
    mutate(`Phone Contrast Asymmetrical`=contrast_label(TGT, OTH))
  contrasts_with_ass <- contrasts %>%
    left_join(assimilation_p1, by="TGT") %>%
    left_join(assimilation_p2, by="OTH")
  return(contrasts_with_ass)
}


assimilation_to_contrast <- function(phones, assimilation, phone_var) {
  assimilation_p1 <- assimilation %>% rename(`Phone 1`={{phone_var}}) %>%
    rename_with(~ paste0(.x, ":Phone 1"), .cols=!`Phone 1`)
  assimilation_p2 <- assimilation %>% rename(`Phone 2`={{phone_var}}) %>%
    rename_with(~ paste0(.x, ":Phone 2"), .cols=!`Phone 2`)
  contrasts <- crossing(A=phones, B=phones) %>% filter(A != B)  %>%
    transmute(`Phone 1`=get_phone1(A, B),
              `Phone 2`=get_phone2(A, B),
              `Phone Contrast`=contrast_label(`Phone 1`, `Phone 2`)) %>%
    unique()
  contrasts_with_ass <- contrasts %>%
    left_join(assimilation_p1, by="Phone 1") %>%
    left_join(assimilation_p2, by="Phone 2")
  return(contrasts_with_ass)
}

assimilation_to_asymm_contrast_grouped <- function(assimilation, phone_var,
                                             value_vars, grouping_vars) {
  assimilation %>%
    reframe(assimilation_to_asymm_contrast(.data[[phone_var]],
                                     pick(all_of(c(value_vars, phone_var))),
                                     phone_var),
            .by=all_of(grouping_vars))
}

assimilation_to_contrast_grouped <- function(assimilation, phone_var,
                                             value_vars, grouping_vars) {
  assimilation %>%
    reframe(assimilation_to_contrast(.data[[phone_var]],
                                     pick(all_of(c(value_vars, phone_var))),
                                     phone_var),
            .by=all_of(grouping_vars))
}

assimilation <- assimilation_vectors %>%
  pivot_wider(id_cols=c("Phone", "Phone Language (Long)",
                        "Phone Language (Code)"),
              names_from=c("Listener Group", "Response"),
              values_from=c("Proportion of Responses", "Goodness")) %>%
  nest(`Assimilation:English`=starts_with("Proportion of Responses_English_"),
       `Assimilation:French`=starts_with("Proportion of Responses_French_"),
       `Goodness:English`=starts_with("Goodness_English_"),
       `Goodness:French`=starts_with("Goodness_French_")) %>%
  pivot_longer(starts_with("Assimilation") | starts_with("Goodness"),
               names_pattern="(.*):(.*)", names_to=c(".value",
                                                     "Listener Group"))

# FIXME: Add random identification simulation for Haskins only

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
                 ~ skld_score(unlist(.x), unlist(.y), 0.0001)),
    CosOverlapGood=map2_dbl(`AssGood:Phone 1`, `AssGood:Phone 2`,
                        ~ sum(.x * .y, na.rm=TRUE)/
                          (sqrt(sum(.x*.x, na.rm=TRUE)))*
                          sqrt(sum(.y*.y, na.rm=TRUE))),
    `Max Top Percentage`=map2_dbl(`Top Percentage:Phone 1`,
                                  `Top Percentage:Phone 2`, max),
    `Min Top Percentage`=map2_dbl(`Top Percentage:Phone 1`,
                                  `Top Percentage:Phone 2`, min)
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

haskins <- assimilation %>%
  assimilation_to_asymm_contrast_grouped("Phone", "Assimilation",
                                         c("Listener Group",
                                           "Phone Language (Long)",
                                           "Phone Language (Code)")) %>%
  mutate(Haskins=map2_dbl(`Assimilation:TGT`, `Assimilation:OTH`,
                          ~ haskins_score(as.matrix(.x), as.matrix(.y))),
         `Phone Contrast Asymmetrical (Language)`=paste0(
           `Phone Contrast Asymmetrical`, " (", `Phone Language (Code)`, ")"))

discr_pam_overlap <- left_join(discriminability_by_contrast,
                               pam_overlap,
                                       by=c("Listener Group",
                                            "Phone Language (Long)",
                                            "Phone Language (Code)",
                                            "Phone Contrast (Language)"))

discr_haskins <- left_join(discriminability_by_asymm_contrast,
                                       haskins,
                                       by=c("Listener Group",
                                            "Phone Language (Long)",
                                            "Phone Language (Code)",
                                            "Phone Contrast Asymmetrical (Language)"))
discr_haskins_collapsed <- discr_haskins %>%
  mutate(`Phone 1`=get_phone1(TGT, OTH),
         `Phone 2`=get_phone2(TGT, OTH),
         `Phone Contrast`=contrast_label(`Phone 1`, `Phone 2`),
         `Phone Contrast (Language)`=paste0(`Phone Contrast`, " (",
                                            `Phone Language (Code)`, ")")) %>%
  group_by(`Listener Group`, `Phone Contrast (Language)`, `Phone Contrast`,
           `Phone Language (Long)`, `Phone Language (Code)`,
           `Phone 1`, `Phone 2`) %>%
  summarize(Haskins=mean(Haskins),
            `Accuracy and Certainty`=mean(`Accuracy and Certainty`),
            Accuracy=mean(Accuracy)) %>%
  ungroup()

ggplot(discr_pam_overlap, aes(x=qlogis(pmin(1-CosOverlap, 0.9999)), y=`Accuracy and Certainty`, fill=`PAM Simplified Type 60 E0.1`, shape=`PAM Simplified Type 60 E0.1`)) + geom_point(size=3) + facet_grid(~ `Listener Group`, scales = "free_x") + theme_bw() + theme(legend.position = "bottom")  + scale_shape_manual(values=c(Other=3, SC=21, CG=21, TC=21)) + scale_fill_manual(values=c(Other="#000000", TC="#ffffff", CG="#aaaaaa", SC="#000000"))
ggplot(discr_pam_overlap, aes(x=qlogis(pmin(1-CosOverlap, 0.9999)), y=`Accuracy and Certainty`, fill=`PAM Simplified Type 50 E0.1`, shape=`PAM Simplified Type 50 E0.1`)) + geom_point(size=3) + facet_grid(~ `Listener Group`, scales = "free_x") + theme_bw() + theme(legend.position = "bottom")  + scale_shape_manual(values=c(Other=3, SC=21, CG=21, TC=21)) + scale_fill_manual(values=c(Other="#000000", TC="#ffffff", CG="#aaaaaa", SC="#000000"))
ggplot(discr_pam_overlap, aes(x=qlogis(pmin(1-CosOverlap, 0.9999)), y=`Accuracy and Certainty`, fill=`Min Top Percentage`)) + geom_point(size=3, pch=21) + facet_grid(~ `Listener Group`, scales = "free_x") + theme_bw() + theme(legend.position = "bottom")  + scale_fill_distiller(palette="Greys", direction=-1)
ggplot(discr_pam_overlap, aes(x=`Min Top Percentage`, y=`Accuracy and Certainty`)) + geom_point(size=3, pch=21, fill="#666666") + facet_grid(~ `Listener Group`, scales = "free_x") + theme_bw()

ggplot(inner_join(discr_haskins_collapsed, discr_pam_overlap), aes(x=Haskins, y=`Accuracy and Certainty`, fill=qlogis(pmin(1-CosOverlap, 0.999)))) + geom_point(pch=21, size=3) + facet_grid(~ `Listener Group`, scales = "free_x") + theme_bw() + theme(legend.position = "bottom")  +  scale_fill_distiller(palette="Greys", direction=1)

ggplot(discr_pam_overlap, aes(x=qlogis(pmin(1-CosOverlap, 0.9999)), y=JSD, fill=`PAM Simplified Type 60 E0.1`, shape=`PAM Simplified Type 60 E0.1`)) + geom_point(size=3) + facet_grid(~ `Listener Group`, scales = "free_x") + theme_bw() + theme(legend.position = "bottom")  + scale_shape_manual(values=c(Other=3, SC=21, CG=21, TC=21)) + scale_fill_manual(values=c(Other="#000000", TC="#ffffff", CG="#aaaaaa", SC="#000000"))
ggplot(discr_pam_overlap, aes(x=(1-CosOverlap), y=JSD, fill=`PAM Simplified Type 60 E0.1`, shape=`PAM Simplified Type 60 E0.1`)) + geom_point(size=3) + facet_grid(~ `Listener Group`, scales = "free_x") + theme_bw() + theme(legend.position = "bottom")  + scale_shape_manual(values=c(Other=3, SC=21, CG=21, TC=21)) + scale_fill_manual(values=c(Other="#000000", TC="#ffffff", CG="#aaaaaa", SC="#000000"))

ggplot(discr_pam_overlap, aes(x=SKLD, y=`Accuracy and Certainty`, fill=`PAM Simplified Type 60 E0.1`, shape=`PAM Simplified Type 60 E0.1`)) + geom_point(size=3) + facet_grid(~ `Listener Group`, scales = "free_x") + theme_bw() + theme(legend.position = "bottom")  + scale_shape_manual(values=c(Other=3, SC=21, CG=21, TC=21)) + scale_fill_manual(values=c(Other="#000000", TC="#ffffff", CG="#aaaaaa", SC="#000000"))
