TOP <- Sys.getenv("CPTOP")
INTERACTIVE <- as.logical(Sys.getenv("CPINT"))
SCRIPTS <- paste0(TOP, "/analysis")
PLOTS <- paste0(TOP, "/analysis")
source(paste0(SCRIPTS, "/pathnames.R"))
source(paste0(SCRIPTS, "/aggregation.R"))
source(paste0(SCRIPTS, "/cleanup.R"))
source(paste0(SCRIPTS, "/discrimination.R"))

library(tidyverse)
library(ggrepel)

group_scatterplot <- function(d, variable_x, variable_y,
                              variable_label, variable_diff, force,
                              xlim=c(-1.1, 3), ylim=c(-1.1, 3),
                              rank_cutoff=NULL,
                              type_0=NULL,
                              type_1=NULL,
                              type_2=NULL) {
  point_types <- rep("D", nrow(d))
  if (!is.null(rank_cutoff)) {
    point_types[rank(-abs(d[[variable_diff]])) <= rank_cutoff] <- "R"
  }
  point_types[d[[variable_label]] %in% type_0] <- "T0"
  point_types[d[[variable_label]] %in% type_1] <- "T1"
  point_types[d[[variable_label]] %in% type_2] <- "T2"
  ggplot(
    d %>% mutate(point_types=point_types),
    aes(x = .data[[variable_x]],
        y = .data[[variable_y]],
        shape = point_types)
  ) +
    geom_abline(colour = "#bbbbbb") +
    geom_label_repel(
      aes(label = ifelse(point_types != "D", .data[[variable_label]], "")),
      family = "Alegreya Sans",
      size = 8 * 0.352777778,
      segment.colour="#888888",
      segment.size=0.3,
      force = force,
      box.padding=0.9,
      label.padding=0.1,
      max.overlaps = NA,
      min.segment.length = 0,
      arrow = arrow(length = unit(1, "mm")),
      seed = 42
    ) +
    scale_shape_manual(values=c(D=1, R=8, T1=24,
                               T0=23, T2=25)) +
    geom_point(size=2, fill="#aaaaaa") +
    coord_cartesian(xlim = xlim, ylim = ylim) +
    theme_bw() +
    theme(
      legend.position = "none",
      text = element_text(family = "Alegreya Sans", size = 12),
      axis.text = element_text(size = 12, colour = "black")
    )
}
  
certaccuracy_by_contrast_plot <- group_scatterplot(
  discriminability_by_contrast_wide,
  "Accuracy and Certainty English",
  "Accuracy and Certainty French",
  "Phone Contrast (Language)",
  "Accuracy and Certainty Difference", 7,
  xlim=c(-0.5,3), ylim=c(-0.5,3),
  rank_cutoff = NULL,
  type_0=c("i–ĩ (pt)",
           "øː–ʏ (de-m)",
           "œ–ɯ (tr)"),
  type_1=c("u–ũ (pt)",
           "ɤː–yː (et)",
           "e–ɛ (fr)",
           "øː–yː (de-m)",
           "uː–yː (de-m)",
           "u–y (tr)"),
  type_2=c("æ–ɑ (en)",
           "æ–ʌ (en)",
           "ɑ–ʌ (en)",
           "eɪ–ɛ (en)",
           "õ–ũ (pt)",
           "aː–æː (et)",
           "e–i (fr)",
           "ɪ–ʏ (de-m)",
           "a–aː (de)",
           "e–i (de)"
           )
)

accuracy_by_contrast_plot <- group_scatterplot(
  discriminability_by_contrast, "Accuracy English",
  "Accuracy French", 
  "Phone Contrast (Language)",
  "Accuracy Difference", 30,
  xlim=c(0.2,1), ylim=c(0.2,1),
  rank_cutoff = 6
)

certaccuracy_accuracy_by_contrast_plot <- ggplot(
  discriminability_by_contrast,
  aes(x=`Accuracy Difference`,
      y=`Accuracy and Certainty Difference`)) +
  geom_point()

if (INTERACTIVE) {
  print(certaccuracy_by_contrast_plot)
  print(accuracy_by_contrast_plot)
  print(certaccuracy_accuracy_by_contrast_plot)
} else {
  ggsave(paste0(PLOTS, "/certaccuracy_by_contrast_plot_600.png"),
         plot=certaccuracy_by_contrast_plot,
         width=5, height=5, units="in", dpi=600)
}

