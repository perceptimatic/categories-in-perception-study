cp_theme <- function() {
  theme_bw() +
    theme(legend.position = "bottom",
          text=element_text(family="Alegreya Sans", size=12),
          legend.text=element_text(size=12),
          legend.box.spacing = unit(0, "inches"),
          strip.background = element_rect(fill="white"))
}