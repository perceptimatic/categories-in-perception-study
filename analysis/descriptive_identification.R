TOP <- Sys.getenv("CPTOP")
INTERACTIVE <- as.logical(Sys.getenv("CPINT"))
SCRIPTS <- paste0(TOP, "/analysis")
PLOTS <- paste0(TOP, "/analysis")
Sys.setlocale(locale="en_US.UTF-8")

library(tidyverse)
source(paste0(SCRIPTS, "/pathnames.R"))
source(paste0(SCRIPTS, "/aggregation.R"))
source(paste0(SCRIPTS, "/cleanup.R"))
source(paste0(SCRIPTS, "/identification.R"))
source(paste0(SCRIPTS, "/plotting.R"))

assimilation_vectors_plot <- ggplot(assimilation_vectors,
       aes(x=`Phone`, y=Response, fill=`Proportion of Responses`)) +
  geom_tile(colour=NA, linewidth=NA) +
  geom_vline(xintercept=seq(1, 60) + 0.5, colour="#555555", linewidth=0.3) +
  facet_grid(`Listener Group` ~ `Phone Language (Long)`, scales="free",
             space="free", labeller=label_wrap_gen(width=13)) +
  scale_fill_viridis_c(limits=c(0,1), option="B") +
  scale_y_discrete(limits=rev) +
  cp_theme() +
  theme(axis.text=element_text(size=8, colour="black"),
        axis.text.x=element_text(angle=90, hjust=1, vjust=0.5)) +
  guides(fill=guide_colourbar(title.position="left",
                              title.vjust=1,
                              barwidth=unit(1.5, "inches"),
                              barheight=unit(0.15, "inches"),
                              frame.colour="black",
                              ticks.colour="black"))


assimilation_vectors_bycontext_french <- 
  read_id(IDENT_DATA) %>%
  filter(`Phone Language (Long)` == "Hexagonal French") %>%
  group_by(`Listener Group`) %>%
  do(get_assimilation_vectors(., c("Phone", "Phone (Language)", 
                                   "Triphone (Language)",
                                    "Triphone", "Phone Language (Long)",
                                    "Phone Language (Code)"),
                              "Response", "Goodness") %>% 
       repeated_average("Triphone (Language)",
                        c("Phone", "Phone Language (Long)",
                          "Phone Language (Code)", "Response"),
                        c("Proportion of Responses", "Goodness"),
                        na.rm=TRUE)) %>%
  ungroup()

assimilation_vectors_without_sucre_french <- 
  read_id(IDENT_DATA) %>%
  filter(`Phone Language (Long)` == "Hexagonal French",
         Triphone != "søk", Phone == "ø",
         `Listener Group` == "French") %>%
  group_by(`Listener Group`) %>%
  do(get_assimilation_vectors(., c("Phone", "Phone (Language)", 
                                   "Context", "Phone Language (Long)",
                                    "Phone Language (Code)"),
                              "Response", "Goodness") %>% 
       repeated_average(c("Context", "Phone (Language)"),
                        c("Phone", "Phone Language (Long)",
                          "Phone Language (Code)", "Response"),
                        c("Proportion of Responses", "Goodness"),
                        na.rm=TRUE)) 

assimilation_vectors_bycontext_french_plot <- ggplot(assimilation_vectors_bycontext_french,
                                    aes(x=`Triphone (Language)`, y=Response, fill=`Proportion of Responses`)) +
  geom_tile(colour=NA, linewidth=NA) +
  geom_vline(xintercept=seq(1, 60) + 0.5, colour="#555555", linewidth=0.3) +
  facet_grid(`Listener Group` ~ Phone, scales="free",
             space="free", labeller=label_wrap_gen(width=13)) +
  scale_fill_viridis_c(limits=c(0,1), option="B") +
  scale_y_discrete(limits=rev) +
  cp_theme() +
  theme(axis.text=element_text(size=10, colour="black"),
        axis.text.x=element_text(angle=90, hjust=1, vjust=0.5, size=7)) +
  guides(fill=guide_colourbar(title.position="left",
                              title.vjust=1,
                              barwidth=unit(1.5, "inches"),
                              barheight=unit(0.15, "inches"),
                              frame.colour="black",
                              ticks.colour="black"))


if (INTERACTIVE) {
  print(assimilation_vectors_plot)
  print(assimilation_vectors_bycontext_french_plot)
  View(assimilation_vectors_without_sucre_french)
} else {
  ggsave(paste0(PLOTS, "/assimilation_vectors_plot_600.png"),
         plot=assimilation_vectors_plot,
          width=6.52, height=4.5, units="in", dpi=600)
}

