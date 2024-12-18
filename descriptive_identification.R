TOP <- Sys.getenv("CATEGORIES_IN_PERCEPTION_TOP")
SCRIPTS <- paste0(TOP, "/analysis")
PLOTS <- paste0(TOP, "/analysis")
source("analysis/pathnames.R")
source("analysis/aggregation.R")
source("analysis/cleanup.R")

MFCC_DATA <- ("mfccclass_vecs_avgs.csv")
W2V_DATA <- ("wav2vec_class_vecs_avgs.csv")
DS_DATA <- ("analysis/wv_deepspeech_class_vectors.csv")
MFCC_DELTAS <- read_csv("mfcc_deltas.csv")
W2V_DELTAS <- read_csv("wav2vec_deltas.csv")

library(tidyverse)

get_response_percentages <- function(response_options, responses_given,
                                     response_counts) {
  all_response_counts <- rep(0, length(response_options))
  names(all_response_counts) <- response_options
  all_response_counts[responses_given] <- response_counts
  all_response_pcts <- all_response_counts/sum(all_response_counts)
  return(all_response_pcts)
}

get_assimilation_vectors <-
  function(id_data, grouping_vars, response_var) {
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
    return(response_pcts)
  }

id_data <- read_csv(
  IDENT_DATA,
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
  rename(
    Participant = individual,
    Goodness = grade,
    Response = assimilation,
    `Trial Number` = nb_stimuli,
    `Listener Group` = language_indiv,
    `Phone Language` = language_stimuli,
    Phone = `#phone`
  ) %>%
  mutate(
    `Listener Group`=str_to_title(`Listener Group`),
    Participant =
      paste0(
        "IDPart",
        ifelse(`Listener Group` == "English", "E", "F"),
        str_pad(
          Participant,
          width = 3,
          side = "left",
          pad = "0"
        )
      ),
    Phone = sub(":", "ː", Phone),
    Context = paste0(prev_phone, ":", next_phone),
    `Phone Language`=fix_phone_languages(`Phone Language`),
    `Phone (Language)`=paste0(Phone, " (", `Phone Language`, ")")
  ) %>%
  select(-code_assim, -language_indiv_code, -language_stimuli_code)

id_data_fr <- dplyr::filter(id_data, `Listener Group` == "French")
id_data_en <- dplyr::filter(id_data, `Listener Group` == "English")

assimilation_vectors_fr <-
  get_assimilation_vectors(id_data_fr,
                           c("Listener Group", "filename", "Context",
                             "Phone (Language)"),
                           "Response")  %>%
  repeated_average(c("Context", "Phone (Language)"),
                   c("Listener Group",  "Response"),
                   "Proportion of Responses")

assimilation_vectors_en <-
  get_assimilation_vectors(id_data_en,
                           c("Listener Group", "filename", "Context", 
                             "Phone (Language)"),
                           "Response")  %>%
  repeated_average(c("Context", "Phone (Language)"),
                   c("Listener Group", "Response"),
                   "Proportion of Responses")   

assimilation_vectors <-
  bind_rows(assimilation_vectors_fr, assimilation_vectors_en)

install.packages('extrafont')
library(extrafont)
font_import(path = "analysis/", pattern = ".ttf")

# original graph
assimilation_vectors_plot_og <- ggplot(assimilation_vectors,
       aes(y=`Phone (Language)`, x=Response, fill=`Proportion of Responses`)) +
  geom_tile(colour="#333333") +
  facet_grid(~ `Listener Group`, scales="free_x") +
  scale_fill_viridis_c(limits=c(0,1), option="B") +
  scale_y_discrete(limits=rev) +
  theme_bw() +
  theme(legend.position = "bottom",
        text=element_text(family="Doulos SIL", size=12),
        axis.text=element_text(size=10, colour="black"),
        legend.text=element_text(size=12),
        legend.box.spacing = unit(0, "inches")) +
  guides(fill=guide_colourbar(title.position="left",
                              title.vjust=1,
                              barwidth=unit(1.5, "inches"),
                              barheight=unit(0.15, "inches"),
                              frame.colour="black",
                              ticks.colour="black"))
# just english:
assimilation_vectors_plot_english <- ggplot(filter(assimilation_vectors, `Listener Group` == 'English')%>% mutate(`Listener Group`=str_replace_all(`Listener Group`, 'English', 'Human (English)'),),
                                    aes(x=`Phone (Language)`, y=Response, fill=`Proportion of Responses`)) +
  geom_tile(colour="#333333") +
  facet_grid(~ `Listener Group`, scales="free_x") +
  scale_fill_viridis_c(limits=c(0,1), option="B") +
  scale_y_discrete(limits=rev) +
  theme_bw() +
  theme(legend.position = "right",
        text=element_text(family="Doulos SIL", size=12),
        axis.text=element_text(size=16, colour="black"),
        axis.text.x=element_text(angle=90, hjust=0.95,vjust=0.2),
        legend.text=element_text(size=16),
        legend.box.spacing = unit(0.5, "inches"),
        axis.title=element_text(size=25),
        strip.text.x = element_text(size = 20),
        legend.title=element_text(size=20)) +
  guides(fill=guide_colourbar(title.position="top",
                              title.vjust=1,
                              barwidth=unit(0.15, "inches"),
                              barheight=unit(1.5, "inches"),
                              frame.colour="black",
                              ticks.colour="black"))

# just french:
assimilation_vectors_plot_french <- ggplot(
  filter(assimilation_vectors, `Listener Group` == 'French') %>%
    mutate(
      Phone = str_extract(`Phone (Language)`, "^[^()]+"), # Extract only the Phone
      Language = str_extract(`Phone (Language)`, "(?<=\\().+?(?=\\))") # Extract only the Language
    ),
  aes(x=Phone, y=Response, fill=`Proportion of Responses`)
) +
  geom_tile(colour="#333333") +
  # Facet by both `Listener Group` and `Language`
  facet_grid(`Listener Group` ~ Language, scales="free_x") +
  scale_fill_viridis_c(limits=c(0,1), option="B") +
  scale_y_discrete(limits=rev) +
  theme_bw() +
  theme(legend.position = "right",
        text=element_text(family="Doulos SIL", size=12),
        axis.text=element_text(size=14, colour="black"),
        axis.text.x=element_text(angle=90, hjust=0.95, vjust=0.2),
        legend.text=element_text(size=16),
        legend.box.spacing = unit(0.5, "inches"),
        axis.title=element_text(size=25),
        strip.text.x = element_text(size = 20),
        legend.title=element_text(size=20)) +
  guides(fill=guide_colourbar(title.position="top",
                              title.vjust=1,
                              barwidth=unit(0.15, "inches"),
                              barheight=unit(1.5, "inches"),
                              frame.colour="black",
                              ticks.colour="black"))


print(assimilation_vectors_plot_french)

# mfcc french classification vectors
mfcc_data <- read_csv(
  MFCC_DATA
) %>%
  rename(`Machine Response` = 'Response') %>%
  mutate(`Listener Group` = 'MFCC') %>%
  mutate(`Phone (Language)` = str_replace_all(`Phone (Language)`, 'de-mun', 'de-muc')) %>%
  # Extract language from the `Phone (Language)` column
  mutate(Phone = str_extract(`Phone (Language)`, "^[^()]+"),
         Language = str_extract(`Phone (Language)`, "(?<=\\().+?(?=\\))")) %>%
  mutate(Language = factor(Language, levels = c("de", "de-muc", "en-us", "et", "fr", "pt-br", "tr")))

mfcc_plot <- ggplot(mfcc_data, 
                    aes(x=Phone, y=`Machine Response`, fill=value)) +
  geom_tile(colour="#333333") +
  scale_fill_viridis_c(limits=c(0,1), option="B") +
  scale_y_discrete(limits=rev) +
  theme_bw() +
  facet_grid(`Listener Group` ~ Language, scales="free_x") +
  theme(legend.position = "right",
        text=element_text(family="Doulos SIL", size=12),
        axis.text=element_text(size=14, colour="black"),
        axis.text.x=element_text(angle=90, hjust=0.95, vjust=0.2),
        legend.text=element_text(size=16),
        legend.box.spacing = unit(0.5, "inches"),
        axis.title=element_text(size=25),
        strip.text.x = element_text(size = 20),
        legend.title=element_text(size=20)) +
  labs(fill = "Proportion of Responses") +
  guides(fill=guide_colourbar(title.position="top",
                              title.vjust=1,
                              barwidth=unit(0.15, "inches"),
                              barheight=unit(1.5, "inches"),
                              frame.colour="black",
                              ticks.colour="black"))

print(mfcc_plot)

mfcc_human_corr <- dplyr::left_join(assimilation_vectors_fr, mfcc_data, by=c(Response="Machine Response", "Phone (Language)")) %>% with(., cor(`Proportion of Responses`, value))
cor((MFCC_DELTAS$bin_user_ans+1)/2, MFCC_DELTAS$mfcc_raw_delta, method = "pearson")
cor((MFCC_DELTAS$bin_user_ans+1)/2, MFCC_DELTAS$mfcc_delta, method = "pearson")

grouped_corr <- MFCC_DELTAS %>%
  group_by(phone_X) %>%  # Replace 'subject_language' with your column of interest
  summarize(group_corr = cor((bin_user_ans + 1) / 2, mfcc_raw_delta, method = "spearman"))

# Calculate the mean of all group correlations
mean_corr <- mean(grouped_corr$group_corr)


#wav2vec2.0 french
wav2vec_data <- read_csv(
  W2V_DATA
) %>%
  rename(`Machine Response` = 'Response') %>%
  mutate(`Listener Group` = 'wav2vec 2.0') %>%
  mutate(`Phone (Language)` = str_replace_all(`Phone (Language)`, 'de-mun', 'de-muc')) %>%
  # Extract language from the `Phone (Language)` column
  mutate(Phone = str_extract(`Phone (Language)`, "^[^()]+"),
         Language = str_extract(`Phone (Language)`, "(?<=\\().+?(?=\\))")) %>%
  mutate(Language = factor(Language, levels = c("de", "de-muc", "en-us", "et", "fr", "pt-br", "tr")))

wav2vec_plot <- ggplot(wav2vec_data, 
                    aes(x=Phone, y=`Machine Response`, fill=value)) +
  geom_tile(colour="#333333") +
  scale_fill_viridis_c(limits=c(0,1), option="B") +
  scale_y_discrete(limits=rev) +
  theme_bw() +
  facet_grid(`Listener Group` ~ Language, scales="free_x") +
  theme(legend.position = "right",
        text=element_text(family="Doulos SIL", size=12),
        axis.text=element_text(size=14, colour="black"),
        axis.text.x=element_text(angle=90, hjust=0.95, vjust=0.2),
        legend.text=element_text(size=16),
        legend.box.spacing = unit(0.5, "inches"),
        axis.title=element_text(size=25),
        strip.text.x = element_text(size = 20),
        legend.title=element_text(size=20)) +
  labs(fill = "Proportion of Responses") +
  guides(fill=guide_colourbar(title.position="top",
                              title.vjust=1,
                              barwidth=unit(0.15, "inches"),
                              barheight=unit(1.5, "inches"),
                              frame.colour="black",
                              ticks.colour="black"))

print(wav2vec_plot)
wav2vec_human_corr <- dplyr::left_join(assimilation_vectors_fr, wav2vec_data, by=c(Response="Machine Response", "Phone (Language)")) %>% with(., cor(`Proportion of Responses`, value))
cor((W2V_DELTAS$bin_user_ans+1)/2, W2V_DELTAS$wav2vec_raw_delta, method = "spearman")
cor((W2V_DELTAS$bin_user_ans+1)/2, W2V_DELTAS$wav2vec_delta, method = "spearman")

w2v_human_corr <- dplyr::left_join(assimilation_vectors_en, w2v_data, by=c(Response="Machine Response", "Phone (Language)")) %>% with(., cor(`Proportion of Responses`, value))


human_and_mfcc <- left_join(assimilation_vectors_fr, mfcc_data, by = join_by(`Response` == `Machine Response`, `Phone (Language)`), suffix = c(".human", ".mfcc")) %>% 
  rename(Proportion.machine=value, Proportion.human=`Proportion of Responses`) %>%
  mutate(Language = factor(Language, levels = c("de", "de-muc", "en-us", "et", "fr", "pt-br", "tr"))) %>% 
  mutate(`MFCC minus Human` = `Proportion.machine`- `Proportion.human`, `Minus Label` = 'MFCC minus Human (French)')

human_minus_mfcc_plot <- ggplot(human_and_mfcc, 
                  aes(x=`Phone`, y=`Response`, fill=`MFCC minus Human`)) +
  geom_tile(colour="#333333") +
  scale_fill_gradient2(limits=c(-1,1), low="purple", mid="black", high="orange") +
  scale_y_discrete(limits=rev) +
  theme_bw() +
  facet_grid(~ `Language`, scales="free_x") +
  theme(legend.position = "right",
        text=element_text(family="Doulos SIL", size=12),
        axis.text=element_text(size=16, colour="black"),
        axis.text.x=element_text(angle=90, hjust=0.95,vjust=0.2),
        legend.text=element_text(size=16),
        legend.box.spacing = unit(0.5, "inches"),
        axis.title=element_text(size=25),
        strip.text.x = element_text(size = 20),
        legend.title=element_text(size=20)) +
  labs(fill = "Machine minus Human") +
  guides(fill=guide_colourbar(title.position="top",
                              title.vjust=1,
                              barwidth=unit(0.15, "inches"),
                              barheight=unit(1.5, "inches"),
                              frame.colour="black",
                              ticks.colour="black"))

print(human_minus_mfcc_plot)

#wav2vec subtraction
human_and_wav2vec <- left_join(assimilation_vectors_fr, wav2vec_data, by = join_by(`Response` == `Machine Response`, `Phone (Language)`), suffix = c(".human", ".wav2vec")
) %>% rename(Proportion.machine=value, Proportion.human=`Proportion of Responses`
) %>% mutate(Language = factor(Language, levels = c("de", "de-muc", "en-us", "et", "fr", "pt-br", "tr"))
) %>% mutate(`wav2vec2.0 minus Human` = `Proportion.machine`- `Proportion.human`,`Minus Label` = 'wav2vec2.0 minus Human (French)') 

human_minus_wav2vec_plot <- ggplot(human_and_wav2vec, 
                                aes(x=`Phone`, y=`Response`, fill=`wav2vec2.0 minus Human`)) +
  geom_tile(colour="#333333") +
  scale_fill_gradient2(limits=c(-1,1), low="purple", mid="black", high="orange") +
  scale_y_discrete(limits=rev) +
  theme_bw() +facet_grid(~ `Language`, scales="free_x") +
  theme(legend.position = "right",
        text=element_text(family="Doulos SIL", size=12),
        axis.text=element_text(size=16, colour="black"),
        axis.text.x=element_text(angle=90, hjust=0.95,vjust=0.2),
        legend.text=element_text(size=16),
        legend.box.spacing = unit(0.5, "inches"),
        axis.title=element_text(size=25),
        strip.text.x = element_text(size = 20),
        legend.title=element_text(size=20)) +
  labs(fill = "Machine minus Human") +
  guides(fill=guide_colourbar(title.position="top",
                              title.vjust=1,
                              barwidth=unit(0.15, "inches"),
                              barheight=unit(1.5, "inches"),
                              frame.colour="black",
                              ticks.colour="black"))

print(human_minus_wav2vec_plot)
