fix_phone_language_codes <- function(x) {
  isoish_names <- c(FR="fr", BR="pt", EN="en", GR="de-m",
                    GL="de", ES="et", TU="tr")
  isoish_names[x]
}

full_phone_languages <- function(x) {
  language_names <- c(FR="Hexagonal French", BR="Brazilian Portuguese",
                      EN="American English", GR="Munich German",
                    GL="Standard German", ES="Estonian", TU="Turkish")
  language_names[x]
}

get_phone1 <- function(a, b) ifelse(a < b, a, b)
get_phone2 <- function(a, b) ifelse(a < b, b, a)
contrast_label <- function(p1, p2)  paste0(p1, "–", p2)