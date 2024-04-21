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