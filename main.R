# Tara Farwana, Heena Mohammed, Annie Henderson, Graham Harwood
library(readxl)

min_year <- 1995
interval <- 5
file_dir <- "./Country_and_Territory_Ratings_and_Statuses_FIW1973-2021.xlsx"
country_stats <- data.frame(read_excel(file_dir, sheet = "Historical distribution"))
five_year_periods <- country_stats[ !is.na(country_stats$Year.s..Under.Review..), ]
five_year_periods$Year.s..Under.Review.. <- as.numeric(lapply(five_year_periods$Year.s..Under.Review.., function(x) {tail(str_split(x, ","), n = 1 ) }) )
five_year_periods <- five_year_periods[five_year_periods$Year]
stringy <- "Jan 1, 2003-Nov 30, 2003"
parts <- tail(strsplit(stringy, ","), 1)
