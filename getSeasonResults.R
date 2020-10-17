library(dplyr)
library(tidyverse)

# pbp <- read.csv('C:/Users/Bill/Desktop/test/data/play_by_play_2017.csv')



season_score_table <- data.table::data.table(season = integer(),
                                             week = integer(),
                                             homeTeam = character(),
                                             awayTeam = character(),
                                             awayDefPoints = integer(),
                                             homeDefPoints = integer(),
                                             awayOffPoints = integer(),
                                             homeOffPoints = integer())

regSeason <- pbp %>% filter(season_type == "REG")
# targets are year, week, home/away off/def points
for (y in 1999:2019){
  pbp <- read.csv(paste0('C:/Users/Bill/Desktop/test/data/play_by_play_', y, '.csv'))
  for (w in 1:17) {
    currWeek <- regSeason %>% filter(week == w)
    # SHOULD ST TDs BE INCLUDED
    for (t in unique(currWeek$home_team)) {
      currGame <- currWeek %>% filter(home_team == t)
      if (currGame %>% nrow() == 0) {next}
      homeTeam <- as.character(currGame$home_team[1])
      awayTeam <- as.character(currGame$away_team[1])
      awayDefPoints <- 7 * (currGame %>% filter(touchdown == 1 & posteam != td_team & 
                                                  posteam == homeTeam) %>% nrow())
      homeDefPoints <- 7 * (currGame %>% filter(touchdown == 1 & posteam != td_team & 
                                                  posteam == awayTeam) %>% nrow())
      awayOffPoints <- 7 * (currGame %>% filter(touchdown == 1 & posteam == td_team & 
                                                  posteam == awayTeam) %>% nrow())
      homeOffPoints <- 7 * (currGame %>% filter(touchdown == 1 & posteam == td_team & 
                                                  posteam == homeTeam) %>% nrow())
      awayOffPoints = awayOffPoints + (3 * (currGame %>% filter(field_goal_attempt == 1 
                                                                & posteam_score_post - posteam_score == 3 & posteam == awayTeam) %>% nrow()))
      homeOffPoints = homeOffPoints + (3 * (currGame %>% filter(field_goal_attempt == 1 
                                                                & posteam_score_post - posteam_score == 3 & posteam == homeTeam) %>% nrow()))
      #season_score_table %>% add_row(season = 2017, week = w, homeTeam = homeTeam, 
      #                              awayTeam = awayTeam, awayDefPoints = awayDefPoints, 
      #                             homeDefPoints = homeDefPoints, awayOffPoints = awayOffPoints,
      #                            homeOffPoints = homeOffPoints)
      currRow <- c(y, w, homeTeam, awayTeam, awayDefPoints, homeDefPoints,
                   awayOffPoints, homeOffPoints)
      print(currRow)
      season_score_table[nrow(season_score_table) + 1,] <- currRow
      # how do we want to account for ST points??
    }
  }
}

write.csv(season_score_table, "allSeasonScores.csv")