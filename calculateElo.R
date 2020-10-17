library(tidyverse)
library(dplyr)

#source: http://andr3w321.com/elo-ratings-part-2-margin-of-victory-adjustments/

winProbability <- function(eloA, eloB) {
  diff = eloA - eloB
  p = 1 - 1 / (1 + 10 ** (diff / 400.0))
  return(p)
}
rate_1vs1 <- function(p1, p2, mov=1, k=20, drawn=F){
  k_multiplier = log(abs(mov) + 1)
  corr_m = 2.2 / ((p1 - p2)*.001 + 2.2)
  rp1 = 10 ** (p1/400)
  rp2 = 10 ** (p2/400)
  exp_p1 = rp1 /(rp1 + rp2)
  exp_p2 = rp2 /(rp1 + rp2)
  if (drawn == T){
    s1 = 0.5
    s2 = 0.5
  }
  else{
    s1 = 1
    s2 = 0
  }
  new_p1 = p1 + k_multiplier * corr_m * k * (s1 - exp_p1)
  new_p2 = p2 + k_multiplier * corr_m * k * (s2 - exp_p2)
  return(c(new_p1, new_p2))
}

df <- read.csv('allSeasonScores.csv')
print(unique(df$homeTeam))
