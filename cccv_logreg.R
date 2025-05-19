library(brms)
library(dplyr)
library(ggplot2)

# teleconnection threshold analysis

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_DMItype2_square4.csv'
dat <- read.csv(panel_data_path)

sum(dat$conflict_count)
unique(dat$conflict_count)
#plot(subset(dat, loc_id=='loc_1')$cindex_lag0, type='l')
#head(dat)
dat <- subset(dat, year!=1989)
dat$bool1989 <- ifelse(dat$year<1989,0,1)
dat$year <- dat$year - min(dat$year)
dat$loc_id <- as.factor(dat$loc_id)

unique_psi <- dat %>%
  group_by(loc_id) %>%
  summarise(
    psi = first(psi),
    total_conflict_counts = sum(conflict_count))
hist(unique_psi$psi, breaks='scott')
hist(unique_psi$total_conflict_counts, breaks='scott')


quantile(unique_psi$psi, c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.925))
dat_help <- subset(dat, psi < 0.5)
sum(dat_help$conflict_count)


dpsi <- 0.15
min.psi <- min(dat$psi)
max.psi <- max(dat$psi)
npsi <- 10
psi_start_seq <- seq(min.psi, max.psi, length.out=npsi)

for (i in 1:(npsi-2)){
  print(paste0("...", i))
  dat_help <- subset(dat, psi >= psi_start_seq[i] & psi < psi_start_seq[i+1])
  
  dat_agg <- dat_help %>%
    group_by(year) %>%
    summarise(
      conflict_count = sum(conflict_count),
      bool1989 = first(bool1989), 
      cindex_lag0y = first(cindex_lag0y), 
      cindex_lag1y = first(cindex_lag1y), 
      cindex_lag2y = first(cindex_lag2y)
    )
  dat_agg$cindex_lag0y[73] <- +1.96000000
  dat_agg$cindex_lag1y[73] <- -0.73333333
  
  pos.mod <- glm(
    conflict_count ~ cindex_lag0y + year + bool1989,
    family = poisson(link = "log"),
    data   = dat_agg
  )
  
  print(pos.mod)
}



########
test <- subset(dat, conflict_count > 0)
quantile(test$psi, c(0.8, 0.95))
nrow(subset(test, psi<0.04))




min.psi <- min(dat$psi)
max.psi <- max(dat$psi)
npsi <- 100
psi_start_seq <- seq(0.04, 0.75, length.out=npsi)



library(broom)
library(dplyr)

results <- tibble(
  bin          = integer(),      # loop index or any id you like
  psi_min      = numeric(),      # lower edge of the psi bin
  psi_max      = numeric(),      # upper edge of the psi bin
  estimate     = numeric(),      # β̂
  conf.low     = numeric(),      # 2.5 % CI
  conf.high    = numeric()       # 97.5 % CI
)

for (i in 1:(npsi - 1)) {
  message("… ", i)
  
  dat_help <- subset(
    dat,
    psi <= psi_start_seq[i]
  )
  
  dat_agg <- dat_help %>%
    group_by(year) %>%
    summarise(
      conflict_count = sum(conflict_count),
      bool1989       = first(bool1989),
      cindex_lag0y   = first(cindex_lag0y),
      cindex_lag1y   = first(cindex_lag1y),
      cindex_lag2y   = first(cindex_lag2y),
      .groups = "drop"
    )
  
  #dat_agg$cindex_lag0y[73] <-  1.96000000
  #dat_agg$cindex_lag1y[73] <- -0.73333333
  
  pos.mod <- glm(
    conflict_count ~ cindex_lag0y + I(cindex_lag0y^2) + year + bool1989,
    family = poisson(link = "log"),
    data   = dat_agg
  )
  
  
  ## tidy() gives one row per coefficient, with CI if requested
  coef_row <- tidy(pos.mod, conf.int = TRUE, conf.level = 0.90) %>%
    filter(term == "I(cindex_lag0y^2)") %>%            # keep only the β you want
    mutate(
      bin     = i,
      psi_min = psi_start_seq[i],
      psi_max = psi_start_seq[i + 1]
    ) %>%
    select(bin, psi_min, psi_max, estimate, conf.low, conf.high)
  
  results <- bind_rows(results, coef_row)
}

library(ggplot2)
ggplot(results, aes(x = psi_min, y = estimate)) +
  geom_line() +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +
  labs(
    x = "ψ bin start",
    y = "β̂ for cindex_lag0y (log-scale)",
    title = "Effect of cindex_lag0y across ψ slices"
  )

