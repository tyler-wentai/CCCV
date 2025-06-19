library(brms)
library(dplyr)
library(ggplot2)

# teleconnection threshold analysis

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_NINO3type2_square4.csv'
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
dat_help <- subset(dat, psi > 0.6)
sum(dat_help$conflict_count)


###
dat <- dat %>%
  arrange(loc_id, year) %>%        # order each loc_id by year
  group_by(loc_id) %>%             # then group
  mutate(
    cindex_lagF1y = lead(cindex_lag0y) # forward lag: next year’s your_column
    # prev_value = lag(your_column)  # backward lag: last year’s your_column
  ) %>%
  ungroup()  %>%
  filter(!is.na(cindex_lagF1y))
View(dat)

########
test <- subset(dat, conflict_count > 0)
quantile(test$psi, c(0.018, 0.980))
nrow(subset(test, psi<=0.08361459))

ecdf_fun <- ecdf(unique_psi$psi)
q <- ecdf_fun(2.9)




ecdf_fun <- ecdf(unique_psi$psi)
min.psi <- 0.08361459
max.psi <- 0.99923759
psi_quantile_seq <- seq(ecdf_fun(min.psi), ecdf_fun(max.psi), 0.01)
psi_start_seq <- quantile(unique_psi$psi, psi_quantile_seq)

library(broom)
library(dplyr)

results <- tibble(
  bin          = integer(),      # loop index or any id you like
  psi_threshold = numeric(),      # lower edge of the psi bin
  psi_quantile = numeric(),
  estimate     = numeric(),      # point estimate
  conf.low     = numeric(),      # 2.5 % CI
  conf.high    = numeric(),       # 97.5 % CI
  nconflicts   = numeric()
)

for (i in 1:length(psi_start_seq)) {
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
      cindex_lagF1y   = first(cindex_lagF1y),
      cindex_lag0y   = first(cindex_lag0y),
      cindex_lag1y   = first(cindex_lag1y),
      cindex_lag2y   = first(cindex_lag2y),
      .groups = "drop"
    )
  
  #dat_agg$cindex_lag0y[73] <-  1.96000000
  #dat_agg$cindex_lag1y[73] <- -0.73333333
  
  pos.mod <- glm(
    conflict_count ~ cindex_lagF1y + year + bool1989,
    family = poisson(link = "log"),
    data   = dat_agg
  )
  
  
  ## tidy() gives one row per coefficient, with CI if requested
  coef_row <- tidy(pos.mod, conf.int = TRUE, conf.level = 0.90) %>%
    filter(term == "cindex_lagF1y") %>%            # keep only the β you want
    mutate(
      bin     = i,
      psi_threshold = psi_start_seq[i],
      psi_quantile  = ecdf_fun(psi_start_seq[i]),
      nconflicts    = sum(dat_agg$conflict_count)
    ) %>%
    select(bin, psi_threshold, psi_quantile, estimate, conf.low, conf.high, nconflicts)
  
  results <- bind_rows(results, coef_row)
}

library(ggplot2)
ggplot(results, aes(x = psi_quantile, y = estimate)) +
  geom_line() +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +
  labs(
    x = "ψ bin start",
    y = "β̂ for cindex_lag0y (log-scale)",
    title = "Effect of cindex_lag0y across ψ slices"
  )

write.csv(results, "/Users/tylerbagwell/Desktop/psithreshold_analysis/data/PsiThreshold_Greater_cindex_lag0y_Onset_Count_Global_ANI_square4_ci90.csv", row.names = FALSE)