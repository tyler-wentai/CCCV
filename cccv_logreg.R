panel_data_path <- '/Users/tylerbagwell/Desktop/panel_data_AFRICA_binary.csv'
dat <- read.csv(panel_data_path)

View(dat)

colnames(dat)

dat$SOVEREIGNT <- as.factor(dat$SOVEREIGNT)
dat$loc_id <- as.factor(dat$loc_id)
dat$year <- dat$year - min(dat$year)


mod <- lm('conflict_binary ~ I(INDEX_lag0y*psi) + I((INDEX_lag0y*psi)^2) +
          I(INDEX_lag1y*psi) + I((INDEX_lag1y*psi)^2) +
          loc_id + loc_id*year', data=dat)
summary(mod)

hist(dat$conflict_binary)