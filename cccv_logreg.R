library(brms)
library(tictoc)

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_data_AFRICA_binary.csv'
dat <- read.csv(panel_data_path)

#View(dat)

colnames(dat)

dat$SOVEREIGNT <- as.factor(dat$SOVEREIGNT)
dat$loc_id <- as.factor(dat$loc_id)
dat$year <- dat$year - min(dat$year)


mod <- lm('conflict_binary ~ I(INDEX_lag0y*psi) + I((INDEX_lag0y*psi)^2) +
          I(INDEX_lag1y*psi) + I((INDEX_lag1y*psi)^2) +
          I(INDEX_lag2y*psi) + I((INDEX_lag2y*psi)^2) +
          loc_id - 1', data=dat)
summary(mod)

hist(dat$conflict_binary)



###### BAYESIAN FITS
# linear model
tic("Brms Model Fitting")
fit1 <- brm(
  conflict_binary ~ 0 + ar(time = year, gr = loc_id, p = 1, cov = FALSE) +
    I(INDEX_lag0y*psi) + I((INDEX_lag0y*psi)^2) +
    I(INDEX_lag1y*psi) + I((INDEX_lag1y*psi)^2) +
    I(INDEX_lag2y*psi) + I((INDEX_lag2y*psi)^2) +
    loc_id,
  data = dat, family = gaussian(), 
  iter = 5000, chains=1, warmup=1000,
  prior = prior(normal(0, 10), class = b) + 
    prior(cauchy(0, 2), class = sigma)
)
toc()

print(summary(fit1), digits = 4)
plot(fit1)

# bernoulli model
tic("Brms Model Fitting")
fit2 <- brm(
  conflict_binary ~ 0 + ar(time = year, gr = loc_id, p = 1, cov = FALSE) +
    I(INDEX_lag0y*psi) + I((INDEX_lag0y*psi)^2) +
    I(INDEX_lag1y*psi) + I((INDEX_lag1y*psi)^2) +
    I(INDEX_lag2y*psi) + I((INDEX_lag2y*psi)^2) +
    loc_id,
  data = dat, family = bernoulli(link = "logit"), 
  iter = 5000, chains=1, warmup=1000,
  prior = prior(normal(0, 10), class = b)
)
toc()

print(summary(fit2), digits = 4)
plot(fit2)


tic("Brms Model Fitting")
fit2 <- brm(
  conflict_binary ~ 0 + ar(time = year, gr = loc_id, p = 1, cov = FALSE) +
    I(INDEX_lag0y*psi) + I((INDEX_lag0y*psi)^2) +
    I(INDEX_lag1y*psi) + I((INDEX_lag1y*psi)^2) +
    I(INDEX_lag2y*psi) + I((INDEX_lag2y*psi)^2) +
    + (1 | loc_id),
  data = dat, family = bernoulli(link = "logit"), 
  iter = 5000, chains=1, warmup=1000,
  prior = prior(normal(0, 10), class = b)
)
toc()

print(summary(fit2), digits = 4)
plot(fit2)


