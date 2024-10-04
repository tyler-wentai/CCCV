library(brms)
library(tictoc)
library(dplyr)
library(ggplot2)

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_data_ASIA_binary.csv'
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


max(dat$psi)


###### BAYESIAN FITS
# linear model
tic("Brms Model Fitting")
fit1 <- brm(
  conflict_binary ~ 1 + conflict_binary_lag1y +
    INDEX_lag0y + I(INDEX_lag0y*psi) +
    INDEX_lag1y + I(INDEX_lag1y*psi) +
    INDEX_lag2y + I(INDEX_lag2y*psi) +
    SOVEREIGNT,
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
  conflict_binary ~ year +
    INDEX_lag0y + I(INDEX_lag0y*psi) +
    INDEX_lag1y + I(INDEX_lag1y*psi) +
    INDEX_lag2y + I(INDEX_lag2y*psi) +
    INDEX_lag3y + I(INDEX_lag3y*psi) +
    SOVEREIGNT,
  data = dat, family = bernoulli(link = "logit"), 
  iter = 4000, chains=1, warmup=1000,
  prior = prior(normal(0, 10), class = b)
)
toc()

print(summary(fit2), digits = 4)
#plot(fit2)



draws_matrix <- as_draws_matrix(fit2)
colnames(draws_matrix)
psi_help <- seq(min(dat$psi), max(dat$psi), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(psi_help)){
  psi_i <- psi_help[i]
  sum_params <- draws_matrix[, "b_INDEX_lag0y"] + psi_i*draws_matrix[, "b_IINDEX_lag0yMUpsi"] +
    draws_matrix[, "b_INDEX_lag1y"] + psi_i*draws_matrix[, "b_IINDEX_lag1yMUpsi"] +
    draws_matrix[, "b_INDEX_lag2y"] + psi_i*draws_matrix[, "b_IINDEX_lag2yMUpsi"]
  results <- rbind(results, c(psi_i,
                              mean(sum_params), 
                              sd(sum_params),
                              quantile(sum_params, 0.025),
                              quantile(sum_params, 0.975)))
  
}
plot(results[,1], results[,2], type='l', col='black', ylim=c(min(results[,4]),max(results[,5])))
lines(results[,1], results[,4], type='l', col='blue')
lines(results[,1], results[,5], type='l', col='blue')
abline(h=0)





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


# negative binomial model
panel_data_path <- '/Users/tylerbagwell/Desktop/panel_data_ASIA_count.csv'
dat <- read.csv(panel_data_path)

dat$SOVEREIGNT <- as.factor(dat$SOVEREIGNT)
dat$loc_id <- as.factor(dat$loc_id)
dat$year <- dat$year - min(dat$year)

colnames(dat)

test = subset(dat, conflict_count<20)

hist(test$conflict_count)

tic("nb_model Fitting")
nb_model <- brm(
  formula = conflict_count ~ SOVEREIGNT + 
    INDEX_lag0y + I(INDEX_lag0y*psi) + 
    INDEX_lag1y + I(INDEX_lag1y*psi) +
    INDEX_lag2y + I(INDEX_lag2y*psi),
  data = dat,
  family = negbinomial(link = "log", link_shape = "log"),
  prior = c(
    set_prior("normal(0, 10)", class = "b"),  # Priors for coefficients
    set_prior("gamma(0.01, 0.01)", class = "shape")  # Prior for dispersion
  ),
  chains = 1,  # Number of Markov chains
  cores = parallel::detectCores(),  # Utilize all available cores
  iter = 4000,  # Number of iterations per chain
  warmup = 1000,  # Number of warmup iterations
)
toc()

print(summary(nb_model), digits = 4)
plot(nb_model)



draws_matrix <- as_draws_matrix(nb_model)
colnames(draws_matrix)
psi_help <- seq(min(dat$psi), max(dat$psi), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(psi_help)){
  psi_i <- psi_help[i]
  sum_params <- draws_matrix[, "b_INDEX_lag2y"] + psi_i*draws_matrix[, "b_IINDEX_lag2yMUpsi"]
  results <- rbind(results, c(psi_i,
                              mean(sum_params), 
                              sd(sum_params),
                              quantile(sum_params, 0.025),
                              quantile(sum_params, 0.975)))
  
}
plot(results[,1], results[,2], type='l', col='black', ylim=c(min(results[,4]),max(results[,5])))
lines(results[,1], results[,4], type='l', col='blue')
lines(results[,1], results[,5], type='l', col='blue')
abline(h=0)







