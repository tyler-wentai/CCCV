library(brms)
library(tictoc)
library(dplyr)
library(ggplot2)

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_data_AFRICA_binary.csv'
dat <- read.csv(panel_data_path)

#View(dat)
colnames(dat)

dat$SOVEREIGNT <- as.factor(dat$SOVEREIGNT)
dat$loc_id <- as.factor(dat$loc_id)
dat$year <- dat$year - min(dat$year)


###### BAYESIAN FITS
# bernoulli model
tic("Brms Model Fitting")
fit1 <- brm(
  conflict_binary ~ conflict_binary_lag1y +
    INDEX_lag0y + I(INDEX_lag0y*psi) + I((INDEX_lag0y*psi)^2) +
    INDEX_lag1y + I(INDEX_lag1y*psi) + I((INDEX_lag1y*psi)^2) +
    INDEX_lag2y + I(INDEX_lag2y*psi) + I((INDEX_lag2y*psi)^2) +
    INDEX_lag3y + I(INDEX_lag3y*psi) + I((INDEX_lag3y*psi)^2) +
    year + SOVEREIGNT + year:SOVEREIGNT,
  data = dat, family = bernoulli(link = "logit"), 
  iter = 4000, chains=1, warmup=1000,
  prior = prior(normal(0, 10), class = b)
)
toc()

print(summary(fit1), digits = 4)
#plot(fit1)


draws_matrix <- as_draws_matrix(fit1)
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
plot(results[,1], results[,2], type='l', col='black', ylim=c(min(results[,4]),max(results[,5])), lwd=2)
lines(results[,1], results[,4], type='l', col='blue', lwd=2)
lines(results[,1], results[,5], type='l', col='blue', lwd=2)
abline(h=0)

