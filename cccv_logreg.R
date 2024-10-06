library(brms)
library(tictoc)
library(dplyr)
library(ggplot2)

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_data_Africa_binary.csv'
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



psi <- 1.2
climindex <- seq(min(dat$INDEX_lag0y), max(dat$INDEX_lag0y), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(climindex)){
  climind <- climindex[i]
  sum_params <- (climind*draws_matrix[, "b_INDEX_lag0y"]) + (psi*climind*draws_matrix[, "b_IINDEX_lag0yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag0yMUpsiE2"]) +
    (climind*draws_matrix[, "b_INDEX_lag1y"]) + (psi*climind*draws_matrix[, "b_IINDEX_lag1yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag1yMUpsiE2"]) +
    (climind*draws_matrix[, "b_INDEX_lag2y"]) + (psi*climind*draws_matrix[, "b_IINDEX_lag2yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag2yMUpsiE2"]) +
    (climind*draws_matrix[, "b_INDEX_lag3y"]) + (psi*climind*draws_matrix[, "b_IINDEX_lag3yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag3yMUpsiE2"])
  results <- rbind(results, c(climind,
                              mean(sum_params), 
                              sd(sum_params),
                              quantile(sum_params, 0.025),
                              quantile(sum_params, 0.975)))
  
}
plot(results[,1], results[,2], type='l', col='black', ylim=c(-0.1,3), lwd=2)
lines(results[,1], results[,4], type='l', col='blue', lwd=2)
lines(results[,1], results[,5], type='l', col='blue', lwd=2)
abline(h=0)





psi <- 1.20
climindex <- seq(min(dat$INDEX_lag0y), max(dat$INDEX_lag0y), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(climindex)){
  climind <- climindex[i]
  sum_params <- exp((climind*draws_matrix[, "b_INDEX_lag0y"]) + (psi*climind*draws_matrix[, "b_IINDEX_lag0yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag0yMUpsiE2"]) +
    (climind*draws_matrix[, "b_INDEX_lag1y"]) + (psi*climind*draws_matrix[, "b_IINDEX_lag1yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag1yMUpsiE2"]) +
    (climind*draws_matrix[, "b_INDEX_lag2y"]) + (psi*climind*draws_matrix[, "b_IINDEX_lag2yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag2yMUpsiE2"]) +
    (climind*draws_matrix[, "b_INDEX_lag3y"]) + (psi*climind*draws_matrix[, "b_IINDEX_lag3yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag3yMUpsiE2"]))
  results <- rbind(results, c(climind,
                              mean(sum_params), 
                              sd(sum_params),
                              quantile(sum_params, 0.025),
                              quantile(sum_params, 0.975)))
  
}
plot(results[,1], results[,2], type='l', col='black', ylim=c(0.9,5), lwd=2)
lines(results[,1], results[,4], type='l', col='blue', lwd=2)
lines(results[,1], results[,5], type='l', col='blue', lwd=2)
abline(h=1)



