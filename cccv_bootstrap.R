library(brms)
library(tictoc)
library(dplyr)
library(ggplot2)

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_datasets/Binary_Africa_NINO3_square2_CON1_notrend.csv'
dat <- read.csv(panel_data_path)

#View(dat)
colnames(dat)

dat$SOVEREIGNT <- as.factor(dat$SOVEREIGNT)
dat$loc_id <- as.factor(dat$loc_id)
dat$tropical_year <- dat$tropical_year - min(dat$tropical_year)



dat_l   <- subset(dat, psi < 0.35)
dat_m   <- subset(dat, psi >= 1 & psi <= 2)
dat_h   <- subset(dat, psi > 2)


reg <- glm(conflict_binary ~ conflict_binary_lag1y + 
             I(psi*INDEX_lagF1y) + I((psi*INDEX_lagF1y)^2) + 
             I(psi*INDEX_lag0y) + I((psi*INDEX_lag0y)^2) + 
             I(psi*INDEX_lag1y) + I((psi*INDEX_lag1y)^2) +
             poly(t2m_lagF1y, 1) + poly(tp_lagF1y, 1) +
             poly(t2m_lag0y, 1) + poly(tp_lag0y, 1) +
             poly(t2m_lag1y, 1) + poly(tp_lag1y, 1) +
             tropical_year + loc_id,
           data = dat,
           family = binomial)
summary(reg)

library(boot)


formula <- as.formula('conflict_binary ~ conflict_binary_lag1y + 
             I(psi*INDEX_lag0y) + I((psi*INDEX_lag0y)^2) + 
             I(psi*INDEX_lag1y) + I((psi*INDEX_lag1y)^2) + 
             t2m_lag0y + tp_lag0y +
             t2m_lag1y + tp_lag1y +
             tropical_year + loc_id')

model <- glm(formula = formula,
           data = dat,
           family = binomial)
summary(model)


####
B <- 20
coef_list <- vector("list", B)
desired_coefficients <- c('(Intercept)', 'conflict_binary_lag1y',
                          'I(psi * INDEX_lag0y)', 'I((psi * INDEX_lag0y)^2)',
                          'I(psi * INDEX_lag1y)', 'I((psi * INDEX_lag1y)^2)',
                          't2m_lag0y', 'tp_lag0y',
                          't2m_lag1y', 'tp_lag1y',
                          'tropical_year')

for (b in 1:B) {
  print(b)
  unique_locs <- unique(dat$loc_id)
  
  bootstrap_locs <- sample(unique_locs, size = length(unique_locs), replace = TRUE)
  
  bootstrap_df_list <- lapply(bootstrap_locs, function(id) {
    dat[dat$loc_id == id, ]
  })
  
  bootstrap_df <- do.call(rbind, bootstrap_df_list)
  
  model <- glm(formula = formula,
               data = bootstrap_df,
               family = binomial)
  
  coef_list[[b]] <- coef(model)[desired_coefficients]
}

# Convert list to matrix
coef_matrix <- do.call(rbind, coef_list)

# Calculate standard errors
bootstrap_mean <- apply(coef_matrix, 2, mean)
conf_int_percentile <- apply(coef_matrix, 2, quantile, probs = c(0.025, 0.5, 0.975))
conf_int_percentile <- t(conf_int_percentile)
conf_int_percentile


coef(model)['conflict_binary_lag1y']







