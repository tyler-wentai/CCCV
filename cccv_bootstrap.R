library(dplyr)
library(ggplot2)
library(parallel)

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_datasets/Binary_Africa_NINO3_square2_CON1_notrend.csv'
dat <- read.csv(panel_data_path)
colnames(dat)

dat$SOVEREIGNT <- as.factor(dat$SOVEREIGNT)
dat$loc_id <- as.factor(dat$loc_id)
dat$tropical_year <- dat$tropical_year - min(dat$tropical_year)

formula <- as.formula('conflict_binary ~ conflict_binary_lag1y + 
             I(psi*INDEX_lag0y) + I((psi*INDEX_lag0y)^2) + 
             I(psi*INDEX_lag1y) + I((psi*INDEX_lag1y)^2) + 
             t2m_lag0y + tp_lag0y +
             t2m_lag1y + tp_lag1y +
             tropical_year + as.factor(loc_id) - 1')

desired_coefficients <- c('conflict_binary_lag1y',
                          'I(psi * INDEX_lag0y)', 'I((psi * INDEX_lag0y)^2)',
                          'I(psi * INDEX_lag1y)', 'I((psi * INDEX_lag1y)^2)',
                          't2m_lag0y', 'tp_lag0y',
                          't2m_lag1y', 'tp_lag1y',
                          'tropical_year')


# bootstrap
df <- as.data.frame(matrix(ncol=length(desired_coefficients), nrow=0))
colnames(df) <- desired_coefficients

csv_file <- "/Users/tylerbagwell/Desktop/cccv_data/bootstrapped_data/boot_Binary_Africa_NINO3_square2_CON1_notrend.csv"
write.csv(df, csv_file, row.names = FALSE)

myFunction <- function() {
  unique_locs <- unique(dat$loc_id)
  
  bootstrap_locs <- sample(unique_locs, size = length(unique_locs), replace = TRUE)
  
  bootstrap_df_list <- lapply(bootstrap_locs, function(id) {
    dat[dat$loc_id == id, ]
  })
  
  # Need to reset the loc_id's so duplicated units don't have the same loc_id:
  for (i in 1:length(unique_locs)){
    loc_string = paste0('loc_', i)
    bootstrap_df_list[[i]]$loc_id = loc_string
  }
  
  bootstrap_df <- do.call(rbind, bootstrap_df_list)
  
  model <- glm(formula = formula,
               data = bootstrap_df,
               family = binomial)
  
  coefficients <- t(as.matrix(coef(model)[desired_coefficients]))
  
  # Check if the CSV file already exists
  if (file.exists(csv_file)) {
    write.table(coefficients, file = csv_file, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
  } else {
    write.table(coefficients, file = csv_file, sep = ",", row.names = FALSE, col.names = TRUE)
  }
}

n_runs <- 250                     # Specify how many times you want to run the function
numCores <- detectCores() - 1   # Leave one core free

system.time({
  mclapply(1:n_runs, function(x) myFunction(), mc.cores = numCores)
})


#system.time({
#  results <- lapply(1:n_runs, function(x) myFunction())
#})



### Read csv file

boot_dat <- read.csv(csv_file)



conf_int_percentile <- apply(boot_dat, 2, quantile, probs = c(0.025, 0.975))
conf_int_percentile <- t(conf_int_percentile)
conf_int_percentile















#############################



dat_l   <- subset(dat, psi < 0.35)
dat_m   <- subset(dat, psi >= 1 & psi <= 2)
dat_h   <- subset(dat, psi > 2)


reg <- glm(conflict_binary ~ conflict_binary_lag1y + 
             I(psi*INDEX_lag0y) + I((psi*INDEX_lag0y)^2) + 
             I(psi*INDEX_lag1y) + I((psi*INDEX_lag1y)^2) +
             poly(t2m_lag0y, 1) + poly(tp_lag0y, 1) +
             poly(t2m_lag1y, 1) + poly(tp_lag1y, 1) +
             tropical_year + loc_id,
           data = dat_l,
           family = binomial)
summary(reg)

reg <- glm(conflict_binary ~ conflict_binary_lag1y + 
             INDEX_lag0y + I(INDEX_lag0y^2) + 
             INDEX_lag1y + I(INDEX_lag1y^2) + 
             poly(t2m_lag0y, 1) + poly(tp_lag0y, 1) +
             poly(t2m_lag1y, 1) + poly(tp_lag1y, 1) +
             tropical_year + loc_id,
           data = dat_h,
           family = binomial)
summary(reg)

library(boot)


formula <- as.formula('conflict_binary ~ conflict_binary_lag1y + 
             I(psi*INDEX_lag0y) + I((psi*INDEX_lag0y)^2) + 
             I(psi*INDEX_lag1y) + I((psi*INDEX_lag1y)^2) + 
             t2m_lag0y + tp_lag0y +
             t2m_lag1y + tp_lag1y +
             tropical_year')

model <- glm(formula = formula,
             data = dat,
             family = binomial)
summary(model)


####
library(tictoc)
B <- 100
coef_list <- vector("list", B)
desired_coefficients <- c('conflict_binary_lag1y',
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
  
  # Need to reset the loc_id's so duplicated units don't have the same loc_id:
  for (i in 1:length(unique_locs)){
    loc_string = paste0('loc_', i)
    bootstrap_df_list[[i]]$loc_id = loc_string
  }
  
  bootstrap_df <- do.call(rbind, bootstrap_df_list)
  
  tic("  fit glm")
  model <- glm(formula = formula,
               data = bootstrap_df,
               family = binomial)
  toc()
  
  coef_list[[b]] <- coef(model)[desired_coefficients]
}

# Convert list to matrix
coef_matrix <- do.call(rbind, coef_list)

# Calculate standard errors
bootstrap_mean <- apply(coef_matrix, 2, mean)
conf_int_percentile <- apply(coef_matrix, 2, quantile, probs = c(0.025, 0.975))
conf_int_percentile <- t(conf_int_percentile)
conf_int_percentile


x_span <- range(dat$INDEX_lag0y)
x_vals <- seq(x_span[1], x_span[2], length.out=100)

psi_val <- quantile(unique(dat$psi), 0.9)
y0 <- exp( bootstrap_mean['I(psi * INDEX_lag0y)']*(psi_val*x_vals) + 
             bootstrap_mean['I((psi * INDEX_lag0y)^2)']*(psi_val*x_vals)^2 )
y1 <- exp( bootstrap_mean['I(psi * INDEX_lag1y)']*(psi_val*x_vals) + 
             bootstrap_mean['I((psi * INDEX_lag1y)^2)']*(psi_val*x_vals)^2 )

plot(x_vals, y0, type='l', lwd=3.0, col='red')
lines(x_vals, y1, type='l', lwd=3.0, col='blue')
abline(h=1.0)





