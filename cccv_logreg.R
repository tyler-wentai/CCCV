library(brms)
library(tictoc)
library(dplyr)
library(ggplot2)

#panel_data_path <- '/Users/tylerbagwell/Desktop/panel_data_Africa_binary.csv'
panel_data_path <- '/Users/tylerbagwell/Desktop/panel_data_Asia_binary.csv'
dat <- read.csv(panel_data_path)

#View(dat)
colnames(dat)

dat$SOVEREIGNT <- as.factor(dat$SOVEREIGNT)
dat$loc_id <- as.factor(dat$loc_id)
dat$year <- dat$year - min(dat$year)


###### BAYESIAN FITS
# bernoulli model
tic("Brms Model Fitting")
fit5a <- brm(
  conflict_binary ~  I(INDEX_lag0y*psi) + I((INDEX_lag0y*psi)^2) + 
    I(INDEX_lag1y*psi) + I((INDEX_lag1y*psi)^2) +
    I(INDEX_lag2y*psi) + I((INDEX_lag2y*psi)^2) +
    year + SOVEREIGNT + year:SOVEREIGNT,
  data = dat, family = bernoulli(link = "logit"), 
  iter = 4000, chains=1, warmup=1000,
  prior = prior(normal(0, 10), class = b)
)
toc()

print(summary(fit5a), digits = 4)
#plot(fit5a)


draws_matrix <- as_draws_matrix(fit5)
colnames(draws_matrix)

psi <- 1.2
climindex <- seq(min(dat$INDEX_lag0y), max(dat$INDEX_lag0y), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(climindex)){
  climind <- climindex[i]
  sum_params <- (psi*climind*draws_matrix[, "b_IINDEX_lag0yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag0yMUpsiE2"]) +
    (psi*climind*draws_matrix[, "b_IINDEX_lag1yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag1yMUpsiE2"])
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







psi <- 0.5
climindex <- seq(min(dat$INDEX_lag0y), max(dat$INDEX_lag0y), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(climindex)){
  climind <- climindex[i]
  sum_params <- exp((climind*draws_matrix[, "b_INDEX_lag0y"]) + (psi*climind*draws_matrix[, "b_IINDEX_lag0yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag0yMUpsiE2"]) +
                      (climind*draws_matrix[, "b_INDEX_lag1y"]) + (psi*climind*draws_matrix[, "b_IINDEX_lag1yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag1yMUpsiE2"]) +
                      (climind*draws_matrix[, "b_INDEX_lag2y"]) + (psi*climind*draws_matrix[, "b_IINDEX_lag2yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag2yMUpsiE2"]))
  results <- rbind(results, c(climind,
                              mean(sum_params), 
                              sd(sum_params),
                              quantile(sum_params, 0.025),
                              quantile(sum_params, 0.975)))
  
}
plot(results[,1], results[,2], type='l', col='black', ylim=c(0.9,10), lwd=2)
lines(results[,1], results[,4], type='l', col='blue', lwd=2)
lines(results[,1], results[,5], type='l', col='blue', lwd=2)
abline(h=1)













##### PLOTTING
library(viridis)

draws_matrix <- as_draws_matrix(fit5a)
colnames(draws_matrix)

psi <- 0.3
climindex <- seq(min(dat$INDEX_lag0y), max(dat$INDEX_lag0y), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(climindex)){
  climind <- climindex[i]
  sum_params <- exp((psi*climind*draws_matrix[, "b_IINDEX_lag0yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag0yMUpsiE2"]) +
                      (psi*climind*draws_matrix[, "b_IINDEX_lag1yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag1yMUpsiE2"]) +
                      (psi*climind*draws_matrix[, "b_IINDEX_lag2yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag2yMUpsiE2"]))
  results <- rbind(results, c(climind,
                              mean(sum_params), 
                              sd(sum_params),
                              quantile(sum_params, 0.025),
                              quantile(sum_params, 0.975)))
  
}
plot(results[,1], results[,2], type='l', col='black', ylim=c(0.9,10), lwd=2)
lines(results[,1], results[,4], type='l', col='blue', lwd=2)
lines(results[,1], results[,5], type='l', col='blue', lwd=2)
abline(h=1)

results_0d3 <- results
results_0d7 <- results
results_1d0 <- results

df <- data.frame(
  x = rep(climindex, 3),
  y = c(results_0d3[,2], results_0d7[,2], results_1d0[,2]),
  ymin = c(results_0d3[,4], results_0d7[,4], results_1d0[,4]),
  ymax = c(results_0d3[,5], results_0d7[,5], results_1d0[,5]),
  group = rep(c("Psi=0.3", "Psi=0.7", "Psi=1.0"), each = length(climindex))
)

p <- ggplot(df, aes(x = x, y = y, color = group, fill = group)) +
  geom_hline(yintercept=0, col = "gray", linewidth = 1.5) + 
  geom_vline(xintercept=0, col = "gray", linewidth = 1.5) +
  geom_ribbon(aes(ymin = ymin, ymax = ymax), alpha = 0.2, color = NA) +  # Credible region
  geom_line(linewidth = 1) +  # Lines
  scale_x_continuous(limits = c(-max(dat$INDEX_lag0y), max(dat$INDEX_lag0y))) +
  coord_cartesian(ylim = c(0.2, 2.25)) + 
  labs(
    title = "Asia, Model, 5",
    x = "NINO3 Index",
    y = "ENSO-induced change in the odds of conflict",
    color = "Teleconnection Strength",
    fill = "Teleconnection Strength"
  ) +
  theme_light() +
  theme(legend.position=c(0.2, 0.8), legend.background = element_rect(fill = NA, color = NA),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))

p

ggsave(
  filename = "/Users/tylerbagwell/Desktop/asia_model5_plot.png",  # File name and format
  plot = p,                             # Plot object
  width = 5,                           # Width in inches
  height = 5,                           # Height in inches
  dpi = 300                             # Resolution (dots per inch)
)




#scale_color_viridis(discrete = TRUE, option = "D") +
#scale_fill_viridis(discrete = TRUE, option = "D") +



