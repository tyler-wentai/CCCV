library(brms)
library(dplyr)
library(ggplot2)

panel_data_path <- '/Users/tylerbagwell/Desktop/Onset_global_nino3.csv'
dat <- read.csv(panel_data_path)

head(dat)

dat$country <- as.factor(dat$country)
dat$year <- dat$year - min(dat$year)


mod <- lm(conflict_onset ~ INDEX_lag0y + I(INDEX_lag0y*psi) + 
            INDEX_lag1y + I(INDEX_lag1y*psi) + 
            INDEX_lag2y + I(INDEX_lag2y*psi) + 
            year + country + year:country - 1,
          dat = dat)
summary(mod)



dat_l <- subset(dat, psi<0.4)
dat_h <- subset(dat, psi>0.8)

mod <- lm(conflict_onset ~ INDEX_lag0y +
            year + country + year:country - 1,
          dat = dat_h)
summary(mod)



###### BAYESIAN FITS
# bernoulli model
fit <- brm(
  conflict_onset ~  0 + 
    INDEX_lag0y + I(INDEX_lag0y*psi) +
    INDEX_lag1y + I(INDEX_lag1y*psi) +
    INDEX_lag2y + I(INDEX_lag2y*psi) +
    year + country,
  data = dat, family = bernoulli(link = "logit"), 
  iter = 5000, chains=1, warmup=1000,
  prior = prior(normal(0, 10), class = b)
)

print(summary(fit), digits = 3)





#
library(scales)

draws_matrix <- as_draws_matrix(fit)
colnames(draws_matrix)

psi_range <- seq(min(dat$psi), max(dat$psi), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(psi_range)){
  psi_i <- psi_range[i]
  sum_params <- exp( draws_matrix[, "b_INDEX_lag2y"] + (psi_i*draws_matrix[, "b_IINDEX_lag2yMUpsi"]) )
  results <- rbind(results, c(psi_i,
                              mean(sum_params), 
                              sd(sum_params),
                              quantile(sum_params, 0.025),
                              quantile(sum_params, 0.975)))
  
}

results_0d3 <- results
results_0d7 <- results
results_1d0 <- results

df <- data.frame(
  x = rep(psi_range, 3),
  y = c(results_0d3[,2], results_0d7[,2], results_1d0[,2]),
  ymin = c(results_0d3[,4], results_0d7[,4], results_1d0[,4]),
  ymax = c(results_0d3[,5], results_0d7[,5], results_1d0[,5]),
  group = rep(c("Lag 0", "Lag 1", "Lag 2"), each = length(psi_range))
)

sd_index <- sd(dat$INDEX_lag0y[1:34])
df$x <- df$x/sd_index

p <- ggplot(df, aes(x = x, y = y, color = group, fill = group)) +
  geom_hline(yintercept=1, col = "gray", linewidth = 1.5) + 
  geom_vline(xintercept=0, col = "gray", linewidth = 1.5) +
  geom_ribbon(aes(ymin = ymin, ymax = ymax), alpha = 0.2, color = NA) +  # Credible region
  geom_line(linewidth = 1) +  # Lines
  scale_x_continuous(limits = c(min(df$x), max(df$x))) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) + 
  coord_cartesian(ylim = c(0.5, 1.7)) + 
  labs(
    title = "Africa, ENSO, Model 3",
    x = "NINO3 Index (s.d.)",
    y = "ENSO-induced change in the odds of conflict (%)",
    color = expression("Teleconnection"~Psi~"=1.0"),
    fill = expression("Teleconnection"~Psi~"=1.0")
  ) +
  theme_light() +
  theme(legend.position=c(0.26, 0.8), legend.background = element_rect(fill = NA, color = NA),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))

p






####################################################################

library(brms)
library(tictoc)
library(dplyr)
library(ggplot2)

#panel_data_path <- '/Users/tylerbagwell/Desktop/panel_data_Africa_binary.csv'
panel_data_path <- '/Users/tylerbagwell/Desktop/panel_data_Global_binary_dmi_1admin.csv'
dat <- read.csv(panel_data_path)

#View(dat)
colnames(dat)

dat$SOVEREIGNT <- as.factor(dat$SOVEREIGNT)
dat$loc_id <- as.factor(dat$loc_id)
dat$year <- dat$year - min(dat$year)


dat_rand <- dat
dat_rand$INDEX_lag0y <- sample(dat$INDEX_lag0y)
dat_rand$INDEX_lag1y <- sample(dat$INDEX_lag1y)
dat_rand$INDEX_lag2y <- sample(dat$INDEX_lag2y)

reg <- lm(conflict_binary ~ conflict_binary_lag1y + 
            I(psi*INDEX_lag0y) + I((psi*INDEX_lag0y)^2) +
            I(psi*INDEX_lag1y) + I((psi*INDEX_lag1y)^2) +
            I(psi*INDEX_lag2y) + I((psi*INDEX_lag2y)^2) +
            year, data=dat_randpsi)
summary(reg)



unique_psi <- dat %>%
  select(loc_id, psi) %>%
  distinct()

#set.seed(123)
shuffled_psi <- unique_psi %>%
  mutate(psi = sample(psi))

dat_randpsi <- dat %>%
  select(-psi) %>%  # Remove the original psi
  left_join(shuffled_psi, by = "loc_id")  # Add the shuffled psi






###### BAYESIAN FITS
# bernoulli model
tic("Brms Model Fitting")
fit1_nino_randind <- brm(
  conflict_binary ~  0 + conflict_binary_lag1y + 
    I(INDEX_lag0y*psi) + I((INDEX_lag0y*psi)^2) + 
    I(INDEX_lag1y*psi) + I((INDEX_lag1y*psi)^2) +
    I(INDEX_lag2y*psi) + I((INDEX_lag2y*psi)^2) +
    year + loc_id,
  data = dat_rand, family = bernoulli(link = "logit"), 
  iter = 5000, chains=1, warmup=1000,
  prior = prior(normal(0, 10), class = b)
)
toc()

print(summary(fit1_nino), digits = 3)
#plot(fit1_nino)


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
library(scales)

draws_matrix <- as_draws_matrix(fit1_nino_randind)
colnames(draws_matrix)

psi <- 1
climindex <- seq(min(dat$INDEX_lag0y), max(dat$INDEX_lag0y), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(climindex)){
  climind <- climindex[i]
  sum_params <- exp( 
    (psi*climind*draws_matrix[, "b_IINDEX_lag0yMUpsi"]) + 
      ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag0yMUpsiE2"])) - 1
  results <- rbind(results, c(climind,
                              mean(sum_params), 
                              sd(sum_params),
                              quantile(sum_params, 0.025),
                              quantile(sum_params, 0.975)))
  
}

results_0d3 <- results
results_0d7 <- results
results_1d0 <- results

df <- data.frame(
  x = rep(climindex, 3),
  y = c(results_0d3[,2], results_0d7[,2], results_1d0[,2]),
  ymin = c(results_0d3[,4], results_0d7[,4], results_1d0[,4]),
  ymax = c(results_0d3[,5], results_0d7[,5], results_1d0[,5]),
  group = rep(c("Lag 0", "Lag 1", "Lag 2"), each = length(climindex))
)

sd_index <- sd(dat$INDEX_lag0y[1:34])
df$x <- df$x/sd_index

p <- ggplot(df, aes(x = x, y = y, color = group, fill = group)) +
  geom_hline(yintercept=0, col = "gray", linewidth = 1.5) + 
  geom_vline(xintercept=0, col = "gray", linewidth = 1.5) +
  geom_ribbon(aes(ymin = ymin, ymax = ymax), alpha = 0.2, color = NA) +  # Credible region
  geom_line(linewidth = 1) +  # Lines
  scale_x_continuous(limits = c(-max(df$x), max(df$x))) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) + 
  coord_cartesian(ylim = c(-0.1, 1.5)) + 
  labs(
    title = "Africa, ENSO, Model 3",
    x = "NINO3 Index (s.d.)",
    y = "ENSO-induced change in the odds of conflict (%)",
    color = expression("Teleconnection"~Psi~"=1.0"),
    fill = expression("Teleconnection"~Psi~"=1.0")
  ) +
  theme_light() +
  theme(legend.position=c(0.26, 0.8), legend.background = element_rect(fill = NA, color = NA),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))

p

ggsave(
  filename = "/Users/tylerbagwell/Desktop/africa_model3_psi1d0.png",  # File name and format
  plot = p,                             # Plot object
  width = 5,                           # Width in inches
  height = 5,                           # Height in inches
  dpi = 300                             # Resolution (dots per inch)
)







sd_index <- sd(dat$INDEX_lag0y[1:34])
index199.ob <- 0.5*sd_index



draws_matrix <- as_draws_matrix(fit1_nino)
colnames(draws_matrix)

psi <- 0.4
d_odds_lag0 <- exp((index199.ob*draws_matrix[, "b_INDEX_lag0y"]) + 
                     (psi*index199.ob*draws_matrix[, "b_IINDEX_lag0yMUpsi"]) + 
                     ((psi^2)*(index199.ob^2)*draws_matrix[, "b_IINDEX_lag0yMUpsiE2"])) - exp(0)
d_odds_lag1 <- exp((index199.ob*draws_matrix[, "b_INDEX_lag1y"]) + 
                     (psi*index199.ob*draws_matrix[, "b_IINDEX_lag1yMUpsi"]) + 
                     ((psi^2)*(index199.ob^2)*draws_matrix[, "b_IINDEX_lag1yMUpsiE2"])) - exp(0)
d_odds_lag2 <- exp((index199.ob*draws_matrix[, "b_INDEX_lag2y"]) + 
                     (psi*index199.ob*draws_matrix[, "b_IINDEX_lag2yMUpsi"]) + 
                     ((psi^2)*(index199.ob^2)*draws_matrix[, "b_IINDEX_lag2yMUpsiE2"])) - exp(0)

results <- matrix(ncol=6, nrow=0)
results <- rbind(results, c(psi, 0,
                            mean(d_odds_lag0), 
                            sd(d_odds_lag0),
                            quantile(d_odds_lag0, 0.025),
                            quantile(d_odds_lag0, 0.975)))
results <- rbind(results, c(psi, 1,
                            mean(d_odds_lag1), 
                            sd(d_odds_lag1),
                            quantile(d_odds_lag1, 0.025),
                            quantile(d_odds_lag1, 0.975)))
results <- rbind(results, c(psi, 2,
                            mean(d_odds_lag2), 
                            sd(d_odds_lag2),
                            quantile(d_odds_lag2, 0.025),
                            quantile(d_odds_lag2, 0.975)))

results_0d4 <- results
results_1d0 <- results


png("/Users/tylerbagwell/Desktop/Africa_weak_nino.png", width = 2000, height = 1800, res = 300)
plot(results_1d0[,2], results_1d0[,3], type='b', ylim=c(-0.1,0.8), lwd=3.5, col='red', lty=2, , xaxt = "n", yaxt = "n",
     ylab='El-Nino induced change in odds of conflict',
     xlab='Lag (years)',
     main='Weak El-Nino Event (0.5 s.d.) (Africa, model 3)')
lines(c(results_1d0[1,2],results_1d0[1,2]), c(results_1d0[1,5],results_1d0[1,6]), lwd=3.5, col='red')
lines(c(results_1d0[2,2],results_1d0[2,2]), c(results_1d0[2,5],results_1d0[2,6]), lwd=3.5, col='red')
lines(c(results_1d0[3,2],results_1d0[3,2]), c(results_1d0[3,5],results_1d0[3,6]), lwd=3.5, col='red')
#
lines(results_0d4[,2]-0.02, results_0d4[,3], type='b', ylim=c(-0.5,2), lwd=3.5, col='gold', lty=2)
lines(c(results_0d4[1,2]-0.02,results_0d4[1,2]-0.02), c(results_0d4[1,5],results_0d4[1,6]), lwd=3.5, col='gold')
lines(c(results_0d4[2,2]-0.02,results_0d4[2,2]-0.02), c(results_0d4[2,5],results_0d4[2,6]), lwd=3.5, col='gold')
lines(c(results_0d4[3,2]-0.02,results_0d4[3,2]-0.02), c(results_0d4[3,5],results_0d4[3,6]), lwd=3.5, col='gold')
#
y_ticks <- seq(-0.0,0.8,0.2)
y_labels <- paste0(y_ticks * 100, "%")
axis(side = 2, at = y_ticks, labels = y_labels)
axis(side = 1, at = c(0, 1, 2))
legend('topright', legend=c('Psi=1.0','Psi=0.4'), lty=1, col=c('red','gold'), lwd=2.5, bty = "n")
abline(h=0)
grid(nx = NA, ny = NULL)
dev.off()











draws_matrix <- as_draws_matrix(fit1)
colnames(draws_matrix)

psi <- 1.0
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









