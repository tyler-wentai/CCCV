library(brms)
library(dplyr)
library(ggplot2)

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_DMI_cindexnosd.csv'
dat <- read.csv(panel_data_path)

sum(dat$conflict_binary)
#plot(subset(dat, loc_id=='loc_1')$cindex_lag0, type='l')
#head(dat)
dat <- subset(dat, year!=1989)
dat$bool1989 <- ifelse(dat$year<1989,0,1)
dat$year <- dat$year - min(dat$year)
dat$loc_id <- as.factor(dat$loc_id)

unique_psi <- dat %>%
  group_by(loc_id) %>%
  summarise(
    psi = first(pop_avg_psi),
    total_conflict_counts = sum(conflict_binary))
hist(unique_psi$psi, breaks='scott')

quantile(unique_psi$psi, c(0.15,0.50,0.75))
dat_help <- subset(dat, pop_avg_psi >= 0.6863374 & pop_avg_psi < 1.2993510)
sum(dat_help$conflict_binary)
dat_help <- subset(dat_help, loc_id!=572)
sum(dat_help$conflict_binary)


#quantile(subset(dat, loc_id=='534')$cindex_lag0y, c(0.01, 0.01,0.99))
#dat_help <- subset(dat, cindex_lag0y <= 2)
#dat_help <- subset(dat_help, psi > 1.4415020)
#sum(dat_help$conflict_binary)

group_agg <- dat_help %>%
  group_by(loc_id) %>%
  summarise(
    psi = first(psi),
    total_conflict_counts = sum(conflict_binary))
View(group_agg)
View(dat)


m <- brm(
  conflict_binary ~ 
    cindex_lag0y +
    (1 + year || loc_id), #year:loc_id + loc_id,
  data = dat_help, family = gaussian(),#bernoulli(link = "logit"), 
  iter = 4000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b)
)
plot(m)
print(summary(m, prob = 0.99), digits = 4)
ce <- conditional_effects(m, effects = "cindex_lag0y", prob = 0.90)
plot(ce, ask = FALSE)
ce_data <- ce$cindex_lag0y

waic(m0)
waic(m1)

sum(dat_help$conflict_binary)


ce_data <- ce$cindex_lag0y

med_psi <- median(dat$psi)
dat_l <- subset(dat, psi<med_psi)
dat_h <- subset(dat, psi>=med_psi)

mod0 <- lm(conflict_binary ~ conflict_binary_lag1y + loc_id + year:loc_id + bool1989 - 1,
           dat = dat_help)
summary(mod0)
mod1 <- lm(conflict_binary ~ + cindex_lag0y + loc_id + year:loc_id + bool1989 - 1,
           dat = dat_help)
summary(mod1)

library(mgcv)
gam_fit <- gam(conflict_binary ~ s(cindex_lag0y) + loc_id + year + bool1989 - 1, data = dat_help)
summary(gam_fit)
plot(gam_fit, pages = 1)

AIC(mod0)
AIC(mod1)

BIC(mod0)
BIC(mod1)


#
dat_agg <- dat_help %>%
  group_by(year) %>%
  summarise(
    conflict_proportion = sum(conflict_binary) / n(),
    bool1989 = first(bool1989), 
    cindex_lag0y = first(cindex_lag0y), 
    cindex_lag1y = first(cindex_lag1y), 
    cindex_lag2y = first(cindex_lag2y)
  )

mod0 <- lm(conflict_proportion ~ year + bool1989, data=dat_agg)
mod1 <- lm(conflict_proportion ~ I(cindex_lag0y^1) + year + bool1989, data=dat_agg)
summary(mod1)

plot(dat_agg$conflict_proportion, type='l')
stud_res <- rstudent(mod1)
plot(dat_agg$year, stud_res)
abline(h=0)

AIC(mod0)
AIC(mod1)

BIC(mod0)
BIC(mod1)

plot(dat_agg$cindex_lag0y, dat_agg$conflict_proportion)
abline(mod1)
abline(v=c(-2,2))

plot(mod1)
plot(cooks.distance(mod1),type="b",pch=18,col="red")
N = 74
k = 4
cutoff = 4/ (N-k-1)
abline(h=cutoff,lty=2)

dat_agg_cleaned <- subset(dat_agg, year != 39)

###### BAYESIAN FITS
fit1 <- brm(
  conflict_proportion ~ cindex_lag0y + year + bool1989,
  data = dat_agg, family = gaussian(), 
  iter = 10000, chains=2, warmup=1000, cores=1,
  prior = c(
    prior(normal(0, 2.5), class = "b"),                # regression coefficients
    prior(normal(0, 5), class = "Intercept"),          # intercept term
    prior(exponential(1), class = "sigma")             # residual standard deviation
  )
)

print(summary(fit1, prob = 0.90), digits = 4)
#plot(fit1)

ce <- conditional_effects(fit1, effects = "cindex_lag0y", prob = 0.90, resolution = 1000)
plot(ce, ask = FALSE)
ce_data <- ce$cindex_lag0y
write.csv(ce_data, file = "/Users/tylerbagwell/Desktop/panel_datasets/results/CE_cindex_lag0y_Onset_Binary_Global_NINO3_square4_mod95.csv", row.names = FALSE)

loo_compare(loo(fit0), loo(fit1))
pp_check(fit1, ndraws=100)

waic(fit0)
waic(fit1)


library(mgcv)
gam_fit <- gam(conflict_proportion ~ s(cindex_lag0y) + year + bool1989, data = dat_agg)
plot(gam_fit, pages = 1)


#
draws_matrix <- as_draws_matrix(fit)
colnames(draws_matrix)

climindex <- seq(min(dat$cindex_lag0y), max(dat$cindex_lag0y), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(climindex)){
  climind <- climindex[i]
  sum_params <- (climind*draws_matrix[, "b_cindex_lag0y"]) + (climind^2*draws_matrix[, "b_Icindex_lag0yE2"]) + (draws_matrix[, "b_Intercept"])
  results <- rbind(results, c(climind,
                              mean(sum_params), 
                              sd(sum_params),
                              quantile(sum_params, 0.05),
                              quantile(sum_params, 0.95)))
  
}

plot(results[,1], results[,2], type='l', col='black', ylim=c(-0.005,0.005), lwd=2)
lines(results[,1], results[,4], type='l', col='blue', lwd=2)
lines(results[,1], results[,5], type='l', col='blue', lwd=2)
abline(h=0)




#
library(scales)

draws_matrix <- as_draws_matrix(fit)
colnames(draws_matrix)

psi_range <- seq(min(dat$psi), max(dat$psi), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(psi_range)){
  psi_i <- psi_range[i]
  sum_params <- exp( draws_matrix[, "b_INDEX_lag0y"] + (psi_i*draws_matrix[, "b_IINDEX_lag0yE2"]) )
  results <- rbind(results, c(psi_i,
                              mean(sum_params), 
                              sd(sum_params),
                              quantile(sum_params, 0.025),
                              quantile(sum_params, 0.975)))
  
}
plot(results[,1], results[,2], type='l', col='black', ylim=c(0,2.5), lwd=2)
lines(results[,1], results[,4], type='l', col='blue', lwd=2)
lines(results[,1], results[,5], type='l', col='blue', lwd=2)
abline(h=1)

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
panel_data_path <- '/Users/tylerbagwell/Desktop/panel_datasets/Binary_Africa_NINO3_square2_CON1_nocontrols.csv'
dat <- read.csv(panel_data_path)

#View(dat)
colnames(dat)

#dat$conflict_binary <- as.factor(dat$conflict_binary)
#dat$conflict_binary_lag1y <- as.factor(dat$conflict_binary_lag1y)
dat$SOVEREIGNT <- as.factor(dat$SOVEREIGNT)
dat$loc_id <- as.factor(dat$loc_id)
dat$tropical_year <- dat$tropical_year - min(dat$tropical_year)

reg <- lm(conflict_binary ~ conflict_binary_lag1y + 
            INDEX_lag0y + I(psi*INDEX_lag0y) + I((psi*INDEX_lag0y)^2) +
            INDEX_lag1y + I(psi*INDEX_lag1y) + I((psi*INDEX_lag1y)^2) +
            poly(t2m_lag0y, 1) + poly(tp_lag0y, 1) +
            poly(t2m_lag1y, 1) + poly(tp_lag1y, 1) +
            loc_id + tropical_year, data=dat)
summary(reg)

reg <- glm(conflict_binary ~ conflict_binary_lag1y + 
             I(psi*INDEX_lagF1y) + I((psi*INDEX_lagF1y)^2) +
             I(psi*INDEX_lag0y) + I((psi*INDEX_lag0y)^2) +
             I(psi*INDEX_lag1y) + I((psi*INDEX_lag1y)^2) +
             poly(t2m_lagF1y, 1) + poly(tp_lagF1y, 1) +
             poly(t2m_lag0y, 1) + poly(tp_lag0y, 1) +
             poly(t2m_lag1y, 1) + poly(tp_lag1y, 1) +
             loc_id + tropical_year,
           data = dat,
           family = binomial)
summary(reg)

#

median(dat$psi)

dat_l   <- subset(dat, psi < 0.5)
dat_m   <- subset(dat, psi > 0.5 & psi < 1.0)
dat_h   <- subset(dat, psi > 1.0 & psi < 2.00)
dat_hh  <- subset(dat, psi > 1.5)

reg <- lm(conflict_binary ~ conflict_binary_lag1y +
            I(psi*INDEX_lagF1y) + I((psi*INDEX_lagF1y)^2) +
            I(psi*INDEX_lag0y) + I((psi*INDEX_lag0y)^2) +
            I(psi*INDEX_lag1y) + I((psi*INDEX_lag1y)^2) +
            poly(t2m_lag0y, 1) + poly(tp_lag0y, 1) +
            poly(t2m_lag1y, 1) + poly(tp_lag1y, 1) +
            loc_id + tropical_year, data=dat_weak)
summary(reg)


dat_help <- subset(dat, psi > 0.6570268)

reg0 <- glm(conflict_binary ~ conflict_binary_lag1y +
              loc_id + tropical_year - 1,
            data = dat_help,
            family = binomial)
#summary(reg0)
reg1 <- glm(conflict_binary ~ conflict_binary_lag1y +
              INDEX_lag0y + I(INDEX_lag0y^2) + 
              INDEX_lag1y + I(INDEX_lag1y^2) + 
              INDEX_lag2y + I(INDEX_lag2y^2) + 
              loc_id + tropical_year - 1,
            data = dat_help,
            family = binomial)
#summary(reg1)
AIC(reg0)
AIC(reg1)

quantile(dat$psi, c(0.25,0.75))

probabilities0 <- predict(reg0, type = "response")
probabilities1 <- predict(reg1, type = "response")
library(pROC)
library(precrec)
library(PRROC)
roc_obj0 <- roc(dat_help$conflict_binary, probabilities0)
roc_obj1 <- roc(dat_help$conflict_binary, probabilities1)
auc_value0 <- pROC::auc(roc_obj0)
auc_value1 <- pROC::auc(roc_obj1)
plot(roc_obj0)
lines(roc_obj1, col='red')
auc_value0
auc_value1
legend("bottomright", legend = paste("AUC =", round(auc_value, 4)), bty = "n")


precrec_obj1 <- evalmod(scores = probabilities1, labels = dat_help$conflict_binary)
precrec_obj0 <- evalmod(scores = probabilities0, labels = dat_help$conflict_binary)
autoplot(precrec_obj0, "PRC")
autoplot(precrec_obj1, "PRC")

scores_list <- list(Model0 = probabilities0, Model1 = probabilities1)
labels_list <- list(dat_help$conflict_binary, dat_help$conflict_binary)
precrec_combined <- evalmod(scores = scores_list, labels = labels_list)
autoplot(precrec_combined, "PRC") +
  ggtitle("Comparison of Precision-Recall Curves") +
  theme_minimal()



x_span  <- range(dat$INDEX_lag0y)
xx <- seq(x_span[1], x_span[2], length.out=100)
x0_high <- -2.285e-01*xx + 9.268e-02*xx^2
x1_high <- -1.172e-01*xx + 1.020e-01*xx^2

x0_low <- -1.405e-01*xx + 2.094e-01*xx^2
x1_low <-  1.986e-01*xx -1.006e-01*xx^2

plot(xx, x1_high, type='l', col='red', lwd=3)
lines(xx, x1_low, type='l', col='blue', lwd=3)
abline(h=0)


library(mgcv)

dat_strongninoremoved <- subset(dat, INDEX_lag0y<2.0)

# Fit a GAM with a smooth term for predictor x
gam_model <- gam(conflict_binary ~ conflict_binary_lag1y + 
                   INDEX_lag0y + s(I(psi*INDEX_lag0y)) +
                   INDEX_lag1y + s(I(psi*INDEX_lag1y)) +
                   poly(t2m_lag0y, 1) + poly(tp_lag0y, 1) +
                   poly(t2m_lag1y, 1) + poly(tp_lag1y, 1) +
                   loc_id + tropical_year:loc_id - 1,
                 family = binomial, data=dat_m)
summary(gam_model)

# Plot the smooth term
plot(gam_model, se = TRUE)
abline(h=0)


###### BAYESIAN FITS
# bernoulli model
log_mod <- brm(
  conflict_binary ~  0 + conflict_binary_lag1y + 
    I(INDEX_lag0y*psi) + I((INDEX_lag0y*psi)^2) + 
    I(INDEX_lag1y*psi) + I((INDEX_lag1y*psi)^2) +
    I(INDEX_lag2y*psi) + I((INDEX_lag2y*psi)^2) +
    tropical_year + loc_id,
  data = dat, family = bernoulli(link = "logit"), 
  iter = 4000, chains=1, warmup=1000,
  prior = prior(normal(0, 10), class = b)
)

print(summary(log_mod), digits = 3)

#
quantile(dat$psi, c(0.33,0.66))
dat_help <- subset(dat, psi < 0.5488111)

mod_l <- brm(
  conflict_binary ~  0 + conflict_binary_lag1y + 
    INDEX_lag0y + I(INDEX_lag0y^2) + 
    INDEX_lag1y + I(INDEX_lag1y^2) + 
    INDEX_lag2y + I(INDEX_lag2y^2) + 
    tropical_year + loc_id,
  data = dat_help, family = bernoulli(link = "logit"), 
  iter = 4000, chains=1, warmup=1000,
  prior = prior(normal(0, 10), class = b)
)

print(summary(mod_h), digits = 3)
#plot(fit1_nino)


draws_matrix <- as_draws_matrix(log_mod)
colnames(draws_matrix)

psi <- 1.2787523
climindex <- seq(min(dat$INDEX_lag0y), max(dat$INDEX_lag0y), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(climindex)){
  climind <- climindex[i]
  sum_params <- exp((psi*climind*draws_matrix[, "b_IINDEX_lag2yMUpsi"]) + ((psi^2)*(climind^2)*draws_matrix[, "b_IINDEX_lag2yMUpsiE2"]))
  results <- rbind(results, c(climind,
                              mean(sum_params), 
                              sd(sum_params),
                              quantile(sum_params, 0.025),
                              quantile(sum_params, 0.975)))
  
}

plot(results[,1], results[,2], type='l', col='black', ylim=c(0,2.5), lwd=2)
lines(results[,1], results[,4], type='l', col='blue', lwd=2)
lines(results[,1], results[,5], type='l', col='blue', lwd=2)
abline(h=1)


results_l <- results
results_h <- results

plot(results_l[,1], results_l[,2], type='l', col='navy', ylim=c(0,2.5), lwd=2)
lines(results_l[,1], results_l[,4], type='l', col='blue', lwd=2)
lines(results_l[,1], results_l[,5], type='l', col='blue', lwd=2)
#
lines(results_h[,1], results_h[,2], type='l', col='red', lwd=2)
lines(results_h[,1], results_h[,4], type='l', col='tomato', lwd=2)
lines(results_h[,1], results_h[,5], type='l', col='tomato', lwd=2)
abline(h=1)





draws_matrix <- as_draws_matrix(mod_l)
colnames(draws_matrix)

climindex <- seq(min(dat$INDEX_lag0y), max(dat$INDEX_lag0y), length.out=100)
results <- matrix(ncol=5, nrow=0)
for (i in 1:length(climindex)){
  climind <- climindex[i]
  sum_params <- exp((climind*draws_matrix[, "b_INDEX_lag0y"]) + ((climind^2)*draws_matrix[, "b_IINDEX_lag0yE2"]))
  results <- rbind(results, c(climind,
                              mean(sum_params), 
                              sd(sum_params),
                              quantile(sum_params, 0.025),
                              quantile(sum_params, 0.975)))
  
}

plot(results[,1], results[,2], type='l', col='black', ylim=c(0,2.5), lwd=2)
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









