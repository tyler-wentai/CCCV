library(brms)
library(dplyr)
library(ggplot2)

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_ANI.csv'
dat <- read.csv(panel_data_path)

sum(dat$conflict_binary)
#head(dat)
dat <- subset(dat, year!=1989)
dat$bool1989 <- ifelse(dat$year<1989,0,1)
dat$year <- dat$year - min(dat$year)
dat$loc_id <- as.factor(dat$loc_id)

unique_psi <- dat %>%
  group_by(loc_id) %>%
  summarise(
    psi = first(pop_avg_psi),
    total_conflict_onsets = sum(conflict_binary))
hist(unique_psi$psi, breaks='scott')
hist(unique_psi$total_conflict_onsets, breaks='scott')


quantile(dat$psi, c(0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95))
dat_help <- subset(dat, pop_avg_psi > 1.35)
sum(dat_help$conflict_binary)


View(dat)

dat$interact_indexpsi <- dat$cindex_lag0y*dat$pop_avg_psi


mod <- brm(
  conflict_binary ~ 
    cindex_lag0y + 
    year + loc_id + 0, #(1 + year || loc_id), #year:loc_id, 
  data = dat, family = gaussian(),# bernoulli(link = "logit"), #gaussian(),
  iter = 3000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b)
)
#plot(mod7)
print(summary(mod, prob = 0.95), digits = 4)
ce <- conditional_effects(mod, effects = "cindex_lag0y", prob = 0.90)
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
mod1 <- lm(conflict_binary ~ cindex_lag0y + I(cindex_lag0y*pop_avg_psi) + loc_id + year:loc_id - 1,
           dat = dat_help)
summary(mod1)

mod_glm <- glm(
  conflict_binary ~ cindex_lag1y + bool1989 + year + loc_id - 1,
  data   = dat_help,
  family = binomial(link = "logit")
)
summary(mod_glm)
exp(coef(mod_glm)) #Odds-ratios and 95% CI
exp(confint(mod_glm))






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
#View(dat_agg)
#dat_agg <- subset(dat_agg, year<73) # NEED THIS BECAUSE SPI6 DOES HAVE 2022...

mod0 <- lm(conflict_proportion ~ year + bool1989, data=dat_agg)
mod1 <- lm(conflict_proportion ~ I(cindex_lag1y^1) + year + bool1989, data=dat_agg)
summary(mod1)
plot(dat_agg$cindex_lag1y, dat_agg$conflict_proportion)
abline(mod1)

summary(mod0)
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
  conflict_proportion ~ I(cindex_lag0y^1) + year + bool1989,
  data = dat_agg, family = gaussian(), 
  iter = 10000, chains=2, warmup=1000, cores=1,
  prior = c(
    prior(normal(0, 2.5), class = "b"),                # regression coefficients
    prior(normal(0, 5), class = "Intercept"),          # intercept term
    prior(exponential(1), class = "sigma")             # residual standard deviation
  )
)

print(summary(fit1, prob = 0.95), digits = 4)
#plot(fit1)

ce <- conditional_effects(fit1, effects = "cindex_lag0y", prob = 0.95, resolution = 500)
plot(ce, ask = FALSE)
ce_data <- ce$cindex_lag0y
write.csv(ce_data, file = "/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Binary_GlobalState_DMI_grouplinear_ci90_remove4pos.csv", row.names = FALSE)

#write.csv(dat_agg, file = "/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/cindex_lag0y_Onset_Binary_Global_DMI_square4.csv", row.names = FALSE)


ce_test <- conditional_effects(fit1, 
                               effects = "cindex_lag0y", 
                               conditions = list(cindex_lag0y = seq(-2, 3, length.out = 100)),
                               prob = 0.95)


loo_compare(loo(fit0), loo(fit1))
pp_check(fit1, ndraws=100)

waic(fit0)
waic(fit1)



##### RANDOMIZE CINDEX
library(combinat)
base_cindex <- dat_agg$cindex_lag0y

B <- 10000
S <- c()
for (i in 1:B){
  dat_agg$cindex_lag0y <- base_cindex[sample(seq_along(base_cindex))]
  fit <- lm(conflict_proportion ~ I(cindex_lag0y^1) + year + bool1989, data=dat_agg)
  S <- append(S, coef(fit)["I(cindex_lag0y^1)"])
}


hist(S, breaks='scott')
abline(v=0.0038766, col='red', lwd=2.0)
#abline(v=0.02324835, col='black', lwd=2.0, lty=1)


quantile(S, 0.903)





