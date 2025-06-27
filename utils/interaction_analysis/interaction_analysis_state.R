################################################################################
# STATE: TELECONNECTION & INDEX INTERACTION ANALYSIS ------------------------- #
################################################################################
library(brms)
library(dplyr)
library(ggplot2)

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_DMItype2.csv'
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
    pop_avg_psi = first(pop_avg_psi),
    total_conflict_onsets = sum(conflict_binary))
hist(unique_psi$pop_avg_psi, breaks='scott')
hist(unique_psi$total_conflict_onsets, breaks='scott')


mod <- brm(
  conflict_binary ~ 
    cindex_lag0y + I(cindex_lag0y*psi) + bool1989 +
    (1 + year || loc_id), # random effects and random trends
  data = dat, family = bernoulli(link = "logit"),
  iter = 3000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b)
)
plot(mod)
print(summary(mod, prob = 0.90), digits = 4)
ce <- conditional_effects(mod, effects = "cindex_lag0y", prob = 0.90)
plot(ce, ask = FALSE)


########################
########################
mod0 <- brm(
  conflict_binary ~ 
    cindex_lag0y + bool1989 +
    (1 + year || loc_id), # random effects and random trends
  data = dat, family = bernoulli(link = "logit"),
  iter = 3000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b),
  save_pars = save_pars(all = TRUE)
)
mod1 <- brm(
  conflict_binary ~ 
    cindex_lag0y + I(cindex_lag0y*psi) + bool1989 +
    (1 + year || loc_id), # random effects and random trends
  data = dat, family = bernoulli(link = "logit"),
  iter = 3000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b),
  save_pars = save_pars(all = TRUE)
)

post_prob(mod1, mod0)
bayes_factor(mod1, mod0)


######
library(boot)

?boot::inv.logit
xx <- seq(0,1.0,length.out=101)
psi_crit <- 0.5
yy <- boot::inv.logit(10*(xx-psi_crit))
plot(xx,yy,type='b')





bform <- bf(
  conflict_binary ~ alpha0 + 
    alpha1*inv_logit(20 * (pop_avg_psi - beta1))*cindex_lag0y +
    alpha2*inv_logit(20 * (pop_avg_psi - beta2))*pow(cindex_lag0y, 2),
  alpha0 ~ 1 + (1 + year || loc_id),  
  alpha1 ~ 1,
  alpha2 ~ 1,
  beta1  ~ 1,
  beta2  ~ 1,
  nl = TRUE
)

priors <- c(
  prior(normal(0, 2),          nlpar = "alpha0", coef = "Intercept"),
  prior(exponential(1),        class = "sd",  nlpar = "alpha0"),   # sd for both REs
  prior(normal(0, 1),          nlpar = "alpha1"),
  prior(normal(0, 1),          nlpar = "alpha2"),
  prior(uniform(0, 1),         nlpar = "beta1", lb = 0, ub = 1),
  prior(uniform(0, 1),         nlpar = "beta2", lb = 0, ub = 1)
)

fit <- brm(
  formula  = bform,
  data     = dat,
  family   = bernoulli(link = "logit"),
  prior    = priors,
  chains   = 2, cores = 2, warmup = 500,
  control  = list(adapt_delta = 0.95)
)

plot(fit)
print(summary(fit, prob = 0.90), digits = 4)





