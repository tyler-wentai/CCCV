################################################################################
# GRID CELL: TELECONNECTION & INDEX INTERACTION ANALYSIS --------------------- #
################################################################################
library(brms)
library(dplyr)
library(ggplot2)

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Binary_Global_DMItype2_square4.csv'
dat <- read.csv(panel_data_path)
#dat <- dat %>% mutate(cindex_lag0y = replace(cindex_lag0y, year == 2023, 1.96000000))

sum(dat$conflict_binary)
#unique(dat$conflict_count)
#plot(subset(dat, loc_id=='loc_1')$cindex_lag0, type='l')
#View(dat)
dat <- subset(dat, year!=1989)
dat$bool1989 <- ifelse(dat$year<1989,0,1)
dat$year <- dat$year - min(dat$year)
dat$loc_id <- as.factor(dat$loc_id)

unique_psi <- dat %>%
  group_by(loc_id) %>%
  summarise(
    psi = first(psi),
    total_conflict_counts = sum(conflict_binary))
hist(unique_psi$psi, breaks='scott')
hist(unique_psi$total_conflict_counts, breaks='scott')


mod <- brm(
  conflict_binary ~ 
    cindex_lag0y + I(cindex_lag0y*psi) + bool1989 +
    (1 + year || loc_id), # random effects and random trends
  data = dat, family = bernoulli(link = "logit"),
  iter = 2000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b)
)
plot(mod0)
print(summary(mod, prob = 0.90), digits = 4)
ce <- conditional_effects(mod, effects = "cindex_lag0y", prob = 0.90)
plot(ce, ask = FALSE)


########################
########################
dat$cindex_x_psi <- dat$cindex_lag0y*dat$psi
View(dat)

mod0 <- brm(
  conflict_binary ~ cindex_lag0y + I(cindex_lag0y^2) + bool1989 +
    (1 + year || loc_id), # random effects and random trends
  data = dat, family = bernoulli(link = "logit"),
  iter = 2000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b),
  save_pars = save_pars(all = TRUE)
)
plot(mod0)
print(summary(mod0, prob = 0.90), digits = 4)

mod1 <- brm(
  conflict_binary ~ cindex_lag0y + I(cindex_lag0y^2) + cindex_x_psi + I(cindex_x_psi^2) + bool1989 +
    (1 + year || loc_id), # random effects and random trends
  data = dat, family = bernoulli(link = "logit"),
  iter = 2000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b),
  save_pars = save_pars(all = TRUE)
)
plot(mod1)
print(summary(mod1, prob = 0.99), digits = 4)

post_prob(mod1, mod0)
bayes_factor(mod0, mod1)





######
library(boot)

?boot::inv.logit
xx <- seq(0,1.5,length.out=101)
psi_crit <- 0.5
yy <- boot::inv.logit(20*(xx-psi_crit))
plot(xx,yy,type='b')





bform <- bf(
  conflict_binary ~ alpha0 + 
    alpha1 * cindex_lag0y * inv_logit(10 * (psi - beta)),
  alpha0 ~ 1 + (1 + year || loc_id),  
  alpha1 ~ 1,
  beta   ~ 1,
  nl = TRUE
)

priors <- c(
  prior(normal(0, 5),          nlpar = "alpha0", coef = "Intercept"),
  prior(exponential(1),        class = "sd",  nlpar = "alpha0"),   # sd for both REs
  prior(normal(0, 5),          nlpar = "alpha1"),
  prior(normal(0, 5),          nlpar = "beta")
)

fit <- brm(
  formula  = bform,
  data     = dat,
  family   = bernoulli(link = "logit"),
  prior    = priors,
  chains   = 2, cores = 2,
  control  = list(adapt_delta = 0.95)
)





