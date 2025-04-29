library(brms)
library(dplyr)
library(ggplot2)

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_DMI_square1.csv'
dat <- read.csv(panel_data_path)

sum(dat$conflict_count)
unique(dat$conflict_count)
#plot(subset(dat, loc_id=='loc_1')$cindex_lag0, type='l')
#head(dat)
#dat <- subset(dat, year!=1989)
dat$bool1989 <- ifelse(dat$year<1989,0,1)
dat$year <- dat$year - min(dat$year)
dat$loc_id <- as.factor(dat$loc_id)

unique_psi <- dat %>%
  group_by(loc_id) %>%
  summarise(
    psi = first(psi),
    total_conflict_counts = sum(conflict_count))
hist(unique_psi$psi, breaks='scott')
hist(unique_psi$total_conflict_counts, breaks='scott')


quantile(unique_psi$psi, c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 0.95))
dat_help <- subset(dat, psi < 0.90)
sum(dat_help$conflict_count)

View(dat_help)

m2 <- brm(
  conflict_binary ~ 
    cindex_lag0y +
    (1 + year || loc_id), #year:loc_id + loc_id,
  data = dat_help, family = gaussian(),#bernoulli(link = "logit"), 
  iter = 4000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b)
)
plot(m2)
print(summary(m2, prob = 0.95), digits = 4)
ce <- conditional_effects(m2, effects = "cindex_lag0y", prob = 0.95)
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
    conflict_count = sum(conflict_count),
    bool1989 = first(bool1989), 
    cindex_lag0y = first(cindex_lag0y), 
    cindex_lag1y = first(cindex_lag1y), 
    cindex_lag2y = first(cindex_lag2y)
  )
#View(dat_agg)
#dat_agg <- subset(dat_agg, year!=47) # NEED THIS BECAUSE SPI6 DOES HAVE 2022...
#dat_agg$cindex_lag0y[73] <- +1.96000000
#dat_agg$cindex_lag1y[73] <- -0.73333333

#write.csv(dat_agg, file = "/Users/tylerbagwell/Desktop/panel_datasets/results/CE_cindex_lag0y_Onset_Count_Global_DMI_square4.csv", row.names = FALSE)

mod0 <- lm(conflict_count ~ year + bool1989, data=dat_agg)
mod1 <- lm(conflict_count ~ I(cindex_lag0y^1) + I(cindex_lag0y^2) + year + bool1989, data=dat_agg)
summary(mod1)
plot(dat_agg$cindex_lag0y, dat_agg$conflict_count)
abline(mod1)

AIC(mod0)
AIC(mod1)

pois_mod <- glm(
  conflict_count ~ I(cindex_lag1y^1) + I(cindex_lag1y^2) + year + bool1989,
  family = poisson(link = "log"),
  data   = dat_agg
)
summary(pois_mod)


plot(dat_agg$conflict_count, type='l')
stud_res <- rstudent(mod1)
plot(dat_agg$year, stud_res)
abline(h=0)


BIC(mod0)
BIC(mod1)

plot(dat_agg$cindex_lag0y, dat_agg$conflict_count)
abline(mod1)
abline(v=c(-2,2))

plot(mod1)
plot(cooks.distance(mod1),type="b",pch=18,col="red")
N = 74
k = 4
cutoff = 4/ (N-k-1)
abline(h=cutoff,lty=2)

dat_agg_cleaned <- subset(dat_agg, year != 39)

###### BAYESIAN POISSON FIT
fit1 <- brm(
  conflict_count ~ cindex_lag0y + I(cindex_lag0y^2) + year + bool1989,
  data = dat_agg,
  family = poisson(), #gaussian(),  # Poisson
  iter = 5000,
  chains = 2,
  warmup = 1000,
  cores = 1,
  prior = c(
    prior(normal(0, 2.5), class = "b"),         # regression coefficients
    prior(normal(0, 5), class = "Intercept")    # intercept term
  )
)

print(summary(fit1, prob = 0.90), digits = 4)
#plot(fit1)

ce <- conditional_effects(fit1, effects = "cindex_lag0y", prob = 0.90, resolution = 500)
plot(ce, ask = FALSE)
ce_data <- ce$cindex_lag0y
write.csv(ce_data, file = "/Users/tylerbagwell/Desktop/panel_datasets/results/CE_cindex_lag0y_Onset_Count_Global_DMI_square4_psihigh2.25_95ci_poisson.csv", row.names = FALSE)

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
  fit <- glm(conflict_count ~ I(cindex_lag0y^1) + I(cindex_lag0y^2) + year + bool1989,
    family = poisson(link = "log"),
    data   = dat_agg)
  S <- append(S, coef(fit)["I(cindex_lag0y^2)"])
}


hist(S, breaks='scott')
abline(v=-0.427453, col='red', lwd=2.0)
#abline(v=0.02324835, col='black', lwd=2.0, lty=1)

F_emp <- ecdf(S)
quantile_of_x0 <- F_emp(-0.427453)
quantile_of_x0


