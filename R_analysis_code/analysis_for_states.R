# R CODE TO ANALYZE STATE-LEVEL PANEL AND GROUPED DATA
# ... THIS CODE IS RAN CHUNK-WISE BASED ON USER'S DESIRED ANALYSIS.
# NOTE: STATE-LEVEL ANALYSES USES TWO KINDS OF RESPONSE VARIABLES:
# ....... [1.] ONSET BINARY (USED FOR DOSE-RESPONSE & GROUPED LOGIT PANEL MODELS);
# ....... [2.] ANNUAL CONFLICT RISK (USED FOR GROUPED LINEAR MODELS AND IS DERIVED FROM ONSET BINARY RESPONSE).
library(brms)
library(dplyr)
library(ggplot2)

panel_data_path <- '<FILE PATH HERE>/Onset_Binary_GlobalState_DMItype2.csv' # <-- PANEL DATA PATH, ALL STATE PANELS HAVE BINARY RESPONSE
dat <- read.csv(panel_data_path)

# PROCESS PANEL DATA
sum(dat$conflict_binary)
#head(dat)
dat <- subset(dat, year!=1989)
dat$bool1989 <- ifelse(dat$year<1989,0,1)
dat$year <- dat$year - min(dat$year)
dat$loc_id <- as.factor(dat$loc_id)

unique_psi <- dat %>%
  group_by(loc_id) %>%
  summarise(
    pop_avg_psi = first(pop_avg_psi), # <-- POP_AVG_PSI REFERS TO POPULATION-AVG. TELECONNECTION STRENGTH/CORR.
    total_conflict_onsets = sum(conflict_binary))
hist(unique_psi$pop_avg_psi, breaks='scott')
hist(unique_psi$total_conflict_onsets, breaks='scott')

#dat <- dat %>% mutate(cindex_lag0y = replace(cindex_lag0y, year == 2023, +1.960)) # NINO3


# PARTITION PANEL BY POPULATION-AVG. TELECONNECTION STRENGTH/CORR., pop_avg_psi
quantile(unique_psi$pop_avg_psi, c(0.1, 0.2, 0.30, 0.45, 0.50, 0.60, 0.7, 0.80, 0.85, 0.90, 0.925))
dat_help <- subset(dat, pop_avg_psi >= 0.42)
sum(dat_help$conflict_binary)


# A. CODE TO ESTIMATE GROUPED LOGISTIC PANEL REGRESSION W/ RANDOM EFFECTS:
mod.logit <- brm(
  conflict_binary ~ cindex_lag0y + I(cindex_lag0y^2) +
    bool1989 + (1 + year || loc_id),
  data = dat_help, family = bernoulli(link = "logit"),  # USING ONE SIDE OF THE PARTITIONED DATA dat_help
  iter = 5000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b)
)
plot(mod.logit)
print(summary(mod.logit, prob = 0.95), digits = 4)

save(mod.logit, file = "<FILE PATH AND NAME HERE>.RData")


# B. CODE TO ESTIMATE LINEAR MODEL FOR ACR:
dat_agg <- dat_help %>%
  group_by(year) %>%
  summarise(
    conflict_proportion = sum(conflict_binary) / n(), # THIS IS OUR ACR VARIABLE
    bool1989 = first(bool1989), 
    cindex_lag0y = first(cindex_lag0y), 
    cindex_lag1y = first(cindex_lag1y), 
    cindex_lag2y = first(cindex_lag2y)
  )
#dat_agg$cindex_lag0y[73] <- +1.96000000 #NINO3
#dat_agg$cindex_lag1y[73] <- -0.73333333 #NINO3
plot(dat_agg$cindex_lag0y, dat_agg$conflict_proportion)

fit.grouped <- brm(
  conflict_proportion ~ cindex_lag0y + I(cindex_lag0y^2) +
    year + bool1989,
  data = dat_agg, family = gaussian(), 
  iter = 5000, chains=2, warmup=500, cores=1,
  prior = c(
    prior(normal(0, 2.5), class = "b"),                # regression coefficients
    prior(normal(0, 5), class = "Intercept"),          # intercept term
    prior(exponential(1), class = "sigma")             # residual standard deviation
  )
)

plot(fit.grouped)
print(summary(fit.grouped, prob = 0.95), digits = 5)


ce <- conditional_effects(fit.grouped, effects = "cindex_lag0y", prob = 0.90, resolution = 500)
plot(ce, ask = FALSE)
ce_data <- ce$cindex_lag0y
write.csv(ce_data, file = "<FILE PATH HERE>/Onset_Binary_GlobalState_DMItype2_strong_ci90_linear.csv", row.names = FALSE)


# C. CODE TO DETERMINE LINEAR, NONLINEAR, CONTEMPORANEOUS, LAGGED, INTERACTION EFFECTS (DOSE-RESPONSE), CI=CLIMATE INDEX
# mod.a: I_t                                                                                            : cindex_lag0y
# mod.b: I_t + I_t-1                                                                                    : cindex_lag0y + cindex_lag1y
# mod.c: I_t + I_t^2                                                                                    : cindex_lag0y + I(cindex_lag0y^2)
# mod.d: I_t + (I_t x Psi)                                                                              : cindex_lag0y + I(cindex_lag0y*pop_avg_psi)
# mod.e: I_t + I_t-1 + (I_t x Psi) + (I_t-1 x Psi)                                                      : cindex_lag0y + cindex_lag1y + I(cindex_lag0y*pop_avg_psi) + I(cindex_lag1y*pop_avg_psi)
# mod.f: I_t + (I_t x Psi) + I_t^2 + (I_t^2 x Psi)                                                      : cindex_lag0y + I(cindex_lag0y*pop_avg_psi) + I(cindex_lag0y^2) + I((cindex_lag0y^2)*pop_avg_psi)
# mod.g: I_t + I_t-1 + I_t^2 + I_t-1^2 + (I_t x Psi) + (I_t-1 x Psi) + (I_t^2 x Psi) + (I_t-1^2 x Psi)  : cindex_lag0y + cindex_lag1y + I(cindex_lag0y^2) + I(cindex_lag1y^2) + I(cindex_lag0y*pop_avg_psi) + I(cindex_lag1y*pop_avg_psi) + I((cindex_lag0y^2)*pop_avg_psi) + I((cindex_lag1y^2)*pop_avg_psi)


mod.a <- brm(
  conflict_binary ~ cindex_lag0y +                # REPLACE BASE ON mod
    bool1989 + (1 + year || loc_id), 
  data = dat, family = bernoulli(link = "logit"), # USING THE ENTIRE GLOBAL PANEL VIA dat
  iter = 5000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b)
)
plot(mod.a)
print(summary(mod.a, prob = 0.90), digits = 4)

save(mod.a, file = "<PATH HERE>/Onset_Binary_GlobalState_DMItype2_moda.RData")


# D. CODE FOR LAGGED EFFECT OF ACR

dat_agg <- dat_agg %>%
  arrange(year) %>%                             
  mutate(conflict_proportion_lag1y = lag(conflict_proportion, 1)) %>%
  filter(!is.na(conflict_proportion_lag1y))

fit.lagged <- brm(
  conflict_proportion ~ conflict_proportion_lag1y + cindex_lag0y + I(cindex_lag0y^2) +
    year + bool1989,
  data = dat_agg, family = gaussian(), 
  iter = 5000, chains=2, warmup=500, cores=1,
  prior = c(
    prior(normal(0, 2.5), class = "b"),                # regression coefficients
    prior(normal(0, 5), class = "Intercept"),          # intercept term
    prior(exponential(1), class = "sigma")             # residual standard deviation
  )
)
plot(fit.lagged)
print(summary(fit.lagged, prob = 0.90), digits = 4)


# E. ADD NINO3 INDEX DATA FOR IOD + ENSO JOINT ANALYSIS
test <- subset(dat, loc_id==534)
nino3_lag0y <- test[c('cindex_lag0y', 'year')]
colnames(nino3_lag0y) <- c('nino3_lag0y', 'year')
dat <- dat %>% left_join(
  nino3_lag0y %>% 
    select(year, nino3_lag0y),
  by = "year"
)


# F. IOD TELECONNECTION TUNING FOR FINDING IOD-CONFLICT RESPONSE
#mod0:  pop_avg_psi >= 0.00, cindex terms applied to all  (0th quantile)
#modA:  pop_avg_psi >= 0.10, cindex terms applied to all  (30th quantile)
#modB:  pop_avg_psi >= 0.15, cindex terms applied to all  (50th quantile)
#modC:  pop_avg_psi >= 0.23, cindex terms applied to all  (70th quantile)
#modD:  pop_avg_psi >= 0.30, cindex terms applied to all  (80th quantile)
#modE:  pop_avg_psi >= 0.36, cindex terms applied to all  (85th quantile)
#modF:  pop_avg_psi >= 0.42, cindex terms applied to all  (90th quantile)
#modG:  pop_avg_psi >= 0.60, cindex terms applied to all  (95th quantile)

# FOR< OF CLIMATE TERMS: #cindex_lag0y:I(pop_avg_psi >= SOME_VAL) + I(cindex_lag0y^2):I(pop_avg_psi >= SOME_VAL)

modA <- brm(conflict_binary ~ cindex_lag0y:I(pop_avg_psi >= 0.10) + I(cindex_lag0y^2):I(pop_avg_psi >= 0.10) + 
              bool1989 + (1 + year || loc_id), 
            data = dat, family = bernoulli(link = "logit"),
            iter = 3500, chains=2, warmup=500, cores=2,
            prior = prior(normal(0, 3), class = b)
)
plot(modA)
print(summary(modA, prob = 0.90), digits = 4)

# COMPUTE ELPD VIA LOO-CV (CAN TAKE A WHILE TO COMPUTE...)
loo.0 <- loo(mod0)
loo.A <- loo(modA)
loo.B <- loo(modB)
loo.C <- loo(modC)
loo.D <- loo(modD)
loo.E <- loo(modE)
loo.F <- loo(modF)
loo.G <- loo(modG)

# COMPARE:
loo_compare(loo.0,
            loo.A,
            loo.B,
            loo.C,
            loo.D,
            loo.E,
            loo.F,
            loo.G)