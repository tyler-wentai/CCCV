# R CODE TO ANALYZE STATE-LEVEL PANEL AND GROUPED DATA
# ... THIS CODE IS RAN CHUNK-WISE BASED ON USER'S DESIRED ANALYSIS.
# NOTE: GRID CELL-LEVEL ANALYSES USES TWO KINDS OF RESPONSE VARIABLES:
# ....... [1.] ONSET COUNTS (USED FOR GROUPED POISSON MODELS);
# ....... [2.] ONSET BINARY (USED FOR DOSE-RESPONSE & GROUPED LOGIT PANEL MODELS).
library(brms)
library(dplyr)
library(ggplot2)

# CODE FOR [1.] ONSET COUNTS (USED FOR GROUPED POISSON MODELS) ---------------------------------------------------------
panel_data_path <- '<FILE PATH HERE>/Onset_Count_Global_DMItype2_square4.csv' # <-- PANEL DATA PATH, EXAMPLE FOR COUNTS
dat <- read.csv(panel_data_path)

# 1.A PROCESS PANEL DATA
sum(dat$conflict_count)
unique(dat$conflict_count)
dat <- subset(dat, year!=1989)
dat$bool1989 <- ifelse(dat$year<1989,0,1)
dat$year <- dat$year - min(dat$year)
dat$loc_id <- as.factor(dat$loc_id)

unique_psi <- dat %>%
  group_by(loc_id) %>%
  summarise(
    psi = first(psi), # <-- PSI REFERS TO TELECONNECTION STRENGTH/CORR.
    total_conflict_counts = sum(conflict_count))
hist(unique_psi$psi, breaks='scott')
hist(unique_psi$total_conflict_counts, breaks='scott')

# 1.B PARTITION PANEL BY TELECONNECTION STRENGTH/CORR., psi
quantile(unique_psi$psi, c(0.1, 0.2, 0.40, 0.45, 0.55, 0.60, 0.7, 0.80, 0.95, 1))
dat_help <- subset(dat, psi >= 0.55) # <-- APPLY TELECONNECTION/CORR. PARITION
sum(dat_help$conflict_count)

# 1.C CODE TO ESTIMATE POISSON MODEL FOR CONFLICT COUNTS:
dat_agg <- dat_help %>%
  group_by(year) %>%
  summarise(
    conflict_count = sum(conflict_count),
    bool1989 = first(bool1989), 
    cindex_lag0y = first(cindex_lag0y), 
    cindex_lag1y = first(cindex_lag1y), 
    cindex_lag2y = first(cindex_lag2y)
  )
#dat_agg$cindex_lag0y[73] <- +1.96000000 #NINO3
#dat_agg$cindex_lag1y[73] <- -0.73333333 #NINO3

fit.poisson <- brm(
  conflict_count ~ cindex_lag0y + I(cindex_lag0y^2) + year + bool1989,
  data = dat_agg,
  family = poisson(), # Poisson
  iter = 5000,
  chains = 2,
  warmup = 500,
  cores = 1,
  prior = c(
    prior(normal(0, 2.5), class = "b"),         # regression coefficients
    prior(normal(0, 5), class = "Intercept")    # intercept term
  )
)
plot(fit.poisson)
print(summary(fit.poisson, prob = 0.95), digits = 4)

ce <- conditional_effects(fit1, effects = "cindex_lag0y", prob = 0.90, resolution = 500)
plot(ce, ask = FALSE)
ce_data <- ce$cindex_lag0y
write.csv(ce_data, file = "<FILE PATH HERE>/CE_cindex_lag0y_Onset_Count_Global_DMItype2_square4_strong_ci90_poisson.csv", row.names = FALSE)


# 1.D. CODE FOR LAGGED EFFECT OF CONFLICT COUNTS
dat_agg <- dat_agg %>%
  arrange(year) %>%                             
  mutate(conflict_count_lag1y = lag(log(conflict_count+1), 1)) %>%
  filter(!is.na(conflict_count_lag1y))

fit.lagged <- brm(
  conflict_count ~ conflict_count_lag1y + cindex_lag0y + I(cindex_lag0y^2) + year + bool1989,
  data = dat_agg,
  family = poisson(),
  iter = 5000,
  chains = 2,
  warmup = 500,
  cores = 1,
  prior = c(
    prior(normal(0, 2.5), class = "b"),         # regression coefficients
    prior(normal(0, 5), class = "Intercept")    # intercept term
  )
)
plot(fit.lagged)
print(summary(fit.lagged, prob = 0.95), digits = 4)

# 1.E ADD NINO3 INDEX DATA FOR IOD + ENSO JOINT ANALYSIS
test <- subset(dat, loc_id==534)
nino3_lag0y <- test[c('cindex_lag0y', 'year')]
colnames(nino3_lag0y) <- c('nino3_lag0y', 'year')
dat <- dat %>% left_join(
  nino3_lag0y %>%
    select(year, nino3_lag0y),
  by = "year"
)


# CODE FOR [2.] ONSET BINARY (USED FOR DOSE-RESPONSE & GROUPED LOGIT PANEL MODELS) -------------------------------------
panel_data_path <- '<FILE PATH HERE>/Onset_Binary_Global_DMItype2_square4.csv' # <-- PANEL DATA PATH
dat <- read.csv(panel_data_path)

# 2.A PROCESS PANEL DATA
sum(dat$conflict_binary)
#View(dat)
dat <- subset(dat, year!=1989)
dat$bool1989 <- ifelse(dat$year<1989,0,1)
dat$year <- dat$year - min(dat$year)
dat$loc_id <- as.factor(dat$loc_id)

unique_psi <- dat %>%
  group_by(loc_id) %>%
  summarise(
    psi = first(psi),   # <-- PSI REFERS TO TELECONNECTION STRENGTH/CORR.
    total_conflict_counts = sum(conflict_binary))
hist(unique_psi$psi, breaks='scott')
hist(unique_psi$total_conflict_counts, breaks='scott')

#dat <- dat %>% mutate(cindex_lag0y = replace(cindex_lag0y, year == 2023, +1.960)) # NINO3

# 2.B PARTITION PANEL BY TELECONNECTION STRENGTH/CORR., psi
quantile(unique_psi$psi, c(0.1, 0.2, 0.30, 0.45, 0.55, 0.60, 0.7, 0.80, 0.95, 1))
dat_help <- subset(dat, psi < 0.41) # <-- APPLY TELECONNECTION/CORR. PARITION
sum(dat_help$conflict_binary)

# 2.C CODE TO ESTIMATE GROUPED LOGISTIC PANEL REGRESSION W/ RANDOM EFFECTS:
mod.logit <- brm(
  conflict_binary ~ cindex_lag0y + I(cindex_lag0y^2) +
    bool1989 + (1 + year || loc_id),
  data = dat_help, family = bernoulli(link = "logit"),  # USING ONE SIDE OF THE PARTITIONED DATA dat_help
  iter = 5000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b)
)
plot(mod.logit)
print(summary(mod.logit, prob = 0.90), digits = 4)

save(mod.logit, file = "<FILE PATH AND NAME HERE>.RData")

# 2.D CODE TO DETERMINE LINEAR, NONLINEAR, CONTEMPORANEOUS, LAGGED, INTERACTION EFFECTS (DOSE-RESPONSE), CI=CLIMATE INDEX
# mod.a: I_t                                                                                            : cindex_lag0y
# mod.b: I_t + I_t-1                                                                                    : cindex_lag0y + cindex_lag1y
# mod.c: I_t + I_t^2                                                                                    : cindex_lag0y + I(cindex_lag0y^2)
# mod.d: I_t + (I_t x Psi)                                                                              : cindex_lag0y + I(cindex_lag0y*psi)
# mod.e: I_t + I_t-1 + (I_t x Psi) + (I_t-1 x Psi)                                                      : cindex_lag0y + cindex_lag1y + I(cindex_lag0y*psi) + I(cindex_lag1y*psi)
# mod.f: I_t + (I_t x Psi) + I_t^2 + (I_t^2 x Psi)                                                      : cindex_lag0y + I(cindex_lag0y*psi) + I(cindex_lag0y^2) + I((cindex_lag0y^2)*psi)
# mod.g: I_t + I_t-1 + I_t^2 + I_t-1^2 + (I_t x Psi) + (I_t-1 x Psi) + (I_t^2 x Psi) + (I_t-1^2 x Psi)  : cindex_lag0y + cindex_lag1y + I(cindex_lag0y^2) + I(cindex_lag1y^2) + I(cindex_lag0y*psi) + I(cindex_lag1y*psi) + I((cindex_lag0y^2)*psi) + I((cindex_lag1y^2)*psi)

mod.a <- brm(
  conflict_binary ~ cindex_lag0y +                  # REPLACE BASE ON mod
    bool1989 + (1 + year || loc_id), 
  data = dat, family = bernoulli(link = "logit"),   # USING THE ENTIRE GLOBAL PANEL VIA dat
  iter = 5000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b)
)
plot(mod.a)
print(summary(mod.a, prob = 0.95), digits = 4)

save(mod.a, file = "<PATH HERE>/Onset_Binary_GlobalState_DMItype2_moda.RData")


# 2.E IOD TELECONNECTION TUNING FOR FINDING IOD-CONFLICT RESPONSE
#mod0:  psi >= 0.00, cindex terms applied to all  (0th quantile)
#modA:  psi >= 0.10, cindex terms applied to all  (50th quantile)
#modB:  psi >= 0.15, cindex terms applied to all  (65th quantile)
#modC:  psi >= 0.22, cindex terms applied to all  (75th quantile)
#modD:  psi >= 0.33, cindex terms applied to all  (85th quantile)
#modE:  psi >= 0.41, cindex terms applied to all  (90th quantile)
#modF:  psi >= 0.47, cindex terms applied to all  (92.5th quantile)
#modG:  psi >= 0.55, cindex terms applied to all  (95th quantile)
#modH:  psi >= 0.71, cindex terms applied to all  (97.5th quantile)

# FORM OF CLIMATE TERMS: #cindex_lag0y:I(pop_avg_psi >= SOME_VAL) + I(cindex_lag0y^2):I(pop_avg_psi >= SOME_VAL)

modA <- brm(conflict_binary ~ cindex_lag0y:I(psi >= 0.10) + I(cindex_lag0y^2):I(psi >= 0.10) + 
              bool1989 + (1 + year || loc_id), 
            data = dat, family = bernoulli(link = "logit"), #gaussian(),
            iter = 1700, chains=2, warmup=200, cores=2,
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
loo.H <- loo(modH)

# COMPARE:
loo_compare(loo.0,
            loo.A,
            loo.B,
            loo.C,
            loo.D,
            loo.E,
            loo.F,
            loo.G,
            loo.F)

save(loo.A, file = "<FILE PATH HERE>/Onset_Binary_Global_DMItype2_square4_modA_loo.RData")
load("<FILE PATH HERE>/Onset_Binary_Global_DMItype2_square4_modA_loo.RData")