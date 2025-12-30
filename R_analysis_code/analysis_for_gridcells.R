# R CODE TO ANALYZE GRID CELL-LEVEL PANEL AND GROUPED DATA
# ... THIS CODE IS RAN CHUNK-WISE BASED ON USER'S DESIRED ANALYSIS.
# NOTE: GRID CELL-LEVEL ANALYSES USES TWO KINDS OF RESPONSE VARIABLES:
# ....... [1.] ONSET COUNTS (USED FOR GROUPED POISSON MODELS);
# ....... [2.] ONSET BINARY (USED FOR DOSE-RESPONSE & GROUPED LOGIT PANEL MODELS).
library(brms)
library(dplyr)
library(ggplot2)

# CODE FOR [1.] ONSET COUNTS (USED FOR GROUPED POISSON MODELS) ---------------------------------------------------------
panel_data_path = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/onset_datasets_grid/'
dat <- read.csv(file.path(panel_data_path, 'Onset_Count_Global_DMItype2_v3_square4_newonsetdata.csv')) # <-- PANEL DATA PATH, ALL STATE PANELS HAVE BINARY RESPONSE


# 1.A PROCESS PANEL DATA
sum(dat$conflict_count)
unique(dat$conflict_count)
dat <- subset(dat, year!=1989)
dat$bool1989 <- ifelse(dat$year<=1989,0,1)
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
quantile(unique_psi$psi, c(0.1, 0.2, 0.40, 0.45, 0.50, 0.66, 0.85, 0.90, 0.95, 0.96, 1))
dat_help <- subset(dat, psi >= 0.64575415) # <-- APPLY TELECONNECTION/CORR. PARITION
sum(dat_help$conflict_count)

unique(dat_help$SOVEREIGNT)
subset(dat_help, SOVEREIGNT=="Chad")
dat_help <- subset(dat_help, year!=65)


dat <- dat %>%
  arrange(loc_id, year) %>%
  group_by(loc_id) %>%
  mutate(
    cindex_fwd1y = if_else(lead(year) == year + 1, lead(cindex_lag0y), NA_real_)
  ) %>%
  ungroup() %>%
  filter(!is.na(cindex_fwd1y))


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
#dat_agg$cindex_lag0y[dat_agg$year==73] <- +1.96000000 #NINO3
#dat_agg$cindex_lag1y[73] <- -0.73333333 #NINO3
plot(dat_agg$cindex_lag0y, dat_agg$conflict_count)
plot(dat_agg$year, dat_agg$conflict_count)
abline(v=1989)
cor(dat_agg$year, dat_agg$conflict_count)

dat_agg <- subset(dat_agg, year!=65)
tail(dat_agg)

fit.poisson <- brm(
  conflict_count ~ cindex_lag0y + I(cindex_lag0y^2) + year + bool1989,
  data = dat_agg,
  family = poisson(), # Poisson
  iter = 10000,
  chains = 2,
  warmup = 500,
  cores = 1,
  prior = c(
    prior(normal(0, 3), class = "b"),           # regression coefficients
    prior(normal(0, 5), class = "Intercept")    # intercept term
  )
)
plot(fit.poisson)
print(summary(fit.poisson, prob = 0.985), digits = 4)

# cindex_lag0y   0.0936    0.0520
# cindex_lag0y   0.1097    0.0546

ce <- conditional_effects(fit.poisson, effects = "cindex_lag0y", prob = 0.90, resolution = 500)
plot(ce, ask = FALSE)
ce_data <- ce$cindex_lag0y
write.csv(ce_data, file = "<FILE PATH HERE>/CE_cindex_lag0y_Onset_Count_Global_DMItype2_square4_strong_ci90_poisson.csv", row.names = FALSE)

### MARGINALEFFECTS
library(marginaleffects)
# --- RESULTS, ENSO and Conflict, Wet v. Dry
avg_comparisons(
  fit.poisson,
  variables = list("cindex_lag0y" = c(-1.5, +1.5)), 
  comparison = "ratio",
  conf_level = 0.90,
  type = "response"
)

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
test <- subset(dat, loc_id=='loc_0')
nino3_dat <- test[c('cindex_lag0y', 'year')]
colnames(nino3_dat) <- c('nino3_lag0y', 'year')
nino3_dat$nino3_lag0y[74] <- +1.96000000 #NINO3
dat <- dat %>% left_join(
  nino3_dat %>%
    dplyr::select(year, nino3_lag0y),
  by = "year"
)
View(dat)

dat <- transform(dat,
                 E0 = nino3_lag0y,
                 E0_2 = I(nino3_lag0y^2)
)


# CODE FOR [2.] ONSET BINARY (USED FOR DOSE-RESPONSE & GROUPED LOGIT PANEL MODELS) -------------------------------------
panel_data_path = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/onset_datasets_grid/'
dat <- read.csv(file.path(panel_data_path, 'Onset_Binary_Global_DMItype2_v3_square4_newonsetdata.csv')) # <-- PANEL DATA PATH, ALL STATE PANELS HAVE BINARY RESPONSE

dat <- transform(dat,
                 I0   = cindex_lag0y,
                 I0_2 = I(cindex_lag0y^2),
                 I1   = cindex_lag1y,
                 I1_2 = I(cindex_lag1y^2)
)

# 2.A PROCESS PANEL DATA
sum(dat$conflict_binary)
#View(dat)
dat <- subset(dat, year!=1989)
dat$bool1989 <- ifelse(dat$year<=1989,0,1)
dat$year <- dat$year - min(dat$year)
dat$loc_id <- as.factor(dat$loc_id)

unique_psi <- dat %>%
  group_by(loc_id) %>%
  summarise(
    psi = first(psi),   # <-- PSI REFERS TO TELECONNECTION STRENGTH/CORR.
    total_conflict_counts = sum(conflict_binary))
hist(unique_psi$psi, breaks='scott')
hist(unique_psi$total_conflict_counts, breaks='scott')

max.psi <- max(dat$psi)
dat$psi <- dat$psi/max.psi

write.csv(unique_psi, "/Users/tylerbagwell/Desktop/DMItype2_grid_psi.csv")


dat <- dat %>%
  arrange(loc_id, year) %>%
  group_by(loc_id) %>%
  mutate(
    mindf1 = if_else(lead(year) == year + 1, lead(cindex_lag0y), NA_real_)
  ) %>%
  ungroup() %>%
  filter(!is.na(mindf1))


# 2.B PARTITION PANEL BY TELECONNECTION STRENGTH/CORR., psi
quantile(unique_psi$psi, c(0.00, 0.33, 0.50, 0.66, 0.75, 0.85, 0.90, 0.95))
dat_help <- subset(dat, psi > 0.37879156) # <-- APPLY TELECONNECTION/CORR. PARITION
sum(dat_help$conflict_binary)
View(dat_help)

unique(dat_help$SOVEREIGNT)

# 2.C CODE TO ESTIMATE GROUPED LOGISTIC PANEL REGRESSION W/ RANDOM EFFECTS:

mod.logit <- brm(
  conflict_binary ~ cindex_lag0y +
    bool1989 + (1 + year || loc_id),
  data = dat_help, family = bernoulli(link = "logit"),  # USING ONE SIDE OF THE PARTITIONED DATA dat_help
  iter = 5000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b)
)
plot(mod.logit)
print(summary(mod.logit, prob = 0.95), digits = 4)

save(mod.logit, file = "<FILE PATH AND NAME HERE>.RData")

# 2.D CODE TO DETERMINE LINEAR, NONLINEAR, CONTEMPORANEOUS, LAGGED, INTERACTION EFFECTS (DOSE-RESPONSE), CI=CLIMATE INDEX
# mod.0: I_t+1                                                                                          : mindf1
# mod.a: I_t                                                                                            : mind0
# mod.b: I_t + I_t-1                                                                                    : cindex_lag0y + cindex_lag1y
# mod.c: I_t + I_t^2                                                                                    : cindex_lag0y + I(cindex_lag0y^2)
# mod.d: I_t + (I_t x Psi)                                                                              : cindex_lag0y + I(cindex_lag0y*psi)
# mod.e: I_t + I_t-1 + (I_t x Psi) + (I_t-1 x Psi)                                                      : cindex_lag0y + cindex_lag1y + I(cindex_lag0y*psi) + I(cindex_lag1y*psi)
# mod.f: I_t + (I_t x Psi) + I_t^2 + (I_t^2 x Psi)                                                      : cindex_lag0y + I(cindex_lag0y*psi) + I(cindex_lag0y^2) + I((cindex_lag0y^2)*psi)
# mod.g: I_t + I_t-1 + I_t^2 + I_t-1^2 + (I_t x Psi) + (I_t-1 x Psi) + (I_t^2 x Psi) + (I_t-1^2 x Psi)  : cindex_lag0y + cindex_lag1y + I(cindex_lag0y^2) + I(cindex_lag1y^2) + I(cindex_lag0y*psi) + I(cindex_lag1y*psi) + I((cindex_lag0y^2)*psi) + I((cindex_lag1y^2)*psi)


#mod.test
dat$psi_lo <- as.integer(dat$psi <= 0.27917394)
dat$psi_md <- as.integer(dat$psi > 0.27917394 & dat$psi < 0.55164385)
dat$psi_hi <- as.integer(dat$psi >= 0.55164385)

#mod.test2
dat$psi_lo <- as.integer(dat$psi <= 0.19555889)
dat$psi_md <- as.integer(dat$psi > 0.19555889 & dat$psi < 0.41895949)
dat$psi_hi <- as.integer(dat$psi >= 0.41895949)

mod.test2 <- brm(
  conflict_binary ~ I0:psi_lo + I0:psi_md + I0:psi_hi +  # REPLACE BASE ON mod
    bool1989 + (1 + year || loc_id), 
  data = dat, family = bernoulli(link = "logit"),   # USING THE ENTIRE GLOBAL PANEL VIA dat
  iter = 5000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b)
)
plot(mod.f)
print(summary(mod.test2, prob = 0.85), digits = 4)

hypothesis(mod.a, "eei_lag0y - cindex_lag0y > 0")

save(mod.a, file = "<PATH HERE>/Onset_Binary_GlobalState_DMItype2_moda.RData")


# 2.E IOD TELECONNECTION TUNING FOR FINDING IOD-CONFLICT RESPONSE
quantile(unique_psi$psi, c(0.00, 0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975))
#mod0:  psi >= (0th quantile)
#modA:  psi >= (30th quantile)
#modB:  psi >= (50th quantile)
#modC:  psi >= (70th quantile)
#modD:  psi >= (80th quantile)
#modE:  psi >= (85th quantile)
#modF:  psi >= (90th quantile)
#modG:  psi >= (95th quantile)
#modH:  psi >= (97.5th quantile)

# FORM OF CLIMATE TERMS: #cindex_lag0y:I(pop_avg_psi >= SOME_VAL) + I(cindex_lag0y^2):I(pop_avg_psi >= SOME_VAL)

dat$psi_on <- as.integer(dat$psi >= 0.31586041) # CHANGE THRESHOLD

t0 <- proc.time()
modD <- brm(conflict_binary ~ I0:psi_on + I0_2:psi_on +
              bool1989 + (1 + year || loc_id), 
            data = dat, family = bernoulli(link = "logit"), #gaussian(),
            iter = 1500, chains=2, warmup=200, cores=2,
            prior = prior(normal(0, 3), class = b)
)
loo.D <- loo(modD)
(proc.time() - t0)["elapsed"]
plot(modE)
print(summary(mod0, prob = 0.90), digits = 4)

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

save(loo.E, file = "/Users/tylerbagwell/Desktop/dmi_parametertunning_20251229/Onset_Count_Global_DMItype2_v3_square4_newonsetdata_modE_loo.RData")
load("/Users/tylerbagwell/Desktop/dmi_parametertunning_20251229/Onset_Count_Global_DMItype2_v3_square4_newonsetdata_modE_loo.RData")

loo_compare(loo.0,loo.E,loo.G,loo.H)

# CREATING PANEL WITH BOTH ECI AND EEI DATA:
# 1. load dat for EEI data first, then run below:
dat_eei <- dat %>%
  group_by(loc_id) %>%
  summarise(
    eei_psi  = first(psi)
  )
test <- subset(dat, loc_id=="loc_0")
eei_lag0y <- test[c('cindex_lag0y', 'cindex_lag1y', 'cindex_fwd1y','year')]
colnames(eei_lag0y) <- c('eei_lag0y', 'eei_lag1y', 'eei_fwd1y','year')
# 2. load dat for ECI data, then run below:
dat <- dat %>% left_join(
  dat_eei %>% 
    dplyr::select(loc_id, eei_psi),
  by = "loc_id"
)
dat <- dat %>% left_join(
  eei_lag0y %>% 
    dplyr::select(year, eei_lag0y, eei_lag1y, eei_fwd1y),
  by = "year"
)
View(dat) # check joined and final panel
colnames(dat)

dat <- transform(dat,
                 CF1 = cindex_fwd1y,
                 EF1 = eei_fwd1y,
                 C0 = cindex_lag0y,
                 C1 = cindex_lag1y,
                 E0 = eei_lag0y,
                 E1 = eei_lag1y,
                 psiC = psi,
                 psiE = eei_psi
)

mod.0 <- brm(
  conflict_binary ~ EF1 + CF1 +
    bool1989 + (1 + year || loc_id), 
  data = dat, family = bernoulli(link = "logit"), # USING THE ENTIRE GLOBAL PANEL VIA dat
  iter = 5000, chains=2, warmup=500, cores=2,
  prior = prior(normal(0, 3), class = b)
)
plot(mod)
print(summary(mod.0, prob = 0.90), digits = 4)






test <- subset(dat, loc_id=='loc_0')
dmi_noenso_lag0y <- test[c('cindex_lag0y', 'year')]
colnames(dmi_noenso_lag0y) <- c('dmi_noenso_lag0y', 'year')
dat <- dat %>% left_join(
  dmi_noenso_lag0y %>% 
    dyplr::select(year, dmi_noenso_lag0y),
  by = "year"
)
colnames(dat)
View(dat)





# factors: neg / neu /pos
tau <- 0.05
band3 <- function(z, tau) {
  cut(z,
      breaks = c(-Inf, -tau, tau, Inf),
      labels = c("neg","neu","pos"),
      right = TRUE)
}
dat$psiE_band <- factor(band3(dat$psiE, tau), levels = c("neg","neu","pos"))
dat$psiC_band <- factor(band3(dat$psiC, tau), levels = c("neg","neu","pos"))

mod.a <- brm(
  conflict_binary ~ C0:psiC_band + E0:psiE_band +
    bool1989 + (1 + year || loc_id),
  data = dat, family = bernoulli("logit"),
  prior = prior(normal(0, 2), class = "b"),
  chains = 4, iter = 5000, warmup = 500, cores = 2
)
plot(mod.a)
print(summary(mod.a, prob = 0.90), digits = 4)


with(dat, sapply(split(loc_id, psiC_band), function(x) length(unique(x))))

#### LOSO
dat <- subset(dat, psi >= 0.0)

locs <- unique(dat$loc_id)
c.params <- c()
e.params <- c()

for (i in 1:length(locs)){
  print(paste0("...",i))
  dat.help <- subset(dat, loc_id!=locs[i])
  
  dat_agg <- dat.help %>%
    group_by(year) %>%
    summarise(
      conflict_count = sum(conflict_count),
      bool1989 = first(bool1989), 
      cindex_lag0y = first(cindex_lag0y), 
      cindex_lag1y = first(cindex_lag1y),
      eei_lag0y    = first(eei_lag0y)
    )
  
  pois.mod <- glm(conflict_count ~ cindex_lag0y + eei_lag0y + year + bool1989,
                family = poisson(link="log"),
               data = dat_agg)
  
  cindex.param <- pois.mod$coefficients["cindex_lag0y"]
  c.params <- append(c.params, cindex.param)
  eindex.param <- pois.mod$coefficients["eei_lag0y"]
  e.params <- append(e.params, eindex.param)
}

results.df <- data.frame(c.lag0 = c.params, e.lag0 = e.params, country = locs)

plot(results.df$c.lag0)
plot((results.df$c.lag0 - mean(results.df$c.lag0))/sd(results.df$c.lag0))
abline(h=c(-3,+3), lty=2)

plot(results.df$e.lag0)
plot((results.df$e.lag0 - mean(results.df$e.lag0))/sd(results.df$e.lag0))
abline(h=c(-3,+3), lty=2)

results.df$country[results.df$c.lag0>0.125]

dat.help <- subset(dat, SOVEREIGNT!="Myanmar")

dat_agg <- dat.help %>%
  group_by(year) %>%
  summarise(
    conflict_count = sum(conflict_count),
    bool1989 = first(bool1989), 
    cindex_lag0y = first(cindex_lag0y), 
    cindex_lag1y = first(cindex_lag1y),
    eei_lag0y    = first(eei_lag0y)
  )
pois.mod <- glm(conflict_count ~ cindex_lag0y + eei_lag0y + year + bool1989,
                family = poisson(link="log"),
                data = dat_agg)
summary(pois.mod)


(0.0024832-0.0030207)/0.0030207




dat$SOVEREIGNT[dat$loc_id=="loc_1134"]



#### LOYO
dat <- subset(dat, psi >= 0.00)

years <- unique(dat$year)
c.params <- c()
e.params <- c()

for (i in 1:length(years)){
  print(paste0("...",i))
  dat.help <- subset(dat, year!=years[i])
  
  dat_agg <- dat.help %>%
    group_by(year) %>%
    summarise(
      conflict_count = sum(conflict_count),
      bool1989 = first(bool1989), 
      cindex_lag0y = first(cindex_lag0y), 
      cindex_lag1y = first(cindex_lag1y),
      eei_lag0y    = first(eei_lag0y)
    )
  
  pois.mod <- glm(conflict_count ~ cindex_lag0y + eei_lag0y + year + bool1989,
                  family = poisson(link="log"),
                  data = dat_agg)
  
  cindex.param <- pois.mod$coefficients["cindex_lag0y"]
  c.params <- append(c.params, cindex.param)
  eindex.param <- pois.mod$coefficients["eei_lag0y"]
  e.params <- append(e.params, eindex.param)
}

results.df <- data.frame(c.lag0 = c.params, e.lag0 = e.params, years = years)

plot(results.df$c.lag0)
plot((results.df$c.lag0 - mean(results.df$c.lag0))/sd(results.df$c.lag0))
abline(h=c(-3,+3), lty=2)


plot(results.df$e.lag0)
plot((results.df$e.lag0 - mean(results.df$e.lag0))/sd(results.df$e.lag0))
abline(h=c(-3,+3), lty=2)

results.df$years[results.df$c.lag0<0.1]


dat.help <- subset(dat, year!=65)

dat_agg <- dat.help %>%
  group_by(year) %>%
  summarise(
    conflict_count = sum(conflict_count),
    bool1989 = first(bool1989), 
    cindex_lag0y = first(cindex_lag0y), 
    cindex_lag1y = first(cindex_lag1y),
    eei_lag0y    = first(eei_lag0y)
  )
pois.mod <- glm(conflict_count ~ cindex_lag0y + year + bool1989,
                family = poisson(link="log"),
                data = dat_agg)
summary(pois.mod)













#### LOYO w./ CI

dat <- subset(dat, psi >= 0.41)

years <- sort(unique(dat$year))
n <- length(years)
c.params <- numeric(n)
lo90 <- numeric(n)
hi90 <- numeric(n)

for (i in seq_along(years)){
  message(sprintf("...%d", i))
  dat.help <- subset(dat, year != years[i])
  
  dat_agg <- dat.help %>%
    group_by(year) %>%
    summarise(
      conflict_count = sum(conflict_count),
      bool1989 = first(bool1989),
      cindex_lag0y = first(cindex_lag0y),
      cindex_lag1y = first(cindex_lag1y),
      .groups = "drop"
    )
  
  pois.mod <- glm(conflict_count ~ cindex_lag0y + year + bool1989,
                  family = poisson(link="log"),
                  data = dat_agg)
  
  c.params[i] <- coef(pois.mod)[["cindex_lag0y"]]
  ci <- suppressMessages(confint(pois.mod, "cindex_lag0y", level = 0.90))
  lo90[i] <- ci[1]
  hi90[i] <- ci[2]
}

results.df <- data.frame(year_left_out = years,
                         beta = c.params,
                         lo90 = lo90,
                         hi90 = hi90)

# Raw coefficients with 90% CIs
op <- par(mar = c(8,4,2,1))
x <- seq_len(n)
plot(x, results.df$beta, pch = 16, xaxt = "n",
     xlab = "Left-out year", ylab = expression(hat(beta)[cindex_lag0y]), ylim=c(-0.05,0.3),
     main="Poisson, LOYO, NINO34, lag0, mrsos Drying")
axis(1, at = x, labels = results.df$year_left_out, las = 2, cex.axis = 0.7)
arrows(x0 = x, y0 = results.df$lo90, x1 = x, y1 = results.df$hi90,
       code = 3, angle = 90, length = 0.03)
abline(h = 0, lty = 2)
par(op)

results.df$beta <- (results.df$beta - mean(results.df$beta))/sd(results.df$beta)
plot(results.df$beta)
abline(h=-3)


### plotting
results.df1 <- results.df
results.df2 <- results.df


plot_panel <- function(df, main) {
  n <- nrow(df)
  x <- seq_len(n)
  plot(x, df$beta, pch = 16, xaxt = "n", cex=0.5,
       xlab = "Left-out year",
       ylab = expression(hat(beta)[NINO3_t]),
       ylim = c(-0.2, 0.3),
       main = main, cex.main = 0.8, cex.lab=1, cex.axis = 0.8)
  axis(1, at = x, labels = df$year_left_out + 1950, las = 2, cex.axis = 0.5)
  arrows(x0 = x, y0 = df$lo90, x1 = x, y1 = df$hi90,
         code = 3, angle = 90, length = 0.01)
  abline(h = 0, lty = 2)
}

png("/Users/tylerbagwell/Desktop/LOYO_poisson_weakvstel_NINO3t.png", width = 8, height = 4, units = "in", res = 300, type = "cairo", antialias = "subpixel")
par(mfrow = c(1, 2), mar = c(8, 4, 2, 1))
plot_panel(results.df1, "Poisson, LOYO, NINO3_t, Weakly affected grid cells")
plot_panel(results.df2, "Poisson, LOYO, NINO3_t, Teleconnected grid cells")
dev.off()




#### LOSO w./ CI

dat <- subset(dat, psi < 0.0 & year!=65)

locs <- sort(unique(dat$SOVEREIGNT))
n <- length(locs)
c.params <- numeric(n)
lo90 <- numeric(n)
hi90 <- numeric(n)

for (i in seq_along(locs)){
  message(sprintf("...%d", i))
  dat.help <- subset(dat, SOVEREIGNT != locs[i])
  
  dat_agg <- dat.help %>%
    group_by(year) %>%
    summarise(
      conflict_count = sum(conflict_count),
      bool1989 = first(bool1989),
      cindex_lag0y = first(cindex_lag0y),
      cindex_lag1y = first(cindex_lag1y),
      .groups = "drop"
    )
  
  pois.mod <- glm(conflict_count ~ cindex_lag0y + year + bool1989,
                  family = poisson(link="log"),
                  data = dat_agg)
  
  c.params[i] <- coef(pois.mod)[["cindex_lag0y"]]
  ci <- suppressMessages(confint(pois.mod, "cindex_lag0y", level = 0.90))
  lo90[i] <- ci[1]
  hi90[i] <- ci[2]
}

results.df <- data.frame(state_left_out = locs,
                         beta = c.params,
                         lo90 = lo90,
                         hi90 = hi90)

# Raw coefficients with 90% CIs
op <- par(mar = c(8,4,2,1))
x <- seq_len(n)
plot(x, results.df$beta, pch = 16, xaxt = "n", col='red',
     xlab = "Left-out state", ylab = expression(hat(beta)[cindex_lag0y]), ylim=c(-0.04,0.25),
     main="Poisson, LOGCO, NINO34, lag0, mrsos Drying")
axis(1, at = x, labels = results.df$state_left_out, las = 2, cex.axis = 0.7)
arrows(x0 = x, y0 = results.df$lo90, x1 = x, y1 = results.df$hi90,
       code = 3, angle = 90, length = 0.03)
abline(h = 0, lty = 2)
par(op)




avg_comparisons(fit.poisson, variables = list(cindex_lag0y = c(-0.1, +0.1)), re_formula = NA, conf_level = 0.90, type="link")


exp(0.00862)



##### LOYO BRMS FINAL LINEAR:
library(brms)
library(posterior)
library(dplyr)
library(purrr)
library(tidyr)
library(readr)
#options(brms.backend = "cmdstanr")  # or set backend="cmdstanr" per fit

dat_help <- transform(dat_help,
                      C0 = cindex_lag0y,
                      C1 = cindex_lag1y,
                      C0_2 = cindex_lag0y^2,
                      C1_2 = cindex_lag1y^2
)


years <- sort(unique(dat_help$year))
terms  <- c("C0","C0_2")
#terms  <- c("C0")
bvars  <- paste0("b_", terms)

# 1) fit once (leave out first year) to compile the model
dat_loyo <- dplyr::filter(dat_help, year != years[1])
dat_agg <- dat_loyo %>%
  group_by(year) %>%
  summarise(
    conflict_count = sum(conflict_count),
    bool1989 = first(bool1989), 
    C0 = first(C0), 
    C0_2 = first(C0_2)
  )
fit0 <- brm(
  conflict_count ~ C0 + C0_2 + year + bool1989,
  data = dat_agg,
  family = poisson(), # Poisson
  iter = 10000,
  chains = 2,
  warmup = 500,
  cores = 1,
  prior = c(
    prior(normal(0, 3), class = "b"),         # regression coefficients
    prior(normal(0, 5), class = "Intercept")    # intercept term
  )
)

# helper to summarize fixed effects for one holdout year
summarize_fit <- function(fit, y_left_out) {
  fe <- as.data.frame(brms::fixef(fit, probs = c(0.05, 0.95)))
  fe$term <- rownames(fe)
  dplyr::filter(fe, term %in% terms) |>
    dplyr::transmute(
      year_left_out = y_left_out,
      term,
      mean = Estimate,
      q05  = Q5,
      q95  = Q95
    )
}

loyo_stats <- summarize_fit(fit0, years[1])

for (y in years[-1]) {
  dat_y <- dplyr::filter(dat_help, year != y)
  dat_y <- dat_y %>%
    group_by(year) %>%
    summarise(
      conflict_count = sum(conflict_count),
      bool1989 = first(bool1989), 
      C0 = first(C0), 
      C0_2 = first(C0_2)
    )
  fit_y <- update(fit0, newdata = dat_y, recompile = FALSE, refresh = 0,
                  seed = 20251019 + which(years == y))
  loyo_stats <- dplyr::bind_rows(loyo_stats, summarize_fit(fit_y, y))
  rm(fit_y); gc()
}

loyo_wide <- loyo_stats |>
  dplyr::arrange(year_left_out, term) |>
  tidyr::pivot_wider(
    id_cols = year_left_out,
    names_from = term,
    values_from = c(mean, q05, q95)
  ) |>
  dplyr::arrange(year_left_out)
#View(loyo_wide)

idx <- seq_len(nrow(loyo_wide))
plot(idx, loyo_wide$mean_C0_2, pch = 16, xlab = "Row", ylab = "C0",
     ylim = range(-0.05, loyo_wide$q95_C0_2))
segments(idx, loyo_wide$q05_C0_2, idx, loyo_wide$q95_C0_2)
abline(h = 0, lty = 3)

readr::write_csv(loyo_wide, "/Users/tylerbagwell/Desktop/Onset_Count_Global_mrsosNINO34_square4_grouped_poisson_dry_loyo.csv")




##### LOSO BRMS FINAL POISSON:
library(brms)
library(posterior)
library(dplyr)
library(purrr)
library(tidyr)
library(readr)
#options(brms.backend = "cmdstanr")  # or set backend="cmdstanr" per fit

dat_help <- transform(dat_help,
                      C0 = cindex_lag0y,
                      C1 = cindex_lag1y,
                      C0_2 = cindex_lag0y^2,
                      C1_2 = cindex_lag1y^2
)


countries <- unique(dat_help$SOVEREIGNT)
#terms  <- c("C0")
terms  <- c("C0","C0_2")
bvars  <- paste0("b_", terms)
colnames(dat)

# 1) fit once (leave out first state) to compile the model
dat_loso <- dplyr::filter(dat_help, SOVEREIGNT != countries[1])

dat_agg <- dat_help %>%
  group_by(year) %>%
  summarise(
    conflict_count = sum(conflict_count),
    bool1989 = first(bool1989), 
    C0 = first(C0), 
    C0_2 = first(C0_2)
  )
fit0 <- brm(
  conflict_count ~ C0 + C0_2 + year + bool1989,
  data = dat_agg,
  family = poisson(), # Poisson
  iter = 10000,
  chains = 2,
  warmup = 500,
  cores = 1,
  prior = c(
    prior(normal(0, 3), class = "b"),         # regression coefficients
    prior(normal(0, 5), class = "Intercept")    # intercept term
  )
)

# helper to summarize fixed effects for one holdout year
summarize_fit <- function(fit, s_left_out) {
  fe <- as.data.frame(brms::fixef(fit, probs = c(0.05, 0.95)))
  fe$term <- rownames(fe)
  dplyr::filter(fe, term %in% terms) |>
    dplyr::transmute(
      state_left_out = s_left_out,
      term,
      mean = Estimate,
      q05  = Q5,
      q95  = Q95
    )
}

loso_stats <- summarize_fit(fit0, countries[1])

for (s in countries[-1]) {
  dat_s <- dplyr::filter(dat_help, SOVEREIGNT != s)
  dat_s <- dat_s %>%
    group_by(year) %>%
    summarise(
      conflict_count = sum(conflict_count),
      bool1989 = first(bool1989), 
      C0 = first(C0), 
      C0_2 = first(C0_2)
    )
  
  fit_s <- update(fit0, newdata = dat_s, recompile = FALSE, refresh = 0,
                  seed = 20251019 + which(countries == s))
  loso_stats <- dplyr::bind_rows(loso_stats, summarize_fit(fit_s, s))
  rm(fit_s); gc()
}

loso_wide <- loso_stats |>
  dplyr::arrange(state_left_out, term) |>
  tidyr::pivot_wider(
    id_cols = state_left_out,
    names_from = term,
    values_from = c(mean, q05, q95)
  ) |>
  dplyr::arrange(state_left_out)
#View(loso_wide)

idx <- seq_len(nrow(loso_wide))
plot(idx, loso_wide$mean_C0_2, pch = 16, xlab = "Row", ylab = "C0",
     ylim = range(-0.1, loso_wide$q95_C0_2))
segments(idx, loso_wide$q05_C0_2, idx, loso_wide$q95_C0_2)
abline(h = 0, lty = 3)

readr::write_csv(loso_wide, "/Users/tylerbagwell/Desktop/Onset_Count_Global_mrsosNINO34_square4_grouped_poisson_wet_loso.csv")




#### PERMUTATION TEST
library(randomizr)

block_perm_3y <- function(x.in){
  n.leftover <- length(x.in) %% 3
  
  xx <- append(x.in, x.in)
  ind.start <- sample(seq(1,length(x.in)-1,1),1)
  xx <- xx[seq(ind.start,ind.start+length(x.in)-1,1)]
  
  ind.remove <- sample(seq(1,length(xx),1),1)
  val.remove <- xx[ind.remove]
  x.final <- xx[-ind.remove]
  if (length(x.final) %% 3 != 0) {
    stop("Vector length must be divisible by 3")
  }
  m <- matrix(x.final, ncol = 3, byrow = TRUE)
  permuted_order <- sample(nrow(m))
  m_permuted <- m[permuted_order, ]
  
  x.permuted <- as.vector(t(m_permuted))
  x.permuted <- append(x.permuted, val.remove)
  
  return(x.permuted)
}

block_perm_5y <- function(x.in){
  n.leftover <- length(x.in) %% 5
  
  xx <- append(x.in, x.in)
  ind.start <- sample(seq(1,length(x.in)-1,1),1)
  xx <- xx[seq(ind.start,ind.start+length(x.in)-1,1)]
  
  ind.remove <- sample(seq(1,length(xx),1),n.leftover)
  val.remove <- xx[ind.remove]
  x.final <- xx[-ind.remove]
  if (length(x.final) %% 5 != 0) {
    stop("Vector length must be divisible by 5")
  }
  m <- matrix(x.final, ncol = 5, byrow = TRUE)
  permuted_order <- sample(nrow(m))
  m_permuted <- m[permuted_order, ]
  
  x.permuted <- as.vector(t(m_permuted))
  x.permuted <- append(x.permuted, val.remove)
  
  return(x.permuted)
}


dat_agg$C0 <- block_perm_5y(dat_agg$cindex_lag0y)

fit0 <- brm(
  conflict_count ~ C0 + I(C0^2) + year + bool1989,
  data   = dat_agg,
  family = poisson(),
  iter   = 5000,
  chains = 2,
  warmup = 500,
  cores  = 1,
  prior  = c(
    prior(normal(0, 3), class = "b"),
    prior(normal(0, 5), class = "Intercept")
  ),
  save_pars = save_pars(all = TRUE)  # makes updating + extracting safer
)

pnames <- variables(fit0)
par_c0 <- "b_C0"
par_sq <- {
  cand <- pnames[pnames %in% c("b_IC0E2", "b_I(C0^2)")]
  if (length(cand) == 0) cand <- pnames[grepl("^b_.*C0.*E2$", pnames)]
  cand[1]
}

n_perm <- 2000
res <- data.frame(
  perm     = seq_len(n_perm),
  b_C0     = NA_real_,
  b_IC0E2  = NA_real_
)

for (i in seq_len(n_perm)) {
  
  if (i%%50==0){print(paste0("Test: ", i))}
  
  # Make permutation reproducible (affects block_perm_3y if it uses RNG)
  set.seed(300000 + i)
  
  dat_perm <- dat_agg
  #dat_perm$C0 <- block_perm_3y(dat_perm$cindex_lag0y)
  dat_perm$C0 <- block_perm_5y(dat_perm$cindex_lag0y)
  
  fit_i <- update(
    fit0,
    newdata   = dat_perm,
    recompile = FALSE,
    refresh   = 0,
    seed      = 20251019 + i
  )
  
  s <- posterior_summary(fit_i, variable = c(par_c0, par_sq))
  res$b_C0[i]    <- s[par_c0, "Estimate"]
  res$b_IC0E2[i] <- s[par_sq, "Estimate"]
}

hist(res$b_IC0E2, breaks='scott')
abline(v=1.96, col='red')
quantile(abs(res$b_IC0E2), c(0.05, 0.90))

hist(res$b_C0, breaks='scott')
abline(v=0.55, col='red')
quantile(abs(res$b_C0), c(0.05, 0.90))







