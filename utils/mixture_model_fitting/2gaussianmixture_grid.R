################################################################################
# GRID CELL TELECONNECTION STRENGTH PARTITIONING ----------------------------- #
# -- 1. BRMS 2-NORMAL MIXTURE MODEL FITTING FOR TELECONNECTION STRENGTH ------ #
# -- 2. EM 2-NORMAL MIXTURE MODEL FITTING FOR TELECONNECTION STRENGTH -------- #
################################################################################

# -- 1. BRMS 2-NORMAL MIXTURE MODEL FITTING FOR TELECONNECTION STRENGTH ------ #

library(brms)
library(dplyr)

path <- '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_NINO3type2_square4_wGeometry.csv'
dat <- read.csv(path)
#View(dat)

unique_psi <- dat %>%
  group_by(loc_id) %>%
  summarise(
    psi = first(psi))
hist(unique_psi$psi, breaks='scott')

mix_family <- mixture(gaussian(), gaussian())

fit_mix <- brm(
  bf(psi ~ 1),  # extra formula: mu2; mu1 is taken from psi ~ 1
  family = mix_family,
  data   = unique_psi,
  prior  = c(
    prior(dirichlet(1, 1),       class = "theta"),
    prior(normal(0, 0.20), Intercept, dpar = mu1),
    prior(normal(1, 0.20), Intercept, dpar = mu2)
  ),
  chains = 2, cores = 2, iter = 5000,
  control = list(adapt_delta = 0.95)
)
plot(fit_mix)
print(fit_mix, digits = 3)

post <- as_draws_df(fit_mix, variable=c('Intercept_mu1', 'Intercept_mu2',
                                        'sigma1', 'sigma2',
                                        'theta1', 'theta2'))

find_threshold <- function(w1, mu1, sd1, w2, mu2, sd2) {
  f <- function(x) w1*dnorm(x, mu1, sd1) - w2*dnorm(x, mu2, sd2)
  uniroot(f, lower=0.0, upper=2.0)$root
}

thresholds <- mapply(find_threshold,
                     post$theta1, post$Intercept_mu1, post$sigma1,
                     post$theta2, post$Intercept_mu2, post$sigma2)

mean_thr <- mean(thresholds)
ci_thr   <- quantile(thresholds, c(0.025, 0.975))
cat("50/50 crossover at ψ ≈", round(mean_thr, 3),
    " (95% CI:", round(ci_thr[1],3), "-", round(ci_thr[2],3), ")\n")

hist(thresholds, breaks='scott', xlim=c(min(unique_psi$psi),max(unique_psi$psi)))


# -- 2. EM 2-NORMAL MIXTURE MODEL FITTING FOR TELECONNECTION STRENGTH -------- #
library(dplyr)
library(mixtools)

path <- '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_NINO3type2_square4_wGeometry.csv'
dat <- read.csv(path)

View(dat)

unique_psi <- dat %>%
  group_by(loc_id) %>%
  summarise(
    psi = first(psi))
hist(unique_psi$psi, breaks='scott')

quantile(unique_psi$psi, 0.66)


dens <- density(unique_psi$psi, kernel="gaussian")
plot(dens,
     main   = "Kernel Density Estimate of ψ",
     xlab   = expression(psi),
     ylab   = "Density",
     lwd    = 2)
polygon(dens, col = "lightblue", border = "blue", density = 20)


mix_fit <- normalmixEM(unique_psi$psi, k = 2, maxit = 1000, epsilon = 1e-8)
mix_fit$lambda    # mixing proportions (weights)
mix_fit$mu        # component means
mix_fit$sigma     # component standard deviations

# 3. Plot the fitted densities on top of your histogram
hist(unique_psi$psi, breaks = "FD", prob = TRUE,
     main = "2-Component Gaussian Mixture",
     xlab = expression(psi))
# overall fitted density
lines(density(unique_psi$psi), lwd = 2, col = "gray50", lty = 2)
# component densities
curve(mix_fit$lambda[1] * dnorm(x, mix_fit$mu[1], mix_fit$sigma[1]),
      add = TRUE, col = "blue", lwd = 2)
curve(mix_fit$lambda[2] * dnorm(x, mix_fit$mu[2], mix_fit$sigma[2]),
      add = TRUE, col = "red",  lwd = 2)
# mixture density
curve(mix_fit$lambda[1] * dnorm(x, mix_fit$mu[1], mix_fit$sigma[1]) +
        mix_fit$lambda[2] * dnorm(x, mix_fit$mu[2], mix_fit$sigma[2]),
      add = TRUE, col = "black", lwd = 2)
legend("topright", legend = c("Comp.1","Comp.2","Mix"),
       col = c("blue","red","black"), lwd = 2)



# suppose you've already done
mix_fit <- normalmixEM(unique_psi$psi, k = 2, maxit = 1000, epsilon = 1e-8)

# extract parameters
λ1 <- mix_fit$lambda[1];   λ2 <- mix_fit$lambda[2]
μ1 <- mix_fit$mu[1];       μ2 <- mix_fit$mu[2]
σ1 <- mix_fit$sigma[1];    σ2 <- mix_fit$sigma[2]

# define the function whose root is the 50/50 point
f <- function(x) {
  λ1 * dnorm(x, mean = μ1, sd = σ1) -
    λ2 * dnorm(x, mean = μ2, sd = σ2)
}

# pick an interval that brackets the root: for instance between the two means
lower <- min(μ1, μ2) - 3 * max(σ1, σ2)
upper <- max(μ1, μ2) + 3 * max(σ1, σ2)

# find x* via uniroot
root <- uniroot(f, lower = 0.0, upper = upper)$root
root





