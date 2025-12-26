################################################################################
# GRID CELL TELECONNECTION STRENGTH PARTITIONING ----------------------------- #
# -- 1. BRMS 2-NORMAL MIXTURE MODEL FITTING FOR TELECONNECTION STRENGTH ------ #
# -- 2. EM 2-NORMAL MIXTURE MODEL FITTING FOR TELECONNECTION STRENGTH -------- #
################################################################################

# -- 1. BRMS 2-NORMAL MIXTURE MODEL FITTING FOR TELECONNECTION STRENGTH ------ #

library(brms)
library(dplyr)
panel_data_path = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/onset_datasets_grid/'
#
dat <- read.csv(file.path(panel_data_path, 'Onset_Count_Global_NINO3type2_v3_newonsetdata_square4.csv')) # <-- PANEL DATA PATH, ALL STATE PANELS HAVE BINARY RESPONSE
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
    prior(normal(0, 0.10), Intercept, dpar = mu1),
    prior(normal(1, 0.10), Intercept, dpar = mu2)
  ),
  chains = 4, cores = 2, iter = 10000,
  control = list(adapt_delta = 0.95)
)
plot(fit_mix)
print(fit_mix, digits = 3)

post <- as_draws_df(fit_mix, variable=c('Intercept_mu1', 'Intercept_mu2',
                                        'sigma1', 'sigma2',
                                        'theta1', 'theta2'))

find_threshold_safe <- function(w1, mu1, sd1, w2, mu2, sd2,
                                lower = 0, upper = 2, grid_n = 401) {
  
  # relabel per draw so mu1 <= mu2 (reduces label-switching headaches)
  if (is.finite(mu1) && is.finite(mu2) && mu1 > mu2) {
    tmp <- w1; w1 <- w2; w2 <- tmp
    tmp <- mu1; mu1 <- mu2; mu2 <- tmp
    tmp <- sd1; sd1 <- sd2; sd2 <- tmp
  }
  
  # guards
  if (any(!is.finite(c(w1, mu1, sd1, w2, mu2, sd2))) || any(c(sd1, sd2) <= 0) ||
      any(c(w1, w2) <= 0)) return(NA_real_)
  
  f <- function(x) w1*dnorm(x, mu1, sd1) - w2*dnorm(x, mu2, sd2)
  
  xs <- seq(lower, upper, length.out = grid_n)
  ys <- f(xs)
  
  # remove non-finite
  ok <- is.finite(ys)
  xs <- xs[ok]; ys <- ys[ok]
  if (length(xs) < 2) return(NA_real_)
  
  s <- sign(ys)
  # indices where sign changes (or hits 0)
  idx <- which(s[-1] * s[-length(s)] <= 0)
  
  if (length(idx) == 0) return(NA_real_)  # no root in [lower, upper]
  
  # choose a sign-change interval; often you'd want the one between means:
  mid <- (mu1 + mu2) / 2
  a_candidates <- xs[idx]
  b_candidates <- xs[idx + 1]
  mids <- (a_candidates + b_candidates) / 2
  j <- which.min(abs(mids - mid))  # pick closest to midpoint between means
  
  a <- a_candidates[j]; b <- b_candidates[j]
  out <- tryCatch(uniroot(f, lower = a, upper = b)$root, error = function(e) NA_real_)
  out
}

thresholds <- mapply(find_threshold_safe,
                     post$theta1, post$Intercept_mu1, post$sigma1,
                     post$theta2, post$Intercept_mu2, post$sigma2)

mean_thr <- mean(thresholds)
ci_thr   <- quantile(thresholds, c(0.025, 0.975))
cat("50/50 crossover at ψ ≈", round(mean_thr, 3),
    " (95% CI:", round(ci_thr[1],3), "-", round(ci_thr[2],3), ")\n")

hist(thresholds, breaks='scott')
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
hist(unique_psi$psi, breaks = "scott", prob = TRUE,
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





