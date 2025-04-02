library(brms)
library(dplyr)
library(ggplot2)

panel_data_path <- '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets/Onset_Binary_Global_NINO3_square4.csv'
dat <- read.csv(panel_data_path)

# just for missing years (there should not be any)
years_list <- sort(unique(dat$year))
full_years <- seq(min(years_list), max(years_list))
missing_years <- setdiff(full_years, years_list)
if (length(missing_years) == 0) {
  cat("All years from", min(years_list), "to", max(years_list), "are present.\n")
} else {
  cat("Missing years:", paste(missing_years, collapse = ", "), "\n")
}

dat$bool1989 <- ifelse(dat$year<=1989,0,1)
dat$year <- dat$year - min(dat$year)
dat$loc_id <- as.factor(dat$loc_id)

quantile(dat$psi, c(0.33,0.80,0.90))
dat_help <- subset(dat, psi > 1.4415020)

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

###
cindex_summary_mat <- matrix(nrow=0, ncol=5)
colnames(cindex_summary_mat) <- c('Estimate','Est.Error','Q5','Q95','window_start_year')

n <- nrow(dat_agg)
prop <- 0.60
nn <- as.integer(n*prop)

for (i in 1:as.integer(n-nn)){
  print(paste0('...Starting ', i, ' iteration'))
  dat_slice <- dat_agg[i:(nn+i),]
  
  # BAYESIAN FITS
  fit <- brm(
    conflict_proportion ~ cindex_lag0y + 
      year + bool1989,
    data = dat_slice, family = gaussian(), 
    iter = 10000, chains=3, warmup=1000,
    prior = c(
      prior(normal(0, 2.5), class = "b"),                # regression coefficients
      prior(normal(0, 5), class = "Intercept"),          # intercept term
      prior(exponential(1), class = "sigma")             # residual standard deviation
    )
  )
  
  fixed_effects <- fixef(fit, probs = c(0.05, 0.95))
  cindex_summary <- fixed_effects["cindex_lag0y", ]
  cindex_summary <- c(cindex_summary, years_list[i])
  cindex_summary_mat <- rbind(cindex_summary_mat, cindex_summary)
}


cindex_summary_df <- as.data.frame(cindex_summary_mat)
plot(cindex_summary_df$Estimate, ylim=c(0,max(cindex_summary_df$Estimate)))
abline(h=0, lwd=2)

write.csv(cindex_summary_df, '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/runningwindow_cindex_lag0y_Onset_Binary_Global_NINO3_square4_geq80_ratio0.60.csv')
 



