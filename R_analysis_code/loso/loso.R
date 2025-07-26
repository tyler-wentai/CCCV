library(brms)
library(dplyr)
library(ggplot2)

panel_data_path <- '<FILE PATH HERE>/Onset_Count_Global_DMItype2_square4.csv'
dat <- read.csv(panel_data_path)

sum(dat$conflict_count)
unique(dat$conflict_count)
#View(dat)
dat <- subset(dat, year!=1989)
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


quantile(unique_psi$psi, c(0.1, 0.2, 0.40, 0.45, 0.55, 0.60, 0.7, 0.80, 0.95, 1))
dat_help <- subset(dat, psi >= 0.55)
sum(dat_help$conflict_count)


###### PERFORM LOSO (LEAVE-ONE-SUBJECT-OUT):
locs <- unique(dat_help$loc_id)

#
for (i in 1:length(locs)){
  
  dat_help2 <- subset(dat_help, loc_id!=locs[i])
  
  dat_agg_loso <- dat_help2 %>%
    group_by(year) %>%
    summarise(
      conflict_count = sum(conflict_count),
      bool1989 = first(bool1989), 
      cindex_lag0y = first(cindex_lag0y), 
      cindex_lag1y = first(cindex_lag1y), 
      cindex_lag2y = first(cindex_lag2y)
    )
  
  fit_loso <- brm(
    conflict_count ~ cindex_lag0y + I(cindex_lag0y^2) + year + bool1989,
    data = dat_agg_loso,
    family = poisson(),
    iter = 5000,
    chains = 2,
    warmup = 500,
    cores = 1,
    prior = c(
      prior(normal(0, 2.5), class = "b"),       
      prior(normal(0, 5), class = "Intercept")
    )
  )
  
  if (i==1){
    sum_mat <- summary(fit_loso, prob = 0.90)$fixed
    sub_mat1 <- sum_mat[c("cindex_lag0y"), ]
    sub_mat2 <- sum_mat[c("Icindex_lag0yE2"), ]
    sub_mat1$loc_id <- locs[i]
    sub_mat2$loc_id <- locs[i]
  } else{
    sum_mat <- summary(fit_loso, prob = 0.90)$fixed
    sub_mat1_help <- sum_mat[c("cindex_lag0y"), ]
    sub_mat2_help <- sum_mat[c("Icindex_lag0yE2"), ]
    sub_mat1_help$loc_id <- locs[i]
    sub_mat2_help$loc_id <- locs[i]
    
    sub_mat1 <- rbind(sub_mat1, sub_mat1_help)
    sub_mat2 <- rbind(sub_mat2, sub_mat2_help)
  }
  
}


sub_mat1$Estimate <- sub_mat1$Estimate*100
sub_mat1$'l-90% CI' <- sub_mat1$'l-90% CI'*100
sub_mat1$'u-90% CI' <- sub_mat1$'u-90% CI'*100

sub_mat2$Estimate <- sub_mat2$Estimate*100
sub_mat2$'l-90% CI' <- sub_mat2$'l-90% CI'*100
sub_mat2$'u-90% CI' <- sub_mat2$'u-90% CI'*100


## beta1
p1 <-ggplot(sub_mat1, aes(x = factor(loc_id), y = Estimate)) +
  geom_linerange(aes(ymin = `l-90% CI`, ymax = `u-90% CI`),
                 size = 1, colour = "steelblue") +
  geom_point(size = 2, shape = 21, fill = "white", stroke = 1) +
  scale_x_discrete(drop = FALSE) +
  geom_hline(yintercept = 0, colour = "grey50") +
  geom_hline(yintercept = 0.55, linetype = "dashed", colour = "red", size = 0.75, show.legend = TRUE) +
  labs(x = "Grid cell removed",
       y = expression(paste("Estimate of ", beta[1], " (90% CI)")),
       title = "Poisson model: IOD-conflict responsive grid cells, leave-one-grid-cell-out") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, size  = 6)  # rotate if needed
  )

p1

## beta2
p2 <-ggplot(sub_mat2, aes(x = factor(loc_id), y = Estimate)) +
  geom_linerange(aes(ymin = `l-90% CI`, ymax = `u-90% CI`),
                 size = 1, colour = "steelblue") +
  geom_point(size = 2, shape = 21, fill = "white", stroke = 1) +
  scale_x_discrete(drop = FALSE) +
  geom_hline(yintercept = 0, colour = "grey50") +
  geom_hline(yintercept = 1.94, linetype = "dashed", colour = "red", size = 0.75, show.legend = TRUE) +
  labs(x = "Grid cell removed",
       y = expression(paste("Estimate of ", beta[2], " (90% CI)")),
       title = " ") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, size  = 6)  # rotate if needed
  )

p2

combined_ps <- grid.arrange(p1, p2, nrow = 2)

ggsave(
  filename = "<FILE PATH HERE>/iodconflict_responsive_gridcells_loso.jpg",
  plot     = combined_ps,
  device   = "jpeg",
  width    = 7,
  height   = 5.5,
  units    = "in",
  dpi      = 300
)




