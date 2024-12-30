################################################################################
# Setup
################################################################################

par(family="serif", las=1, bty="l",
    cex.axis=1, cex.lab=1, cex.main=1,
    xaxs="i", yaxs="i", mar = c(5, 5, 3, 1))

library(rstan)
rstan_options(auto_write = TRUE)            # Cache compiled Stan programs
options(mc.cores = parallel::detectCores()) # Parallelize chains
parallel:::setDefaultClusterOptions(setup_strategy = "sequential")

util <- new.env()
source('mcmc_analysis_tools_rstan.R', local=util)
source('mcmc_visualization_tools.R', local=util)

################################################################################
# Cleanup Data
################################################################################

raw_data <- data.frame(matrix(ncol=3, nrow=0))
names(raw_data) <- c('movie_ids', 'customer_ids', 'ratings')

for (n in 1:1000) {
  name <- as.character(n)
  name <- paste0('raw_data/netflix/training_set/mv_',
                 do.call(paste0, as.list(rep(0, 7 - nchar(name)))), 
                 name, '.txt')
  df <- read.csv(name, header=FALSE, skip=1)
  df <- cbind(rep(n, nrow(df)), df)

  raw_data <- rbind(raw_data, df[,1:3])
}

names(raw_data) <- c('movie_ids', 'customer_ids', 'ratings')

# Subset customers

set.seed(49483328)

customer_counts <- table(raw_data$customer_ids)
customer_ids <- as.numeric(names(customer_counts))

customer_freqs <- customer_counts / sum(customer_counts)
selected_customer_ids <- sample(customer_ids, 100, prob=customer_freqs)

subset_data <- raw_data[raw_data$customer_ids %in% selected_customer_ids,]

table(subset_data$customer_ids)
length(table(subset_data$customer_ids))

table(subset_data$movie_ids)
hist(table(subset_data$movie_ids), breaks=50)
length(table(subset_data$movie_ids))

movie_counts <- table(subset_data$movie_ids)
movie_ids <- as.numeric(names(movie_counts))

movie_freqs <- movie_counts / sum(customer_counts)
selected_movie_ids <- sample(movie_ids, 200, prob=movie_freqs)

subset_data <- subset_data[subset_data$movie_ids %in% selected_movie_ids,]


N_ratings <- nrow(subset_data)
N_customers <- length(unique(subset_data$customer_ids))
N_movies <- length(unique(subset_data$movie_ids))

customer_idxs <- factor(subset_data$customer_ids, 
                     levels=sort(unique(subset_data$customer_ids)),
                     labels=1:N_customers)

movie_idxs <- factor(subset_data$movie_ids, 
                     levels=sort(unique(subset_data$movie_ids)),
                     labels=1:N_movies)

data <- list('N_customers'=N_customers,
             'nflx_customer_ids'=subset_data$customer_ids,
             'customer_idxs'=customer_idxs,
             'N_movies'=N_movies,
             'nflx_movie_ids'=subset_data$movie_ids,
             'movie_idxs'=movie_idxs,
             'N_ratings'=N_ratings,
             'ratings'=subset_data$ratings)

nflx_customer_ids <- subset_data$customer_ids
nflx_movie_ids <- subset_data$movie_ids
ratings <- subset_data$ratings

stan_rdump(c('N_customers', 'nflx_customer_ids', 'customer_idxs',
             'N_movies', 'nflx_movie_ids', 'movie_idxs',
             'N_ratings', 'ratings'), 
           'data/ratings.data.R')

################################################################################
# Explore Data
################################################################################

data <- read_rdump('data/ratings.data.R')

cat(sprintf("%s Customers", data$N_customers))
cat(sprintf("%s Movies", data$N_movies))
cat(sprintf("%s Total Ratings", data$N_ratings))

# Most customers rate only a few movies.
xs <- seq(1, data$N_movies, 1)
ys <- seq(1, data$N_customers, 1)
zs <- matrix(0, nrow=data$N_movies, ncol=data$N_customers)

for (n in 1:data$N_ratings) {
  zs[data$movie_idxs[n], data$customer_idxs[n]] <- 1
}

par(mfrow=c(1, 1), mar = c(5, 5, 1, 1))

image(xs, ys, zs, col=c("white", util$c_dark_teal),
      xlab="Movie", ylab="customer")

# Aggregate ratings

par(mfrow=c(1, 1), mar=c(5, 5, 2, 1))

util$plot_line_hist(data$ratings, 
                    -0.5, 6.5, 1, xlab="Ratings")

# Number of ratings per customer

par(mfrow=c(1, 1), mar=c(5, 5, 2, 1))

util$plot_line_hist(table(data$customer_idxs), 
                    -0.5, 95.5, 5, 
                    xlab="Number of Ratings Per Customer")

# Number of ratings per movie

par(mfrow=c(1, 1), mar=c(5, 5, 2, 1))

util$plot_line_hist(table(data$movie_idxs), 
                    -0.5, 55.5, 2, 
                    xlab="Number of Ratings Per Movie")

# Lots of variation in ratings.

# Customer 70 vary generous, Customer 23 less so.

par(mfrow=c(2, 3), mar=c(5, 5, 1, 1))

for (c in c(7, 23, 40, 70, 77, 100)) {
  util$plot_line_hist(data$ratings[data$customer_idxs == c], 
                      -0.5, 6.5, 1,
                      xlab="Rating", main=paste('Customer', c))
}

# Movies 117 and  180 is pretty well-reviewed, others more mixed.

par(mfrow=c(2, 3), mar=c(5, 5, 1, 1))

for (m in c(33, 53, 61, 80, 117, 180)) {
  util$plot_line_hist(data$ratings[data$movie_idxs == m], 
                      -0.5, 6.5, 1,
                      xlab="Rating", main=paste('Movie', m))
}

# Average ratings stratified by customer and movie

par(mfrow=c(1, 2), mar=c(5, 5, 2, 1))

mean_rating_customer <- sapply(1:data$N_customers, 
                           function(c) mean(data$ratings[data$customer_idxs == c]))
util$plot_line_hist(mean_rating_customer, 
                    0, 6, 1, 
                    xlab="Customer-wise Average Ratings")


mean_rating_movie <- sapply(1:data$N_movies, 
                           function(m) mean(data$ratings[data$movie_idxs == m]))
util$plot_line_hist(mean_rating_movie, 
                    0, 6, 0.5, 
                    xlab="Movie-wise Average Ratings")

# Rating variance stratified by customer and movie

safe_var <- function(vals) {
  if (length(vals) == 1)
    (0)
  else
    (var(vals))
}

par(mfrow=c(1, 2), mar=c(5, 5, 2, 1))

var_rating_customer <- sapply(1:data$N_customers, 
                           function(c) safe_var(data$ratings[data$customer_idxs == c]))
util$plot_line_hist(var_rating_customer, 
                    0, 5, 0.5, 
                    xlab="customer-wise Rating Variances")

var_rating_movie <- sapply(1:data$N_movies, 
                           function(m) safe_var(data$ratings[data$movie_idxs == m]))
util$plot_line_hist(var_rating_movie, 
                    0, 5, 0.5, 
                    xlab="Movie-wise Rating Variances")

# The limitation with these stratified summary statistics is that they
# are mostly sensitive to the marginal variation across movies and 
# customers.  In other words they are senstive to heterogeneity across
# movies and customers but not heterogeneity across movies and customers 
# at the same time.

# In order to interrogate these more subtle variations we need to 
# consider higher-order summaries such as covariations.  For example 
# we might consider movie-pair covariances

# EQN.

# Implementing these statistics, however, is immediately frustrated by
# the fact that not every customer has rated every movie.  The best we
# can do is sum over the customers that have happened to rated both 
# movies in question.

# EQN.

# Most movie pairs will not have been rated by any customers.  Of 
# those that have any ratings most will have only a few, resulting 
# in not particularly informative values.  In order to isolate more 
# informative covariances we can require a minimum number of 
# contributing customers.  This also has the added benefit of reducing
# the total number of covariances to visualize.

# Compute empirical sums of squares and counts.

covar_rating_movie <- matrix(0, 
                             nrow=data$N_movies, 
                             ncol=data$N_movies)
movie_pair_counts <- matrix(0, 
                            nrow=data$N_movies, 
                            ncol=data$N_movies)

for (n1 in 1:data$N_ratings) {
  for (n2 in 1:data$N_ratings) {
    if (data$customer_idxs[n1] == data$customer_idxs[n2]) {
      m1 <- data$movie_idxs[n1]
      m2 <- data$movie_idxs[n2]
      y <- (data$ratings[n1] - mean_rating_movie[m1]) * 
           (data$ratings[n2] - mean_rating_movie[m2])
      covar_rating_movie[m1, m2] <- covar_rating_movie[m1, m2] + y
      covar_rating_movie[m2, m1] <- covar_rating_movie[m2, m1] + y
      movie_pair_counts[m1, m2] <- movie_pair_counts[m1, m2] + 1
      movie_pair_counts[m2, m1] <- movie_pair_counts[m2, m1] + 1
    }
  }
}

# Compute empirical covariances for movie pairs with more than seven 
# concurrent ratings.

m_pairs <- list()
covar_rating_movie_filt <- c()

for (m1 in 2:data$N_movies) {
  for (m2 in 1:(m1 - 1)) {
    if (movie_pair_counts[m1, m2] > 7) {
      m_pairs[[length(m_pairs) + 1]] <- c(m1, m2)
      covar_rating_movie_filt <- c(covar_rating_movie_filt, 
                                   covar_rating_movie[m1, m2] / 
                                     (movie_pair_counts[m1, m2] - 1))
    }
  }
}

# We can use a histogram to visualize the remaining covariances.

par(mfrow=c(1, 1), mar=c(5, 5, 2, 1))

util$plot_line_hist(covar_rating_movie_filt,
                    -4, 4, 0.25, 
                    xlab="Movie-wise Rating Covariances")

# Finally we'll need to save the names of the relevant covariances
# so that we can construct appropriate posterior predictive values.

covar_rating_movie_filt_names <- sapply(m_pairs, 
                                        function(p) 
                                          paste0('covar_rating_movie_pred[', 
                                                 p[1], ',', p[2], ']'))

################################################################################
# Homogenous Customer Model Attempt 1
################################################################################

# Modeling observed ratings with an ordinal model, using a latent logistic density 
# function and cut points to derive the rating probabilities.

# Assume homogeneous customer behavior.

# Assume exchangeable movie affinities.  Zero population location to avoid any
# degeneracy with the cutpoints.

# Start with a monolithic non-centered parameterization.

fit <- stan(file="stan_programs/model1.stan",
            data=data, seed=8438338,
            warmup=1000, iter=2024, refresh=0)

# Diagnostics clean!

diagnostics <- util$extract_hmc_diagnostics(fit)
util$check_all_hmc_diagnostics(diagnostics)

samples1 <- util$extract_expectand_vals(fit)
base_samples <- util$filter_expectands(samples1,
                                       c('gamma_ncp', 
                                         'tau_gamma',
                                         'cut_points'),
                                       check_arrays=TRUE)
util$check_all_expectand_diagnostics(base_samples)


# Retrodictive checks.

# All ratings aggregated together look good.

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

util$plot_hist_quantiles(samples1, 'rating_pred', -0.5, 6.5, 1,
                         baseline_values=data$ratings,
                         xlab="All Ratings")

# Spot checks some customer ratings.
# Posterior predictive behavior is far more homogeneous than the observed behavior.

par(mfrow=c(2, 3), mar=c(5, 5, 1, 1))

for (c in c(7, 23, 40, 70, 77, 100)) {
  names <- sapply(which(data$customer_idxs == c), 
                  function(n) paste0('rating_pred[', n, ']'))
  filtered_samples <- util$filter_expectands(samples1, names)
  
  util$plot_hist_quantiles(filtered_samples, 'rating_pred', -0.5, 6.5, 1,
                           baseline_values=data$ratings[data$customer_idxs == c],
                           xlab="Ratings",
                           main=paste('Customer', c))
}

# Spot checks some movie ratings.
# Reasonable agreement.

par(mfrow=c(2, 3), mar=c(5, 5, 1, 1))

for (m in c(33, 53, 61, 80, 117, 180)) {
  names <- sapply(which(data$movie_idxs == m), 
                  function(n) paste0('rating_pred[', n, ']'))
  filtered_samples <- util$filter_expectands(samples1, names)
  
  util$plot_hist_quantiles(filtered_samples, 'rating_pred', -0.5, 6.5, 1,
                           baseline_values=data$ratings[data$movie_idxs == m],
                           xlab="Ratings",
                           main=paste('Movie', m))
}

# Posterior predictive customer-wise means are more narrowly distributed 
# thanwhat we see in the observed data.  At the same time the posterior
# predictive customer-wise variances concentrate at larger values 
# than the observed behaviors.

par(mfrow=c(2, 2), mar=c(5, 5, 1, 1))

util$plot_hist_quantiles(samples1, 'mean_rating_customer_pred', 0, 6, 0.5,
                         baseline_values=mean_rating_customer,
                         xlab="Customer-wise Average Ratings")

util$plot_hist_quantiles(samples1, 'mean_rating_movie_pred', 0, 6, 0.6,
                         baseline_values=mean_rating_movie,
                         xlab="Movie-wise Average Ratings")

util$plot_hist_quantiles(samples1, 'var_rating_customer_pred', 0, 7, 0.5,
                         baseline_values=var_rating_customer,
                         xlab="Customer-wise Rating Variances")

util$plot_hist_quantiles(samples1, 'var_rating_movie_pred', 0, 7, 0.5,
                         baseline_values=var_rating_movie,
                         xlab="Movie-wise Rating Variances")

# Filtered movie covariances are more heavy-tailed in the observed data
# than in the posterior predictions.

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

filtered_samples <- util$filter_expectands(samples1, 
                                           covar_rating_movie_filt_names)

util$plot_hist_quantiles(filtered_samples, 'covar_rating_movie_pred', -4.25, 4.25, 0.25,
                         baseline_values=covar_rating_movie_filt,
                         xlab="Filtered Movie-wise Rating Covariances")
 
################################################################################
# Independent Heterogeneous Customer Model
################################################################################

# Allow each customer to have their own cutpoints.

fit <- stan(file="stan_programs/model2.stan",
            data=data, seed=8438338,
            warmup=1000, iter=2024, refresh=0)

# A few stray divergences.

diagnostics <- util$extract_hmc_diagnostics(fit)
util$check_all_hmc_diagnostics(diagnostics)

samples2 <- util$extract_expectand_vals(fit)
base_samples <- util$filter_expectands(samples2,
                                       c('gamma_ncp', 
                                         'tau_gamma',
                                         'cut_points'),
                                       check_arrays=TRUE)
util$check_all_expectand_diagnostics(base_samples)

# No obvious inverted funnels on a quick spot-check of the
# movies with the most ratings, and hence most susceptible 
# to inverted funnels in the non-centered parameterization.

idxs <- as.numeric(names(tail(sort(table(data$movie_idxs)), 9)))
names <- sapply(idxs, function(m) paste0('gamma[', m, ']'))
util$plot_div_pairs(names, 'tau_gamma', samples2, diagnostics)

# Run again with less aggressive step size adaptation.

fit <- stan(file="stan_programs/model2.stan",
            data=data, seed=8438338,
            warmup=1000, iter=2024, refresh=0,
            control=list('adapt_delta'=0.9))

# No more divergences.

diagnostics <- util$extract_hmc_diagnostics(fit)
util$check_all_hmc_diagnostics(diagnostics)

samples2 <- util$extract_expectand_vals(fit)
base_samples <- util$filter_expectands(samples2,
                                       c('gamma_ncp', 
                                         'tau_gamma',
                                         'cut_points'),
                                       check_arrays=TRUE)
util$check_all_expectand_diagnostics(base_samples)

# Retrodictive performance.

# Interestingly aggregate ratings aren't as consistent.

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

util$plot_hist_quantiles(samples2, 'rating_pred', -0.5, 6.5, 1,
                         baseline_values=data$ratings,
                         xlab="All Ratings")

# Spot checks some customer ratings.  Far better agreement.

par(mfrow=c(2, 3), mar=c(5, 5, 1, 1))

for (c in c(7, 23, 40, 70, 77, 100)) {
  names <- sapply(which(data$customer_idxs == c), 
                  function(n) paste0('rating_pred[', n, ']'))
  filtered_samples <- util$filter_expectands(samples2, names)
  
  util$plot_hist_quantiles(filtered_samples, 'rating_pred', -0.5, 6.5, 1,
                           baseline_values=data$ratings[data$customer_idxs == c],
                           xlab="Ratings",
                           main=paste('Customer', c))
}

# Spot checks some movie ratings.  Same reasonable agreement.

par(mfrow=c(2, 3), mar=c(5, 5, 1, 1))

for (m in c(33, 53, 61, 80, 117, 180)) {
  names <- sapply(which(data$movie_idxs == m), 
                  function(n) paste0('rating_pred[', n, ']'))
  filtered_samples <- util$filter_expectands(samples2, names)
  
  util$plot_hist_quantiles(filtered_samples, 'rating_pred', -0.5, 6.5, 1,
                           baseline_values=data$ratings[data$movie_idxs == m],
                           xlab="Ratings",
                           main=paste('Movie', m))
}

# Customer-wise means and variances exhibit better agreement, 
# but still a bit lacking.

par(mfrow=c(2, 2), mar=c(5, 5, 1, 1))

util$plot_hist_quantiles(samples2, 'mean_rating_customer_pred', 0, 6, 0.5,
                         baseline_values=mean_rating_customer,
                         xlab="Customer-wise Average Ratings")

util$plot_hist_quantiles(samples2, 'mean_rating_movie_pred', 0, 6, 0.6,
                         baseline_values=mean_rating_movie,
                         xlab="Movie-wise Average Ratings")

util$plot_hist_quantiles(samples2, 'var_rating_customer_pred', 0, 7, 0.5,
                         baseline_values=var_rating_customer,
                         xlab="Customer-wise Rating Variances")

util$plot_hist_quantiles(samples2, 'var_rating_movie_pred', 0, 7, 0.5,
                         baseline_values=var_rating_movie,
                         xlab="Movie-wise Rating Variances")

# Filtered movie covariances still more heavily tailed in the observed 
# data relative to the posterior predictions.

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

filtered_samples <- util$filter_expectands(samples2, 
                                           covar_rating_movie_filt_names)

util$plot_hist_quantiles(filtered_samples, 'covar_rating_movie_pred', -4.25, 4.25, 0.25,
                         baseline_values=covar_rating_movie_filt,
                         xlab="Filtered Movie-wise Rating Covariances")

# What if we were content with this model?

# Customer inferences

par(mfrow=c(4, 1), mar=c(5, 5, 1, 1))

for (k in 1:4) {
  names <- sapply(1:data$N_customers,
                  function(r) paste0('cut_points[', r, ',', k, ']'))
  util$plot_disc_pushforward_quantiles(samples2, names,
                                       xlab="Customer", 
                                       display_ylim=c(-6, 6),
                                       ylab=paste0('cut_point[', k, ']'))
}

# Customer 23 is pretty stingy; a movie affinity needs to be really large
# in order for the ratings to be large.

# On the other hand Customer 70 is much more generous; they give even a 
# mediacre movie high ratings.

par(mfrow=c(2, 1), mar=c(5, 5, 1, 1))

cols <- c(util$c_mid, util$c_mid_highlight, 
          util$c_dark, util$c_dark_highlight)

c <- 23

k <- 1
name <-paste0('cut_points[', c, ',', k, ']')
util$plot_expectand_pushforward(samples2[[name]],
                                50, flim=c(-8, 8), ylim=c(0, 2),
                                col=cols[k], display_name='Cut Points',
                                main=paste('Customer', c))


for (k in 2:4) {
  name <-paste0('cut_points[', c, ',', k, ']')
  util$plot_expectand_pushforward(samples2[[name]],
                                  50, flim=c(-8, 8),
                                  col=cols[k], border="#BBBBBB88",
                                  add=TRUE)
}

text(0, 1.65, "cut_points[1]", col=util$c_mid)
text(2, 1.4, "cut_points[2]", col=util$c_mid_highlight)
text(4, 1, "cut_points[3]", col=util$c_dark)
text(6, 0.5, "cut_points[4]", col=util$c_dark_highlight)

c <- 70

k <- 1
name <-paste0('cut_points[', c, ',', k, ']')
util$plot_expectand_pushforward(samples2[[name]],
                                50, flim=c(-8, 8), ylim=c(0, 2),
                                col=cols[k], display_name='Cut Points',
                                main=paste('Customer', c))


for (k in 2:4) {
  name <-paste0('cut_points[', c, ',', k, ']')
  util$plot_expectand_pushforward(samples2[[name]],
                                  50, flim=c(-8, 8),
                                  col=cols[k], border="#BBBBBB88",
                                  add=TRUE)
}

text(-5.5, 0.35, "cut_points[1]", col=util$c_mid)
text(-3, 0.75, "cut_points[2]", col=util$c_mid_highlight)
text(-2, 1.0, "cut_points[3]", col=util$c_dark)
text(1.0, 1.25, "cut_points[4]", col=util$c_dark_highlight)

# Movie inferences

par(mfrow=c(2, 1), mar=c(5, 5, 1, 1))

util$plot_expectand_pushforward(samples2[['tau_gamma']],
                                25, flim=c(0, 1),
                                display_name='tau_gamma')

names <- sapply(1:data$N_movies,
                function(m) paste0('gamma[', m, ']'))
util$plot_disc_pushforward_quantiles(samples2, names,
                                     xlab="Movie",
                                     ylab="Affinity")

################################################################################
# Hierarchical customer Model
################################################################################

# We're already modeling the movie affinity hierarchically -- why not
# model the cut points hierarchically?  We don't have any information
# that would obstruct the exchangeability of the customers.  All we 
# need is an appropriate multivariate population model.

# Fortuantely the induced Dirichlet prior naturally composes with a
# hyper Dirichlet population model.

# One substantial assumption that we're making here is that the 
# heterogeneity in the cut points and the heterogeneity in the 
# movie affinities are independent of each other.

fit <- stan(file="stan_programs/model3.stan",
            data=data, seed=8438338,
            warmup=1000, iter=2024, refresh=0)

# Diagnostics are clean!

diagnostics <- util$extract_hmc_diagnostics(fit)
util$check_all_hmc_diagnostics(diagnostics)

samples3 <- util$extract_expectand_vals(fit)
base_samples <- util$filter_expectands(samples3,
                                       c('gamma_ncp', 
                                         'tau_gamma',
                                         'cut_points', 
                                         'mu_q', 'tau_q'),
                                       check_arrays=TRUE)
util$check_all_expectand_diagnostics(base_samples)


# Retrodictive performance.

# Aggregate ratings better.

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

util$plot_hist_quantiles(samples3, 'rating_pred', -0.5, 6.5, 1,
                         baseline_values=data$ratings,
                         xlab="All Ratings")

# Spot checks some customer ratings.
# Reasonable agreement.

par(mfrow=c(2, 3), mar=c(5, 5, 1, 1))

for (c in c(7, 23, 40, 70, 77, 100)) {
  names <- sapply(which(data$customer_idxs == c), 
                  function(n) paste0('rating_pred[', n, ']'))
  filtered_samples <- util$filter_expectands(samples3, names)
  
  util$plot_hist_quantiles(filtered_samples, 'rating_pred', -0.5, 6.5, 1,
                           baseline_values=data$ratings[data$customer_idxs == c],
                           xlab="Ratings",
                           main=paste('Customer', c))
}

# Spot checks some movie ratings.
# Reasonable agreement.

par(mfrow=c(2, 3), mar=c(5, 5, 1, 1))

for (m in c(33, 53, 61, 80, 117, 180)) {
  names <- sapply(which(data$movie_idxs == m), 
                  function(n) paste0('rating_pred[', n, ']'))
  filtered_samples <- util$filter_expectands(samples3, names)
  
  util$plot_hist_quantiles(filtered_samples, 'rating_pred', -0.5, 6.5, 1,
                           baseline_values=data$ratings[data$movie_idxs == m],
                           xlab="Ratings",
                           main=paste('Movie', m))
}

# Movie-wise empirical means and variances a bit better.

par(mfrow=c(2, 2), mar=c(5, 5, 1, 1))

util$plot_hist_quantiles(samples3, 'mean_rating_customer_pred', 0, 6, 0.5,
                         baseline_values=mean_rating_customer,
                         xlab="Customer-wise Average Ratings")

util$plot_hist_quantiles(samples3, 'mean_rating_movie_pred', 0, 6, 0.6,
                         baseline_values=mean_rating_movie,
                         xlab="Movie-wise Average Ratings")

util$plot_hist_quantiles(samples3, 'var_rating_customer_pred', 0, 7, 0.5,
                         baseline_values=var_rating_customer,
                         xlab="Customer-wise Rating Variances")

util$plot_hist_quantiles(samples3, 'var_rating_movie_pred', 0, 7, 0.5,
                         baseline_values=var_rating_movie,
                         xlab="Movie-wise Rating Variances")

# Movie-wise empirical covariances are still more heavy-tailed.

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

filtered_samples <- util$filter_expectands(samples3, 
                                           covar_rating_movie_filt_names)

util$plot_hist_quantiles(filtered_samples, 'covar_rating_movie_pred', -4.25, 4.25, 0.25,
                         baseline_values=covar_rating_movie_filt,
                         xlab="Filtered Movie-wise Rating Covariances")

# Not a terribly inadequate model.  What would we learn from it?

# Customer inferences

# Small but non-negligible cutpoint heterogeneity.
# Baseline probabilities are relatively optimistic, concentrating
# at a rating of four.

par(mfrow=c(2, 1), mar=c(5, 5, 1, 1))

util$plot_expectand_pushforward(samples3[['tau_q']],
                                25, flim=c(0, 0.3),
                                display_name='tau_q')

names <- sapply(1:5, function(k) paste0('mu_q[', k, ']'))
util$plot_disc_pushforward_quantiles(samples3, names,
                                     xlab="Rating",
                                     ylab="Baseline Rating Probability")

# The hierarchy strongly regularizes the cut-points of infrequent raters.

c <- 42
table(data$customer_idxs)[c]

par(mfrow=c(2, 2), mar=c(5, 5, 1, 1))

lab2_xs <- c(1.5, 2, -4, -3)
lab2_ys <- c(0.15, 0.25, 0.35, 0.35)

lab3_xs <- c(-6, -5, 3, 3.5)
lab3_ys <- c(0.25, 0.4, 0.5, 0.6)

for (k in 1:4) {
  name <- paste0('cut_points[', c, ',', k, ']')
  util$plot_expectand_pushforward(samples2[[name]],
                                50, flim=c(-10, 5), ylim=c(0, 1.0),
                                col=util$c_light, display_name='Cut Points',
                                main=paste('Customer', c))
  
  name <- paste0('cut_points[', c, ',', k, ']')
  util$plot_expectand_pushforward(samples3[[name]],
                                  50, flim=c(-10, 5),
                                  col=util$c_dark, border="#BBBBBB88",
                                  add=TRUE)
  
  text(lab2_xs[k], lab2_ys[k], "Model 2", col=util$c_light)
  text(lab3_xs[k], lab3_ys[k], "Model 3", col=util$c_dark)
}

# Some systematic patterns in the cutpoints.

par(mfrow=c(4, 1), mar=c(5, 5, 1, 1))

for (k in 1:4) {
  names <- sapply(1:data$N_customers,
                  function(c) paste0('cut_points[', c, ',', k, ']'))
  util$plot_disc_pushforward_quantiles(samples3, names,
                                       xlab="customer", 
                                       display_ylim=c(-6, 6),
                                       ylab=paste0('cut_points[', k, ']'))
}

# Customer 23 is stillstingy;  Customer 70 is still generous.

par(mfrow=c(2, 1), mar=c(5, 5, 1, 1))

cols <- c(util$c_mid, util$c_mid_highlight, 
          util$c_dark, util$c_dark_highlight)

c <- 23

k <- 1
name <-paste0('cut_points[', c, ',', k, ']')
util$plot_expectand_pushforward(samples3[[name]],
                                50, flim=c(-8, 8), ylim=c(0, 2),
                                col=cols[k], display_name='Cut Points',
                                main=paste('Customer', c))


for (k in 2:4) {
  name <-paste0('cut_points[', c, ',', k, ']')
  util$plot_expectand_pushforward(samples3[[name]],
                                  50, flim=c(-8, 8),
                                  col=cols[k], border="#BBBBBB88",
                                  add=TRUE)
}

text(0, 1.65, "cut_points[1]", col=util$c_mid)
text(2, 1.45, "cut_points[2]", col=util$c_mid_highlight)
text(4, 1, "cut_points[3]", col=util$c_dark)
text(6, 0.5, "cut_points[4]", col=util$c_dark_highlight)

c <- 70

k <- 1
name <-paste0('cut_points[', c, ',', k, ']')
util$plot_expectand_pushforward(samples3[[name]],
                                50, flim=c(-8, 8), ylim=c(0, 2),
                                col=cols[k], display_name='Cut Points',
                                main=paste('Customer', c))


for (k in 2:4) {
  name <-paste0('cut_points[', c, ',', k, ']')
  util$plot_expectand_pushforward(samples3[[name]],
                                  50, flim=c(-8, 8),
                                  col=cols[k], border="#BBBBBB88",
                                  add=TRUE)
}

text(-5.5, 0.4, "cut_points[1]", col=util$c_mid)
text(-3, 0.85, "cut_points[2]", col=util$c_mid_highlight)
text(-2, 1.15, "cut_points[3]", col=util$c_dark)
text(1.0, 1.35, "cut_points[4]", col=util$c_dark_highlight)


# Movie inferences assume that all customers have the same taste in movies.

par(mfrow=c(2, 1), mar=c(5, 5, 1, 1))

util$plot_expectand_pushforward(samples3[['tau_gamma']],
                                50, flim=c(0, 1),
                                display_name='tau_gamma')

names <- sapply(1:data$N_movies,
                function(m) paste0('gamma[', m, ']'))
util$plot_disc_pushforward_quantiles(samples3, names,
                                     xlab="Movie",
                                     ylab="Affinity")

# We can use these movie inferences to rank the movies by their expected
# affinities.  Faster than ranking by pair-wise comparison probabilities.

expected_affinity <- function(m) {
  util$ensemble_mcmc_est(samples3[[paste0('gamma[', m, ']')]])[1]
}

expected_affinities <- sapply(1:data$N_movies,
                             function(m) expected_affinity(m))

post_mean_ordering <- sort(expected_affinities, index.return=TRUE)$ix

# Five worst movies

print(data.frame("Rank"=200:196, 
                 "Movie"=head(post_mean_ordering, 5)),
      row.names=FALSE)

# Five best movies.

print(data.frame("Rank"=5:1,
                 "Movie"=tail(post_mean_ordering, 5)),
      row.names=FALSE)

# Affinity of the best verses the worst movie.

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

m1 <- head(post_mean_ordering, 1)
name <-paste0('gamma[', m1, ']')
util$plot_expectand_pushforward(samples3[[name]],
                                50, flim=c(-3, 3), 
                                ylim=c(0, 1.3),
                                col=util$c_mid, 
                                display_name='Affinity')
text(-1.25, 1.2, paste('Movie', m1), col=util$c_mid)

m2 <- tail(post_mean_ordering, 1)
name <-paste0('gamma[', m2, ']')
util$plot_expectand_pushforward(samples3[[name]],
                                50, flim=c(-3, 3),
                                col=util$c_dark, 
                                border="#BBBBBB88",
                                add=TRUE)
text(1.25, 1.2, paste('Movie', m2), col=util$c_dark)

# Little ambiuity that Movie 180 is better than Movie 31.

var_repl <- list('g1' = paste0('gamma[', m1,']'),
                 'g2' = paste0('gamma[', m2,']'))

p_est <-
  util$implicit_subset_prob(samples3,
                            function(g1, g2) g1 < g2,
                            var_repl)

format_string <- paste0("Posterior probability that movie %i affinity ",
                        "> movie %i affinity = %.3f +/- %.3f.")
cat(sprintf(format_string, m1, m2, p_est[1], 2 * p_est[2]))

# Movie affinities ranked by expected affinity.

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

names <- sapply(post_mean_ordering,
                function(m) paste0('gamma[', m, ']'))
util$plot_disc_pushforward_quantiles(samples3, names,
                                     xlab="Movies Ordered by Expected Affinity",
                                     ylab="Affinity")

################################################################################
# Hierarchical customer Model With Varying Tastes
################################################################################

# Our last model exhibits some weak retrodictive tension.  At the same time
# it made the strong assumption that, once the different interpretation of 
# ratings is taken into account, all customers have the same relative 
# opinions about each movie.

# Allowing for individual taste allows us to make much more nuanced inferences
# and predictions.  Let's see if it resolves some of the retrodictive tensions.

# The challenge with trying to infer personal movie preferences is that all 
# customers rate only a very small proportion of the available movies.  We 
# can visualize this with a two-dimensional array with binary entries.

xs <- seq(1, data$N_movies, 1)
ys <- seq(1, data$N_customers, 1)
zs <- matrix(0, nrow=data$N_movies, ncol=data$N_customers)

for (n in 1:data$N_ratings) {
  zs[data$movie_idxs[n], data$customer_idxs[n]] <- 1
}

par(mfrow=c(1, 1), mar = c(5, 5, 1, 1))

image(xs, ys, zs, col=c("white", util$c_dark_teal),
      xlab="Movie", ylab="customer")

# In the machine learning literature the problem of predicting customer 
# preference for unrated movies is often known as "matrix completion".

# Here we'll use a multivariate normal population model to pool 
# individual preferences across customers.

# EQN.

# We can still model the elements of the population baseline with 
# their own, one-dimensional, population model,

# EQN.

# We'll begin with a non-centered implementation of both movie 
# affinity population models.

fit <- stan(file="stan_programs/model4.stan",
            data=data, seed=8438338, init=0,
            warmup=1000, iter=2024, refresh=0)

# Clean diagnostics.

diagnostics <- util$extract_hmc_diagnostics(fit)
util$check_all_hmc_diagnostics(diagnostics)

samples4 <- util$extract_expectand_vals(fit)
base_samples <- util$filter_expectands(samples4,
                                       c('gamma0_ncp', 
                                         'tau_gamma0',
                                         'delta_gamma_ncp',
                                         'tau_delta_gamma', 
                                         'L_delta_gamma',
                                         'cut_points', 
                                         'mu_q', 'tau_q'),
                                       check_arrays=TRUE)
util$check_all_expectand_diagnostics(base_samples,
                                     exclude_zvar=TRUE)

# No compromises to retrodictive performance, 
# although not much improvement either.

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

util$plot_hist_quantiles(samples4, 'rating_pred', -0.5, 6.5, 1,
                         baseline_values=data$ratings,
                         xlab="All Ratings")



par(mfrow=c(2, 3), mar=c(5, 5, 1, 1))

for (c in c(7, 23, 40, 70, 77, 100)) {
  names <- sapply(which(data$customer_idxs == c), 
                  function(n) paste0('rating_pred[', n, ']'))
  filtered_samples <- util$filter_expectands(samples4, names)
  
  util$plot_hist_quantiles(filtered_samples, 'rating_pred', -0.5, 6.5, 1,
                           baseline_values=data$ratings[data$customer_idxs == c],
                           xlab="Ratings",
                           main=paste('Customer', c))
}



par(mfrow=c(2, 3), mar=c(5, 5, 1, 1))

for (m in c(33, 53, 61, 80, 117, 180)) {
  names <- sapply(which(data$movie_idxs == m), 
                  function(n) paste0('rating_pred[', n, ']'))
  filtered_samples <- util$filter_expectands(samples4, names)
  
  util$plot_hist_quantiles(filtered_samples, 'rating_pred', -0.5, 6.5, 1,
                           baseline_values=data$ratings[data$movie_idxs == m],
                           xlab="Ratings",
                           main=paste('Movie', m))
}



par(mfrow=c(2, 2), mar=c(5, 5, 1, 1))

util$plot_hist_quantiles(samples4, 'mean_rating_customer_pred', 0, 6, 0.5,
                         baseline_values=mean_rating_customer,
                         xlab="Customer-wise Average Ratings")

util$plot_hist_quantiles(samples4, 'mean_rating_movie_pred', 0, 6, 0.6,
                         baseline_values=mean_rating_movie,
                         xlab="Movie-wise Average Ratings")

util$plot_hist_quantiles(samples4, 'var_rating_customer_pred', 0, 7, 0.5,
                         baseline_values=var_rating_customer,
                         xlab="Customer-wise Rating Variances")

util$plot_hist_quantiles(samples4, 'var_rating_movie_pred', 0, 7, 0.5,
                         baseline_values=var_rating_movie,
                         xlab="Movie-wise Rating Variances")



par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

filtered_samples <- util$filter_expectands(samples4, 
                                           covar_rating_movie_filt_names)

util$plot_hist_quantiles(filtered_samples, 'covar_rating_movie_pred', -4.25, 4.25, 0.25,
                         baseline_values=covar_rating_movie_filt,
                         xlab="Filtered Movie-wise Rating Covariances")

# Cutpoint inferences.

# Population inferences mostly consistent with previous model.
# Baseline rating probabilities slightly shift from 4 to 3.

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

util$plot_expectand_pushforward(samples3[['tau_q']],
                                25, flim=c(0, 0.3),
                                col=util$c_light,
                                display_name='tau_q')
text(0.05, 10, "Model 2", col=util$c_light)

util$plot_expectand_pushforward(samples4[['tau_q']],
                                25, flim=c(0, 0.3),
                                col=util$c_dark,
                                border="#BBBBBB88", 
                                add=TRUE)
text(0.2, 10, "Model 3", col=util$c_dark)


par(mfrow=c(2, 1), mar=c(5, 5, 1, 1))

names <- sapply(1:5, function(k) paste0('mu_q[', k, ']'))
util$plot_disc_pushforward_quantiles(samples3, names,
                                     xlab="Rating",
                                     ylab="Baseline Rating Probability",
                                     main="Model 3")

names <- sapply(1:5, function(k) paste0('mu_q[', k, ']'))
util$plot_disc_pushforward_quantiles(samples4, names,
                                     xlab="Rating",
                                     ylab="Baseline Rating Probability",
                                     main="Model 4")

# Individual customer cutpoints, however, change slightly
# for example shifting towards larger values for Customer 23.

par(mfrow=c(2, 2), mar=c(5, 5, 1, 1))

c <- 23

lab3_xs <- c(0, -0.5, 0, 1)
lab3_ys <- c(1.75, 0.5, 0.5, 0.25)

lab4_xs <- c(2, 4, 5.5, 8)
lab4_ys <- c(0.5, 0.5, 0.5, 0.25)

for (k in 1:4) {
  name <-paste0('cut_points[', c, ',', k, ']')
  util$plot_expectand_pushforward(samples3[[name]],
                                  40, flim=c(-2, 10),
                                  col=util$c_light,
                                  display_name=name)
  util$plot_expectand_pushforward(samples4[[name]],
                                  40, flim=c(-2, 10),
                                  col=util$c_dark,
                                  border="#BBBBBB88", 
                                  add=TRUE)
  
  text(lab3_xs[k], lab3_ys[k], "Model 3", col=util$c_light)
  text(lab4_xs[k], lab4_ys[k], "Model 4", col=util$c_dark)
}




# Movie inferences

# We can use the baseline affinities to emulate the universal
# movie preferences from the previous model.

par(mfrow=c(2, 1), mar=c(5, 5, 1, 1))

util$plot_expectand_pushforward(samples4[['tau_gamma0']],
                                25, flim=c(0, 1.25),
                                display_name='tau_gamma0')

names <- sapply(1:data$N_movies,
                function(m) paste0('gamma0[', m, ']'))
util$plot_disc_pushforward_quantiles(samples4, names,
                                     xlab="Movie",
                                     ylab="Baseline Affinities")

# Now we can investigate the preferences ideosyncratic to each customer.

# Individual affinity scales.
# The larger the scale is the more heterogeneity there is in 
# the individual customer preferences.

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

names <- sapply(1:data$N_movies,
                function(m) paste0('tau_delta_gamma[', m, ']'))
util$plot_disc_pushforward_quantiles(samples4, names,
                                     xlab="Movie",
                                     ylab="Affinity Variation Scales")
abline(v=159)

# For example the customers in this data set disagree about Movie 159

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

m <- 159
name <- paste0('tau_delta_gamma[', m, ']')
util$plot_expectand_pushforward(samples4[[name]],
                                25, flim=c(0, 10),
                                display_name=name)

# The correlations in the multinormal population model 
# inform individual preferences for unrated movies.
# Too many elements of the correlation matrix to
# effectively plot in a reasonable amount of time.

# In order to determine individual movie affinities for a 
# particular customer we need to combine the baseline 
# affinities with the individual preferences.

par(mfrow=c(2, 1), mar=c(5, 5, 1, 1))

c <- 23

names <- sapply(1:data$N_movies,
                function(m) paste0('gamma0[', m, ']'))
util$plot_disc_pushforward_quantiles(samples4, names,
                                     xlab="Movie",
                                     ylab="Baseline Affinity",
                                     main="Baseline")

names <- sapply(1:data$N_movies,
                function(m) paste0('delta_gamma[', c, ',', m, ']'))
util$plot_disc_pushforward_quantiles(samples4, names,
                                     xlab="Movie",
                                     ylab="Change in Affinity",
                                     main=paste0('Customer', c))



expectands <- sapply(1:data$N_movies, 
                     function(m) 
                       local({ idx = m; function(x1, x2) x1[idx] + x2[idx] }) )
names(expectands) <- sapply(1:data$N_movies,
                            function(m) paste0('gamma[', c, ',', m, ']'))

var_repl <- list('x1'=array(sapply(1:data$N_movies,
                                   function(m) paste0('gamma0[', m, ']'))),
                 'x2'=array(sapply(1:data$N_movies,
                                   function(m) paste0('delta_gamma[', c, ',', m, ']'))))

affinity_samples <-
  util$eval_expectand_pushforwards(samples4,
                                   expectands,
                                   var_repl)




par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

names <- sapply(1:data$N_movies,
                function(m) paste0('gamma[', c, ',', m, ']'))

util$plot_disc_pushforward_quantiles(affinity_samples, names,
                                     xlab="Movie",
                                     ylab="Affinity",
                                     main=paste('Customer', c))

# We can isolate the affinities directly informed by observed ratings and 
# those informed by only the multinormal hierarchical model by separately 
# visualizing the movies that have been rated and unrated.

rated_movie_idxs <- data$movie_idxs[data$customer_idxs == c]
unrated_movie_idxs <- setdiff(1:data$N_movies, rated_movie_idxs)

par(mfrow=c(2, 1), mar=c(5, 5, 1, 1))

names <- sapply(1:data$N_movies,
                function(m) paste0('gamma[', c, ',', m, ']'))
util$plot_disc_pushforward_quantiles(affinity_samples, names,
                                     xlab="Rated Movie",
                                     ylab="Customer Affinity",
                                     main=paste0('Customer', c))
for (m in unrated_movie_idxs) {
  polygon(c(m - 0.5, m + 0.5, m + 0.5, m- 0.5),
          c(-4.75, -4.75, 4.75, 4.75), col="white", border=NA)
}

util$plot_disc_pushforward_quantiles(affinity_samples, names,
                                     xlab="Unrated Movie",
                                     ylab="Customer Affinity",
                                     main=paste0('Customer', c))
for (m in rated_movie_idxs) {
  polygon(c(m - 0.5, m + 0.5, m + 0.5, m- 0.5),
          c(-4.75, -4.75, 4.75, 4.75), col="white", border=NA)
}


# We can then use these individual preferences to make movie recommendations
# for Customer 23.

expected_affinity <- function(m) {
  util$ensemble_mcmc_est(affinity_samples[[paste0('gamma[', c, ',', m, ']')]])[1]
}

expected_affinities <- sapply(1:data$N_movies,
                             function(m) expected_affinity(m))

post_mean_ordering <- sort(expected_affinities, index.return=TRUE)$ix



par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

names <- sapply(post_mean_ordering,
                function(m) paste0('gamma[', c, ',', m, ']'))
util$plot_disc_pushforward_quantiles(affinity_samples, names,
                                     xlab="Movies Ordered by Expected Affinity",
                                     ylab="Affinity")


# Worst five movies.

print(data.frame("Rank"=200:196, 
                 "Movie"=head(post_mean_ordering, 5)),
      row.names=FALSE)

# Best five movies.

print(data.frame("Rank"=5:1,
                 "Movie"=tail(post_mean_ordering, 5)),
      row.names=FALSE)


# Can we recommend some new movies to customer 13?
# Assume unrated is unseen.

expected_affinities <- sapply(unrated_movie_idxs,
                             function(m) expected_affinity(m))

post_mean_ordering <- sort(expected_affinities, index.return=TRUE)$ix

# Top 10 recommendations

print(data.frame("Rank"=10:1,
                 "Movie"=tail(unrated_movie_idxs[post_mean_ordering], 10)),
      row.names=FALSE)

# Relatively mild confidence in these recommendations given the large uncertainties.

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

names <- sapply(unrated_movie_idxs[post_mean_ordering],
                function(m) paste0('gamma[', c, ',', m, ']'))
util$plot_disc_pushforward_quantiles(affinity_samples, names,
                                     xlab="Unrated Movies Ordered by Expected Affinity",
                                     ylab="Affinity")


# How would Customer 23 rate our top recommendation?

movie_idx <- tail(unrated_movie_idxs[post_mean_ordering], 1)

logistic <- function(x) {
  if (x > 0) {
    1 / (1 + exp(-x))
  } else {
    e <- exp(x)
    e / (1 + e)
  }
}

expectands <- list(function(c, gamma) 1 - logistic(gamma - c[1]),
                   function(c, gamma) logistic(gamma - c[1]) - logistic(gamma - c[2]),
                   function(c, gamma) logistic(gamma - c[2]) - logistic(gamma - c[3]),
                   function(c, gamma) logistic(gamma - c[3]) - logistic(gamma - c[4]),
                   function(c, gamma) logistic(gamma - c[4]))
names(expectands) <- c('p[1]', 'p[2]', 'p[3]', 'p[4]', 'p[5]')

var_repl <- list('c'=array(sapply(1:4, function(k) paste0('cut_points[', c, ',', k, ']'))),
                 'gamma'=paste0('gamma[', c, ',', movie_idx, ']'))

for (k in 1:4) {
  name <- paste0('cut_points[', c, ',', k, ']')
  affinity_samples[[name]] <- samples4[[name]]
}

prob_samples <-util$eval_expectand_pushforwards(affinity_samples,
                                                expectands,
                                                var_repl)


par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

util$plot_disc_pushforward_quantiles(prob_samples, names(expectands),
                                     xlab="Rating",
                                     ylab="Posterior Probability")

# We're not predicting a high rating for the best recommendaton, but 
# this isn't unexpected given the rather stingy reviews from Customer 23
# anyways!

par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

util$plot_line_hist(data$ratings[data$customer_idxs == c], -0.5, 6.5, 1,
                    xlab="Rating", main=paste('customer', c))

# With the hierarchical models we're not even limited to the observed
# customer.  We can make predictions for a new customer by drawing new 
# cut point and movie affinities from the respective hierarchical 
# population model inferences.  With so few ratings across the observed 
# customers these predictions will be highly uncertain, but that 
# uncertainty also prevents us from making overly confident claims.
