functions {
  // Log probability density function over cut point
  // induced by a Dirichlet probability density function
  // over baseline probabilities and latent logistic
  // density function.
  real induced_dirichlet_lpdf(vector c, vector alpha) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);

    // Log Jacobian correction
    real logJ = 0;
    for (k in 1:(K - 1)) {
      if (c[k] >= 0)
        logJ += -c[k] - 2 * log(1 + exp(-c[k]));
      else
        logJ += +c[k] - 2 * log(1 + exp(+c[k]));
    }

    return dirichlet_lpdf(p | alpha) + logJ;
  }
}

data {
  int<lower=1> N_ratings;
  array[N_ratings] int<lower=1, upper=5> ratings;

  int<lower=1> N_customers;
  array[N_ratings] int<lower=1, upper=N_customers> customer_idxs;

  int<lower=1> N_movies;
  array[N_ratings] int<lower=1, upper=N_movies> movie_idxs;
}

parameters {
  vector[N_movies] gamma_ncp; // Non-centered movie affinities
  real<lower=0> tau_gamma;    // Movie affinity population scale

  array[N_customers] ordered[4] cut_points; // Customer rating cut points

  simplex[5] mu_q;     // Rating simplex population location
  real<lower=0> tau_q; // Rating simplex population scale
}

transformed parameters {
  // Centered movie affinities
  vector[N_movies] gamma = tau_gamma * gamma_ncp;
}

model {
  vector[5] ones = rep_vector(1, 5);
  vector[5] alpha = mu_q / tau_q + ones;

  // Prior model
  gamma_ncp ~ normal(0, 1);
  tau_gamma ~ normal(0, 5 / 2.57);

  mu_q ~ dirichlet(5 * ones);
  tau_q ~ normal(0, 1);

  for (c in 1:N_customers)
    cut_points[c] ~ induced_dirichlet(alpha);

  // Observational model
  for (n in 1:N_ratings) {
    int c = customer_idxs[n];
    int m = movie_idxs[n];
    ratings[n] ~ ordered_logistic(gamma[m], cut_points[c]);
  }
}

generated quantities {
  array[N_ratings] int<lower=1, upper=5> rating_pred;

  array[N_customers] real mean_rating_customer_pred
    = rep_array(0, N_customers);
  array[N_customers] real var_rating_customer_pred
    = rep_array(0, N_customers);

  array[N_movies] real mean_rating_movie_pred = rep_array(0, N_movies);
  array[N_movies] real var_rating_movie_pred = rep_array(0, N_movies);

  matrix[N_movies, N_movies] covar_rating_movie_pred;

  {
    array[N_customers] real C = rep_array(0, N_customers);
    array[N_movies] real M = rep_array(0, N_movies);

    for (n in 1:N_ratings) {
      real delta = 0;
      int c = customer_idxs[n];
      int m = movie_idxs[n];

      rating_pred[n] = ordered_logistic_rng(gamma[m], cut_points[c]);

      C[c] += 1;
      delta = rating_pred[n] - mean_rating_customer_pred[c];
      mean_rating_customer_pred[c] += delta / C[c];
      var_rating_customer_pred[c]
        += delta * (rating_pred[n] - mean_rating_customer_pred[c]);

      M[m] += 1;
      delta = rating_pred[n] - mean_rating_movie_pred[m];
      mean_rating_movie_pred[m] += delta / M[m];
      var_rating_movie_pred[m]
        += delta * (rating_pred[n] - mean_rating_movie_pred[m]);
    }

    for (c in 1:N_customers) {
      if (C[c] > 1)
        var_rating_customer_pred[c] /= C[c] - 1;
      else
        var_rating_customer_pred[c] = 0;
    }
    for (m in 1:N_movies) {
      if (M[m] > 1)
        var_rating_movie_pred[m] /= M[m] - 1;
      else
        var_rating_movie_pred[m] = 0;
    }
  }

  {
    matrix[N_movies, N_movies] counts;

    for (m1 in 1:N_movies) {
      for (m2 in 1:N_movies) {
        counts[m1, m2] = 0;
        covar_rating_movie_pred[m1, m2] = 0;
      }
    }

    for (n1 in 1:N_ratings) {
      for (n2 in 1:N_ratings) {
        if (customer_idxs[n1] == customer_idxs[n2]) {
          int m1 = movie_idxs[n1];
          int m2 = movie_idxs[n2];
          real y =   (ratings[n1] - mean_rating_movie_pred[m1])
                   * (ratings[n2] - mean_rating_movie_pred[m2]);
          covar_rating_movie_pred[m1, m2] += y;
          covar_rating_movie_pred[m2, m1] += y;
          counts[m1, m2] += 1;
          counts[m2, m1] += 1;
        }
      }
    }

    for (m1 in 1:N_movies) {
      for (m2 in 1:N_movies) {
        if (counts[m1, m2] > 1)
          covar_rating_movie_pred[m1, m2] /= counts[m1, m2] - 1;
      }
    }
  }
}
