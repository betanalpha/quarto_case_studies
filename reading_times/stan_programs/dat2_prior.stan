data {
  int<lower=1> N; // Number of observations

  // Item configuration
  int<lower=1> N_items;
  array[N] int<lower=1, upper=N_items> item;

  // Subject configuration
  int<lower=1> N_subjects;
  array[N] int<lower=1, upper=N_subjects> subject;

  // Item variant
  //   Object Relative:  subj_rel = 0
  //   Subject Relative: subj_rel = 1
  array[N] int<lower=0, upper=1> subj_rel;

  vector<lower=0>[N] reading_time; // Reading times (ms)
}

generated quantities {
  // Log reading time baseline
  real nu = normal_rng(5.76, 0.50);

  // Relative item difficulties
  vector[N_items - 1] delta_free
    = to_vector(normal_rng(zeros_vector(N_items - 1), 0.99));

  // Relative subject skills
  vector[N_subjects - 1] zeta_free
    = to_vector(normal_rng(zeros_vector(N_subjects - 1), 0.99));

  // Initial failure difference
  real<lower=0> omega = abs(normal_rng(0, 0.90));

  // Measurement scales
  real<lower=0> phi1 = abs(normal_rng(0, 3.89));
  real<lower=0> phi2 = abs(normal_rng(0, 3.89));

  // Initial failure probabilities
  real<lower=0, upper=1> lambda_SR = beta_rng(1, 1);
  real<lower=0, upper=1> lambda_OR = beta_rng(1, 1);

  // Relative skills for all items and subjects
  vector[N_items] delta
    = append_row([0]', delta_free);
  vector[N_subjects] zeta
    = append_row([0]', zeta_free);

  array[N] real log_reading_time_pred;

  for (n in 1:N) {
    int i = item[n];
    int s = subject[n];
    real log_mu = nu + delta[i] - zeta[s];

    if (subj_rel[n] == 1) {
      if (bernoulli_rng(lambda_SR)) {
        log_reading_time_pred[n]
          = log(lognormal_rng(log_mu,         phi1));
      } else {
        log_reading_time_pred[n]
          = log(lognormal_rng(log_mu + omega, phi2));
      }
    } else {
       if (bernoulli_rng(lambda_OR)) {
        log_reading_time_pred[n]
          = log(lognormal_rng(log_mu,         phi1));
      } else {
        log_reading_time_pred[n]
          = log(lognormal_rng(log_mu + omega, phi2));
      }
    }
  }
}
