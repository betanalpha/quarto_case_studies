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

parameters {
  // Log reading time baseline for successful retrieval
  real nu;

  // Relative item difficulties
  vector[N_items - 1] delta_free;

  // Relative subject skills
  vector[N_subjects - 1] zeta_free;

  // Log offset for initial failure before successful retrieval
  real<lower=0> omega;

  // Measurement scales
  real<lower=0> phi1;
  real<lower=0> phi2;

  // Mixture probabilities
  real<lower=0, upper=1> lambda_SR;
  real<lower=0, upper=1> lambda_OR;
}

transformed parameters {
  // Relative skills for all items and subjects
  vector[N_items] delta
    = append_row([0]', delta_free);
  vector[N_subjects] zeta
    = append_row([0]', zeta_free);
}

model {
  // Prior model

  // 100 <~ exp(nu) <~ 1000
  target += normal_lpdf(nu | 5.76, 0.50);

  // 0.1 <~ exp(delta) <~ 10
  target += normal_lpdf(delta_free | 0, 0.99);

  // 0.1 <~ exp(zeta) <~ 10
  target += normal_lpdf(zeta_free| 0, 0.99);

  // 1 <~ exp(omega) <~ 10
  target += normal_lpdf(omega | 0, 0.90);

  // 0 <~ phi <~ 10
  target += normal_lpdf(phi1 | 0, 3.89);
  target += normal_lpdf(phi2 | 0, 3.89);

  // Uniform prior density functions
  target += beta_lpdf(lambda_SR | 1, 1);
  target += beta_lpdf(lambda_OR | 1, 1);

  // Observational model
  for (n in 1:N) {
    int i = item[n];
    int s = subject[n];
    real log_mu = nu + delta[i] - zeta[s];

    if (subj_rel[n] == 1) {
      real lpd1
        =   log(lambda_SR)
          + lognormal_lpdf(reading_time[n] | log_mu,         phi1);
      real lpd2
        =   log(1 - lambda_SR)
          + lognormal_lpdf(reading_time[n] | log_mu + omega, phi2);

      target += log_sum_exp(lpd1, lpd2);
    } else {
      real lpd1
        =   log(lambda_OR)
          + lognormal_lpdf(reading_time[n] | log_mu,         phi1);
      real lpd2
        =   log(1 - lambda_OR)
          + lognormal_lpdf(reading_time[n] | log_mu + omega, phi2);

      target += log_sum_exp(lpd1, lpd2);
    }
  }
}

generated quantities {
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
