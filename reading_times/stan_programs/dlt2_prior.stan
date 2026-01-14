functions {
  // Mean-dispersion parameterization of inverse gamma family
  real inv_gamma_md_lpdf(real x, real log_mu, real psi) {
    return inv_gamma_lpdf(x | inv(psi) + 2,
                              exp(log_mu) * (inv(psi) + 1));
  }

  real inv_gamma_md_rng(real log_mu, real psi) {
    return inv_gamma_rng(inv(psi) + 2,
                         exp(log_mu) * (inv(psi) + 1));
  }
}

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
  // Log Reading Time Baseline
  real kappa = normal_rng(5.76, 0.50);

  // Relative item difficulties
  vector[N_items - 1] delta_free
    = to_vector(normal_rng(zeros_vector(N_items - 1), 0.99));

  // Relative subject skills
  vector[N_subjects - 1] zeta_free
    = to_vector(normal_rng(zeros_vector(N_subjects - 1), 0.99));

  // Subject Relative Difference
  real chi = normal_rng(0, 0.99);

  // Measurement scale
  real<lower=0> phi = abs(normal_rng(0, 3.89));

  // Relative skills for all items and subjects
  vector[N_items] delta
    = append_row([0]', delta_free);
  vector[N_subjects] zeta
    = append_row([0]', zeta_free);

  array[N] real log_reading_time_pred;

  for (n in 1:N) {
    int i = item[n];
    int s = subject[n];

    real log_mu = kappa + delta[i] - zeta[s];
    if (subj_rel[n] == 0) log_mu += chi;

    log_reading_time_pred[n] = log(inv_gamma_md_rng(log_mu, phi));
  }
}
