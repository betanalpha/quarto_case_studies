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
  // Log Reading Time Baseline
  real kappa;

  // Relative item difficulties
  vector[N_items - 1] delta_free;

  // Relative subject skills
  vector[N_subjects - 1] zeta_free;

  // Subject Relative Difference
  real chi;

  // Measurement scale
  real<lower=0> phi;
}

transformed parameters {
  // Relative skills for all items and subjects
  vector[N_items] delta
    = append_row([0]', delta_free);
  vector[N_subjects] zeta
    = append_row([0]', zeta_free);
}

model {
  // Location configurations
  vector[N] log_mu =   kappa
                     + delta[item] - zeta[subject]
                     + (1 - to_vector(subj_rel)) * chi;

  // Prior model

  // 100 <~ exp(kappa) <~ 1000
  target += normal_lpdf(kappa | 5.76, 0.50);

  // 0.1 <~ exp(delta) <~ 10
  target += normal_lpdf(delta_free | 0, 0.99);

  // 0.1 <~ exp(zeta) <~ 10
  target += normal_lpdf(zeta_free| 0, 0.99);

  // 0.1 <~ exp(chi) <~ 10
  target += normal_lpdf(chi | 0, 0.99);

  // 0 <~ phi <~ 10
  target += normal_lpdf(phi | 0, 3.89);

  // Observational model
  target += lognormal_lpdf(reading_time | log_mu, phi);
}

generated quantities {
  array[N] real log_reading_time_pred;

  for (n in 1:N) {
    int i = item[n];
    int s = subject[n];

    real log_mu = kappa + delta[i] - zeta[s];
    if (subj_rel[n] == 0) log_mu += chi;

    log_reading_time_pred[n] = log(lognormal_rng(log_mu, phi));
  }
}
