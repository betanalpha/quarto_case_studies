data {
  // Number of observations
  int<lower=1> N;

  // Relationship status
  // k = 1: Stable partner
  // k = 2: No partner
  int<lower=1> K_rel;

  // Cancer stage
  // k = 1: No cancer
  // k = 2: Early stage cancer
  // k = 3: Advanced stage cancer
  int<lower=1> K_stg;

  // Toxicity status
  // k = 1: None
  // k = 2: Low
  // k = 3: Medium
  // k = 4: High
  int<lower=1> K_tox;

  // Observed conception status
  // y = 0: No conception
  // y = 1: Conception
  array[N] int<lower=0, upper=1> y;

  // Observed relationship status;
  array[N] int<lower=1, upper=K_rel> k_rel;

  // Observed cancer stage;
  array[N] int<lower=1, upper=K_stg> k_stg;

  // Observed toxicity status;
  array[N] int<lower=1, upper=K_tox> k_tox;
}

parameters {
  // Probability of conception for baseline patients in a stable
  // relationship, no cancer, and no toxicity
  real<lower=0, upper=1> q_C_0;

  // Proportional decreases in conception probability due to
  // non-baseline relationship status, cancer stage, and toxicity
  // status.
  positive_ordered[K_rel - 1] alpha_rel;
  positive_ordered[K_stg - 1] alpha_stg;
  positive_ordered[K_tox - 1] alpha_tox;
}

transformed parameters {
  vector[K_rel] alpha_rel_buff = append_row([0]', alpha_rel);
  vector[K_stg] alpha_stg_buff = append_row([0]', alpha_stg);
  vector[K_tox] alpha_tox_buff = append_row([0]', alpha_tox);
}

model {
  // Prior model
  target += beta_lpdf(q_C_0 | 12.7, 3.7); // 0.50 <~ q_C_0 <~ 0.95

  target += normal_lpdf(alpha_rel | 0, 3 / 2.32); // 0 <~ alpha <~ -log(0.05)
  target += normal_lpdf(alpha_stg | 0, 3 / 2.32); // 0 <~ alpha <~ -log(0.05)
  target += normal_lpdf(alpha_tox | 0, 3 / 2.32); // 0 <~ alpha <~ -log(0.05)

  // Observational model
  target += bernoulli_lpmf(y | q_C_0 * exp(-alpha_rel_buff[k_rel]
                                           -alpha_stg_buff[k_stg]
                                           -alpha_tox_buff[k_tox]));
}

generated quantities {
  // Proportional decreases in conception probability
  vector[K_rel] gamma_rel_buff = exp(-alpha_rel_buff);
  vector[K_stg] gamma_stg_buff = exp(-alpha_stg_buff);
  vector[K_tox] gamma_tox_buff = exp(-alpha_tox_buff);

  // Posterior predictive data
  array[N] int<lower=0, upper=1> y_pred;

  for (n in 1:N) {
    y_pred[n] = bernoulli_rng(q_C_0 * exp(-alpha_rel_buff[k_rel[n]]
                                          -alpha_stg_buff[k_stg[n]]
                                          -alpha_tox_buff[k_tox[n]]));
  }
}
