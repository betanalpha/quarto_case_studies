data {
  // Number of observations
  int<lower=1> N;

  // Number of predictions
  int<lower=1> N_pred;

  // Relationship status
  // k = 1: Stable partner
  // k = 2: No partner
  int<lower=1> K_rel;

  // Cancer stage
  // k = 1: No cancer
  // k = 2: Early stage cancer
  // k = 3: Advanced stage cancer
  int<lower=1> K_stg;

  // Treatment status
  // k = 1: No treatment
  // k = 2: Treatment
  int<lower=1> K_trt;

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

  // Observed treatment status;
  array[N] int<lower=1, upper=K_trt> k_trt;

  // Observed toxicity status;
  array[N] int<lower=1, upper=K_tox> k_tox;

  // Observed assistive reproductive technology (ART) status
  // k = 0: No ART
  // k = 1: ART
  array[N] int<lower=0, upper=1> k_art;

  // Hypothetical toxicity distribution configurations
  vector[K_tox] alpha_tox_hyp1;
  vector[K_tox] alpha_tox_hyp2;
}

parameters {
  // Marginal probability of cancer stage
  simplex[K_stg] q_stg;

  // Conditional probability of relationship status given cancer stage
  array[K_stg] simplex[K_rel] q_rel;

  // Conditional probability of treatment status given active cancer stage
  array[K_stg - 1] simplex[K_trt] q_trt_active_stg;

  // Conditional probability of toxicity status given cancer stage and
  // active treatment status
  array[K_stg] simplex[K_tox] q_tox_active_trt;

  // Probability of natural conception for baseline patients in a stable
  // relationship, no cancer, and no toxicity
  real<lower=0, upper=1> q_NC_0;

  // Proportional decreases in conception probability due to
  // non-baseline relationship status, cancer stage, and toxicity
  // status.
  positive_ordered[K_rel - 1] alpha_rel;
  positive_ordered[K_stg - 1] alpha_stg;
  positive_ordered[K_tox - 1] alpha_tox;

  // Probability of ART conception
  real<lower=0, upper=1> q_AC;
}

transformed parameters {
  vector[K_rel] alpha_rel_buff = append_row([0]', alpha_rel);
  vector[K_stg] alpha_stg_buff = append_row([0]', alpha_stg);
  vector[K_tox] alpha_tox_buff = append_row([0]', alpha_tox);

  // Conditional probability of treatment status given cancer stage
  array[K_stg] simplex[K_trt] q_trt = append_array({ [1.0, 0.0]' },
                                                   q_trt_active_stg);

  // Conditional probability of toxicity status given cancer stage and
  // treatment status
  array[K_stg, K_trt] simplex[K_tox] q_tox;
  q_tox[, 1] = { [1, 0, 0, 0]',
                 [1, 0, 0, 0]',
                 [1, 0, 0, 0]' };
  q_tox[, 2] = q_tox_active_trt;
}

model {
  // Prior model
  target += dirichlet_lpdf(q_stg | [4, 3, 1]');
  target += dirichlet_lpdf(q_rel[1] | [4, 1]');
  target += dirichlet_lpdf(q_rel[2] | [2, 1]');
  target += dirichlet_lpdf(q_rel[3] | [1, 1]');
  target += dirichlet_lpdf(q_trt_active_stg[1] | [1, 3]');
  target += dirichlet_lpdf(q_trt_active_stg[2] | [0.5, 4]');
  target += dirichlet_lpdf(q_tox_active_trt[1] | [4, 3, 2, 1]');
  target += dirichlet_lpdf(q_tox_active_trt[2] | [2, 3, 3, 1]');
  target += dirichlet_lpdf(q_tox_active_trt[3] | [1, 2, 3, 3]');

  target += beta_lpdf(q_NC_0 | 12.7, 3.7); // 0.50 <~ q_NC_0 <~ 0.95

  target += normal_lpdf(alpha_rel | 0, 3 / 2.32); // 0 <~ alpha <~ -log(0.05)
  target += normal_lpdf(alpha_stg | 0, 3 / 2.32); // 0 <~ alpha <~ -log(0.05)
  target += normal_lpdf(alpha_tox | 0, 3 / 2.32); // 0 <~ alpha <~ -log(0.05)

  target += beta_lpdf(q_AC | 18.2, 11.6); // 0.4 <~ q_AC_0 <~ 0.8

  // Observational model
  for (n in 1:N) {
    real q_NC = q_NC_0 * exp(-alpha_rel_buff[k_rel[n]]
                             -alpha_stg_buff[k_stg[n]]
                             -alpha_tox_buff[k_tox[n]]);

    target += categorical_lpmf(k_stg[n] | q_stg);
    target += categorical_lpmf(k_rel[n] | q_rel[k_stg[n]]);
    target += categorical_lpmf(k_trt[n] | q_trt[k_stg[n]]);
    target += categorical_lpmf(k_tox[n] | q_tox[k_stg[n],
                                                k_trt[n]]);

    if (k_art[n]) {
      target += bernoulli_lpmf(y[n] | q_AC * (1 - q_NC) + q_NC);
    } else {
      target += bernoulli_lpmf(y[n] | q_NC);
    }
  }
}

generated quantities {
  // Posterior predictive data
  array[N_pred] real<lower=0, upper=1> q_pred;
  array[N_pred] real<lower=0, upper=1> q_hyp_pred;

  for (n in 1:N_pred) {
    int k_stg_pred = categorical_rng(q_stg);
    int k_rel_pred = categorical_rng(q_rel[k_stg_pred]);
    int k_trt_pred = categorical_rng(q_trt[k_stg_pred]);
    int k_tox_pred = categorical_rng(q_tox[k_stg_pred,
                                           k_trt_pred]);

    q_pred[n] = q_NC_0 * exp(-alpha_rel_buff[k_rel_pred]
                             -alpha_stg_buff[k_stg_pred]
                             -alpha_tox_buff[k_tox_pred]);
  }

  {
    array[K_stg, K_trt] vector[K_tox] q_tox_hyp;
    q_tox_hyp[, 1] = q_tox[, 1];
    q_tox_hyp[, 2] = { [1, 0, 0, 0]',
                       dirichlet_rng(alpha_tox_hyp1),
                       dirichlet_rng(alpha_tox_hyp2) };

    for (n in 1:N_pred) {
      int k_stg_pred = categorical_rng(q_stg);
      int k_rel_pred = categorical_rng(q_rel[k_stg_pred]);
      int k_trt_pred = categorical_rng(q_trt[k_stg_pred]);
      int k_tox_pred = categorical_rng(q_tox_hyp[k_stg_pred,
                                                 k_trt_pred]);

      q_hyp_pred[n] = q_NC_0 * exp(-alpha_rel_buff[k_rel_pred]
                                   -alpha_stg_buff[k_stg_pred]
                                   -alpha_tox_buff[k_tox_pred]);
    }
  }
}
