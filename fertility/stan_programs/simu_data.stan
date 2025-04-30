transformed data {
  // Number of observations
  int<lower=1> N = 5000;

  // Relationship status
  // k = 1: Stable partner
  // k = 2: No partner
  int<lower=1> K_rel = 2;

  // Cancer stage
  // k = 1: No cancer
  // k = 2: Early stage cancer
  // k = 3: Advanced stage cancer
  int<lower=1> K_stg = 3;

  // Treatment status
  // k = 1: No treatment
  // k = 2: Treatment
  int<lower=1> K_trt = 2;

  // Toxicity status
  // k = 1: None
  // k = 2: Low
  // k = 3: Medium
  // k = 4: High
  int<lower=1> K_tox = 4;

  // Marginal probability of cancer stage
  simplex[K_stg] q_stg = [0.5, 0.35, 0.15]';

  // Conditional probability of relationship status given cancer stage
  array[K_stg] simplex[K_rel] q_rel = { [0.9, 0.1]',
                                        [0.8, 0.2]',
                                        [0.5, 0.5]' };

  // Conditional probability of treatment status given cancer stage
  array[K_stg] simplex[K_trt] q_trt = { [1.0, 0.0]',
                                        [0.3, 0.7]',
                                        [0.1, 0.9]' };

  // Conditional probability of toxicity status given cancer stage and
  // treatment status
  array[K_stg, K_trt] simplex[K_tox] q_tox;
  q_tox[, 1] = { [1, 0, 0, 0]',
                 [1, 0, 0, 0]',
                 [1, 0, 0, 0]' };
  q_tox[, 2] = { [1, 0.00, 0.00, 0.00]',
                 [0, 0.50, 0.45, 0.05]',
                 [0, 0.10, 0.30, 0.60]' };

  // Probability of natural conception for patients in a stable
  // relationship, no cancer, and no toxicity
  real<lower=0, upper=1> q_NC_0 = 0.8;

  // Proportional decreases in natural conception probability due to
  // relationship status, cancer stage, and toxicity status.
  positive_ordered[K_rel] alpha_rel = [0, -log(0.1)]';
  positive_ordered[K_stg] alpha_stg = [0, -log(0.9), -log(0.4)]';
  positive_ordered[K_tox] alpha_tox = [0, -log(0.8), -log(0.4), -log(0.1)]';

  // Probability of ART conception
  real<lower=0, upper=1> q_AC = 0.6;

  // Conditional probability of ART given treatment status
  array[K_trt] real<lower=0, upper=1> q_art = { 0.1, 0.6 };
}

generated quantities {
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

  // Observed ART status;
  array[N] int<lower=0, upper=1> k_art;

  for (n in 1:N) {
    k_stg[n] = categorical_rng(q_stg);
    k_rel[n] = categorical_rng(q_rel[k_stg[n]]);
    k_trt[n] = categorical_rng(q_trt[k_stg[n]]);
    k_tox[n] = categorical_rng(q_tox[k_stg[n], k_trt[n]]);

    k_art[n] = bernoulli_rng(q_art[k_trt[n]]);

    if (k_art[n] == 1) {
      real q_NC = q_NC_0 * exp(-alpha_rel[k_rel[n]]
                               -alpha_stg[k_stg[n]]
                               -alpha_tox[k_tox[n]]);
      y[n] = bernoulli_rng(q_AC * (1 - q_NC) + q_NC);
    } else {
      real q_NC = q_NC_0 * exp(-alpha_rel[k_rel[n]]
                               -alpha_stg[k_stg[n]]
                               -alpha_tox[k_tox[n]]);
      y[n] = bernoulli_rng(q_NC);
    }
  }
}
