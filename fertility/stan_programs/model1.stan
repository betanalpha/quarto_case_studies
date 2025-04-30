data {
  // Number of observations
  int<lower=1> N;

  // Observed conception status
  // y = 0: No conception
  // y = 1: Conception
  array[N] int<lower=0, upper=1> y;
}

parameters {
  real<lower=0, upper=1> q_C; // Conception probability
}

model {
  // Prior model
  target += beta_lpdf(q_C | 2.5, 2.0); // 0.10 <~ q_C <~ 0.95

  // Observational model
  target += bernoulli_lpmf(y | q_C);

  // Also valid but slightly less efficient
  // for (n in 1:N) {
  //   target += bernoulli_lpmf(y[n] | q_C);
  // }
}

generated quantities {
  // Posterior predictive data
  array[N] int<lower=0, upper=1> y_pred;

  for (n in 1:N) {
    y_pred[n] = bernoulli_rng(q_C);
  }
}
