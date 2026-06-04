functions {
  // Log stimulus function
  real log_lambda(real t, real gamma) {
    return log(gamma);
  }
  
  // Cumulative stimulus function
  real Lambda(real t, real gamma) {
    return gamma * t;
  }

  // Log survival function
  real log_survival(real t, real gamma) {
    return -Lambda(t, gamma);
  }

  // Inverse cumulative stimulus function
  real Lambda_inv(real l, real gamma) {
    return l / gamma;
  }
  
  // Inverse survival function
  real inv_survival(real u, real gamma) {
    return Lambda_inv(-log(u), gamma);
  }
}

data {
  int<lower=0> N;
  real<lower=0> t_upper;
}

parameters {
  real<lower=0> eta; // Inverse scale parameter (Months)
}

model {
  // Prior Model
  // 0 <~ eta <~ 50
  target += normal_lpdf(eta | 0, 1 / 2.57);
}

generated quantities {
  array[N] real<lower=0> tenure_pred;
  real upper_tail_p_hat = 0;

  for (n in 1:N) {
    tenure_pred[n]
      = inv_survival(uniform_rng(0, 1), 1 / eta);
    if (tenure_pred[n] > t_upper)
      upper_tail_p_hat += 1;
  }
  upper_tail_p_hat /= N;
}
