functions {
  // Log stimulus function
  real log_lambda(real t, real alpha, real sigma) {
    return log(alpha) - log(sigma) + (alpha - 1) * log(t / sigma);
  }
  
  // Cumulative stimulus function
  real Lambda(real t, real alpha, real sigma) {
    return pow(t / sigma, alpha);
  }

  // Log survival function
  real log_survival(real t, real alpha, real sigma) {
    return -Lambda(t, alpha, sigma);
  }

  // Inverse cumulative stimulus function
  real Lambda_inv(real l, real alpha, real sigma) {
    return sigma * pow(l, 1 / alpha);
  }
  
  // Inverse survival function
  real inv_survival(real u, real alpha, real sigma) {
    return Lambda_inv(-log(u), alpha, sigma);
  }
}

data {
  int<lower=0> N;
  real<lower=0> t_lower;
  real<lower=0> t_upper;
}

parameters {
  real<lower=0> omega; // Inverse shape parameter (Unitless)
  real<lower=0> sigma; // Scale parameter (Months)
}

model {
  // Prior Model
  // 1 <~ alpha
  // 0 <~ omega = 1 / alpha <~ 1 / 1
  target += normal_lpdf(omega | 0, 1 / 2.57);

  // 0 <~ sigma <~ 75
  target += normal_lpdf(sigma | 0, 75 / 2.57);
}

generated quantities {
  array[N] real<lower=0> tenure_pred;
  real lower_tail_p_hat = 0;
  real upper_tail_p_hat = 0;

  for (n in 1:N) {
    tenure_pred[n]
      = inv_survival(uniform_rng(0, 1), 1 / omega, sigma);
    if (tenure_pred[n] < t_lower)
      lower_tail_p_hat += 1;
    if (tenure_pred[n] > t_upper)
      upper_tail_p_hat += 1;
  }
  lower_tail_p_hat /= N;
  upper_tail_p_hat /= N;
}
