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
  real<lower=0> t_lower;
  real<lower=0> t_upper;

  int I;            // Number of covariates
  row_vector[I] z0; // Covariate baselines

  // Churned customers
  int<lower=0> N_churn;
  matrix[N_churn, I] Z_churn;
  array[N_churn] int online_banking_churn;
}

parameters {
  real<lower=0> omega; // Inverse shape parameter (Unitless)
  real<lower=0> sigma; // Scale parameter (Months)

  vector[I] beta;    // Covariate slopes (1 / covariate units)
  real delta_online; // Online banking contribution
}

transformed parameters {
  real<lower=0> alpha = 1 / omega;
}

model {
  // Prior Model
  // 0 <~ omega <~ 1
  target += normal_lpdf(omega | 0, 1 / 2.57);

  // 0 <~ sigma <~ 75
  target += normal_lpdf(sigma | 0, 75 / 2.57);

  // -log(5) / 10 <~ beta[1] <~ +log(5) / 10
  target += normal_lpdf(beta[1] | 0, 0.07);

  // -log(5) / 1 <~ beta[2] <~ +log(5) / 1
  target += normal_lpdf(beta[2] | 0, 0.70);

  // -log(5) / 2 <~ beta[3] <~ +log(5) / 2
  target += normal_lpdf(beta[3] | 0, 0.35);

  // -log(5) / log(2) <~ beta[4] <~ +log(5) / log(2)
  target += normal_lpdf(beta[4] | 0, 1);

  // 0.1 <~ exp(delta_online) <~ 10
  target += normal_lpdf(delta_online | 0, 1);
}


generated quantities {
  array[N_churn] real<lower=0> tenure_pred;
  real lower_tail_p_hat = 0;
  real upper_tail_p_hat = 0;

  for (n in 1:N_churn) {
    real mu = (Z_churn[n,] - z0) * beta;
    if (online_banking_churn[n])
      mu += delta_online;
    real tau = sigma * exp(-mu / alpha);

    tenure_pred[n]
      = inv_survival(uniform_rng(0, 1), alpha, tau);

    if (tenure_pred[n] < t_lower)
      lower_tail_p_hat += 1;
    if (tenure_pred[n] > t_upper)
      upper_tail_p_hat += 1;
  }
  lower_tail_p_hat /= N_churn;
  upper_tail_p_hat /= N_churn;
}
