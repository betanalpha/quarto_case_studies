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
  real           T0; // Months
  real<lower=T0> T1; // Months

  int I;            // Number of covariates
  row_vector[I] z0; // Covariate baselines

  // Churned customers
  int<lower=0> N_churn;
  array[N_churn] real<lower=T0, upper=T1> t_join_churn;  // Months
  array[N_churn] real<lower=T0, upper=T1> t_leave_churn; // Months

  matrix[N_churn, I] Z_churn;
  array[N_churn] int online_banking_churn;

  // Active customers
  int<lower=0> N_active;
  array[N_active] real<lower=T0, upper=T1> t_join_active; // Months

  matrix[N_active, I] Z_active;
  array[N_active] int online_banking_active;
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

  // Observational Model
  for (n in 1:N_churn) {
    real tenure = t_leave_churn[n] - t_join_churn[n];

    real mu = (Z_churn[n,] - z0) * beta;
    if (online_banking_churn[n])
      mu += delta_online;
    real tau = sigma * exp(-mu / alpha);

    target +=   log_lambda(tenure, alpha, tau)
              + log_survival(tenure, alpha, tau);
  }

  for (n in 1:N_active) {
    real mu = (Z_active[n,] - z0) * beta;
    if (online_banking_active[n])
      mu += delta_online;
    real tau = sigma * exp(-mu / alpha);

    target += log_survival(T1 - t_join_active[n], alpha, tau);
  }
}

generated quantities {
  array[N_churn] real<lower=0> tenure_churn_pred;
  array[N_churn] real<lower=T0> t_leave_churn_pred;

  for (n in 1:N_churn) {
    real mu = (Z_churn[n,] - z0) * beta;
    if (online_banking_churn[n])
      mu += delta_online;
    real tau = sigma * exp(-mu / alpha);

    real S = exp(log_survival(T1 - t_join_churn[n],
                              alpha, tau));
    real u = uniform_rng(S, 1);
    tenure_churn_pred[n]
      = inv_survival(u, alpha, tau);
    t_leave_churn_pred[n]
      = t_join_churn[n] + tenure_churn_pred[n];
  }
}
