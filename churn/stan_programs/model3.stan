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

  // Churned customers
  int<lower=0> N_churn;
  array[N_churn] real<lower=T0, upper=T1> t_join_churn;  // Months
  array[N_churn] real<lower=T0, upper=T1> t_leave_churn; // Months

  // Active customers
  int<lower=0> N_active;
  array[N_active] real<lower=T0, upper=T1> t_join_active; // Months
}

parameters {
  real<lower=0> omega; // Inverse shape parameter (Unitless)
  real<lower=0> sigma; // Scale parameter (Months)
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

  // Observational Model
  for (n in 1:N_churn) {
    real tenure = t_leave_churn[n] - t_join_churn[n];
    target +=   log_lambda(tenure, alpha, sigma)
              + log_survival(tenure, alpha, sigma);
  }

  for (n in 1:N_active) {
    target += log_survival(T1 - t_join_active[n], alpha, sigma);
  }
}

generated quantities {
  array[N_churn] real<lower=0> tenure_churn_pred;
  array[N_churn] real<lower=T0> t_leave_churn_pred;

  for (n in 1:N_churn) {
    real S = exp(log_survival(T1 - t_join_churn[n],
                              alpha, sigma));
    real u = uniform_rng(S, 1);
    tenure_churn_pred[n]
      = inv_survival(u, alpha, sigma);
    t_leave_churn_pred[n]
      = t_join_churn[n] + tenure_churn_pred[n];
  }
}
