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

  // Prospective customers
  int<lower=0> N_pred;

  matrix[N_pred, I] Z_pred;
  array[N_pred] int online_banking_pred;

  int<lower=0> S; // Number of segments
  array[N_pred] int<lower=1, upper=S> segment_pred;

  array[S] real<lower=0> value_per_month; // Euros / Month
  real<lower=0, upper=1> zero_rate; // Percentage / Year

  array[S] real<lower=0> acq_cost;  // Euros
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
  target += normal_lpdf(omega | 0, 1 / 2.57);
  target += normal_lpdf(sigma | 0, 75 / 2.57);
  target += normal_lpdf(beta[1] | 0, 0.07);
  target += normal_lpdf(beta[2] | 0, 0.70);
  target += normal_lpdf(beta[3] | 0, 0.35);
  target += normal_lpdf(beta[4] | 0, 1);
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
  array[N_pred] real<lower=0> clv_discount;
  array[N_pred] real profit;

  for (n in 1:N_pred) {
    real mu = (Z_pred[n,] - z0) * beta;
    if (online_banking_pred[n])
      mu += delta_online;
    real tau = sigma * exp(-mu / alpha);

    real u = uniform_rng(0, 1);
    real tenure = inv_survival(u, alpha, tau);

    int s = segment_pred[n];

    clv_discount[n] = value_per_month[s] * tenure /
                      pow(1 + zero_rate / 360, 12 * tenure);
    profit[n] = clv_discount[n] - acq_cost[s];
  }
}
