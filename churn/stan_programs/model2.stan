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
  real<lower=0> eta; // Inverse scale parameter (Months)
}

transformed parameters {
  real<lower=0> gamma = 1 / eta;
}

model {
  // Prior Model
  // 0 <~ eta <~ 40
  target += normal_lpdf(eta | 0, 40 / 2.57);

  // Observational Model
  for (n in 1:N_churn) {
    real tenure = t_leave_churn[n] - t_join_churn[n];
    target +=   log_lambda(tenure, gamma)
              + log_survival(tenure, gamma);
  }

  for (n in 1:N_active) {
    target += log_survival(T1 - t_join_active[n], gamma);
  }
}

generated quantities {
  array[N_churn] real<lower=0> tenure_churn_pred;
  array[N_churn] real<lower=T0> t_leave_churn_pred;

  for (n in 1:N_churn) {
    real S = exp(log_survival(T1 - t_join_churn[n], gamma));
    real u = uniform_rng(S, 1);
    tenure_churn_pred[n]
      = inv_survival(u, gamma);
    t_leave_churn_pred[n]
      = t_join_churn[n] + tenure_churn_pred[n];
  }
}
