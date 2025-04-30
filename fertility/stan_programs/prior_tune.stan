functions {
  // Differences between inverse gamma tail
  // probabilities and target probabilities
  vector tail_delta(vector y, vector theta,
                    array[] real x_r, array[] int x_i) {
    vector[2] deltas;
    deltas[1] = inv_gamma_cdf(theta[1] | exp(y[1]), exp(y[2])) - 0.01;
    deltas[2] = 1 - inv_gamma_cdf(theta[2] | exp(y[1]), exp(y[2])) - 0.01;
    return deltas;
  }
}

data {
  real<lower=0>     y_low;
  real<lower=y_low> y_high;
}

transformed data {
  // Initial guess at inverse gamma parameters
  vector[2] y_guess = [log(2), log(5)]';
  // Target quantile
  vector[2] theta = [y_low, y_high]';
  vector[2] y;
  array[0] real x_r;
  array[0] int x_i;

  // Find inverse Gamma density parameters that ensure 
  // 1% probability below y_low and 1% probability above y_high
  y = algebra_solver(tail_delta, y_guess, theta, x_r, x_i);

  print("alpha = ", exp(y[1]));
  print("beta = ", exp(y[2]));
}

generated quantities {
  real alpha = exp(y[1]);
  real beta = exp(y[2]);
}
