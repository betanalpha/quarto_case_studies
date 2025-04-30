functions {
  // Differences between beta tail probabilities
  // and target probabilities
  vector tail_delta(vector y, vector theta,
                    array[] real x_r, array[] int x_i) {
    vector[2] deltas;
    deltas[1] =     beta_cdf(theta[1] | exp(y[1]), exp(y[2])) - 0.01;
    deltas[2] = 1 - beta_cdf(theta[2] | exp(y[1]), exp(y[2])) - 0.01;
    return deltas;
  }
}

data {
  real<lower=0, upper=1>     q_low;  // Lower threshold
  real<lower=q_low, upper=1> q_high; // Upper threshold
}

transformed data {
  vector[2] y_guess = [log(5), log(5)]'; // Initial guess at beta parameters
  vector[2] theta = [q_low, q_high]';    // Target quantiles
  vector[2] y;
  array[0] real x_r;
  array[0] int x_i;

  // Find beta parameters that ensure
  // 1% probability below lower threshold
  // and 1% probability above upper threshold
  y = algebra_solver(tail_delta, y_guess, theta, x_r, x_i);

  print("alpha = ", exp(y[1]));
  print("beta = ", exp(y[2]));
}

generated quantities {
  real alpha = exp(y[1]);
  real beta = exp(y[2]);
}
