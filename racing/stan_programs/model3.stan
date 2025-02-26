functions {
  // Mean-dispersion parameterization of inverse gamma family
  real inv_gamma_md_lpdf(real x, real mu, real psi) {
    return inv_gamma_lpdf(x | inv(psi) + 2, mu * (inv(psi) + 1));
  }

  real inv_gamma_md_rng(real mu, real psi) {
    return inv_gamma_rng(inv(psi) + 2, mu * (inv(psi) + 1));
  }
}

data {
  int<lower=1> N_races;    // Total number of races
  int<lower=1> N_entrants; // Total number of entrants
  // Each entrant is assigned a unique index in [1, N_entrants]

  // Number of entrants in each race who finished
  array[N_races] int<lower=1, upper=N_entrants> race_N_entrants_f;

  // Indices for extracting finished entrant information in each race
  array[N_races] int race_f_start_idxs;
  array[N_races] int race_f_end_idxs;

  // Number of entrants in each race who forfeit and did not finish
  array[N_races] int<lower=0, upper=N_entrants> race_N_entrants_dnf;

  // Indices for extracting forfeited entrant information in each race
  array[N_races] int race_dnf_start_idxs;
  array[N_races] int race_dnf_end_idxs;

  // Total number of finishes across all races
  int <lower=1> N_entrances_fs;

  // Finished entrant indices within each race
  array[N_entrances_fs] int race_entrant_f_idxs;

  // Entrant finish times within each race
  array[N_entrances_fs] real race_entrant_f_times;

  // Total number of forfeits across all races
  int<lower=0> N_entrances_dnfs;

  // Forfeited entrant indices within each race
  array[N_entrances_dnfs] int race_entrant_dnf_idxs;

  // Anchor configuration
  int<lower=1, upper=N_races> anchor_race_idx;
  int<lower=1, upper=N_entrants> anchor_entrant_idx;
}

parameters {
  real eta; // Log baseline finish time (log seconds)

  // Relative seed difficulties
  array[N_races] real rel_difficulties_free;

  // Relative entrant skills
  array[N_entrants] real rel_skills_free;

  real<lower=0> psi; // Inverse gamma dispersion configuration

  array[N_entrants] real kappas;         // Forfeit thresholds
  array[N_entrants] real<lower=0> betas; // Forfeit scalings
}

transformed parameters {
  array[N_races] real rel_difficulties;
  array[N_entrants] real rel_skills;

  rel_difficulties[1:(anchor_race_idx - 1)]
    =  rel_difficulties_free[1:(anchor_race_idx - 1)];
  rel_difficulties[anchor_race_idx] = 0;
  rel_difficulties[(anchor_race_idx + 1):N_races]
    = rel_difficulties_free[anchor_race_idx:(N_races - 1)];

  rel_skills[1:(anchor_entrant_idx - 1)]
    =  rel_skills_free[1:(anchor_entrant_idx - 1)];
  rel_skills[anchor_entrant_idx] = 0;
  rel_skills[(anchor_entrant_idx + 1):N_entrants]
    = rel_skills_free[anchor_entrant_idx:(N_entrants - 1)];
}

model {
  // Prior model

  // log(1800 s) < eta < log(5400 s)
  eta ~ normal(8.045, 0.237);

  // -sqrt(2) * log(2) <~ difficulties <~ +sqrt(2) * log(2)
  rel_difficulties_free ~ normal(0, 0.423);

  // -sqrt(2) * log(2) <~ rel_skills <~ +sqrt(2) * log(2)
  rel_skills_free ~ normal(0, 0.423);

  psi ~ normal(0, 0.389);   //  0 <~ psi    <~ 1
  kappas ~ normal(0, 2.16); // -5 <~ kappas <~ +5
  betas ~ normal(0, 1.95);  //  0 <~ betas  <~ +5

  // Observational model
  for (r in 1:N_races) {
    // Extract details for entrants who finished
    int N_entrants_f = race_N_entrants_f[r];
    array[N_entrants_f] int f_idxs
      = linspaced_int_array(N_entrants_f,
                            race_f_start_idxs[r],
                            race_f_end_idxs[r]);
    array[N_entrants_f] int entrant_f_idxs
      = race_entrant_f_idxs[f_idxs];
    array[N_entrants_f] real entrant_f_times
      = race_entrant_f_times[f_idxs];

    // Finished entrant model
    for (n in 1:N_entrants_f) {
      int entrant_idx = entrant_f_idxs[n];
      real delta = rel_difficulties[r] - rel_skills[entrant_idx];
      real mu = exp(eta + delta);
      real logit_q = betas[entrant_idx] * (delta - kappas[entrant_idx]);

      entrant_f_times[n] ~ inv_gamma_md(mu, psi);
      0 ~ bernoulli_logit(logit_q); // Did not forfeit
    }

    if (race_N_entrants_dnf[r] > 0) {
      // Extract details for entrants who forfeited
      int N_entrants_dnf = race_N_entrants_dnf[r];
      array[N_entrants_dnf]
        int dnf_idxs = linspaced_int_array(N_entrants_dnf,
                                           race_dnf_start_idxs[r],
                                           race_dnf_end_idxs[r]);
      array[N_entrants_dnf]
        int entrant_dnf_idxs = race_entrant_dnf_idxs[dnf_idxs];

      // Forfeited entrant model
      for (n in 1:N_entrants_dnf) {
        int entrant_idx = entrant_dnf_idxs[n];
        real delta = rel_difficulties[r] - rel_skills[entrant_idx];
        real logit_q = betas[entrant_idx] * (delta - kappas[entrant_idx]);

        1 ~ bernoulli_logit(logit_q); // Did forfeit
      }
    }
  }
}

generated quantities {
  // Posterior predictions
  array[N_entrances_fs] real race_entrant_f_times_pred;
  array[N_races] int race_N_entrants_dnf_pred = rep_array(0, N_races);

  for (r in 1:N_races) {
    // Extract details for entrants who forfeited
    int N_entrants_f = race_N_entrants_f[r];
    array[N_entrants_f] int f_idxs
      = linspaced_int_array(N_entrants_f,
                            race_f_start_idxs[r],
                            race_f_end_idxs[r]);
    array[N_entrants_f] int entrant_f_idxs
      = race_entrant_f_idxs[f_idxs];

    for (n in 1:N_entrants_f) {
      int entrant_idx = entrant_f_idxs[n];
      real delta = rel_difficulties[r] - rel_skills[entrant_idx];
      real mu = exp(eta + delta);
      real logit_q = betas[entrant_idx] * (delta - kappas[entrant_idx]);

      // Finish time prediction conditioned on not forfeiting
      race_entrant_f_times_pred[race_f_start_idxs[r] + n - 1]
        = inv_gamma_md_rng(mu, psi);

      // Forfeit prediction
      race_N_entrants_dnf_pred[r] += bernoulli_logit_rng(logit_q);
    }

    if (race_N_entrants_dnf[r] > 0) {
      // Extract details for entrants who forfeited
      int N_entrants_dnf = race_N_entrants_dnf[r];
      array[N_entrants_dnf]
        int dnf_idxs = linspaced_int_array(N_entrants_dnf,
                                           race_dnf_start_idxs[r],
                                           race_dnf_end_idxs[r]);
      array[N_entrants_dnf]
        int entrant_dnf_idxs = race_entrant_dnf_idxs[dnf_idxs];

      for (n in 1:N_entrants_dnf) {
        int entrant_idx = entrant_dnf_idxs[n];
        real delta = rel_difficulties[r] - rel_skills[entrant_idx];
        real logit_q = betas[entrant_idx] * (delta - kappas[entrant_idx]);

        // Forfeit prediction
        race_N_entrants_dnf_pred[r] += bernoulli_logit_rng(logit_q);
      }
    }
  }
}
