// Model for joint efficacy and toxity modeling using Stan
// Ref implementation: https://mc-stan.org/docs/stan-users-guide/multivariate-outcomes.html#ref-AlbertChib:1993
// Ref paper: https://apps.olin.wustl.edu/faculty/chib/papers/albertchib93.pdf

data {
    int<lower=0> N; // number of observations
    vector[N] Yc; // Continuous data
    array[N] int Yb; // Binary data
}

transformed data {
    int<lower=0> N_pos; // Number of positive (Yb == 1)
    N_pos = sum(Yb);
    int<lower=0> N_neg; // Number of negative (Yb == 0)
    N_neg = N - N_pos;
    array[N_pos] int<lower=1, upper=N> n_pos; // Vectors specifying indices
    array[N_neg] int<lower=1, upper=N> n_neg;

    // this is similar to a while loop
    {
        int i;
        int j;
        i = 1;
        j = 1;
        for (n in 1:N){
            if (Yb[n] == 1){
                n_pos[i] = n;
                i += 1;
            } else {
                n_neg[j] = n;
                j += 1;
            }
        }
    }
}

parameters {
    row_vector[2] mu; // A row-vector of multivariate means
    vector<lower=0>[N_pos] z_pos; // Latent variables that are positive
    vector<upper=0>[N_neg] z_neg; // Latent variables that are negative
    cholesky_factor_corr[2] L_sigma;
    //real<lower=0> var_w;
    //real<lower=0> rho;
}

transformed parameters{
    array[N] vector[2] z;

    // for the first component of the latent variable, just the
    // continous outcome
    for (i in 1:N){
        z[i,1] = Yc[i];
    }

    // for the second component, assign to the constrained neg
    // and pos vectors as specified
    for (i in 1:N_pos){
        z[n_pos[i],2] = z_pos[i];
    }

    for (i in 1:N_neg){
        z[n_neg[i],2] = z_neg[i];
    }

    // Set up covariance matrix.
    //real<lower=0>sigma_w = sqrt(var_w);
    //cov_matrix[2] Sigma;
    //Sigma[1,1] = sigma_w;
    //Sigma[1,2] = rho * sigma_w;
    //Sigma[2,1] = rho * sigma_w;
    //Sigma[2,2] = 1;
}


model {
    // priors
    L_sigma ~ lkj_corr_cholesky(4);
    //var_w ~ inv_gamma(1e-4, 1e-4);
    to_vector(mu) ~ normal(0,100);

    // model
    // multivariate normal
    //z ~ multi_normal(mu, Sigma);
    z ~ multi_normal_cholesky(mu, L_sigma);
}

generated quantities {
    corr_matrix[2] Sigma;
    Sigma = multiply_lower_tri_self_transpose(L_sigma);
    real<lower=0> pt;
    pt = normal_cdf(mu[2], 0, 1);
}
