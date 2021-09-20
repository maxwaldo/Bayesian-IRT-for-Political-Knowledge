
data {
  int<lower=1> N; // Number of observation
  int<lower=1> I; // Number of item question 
  int<lower=1> J; // Number of respondents
  int<lower=1> D; // Number of dimension of political knowledge
  int<lower=0, upper=1> y[N]; // outcome of N observation--> here binary but can also be ordinal.
  int<lower=1> i[N]; // Item i for observation N
  int<lower=1> j[N]; // Actor j for observation N
  int<lower=1> d[N]; // dimension of observation N
  
  
}

parameters {
  vector[I] alpha; // Difficulty parameter 
  vector[I] beta;  // Discrimination parameter
  vector[J] theta[D];  // ideal position of actor j on dimension d
}

// This block exists for computational reasons
// It performs a non-centered parameterization
transformed parameters {
  vector[N] p_binary;
  
  for (n in 1:N) {
    p_binary[n] = inv_logit(theta[d[n], j[n]] * beta[i[n]] + alpha[i[n]]); 
  }
  
}


model {
  beta ~ normal(0, 5); // Priors for the discrimination of items
  alpha ~ normal(0, 5); // Priors for the difficulty of items
  for (n in 1:D) {
     theta[n] ~ std_normal(); // Priors for the latent ability of actors on dimension d
  }
  y ~ bernoulli(p_binary);
  
}


