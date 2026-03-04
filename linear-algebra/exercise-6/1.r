# Parameters
p_true <- 0.5        # The true parameter value for simulation (under H0)
alpha <- 0.1         # Significance level
n_values <- c(10, 100, 1000) # Sample sizes to simulate
M <- 10000           # Number of Monte Carlo simulations

# Critical values for the two-sided test at level alpha
# We reject H0 if Tn < z_alpha/2 or Tn > z_1-alpha/2
z_lower <- qnorm(alpha / 2)
z_upper <- qnorm(1 - alpha / 2)

# Set up the plotting area (1 row, 3 columns)
par(mfrow = c(1, 3))

# Loop over sample sizes
for (n in n_values) {
  # 1. Initialize a vector to store the test statistics Tn
  Tn_values <- numeric(M)
  
  # 2. Run the simulation loop M times
  for (i in 1:M) {
    # Generate random data from Geometric(p_true)
    # Note: rgeom in R uses parameter prob. 
    # R definition: number of failures before first success (0, 1, 2...)
    # This matches the definition in the exercise (support N_0).
    X <- rgeom(n, prob = p_true)
    
    # Compute MLE p_hat
    X_bar <- mean(X)
    p_hat <- 1 / (1 + X_bar)
    
    # Compute the estimator for the standard deviation s_n
    # s_n^2 = p_hat^2 * (1 - p_hat)
    s_n <- sqrt(p_hat^2 * (1 - p_hat))
    
    # Compute the test statistic Tn
    # Tn = sqrt(n) * (p_hat - p0) / s_n
    # Here p0 is p_true because we are simulating under H0
    if (s_n > 0) {
      Tn_values[i] <- sqrt(n) * (p_hat - p_true) / s_n
    } else {
      # Handle edge case where s_n might be 0 (e.g., if p_hat=1)
      Tn_values[i] <- NA 
    }
  }
  
  # Remove any NA values
  Tn_values <- Tn_values[!is.na(Tn_values)]
  
  # 3. Create QQ-Plot
  qqnorm(Tn_values, main = paste("QQ-Plot for n =", n))
  qqline(Tn_values, col = "red", lwd = 2)
  
  # 4. Calculate Type I Error rate
  # Proportion of times Tn falls outside the critical region
  reject_count <- sum(Tn_values < z_lower | Tn_values > z_upper)
  type1_error <- reject_count / length(Tn_values)
  
  # Print results to console
  cat("Sample size n =", n, "\n")
  cat("Type I Error rate:", type1_error, "\n")
  cat("Expected Error rate:", alpha, "\n")
  cat("----------------------------------\n")
}

# Reset plotting parameters
par(mfrow = c(1, 1))
