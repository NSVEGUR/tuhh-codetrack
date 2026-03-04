library(nleqslv) # Standard library

# Parameters
c0 <- 0.0015
a1_0 <- 2
a2_0 <- 1
sigma0 <- 2.5
alpha_level <- 0.1
sample_sizes <- c(10, 31, 100, 1000)
M <- 1000  # Number of simulations (Problem asks for "many", 1000 is standard)

# Critical values for t-test
# Note: We calculate p-values inside the loop, so we check if p_val < alpha_level

# Store results
results_a1 <- list()
results_a2 <- list()

par(mfrow = c(2, 4)) # Setup plot area for QQ plots

for (n in sample_sizes) {
  
  t_stats_a1 <- numeric(M)
  t_stats_a2 <- numeric(M)
  reject_a1 <- 0
  reject_a2 <- 0
  valid_sims <- 0
  
  for (i in 1:M) {
    # 1. Generate Predictors
    U <- runif(n)
    N_err <- rnorm(n)
    Girth <- 8 + 10 * U
    Height <- 62 + Girth + 5.5 * N_err
    
    # 2. Generate Response
    # Volume = c * G^a1 * H^a2 + epsilon
    epsilon <- rnorm(n, mean = 0, sd = sigma0)
    Volume <- c0 * Girth^a1_0 * Height^a2_0 + epsilon
    
    # Create data frame
    sim_data <- data.frame(Volume = Volume, Girth = Girth, Height = Height)
    
    # 3. Fit Model
    # Use tryCatch to handle cases where nls fails to converge (common with small n)
    fit <- tryCatch({
      nls(Volume ~ c * Girth^a1 * Height^a2, 
          data = sim_data, 
          start = list(c = 0.0015, a1 = 2, a2 = 1), # Start at truth to aid convergence
          control = nls.control(maxiter = 100, warnOnly = TRUE))
    }, error = function(e) return(NULL))
    
    if (!is.null(fit)) {
      sum_fit <- summary(fit)
      coeffs <- coef(sum_fit)
      
      # Check if we have all coefficients estimated
      if (all(c("a1", "a2") %in% rownames(coeffs))) {
        valid_sims <- valid_sims + 1
        
        # Test H0: a1 = 2
        est_a1 <- coeffs["a1", "Estimate"]
        se_a1 <- coeffs["a1", "Std. Error"]
        t_a1 <- (est_a1 - 2) / se_a1
        t_stats_a1[valid_sims] <- t_a1
        if (2 * pt(-abs(t_a1), df = df.residual(fit)) < alpha_level) {
          reject_a1 <- reject_a1 + 1
        }
        
        # Test H0: a2 = 1
        est_a2 <- coeffs["a2", "Estimate"]
        se_a2 <- coeffs["a2", "Std. Error"]
        t_a2 <- (est_a2 - 1) / se_a2
        t_stats_a2[valid_sims] <- t_a2
        if (2 * pt(-abs(t_a2), df = df.residual(fit)) < alpha_level) {
          reject_a2 <- reject_a2 + 1
        }
      }
    }
  }
  
  # Trim unused slots
  t_stats_a1 <- t_stats_a1[1:valid_sims]
  t_stats_a2 <- t_stats_a2[1:valid_sims]
  
  # 4. QQ-Plots
  qqnorm(t_stats_a1, main = paste("n =", n, " (alpha1)"), cex = 0.5)
  qqline(t_stats_a1, col = "red")
  
  # Print Type I error rates
  cat("Sample size:", n, "\n")
  cat("Type I Error (alpha1=2):", reject_a1 / valid_sims, "\n")
  cat("Type I Error (alpha2=1):", reject_a2 / valid_sims, "\n")
  cat("--------------------------------------\n")
}
par(mfrow = c(1, 1))