# Load data
data(trees)

# 1. Fit the non-linear model
# We need reasonable start values. 
# From Exercise Sheet 5, we know log-linearizing gives approx alpha1=2, alpha2=1.
# c implies Volume approx cylinder/cone. 
nls_fit <- nls(Volume ~ c * Girth^a1 * Height^a2, 
               data = trees, 
               start = list(c = 0.001, a1 = 2, a2 = 1))

# 2. Extract estimates and standard errors
sum_fit <- summary(nls_fit)
coeffs <- coef(sum_fit)
print(sum_fit)

# 3. Perform t-tests
# H0: a1 = 2
est_a1 <- coeffs["a1", "Estimate"]
se_a1 <- coeffs["a1", "Std. Error"]
t_stat_a1 <- (est_a1 - 2) / se_a1
p_val_a1 <- 2 * pt(-abs(t_stat_a1), df = df.residual(nls_fit))

# H0: a2 = 1
est_a2 <- coeffs["a2", "Estimate"]
se_a2 <- coeffs["a2", "Std. Error"]
t_stat_a2 <- (est_a2 - 1) / se_a2
p_val_a2 <- 2 * pt(-abs(t_stat_a2), df = df.residual(nls_fit))

# Output results
cat("Test H0: alpha(1) = 2\n")
cat("t-statistic:", t_stat_a1, " p-value:", p_val_a1, "\n")
cat("Test H0: alpha(2) = 1\n")
cat("t-statistic:", t_stat_a2, " p-value:", p_val_a2, "\n")