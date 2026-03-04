# Load the necessary package and data
# Note: You may need to install the package first if not present
# install.packages("statforbiology") 
library(statforbiology)
data("beetGrowth")

# b) Plot data to find starting values
cat("\n--- Part b) Plot data to find starting values ---\n")
plot(beetGrowth$DAE, beetGrowth$weightFree, 
     xlab = "Days After Emergence (DAE)", ylab = "Weight Free",
     main = "Beet Growth Data")

# Eyeball starting values from the plot:
# beta1 (Asymptote): Look at the max weight, approx 20-25?
# beta2: Related to intercept. If t=0, y = beta1/(1+beta2). 
# beta3 (Growth rate): Controls steepness.

# Suggested starting values (you might need to adjust these based on the plot)
# Let's assume beta1 ≈ 25, beta2 ≈ 10, beta3 ≈ 0.1
start_values <- list(beta1 = 25, beta2 = 10, beta3 = 0.1)

# Estimate beta using nls
model_growth <- nls(weightFree ~ beta1 / (1 + beta2 * exp(-beta3 * DAE)), 
                    data = beetGrowth, 
                    start = start_values)

summary(model_growth)

# Optional: Add the fitted curve to the plot
lines(beetGrowth$DAE, fitted(model_growth), col = "red", lwd = 2)