# Load data (it is built-in, but good practice to call data())
data(trees)

cat("\n--- Part b) Estimate parameters---\n")

model_log <- lm(log(Volume) ~ log(Girth) + log(Height), data = trees)

# First, get starting values from the linear model in Problem 2
start_vals <- coef(model_log)
c_start <- exp(start_vals[1])  # Intercept from log model is ln(c)
a1_start <- start_vals[2]
a2_start <- start_vals[3]

# Fit the non-linear model
model_nls <- nls(Volume ~ c * Girth^alpha1 * Height^alpha2, 
                 data = trees,
                 start = list(c = c_start, alpha1 = a1_start, alpha2 = a2_start))
summary(model_nls)

# c) Plot Volume against fitted values for both models
cat("\n--- Part c) Plot Volume against fitted values for both models ---\n")
# Create a plot to compare fits
			 # Back-transform log model
fitted_log_original <- exp(fitted(model_log))

plot(trees$Volume, fitted_log_original,
     col = "blue", pch = 19,
     xlab = "Observed Volume",
     ylab = "Fitted Volume",
     main = "Comparison of Linear (Log) vs Non-Linear Fit")

points(trees$Volume, fitted(model_nls),
       col = "red", pch = 19)

abline(0, 1, lty = 2)

legend("topleft",
       legend = c("Log-Linear Model", "Non-Linear Model"),
       col = c("blue", "red"),
       pch = 19)
