# Load data (it is built-in, but good practice to call data())
data(trees)

# a) Estimate parameters using least squares on the log-linear model
cat("\n--- Part a) Estimate parameters using least squares on the log-linear model ---\n")
model_log <- lm(log(Volume) ~ log(Girth) + log(Height), data = trees)
summary(model_log)

# Extract coefficients for alpha1 and alpha2
coef(model_log)

# b) Test H0: alpha1 = 2 and alpha2 = 1 at level 0.01
# We define a transformed dependent variable according to the hint
trees$transformed_Y <- log(trees$Volume) - 2 * log(trees$Girth) - log(trees$Height)

# Fit the model with the transformed variable
model_test <- lm(transformed_Y ~ log(Girth) + log(Height), data = trees)

# The standard F-statistic in the summary tests if all slope coefficients are zero.
# This corresponds exactly to H0: (alpha1 - 2) = 0 and (alpha2 - 1) = 0.
summary(model_test)