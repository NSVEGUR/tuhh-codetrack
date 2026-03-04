# Load necessary packages
if(!require(car)) install.packages("car")
if(!require(MASS)) install.packages("MASS") # For LDA
library(car)
library(MASS)

# Load the data
data(Davis)

# Fix the mistake by swapping weight and height for row 12
temp <- Davis[12, "weight"]
Davis[12, "weight"] <- Davis[12, "height"]
Davis[12, "height"] <- temp

plot(Davis$weight, Davis$height, 
     col = ifelse(Davis$sex == "F", "red", "blue"), 
     pch = 19, 
     xlab = "Weight (kg)", 
     ylab = "Height (cm)",
     main = "Height vs Weight by Sex")
legend("topleft", legend = c("Female", "Male"), col = c("red", "blue"), pch = 19)

#Fit Logistic Regression Model
# family = binomial handles binary outcomes
model_logit <- glm(sex ~ weight + height, family = binomial, data = Davis)

# Display estimates and test results
summary(model_logit)

# Extract coefficients
beta <- coef(model_logit)
b0 <- beta[1]  # Intercept (was beta[2])
b1 <- beta[2]  # weight (was beta[3])
b2 <- beta[3]

# Calculate slope and intercept for the line height = slope * weight + intercept
slope_logit <- -b1 / b2
intercept_logit <- -b0 / b2

# Add the line to the existing plot
abline(a = intercept_logit, b = slope_logit, col = "black", lwd = 2, lty = 2)

# 1. Fit the LDA model
# We specify prior = c(0.5, 0.5) to use the "actual" probabilities mentioned in the text
# Note: The levels are F, M. So prior c(0.5, 0.5) assigns 0.5 to F and 0.5 to M.
model_lda <- lda(sex ~ weight + height, data = Davis, prior = c(0.5, 0.5))
print(model_lda)

# 2. Compute the boundary line parameters manually
# LDA boundary is linear: a*weight + b*height + c = 0

# Extract means (mu_k) and prior probabilities
mu <- model_lda$means
mu_F <- mu["F", ]
mu_M <- mu["M", ]

# Calculate common covariance matrix (Sigma)
# LDA assumes one common covariance matrix. We can estimate it manually or use heuristics.
# A robust way in R context:
n <- nrow(Davis)
# Separate groups
d_F <- subset(Davis, sex == "F")[, c("weight", "height")]
d_M <- subset(Davis, sex == "M")[, c("weight", "height")]
# Weighted average of covariance matrices (standard LDA formula)
S_F <- cov(d_F)
S_M <- cov(d_M)
n_F <- nrow(d_F)
n_M <- nrow(d_M)
# Pooled Covariance Matrix
Sigma <- ((n_F - 1) * S_F + (n_M - 1) * S_M) / (n - 2)

# Calculate coefficients for the boundary line
# Normal vector w = Sigma_inv * (mu_M - mu_F)
Sigma_inv <- solve(Sigma)
w <- Sigma_inv %*% (mu_M - mu_F) # This gives a vector with 2 elements: w_weight and w_height

# Calculate the constant term K
# K = -0.5 * (mu_M - mu_F)^T * Sigma_inv * (mu_M + mu_F) + log(pi_M/pi_F)
# Since pi_M = pi_F = 0.5, log term is 0.
K <- -0.5 * t(mu_M - mu_F) %*% Sigma_inv %*% (mu_M + mu_F)

# The line equation is: w[2]*weight + w[3]*height + K = 0
# height = (-w[2]/w[3]) * weight - (K/w[3])
slope_lda <- -w[2] / w[3]
intercept_lda <- -K / w[3]

# 3. Add LDA boundary to the plot
abline(a = intercept_lda, b = slope_lda, col = "green", lwd = 2)

# Add Legend for boundaries
legend("bottomright", legend = c("Logistic Boundary", "LDA Boundary (Prior=0.5)"),
       col = c("black", "green"), lty = c(2, 1), lwd = 2)