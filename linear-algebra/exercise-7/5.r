# Load the data
data("USArrests")

# --- 1. PCA on Covariance Matrix (scale = FALSE) ---
# This looks at raw variance.
pca_cov <- prcomp(USArrests, scale = FALSE)

cat("--- PCA on Covariance Matrix ---\n")
print(summary(pca_cov))
print(pca_cov$rotation) 
# Note: Look at PC1 rotation. The loading for 'Assault' will be very high (close to 1 or -1)
# because it has the largest numerical variance (0-300 range vs 0-100 or 0-20).


# --- 2. PCA on Correlation Matrix (scale = TRUE) ---
# This standardizes variables first.
pca_cor <- prcomp(USArrests, scale = TRUE)

cat("\n--- PCA on Correlation Matrix ---\n")
print(summary(pca_cor))
print(pca_cor$rotation)
# Note: The loadings in PC1 will be more balanced across the variables,
# reflecting the correlation structure rather than just the magnitude of the numbers.