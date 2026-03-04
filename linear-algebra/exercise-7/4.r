# Load necessary packages
if(!require(kernlab)) install.packages("kernlab")
if(!require(MASS)) install.packages("MASS")
library(kernlab)
library(MASS)

# Load the data
data(spam)

# Fit the Linear Discriminant Analysis model
# We predict 'type' using all other variables (denoted by '.')
lda_model <- lda(type ~ ., data = spam)

# Display the coefficients of the linear discriminant
# These are stored in the 'scaling' component of the model object
# Since there is only 1 discriminant, this will be a matrix with 1 column and 57 rows.
print(head(lda_model$scaling)) # Showing just the first few for brevity

# Predict classes for the original data
# The predict function returns a list; component 'class' contains the predictions
lda_pred <- predict(lda_model, spam)

# Create a confusion matrix (Actual vs Predicted)
conf_matrix <- table(Actual = spam$type, Predicted = lda_pred$class)
print(conf_matrix)

# Count the number of errors
# Errors are the off-diagonal elements (non-spam predicted as spam, spam predicted as non-spam)
num_errors <- sum(lda_pred$class != spam$type)
total_obs <- nrow(spam)
error_rate <- num_errors / total_obs

cat("Total number of errors:", num_errors, "\n")
cat("Error rate:", round(error_rate * 100, 2), "%\n")

# Extract the linear discriminant values (LD1)
# 'x' contains the scores; since we have 1 discriminant, we select the first column
discriminant_scores <- lda_pred$x[, 1]

# 1. Box-plot of discriminant values by group
boxplot(discriminant_scores ~ spam$type,
        main = "LDA Scores by Email Type",
        xlab = "Email Type",
        ylab = "Linear Discriminant Score (LD1)",
        col = c("lightblue", "salmon"))

# 2. QQ-plots for each group
# We want to check if the discriminant scores within each group are approximately normal.
# LDA assumes the features are normal within groups, which implies the linear combination
# should also be normal within groups.

# Split scores by type
scores_spam <- discriminant_scores[spam$type == "spam"]
scores_nonspam <- discriminant_scores[spam$type == "nonspam"]

# Set up the plotting area for two side-by-side plots
par(mfrow = c(1, 2))

# QQ-plot for Non-Spam
qqnorm(scores_nonspam, main = "QQ-Plot: Non-Spam Scores")
qqline(scores_nonspam, col = "red")

# QQ-plot for Spam
qqnorm(scores_spam, main = "QQ-Plot: Spam Scores")
qqline(scores_spam, col = "red")

# Reset plotting area
par(mfrow = c(1, 1))