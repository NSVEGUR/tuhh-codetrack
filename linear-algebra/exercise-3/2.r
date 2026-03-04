# 1. Read the data from the CSV file
# Ensure the file is in your current working directory
data <- read.csv("Residuals_Gaussian.csv")

# -------------------------------------------------------
# Analysis for Y1
# -------------------------------------------------------
# Fit the linear model explaining Y1 by U
lm1 <- lm(Y1 ~ U, data = data)

# Graphical Tool 1: Normal QQ-plot of residuals
# Checks if residuals are normally distributed
qqnorm(lm1$residuals, main = "Normal QQ-plot for Y1")
qqline(lm1$residuals) # Adds a reference line

# Graphical Tool 2: Residuals vs Fitted Values
# Checks for independence/homoscedasticity (no pattern should be visible)
plot(lm1$fitted.values, lm1$residuals, 
     main = "Residuals vs Fitted for Y1",
     xlab = "Fitted Values", ylab = "Residuals")

# -------------------------------------------------------
# Analysis for Y2
# -------------------------------------------------------
# Fit the linear model explaining Y2 by U
lm2 <- lm(Y2 ~ U, data = data)

# Normal QQ-plot
qqnorm(lm2$residuals, main = "Normal QQ-plot for Y2")
qqline(lm2$residuals)

# Residuals vs Fitted Values
plot(lm2$fitted.values, lm2$residuals, 
     main = "Residuals vs Fitted for Y2",
     xlab = "Fitted Values", ylab = "Residuals")

# -------------------------------------------------------
# Analysis for Y3
# -------------------------------------------------------
# Fit the linear model explaining Y3 by U
lm3 <- lm(Y3 ~ U, data = data)

# Normal QQ-plot
qqnorm(lm3$residuals, main = "Normal QQ-plot for Y3")
qqline(lm3$residuals)

# Residuals vs Fitted Values
plot(lm3$fitted.values, lm3$residuals, 
     main = "Residuals vs Fitted for Y3",
     xlab = "Fitted Values", ylab = "Residuals")