library(multcomp)

# 1. Define the Data
mortality <- c(200, 128, 117, 129, 182, 229, 134)
latitude <- c(39, 42, 44, 44, 35, 31, 43)
n <- length(mortality)

# Fit Gaussian linear model
LM <- lm(mortality ~ latitude)
summary(LM)

# Define hypothesis: slope = 0
glht_slope <- glht(LM, linfct = c("latitude = 0"))

# Summary (two-sided by default)
summary(glht_slope, test = adjusted("none"))

# Linear combination for prediction at latitude 40
glht_ohio <- glht(LM, linfct = matrix(c(1,40), nrow=1))

confint(glht_ohio, level = 0.90)
