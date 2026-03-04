# Install package if not already installed
# install.packages("car") 
library(car)
data(Salaries)

# Fit the linear model
# rank, discipline, and sex are factors. R handles them by creating dummy variables.
cat("\n--- Part a) Estimate intercept, coefficients and standard error + Part b) Hypothesis and p-value comparisons---\n")
model <- lm(salary ~ rank + discipline + yrs.since.phd + sex, data = Salaries)
summary(model)

anova(model)