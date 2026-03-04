library(faraway)
data(babyfood)

# 1. Inspect the data to confirm column names
# usually columns are: disease, nondisease, sex, food
print(head(babyfood))

# 2. Fit the main effects model
# Response is cbind(successes, failures)
model_main <- glm(cbind(disease, nondisease) ~ sex + food, 
                  family = binomial, 
                  data = babyfood)

# 3. View estimates
summary(model_main)

# Fit model with interaction
model_interaction <- glm(cbind(disease, nondisease) ~ sex * food,
                         family = binomial,
                         data = babyfood)
summary(model_interaction)

# Get fitted values
fitted_probs <- fitted(model_interaction)
print(fitted_probs)
