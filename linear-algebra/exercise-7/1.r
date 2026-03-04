# Load the necessary package
if(!require(faraway)) install.packages("faraway")
library(faraway)

# Load the data
data(gala)

# Fit the Poisson regression model
# We explain Species by Area, Elevation, Nearest, Scruz, and Adjacent
model_pois <- glm(Species ~ Area + Elevation + Nearest + Scruz + Adjacent, 
                  family = poisson, 
                  data = gala)

# Display the estimated parameters
summary(model_pois)

# --- Method 1: Using predict() ---
# Extract the data for Coamano (row name is usually the island name)
coamano_data <- gala["Coamano", ]

# Predict expected count (type="response" gives values on the scale of Y, i.e., exp(linear predictor))
pred_r <- predict(model_pois, newdata = coamano_data, type = "response")

cat("Prediction using predict():", pred_r, "\n")


# --- Method 2: By hand ---
# Extract the estimated coefficients
betas <- coef(model_pois)

# Extract the values for Coamano
# Note: The model matrix includes a 1 for the intercept
# We construct the vector x corresponding to (1, Area, Elevation, Nearest, Scruz, Adjacent)
x_coamano <- c(1, 
               coamano_data$Area, 
               coamano_data$Elevation, 
               coamano_data$Nearest, 
               coamano_data$Scruz, 
               coamano_data$Adjacent)

# Calculate linear predictor (eta = beta * x)
linear_predictor <- sum(betas * x_coamano)

# Apply inverse link function (exponential)
pred_hand <- exp(linear_predictor)

cat("Prediction by hand:", pred_hand, "\n")
