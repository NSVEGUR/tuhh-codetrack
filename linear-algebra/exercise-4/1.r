# Load the data
data <- read.csv("Greenberg.csv")



# a) Linear model with school as the only independent variable
cat("\n--- Part a) Linear model with school ---\n")
model_a <- lm(height ~ school, data = data)
summary(model_a)



# b) Linear model with school and age
cat("\n--- Part b) Linear model with school and age ---\n")
model_b <- lm(height ~ school + age, data = data)
summary(model_b)