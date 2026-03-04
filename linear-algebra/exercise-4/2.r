# Load the dataset
data("PlantGrowth")

names(PlantGrowth) # Check column names to ensure they match 
#Print their groups/levels
print(unique(PlantGrowth$group))

cat("\n--- Part a) All at once ---\n")
# Fit the linear model to compare the three groups
lm_model <- lm(weight ~ group, data = PlantGrowth)
summary(lm_model)
# Perform ANOVA to test for overall differences between groups
aov_model <- aov(weight ~ group, data = PlantGrowth)
summary(aov_model)

cat("\n--- Part b) Pairwise comparisons ---\n")
# Perform Tukey-Kramer test
TukeyHSD(aov_model)

