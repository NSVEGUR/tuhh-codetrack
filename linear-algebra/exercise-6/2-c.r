data(trees)
set.seed(123)
n_sim <- 1000

# Simulation based on assumptions
U <- runif(n_sim)
N_norm <- rnorm(n_sim)
G_sim <- 8 + 10 * U
H_sim <- 62 + G_sim + 5.5 * N_norm

# Plotting
par(mfrow = c(2, 2))
hist(trees$Girth, main = "Real Girth", xlab = "Inches", col="lightblue")
hist(G_sim, main = "Simulated Girth (Uniform)", xlab = "Inches", col="lightgreen")
plot(trees$Girth, trees$Height, main = "Real Relation", pch=19)
plot(G_sim, H_sim, main = "Simulated Relation", pch=19, cex=0.5)
par(mfrow = c(1, 1))