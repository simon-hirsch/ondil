
library(gamCopula)
library(copula)
library(MASS)




  covariates.distr <- mvdc(normalCopula(0.2, dim = 3),
  c("unif"), list(list(min = -2, max = 2)),
  marginsIdentical = TRUE
  )

  X <- rMvdc(2000, covariates.distr)     

# Transform X to uniform margins using the empirical cumulative distribution function (ECDF)
# X_uniform <- apply(x_data, 2, function(column) {
#  ecdf(column)(column)
# })


  U <- condBiCopSim(1, function(x1, x2, x3) {
   2+ 0.7*x1 + 4*x2 
  }, X, return.par = TRUE)



# Transform U$data (which is on [0,1]) to normal margins using the inverse normal CDF (qnorm)
U_normal <- as.data.frame(qnorm(U$data))
colnames(U_normal) <- colnames(U$data)

merged_data <- data.frame(U$data, X)
names(merged_data) <- c(paste("u", 1:2, sep = ""), paste("x", 1:3, sep = ""))


# Display the current working directory
write.csv(merged_data, "C:/Users/OEK-admin/OneDrive/Arbeit_Uni/Uni_Due/ProjectII/merged_data_1.csv", row.names = FALSE)

fit = gamBiCopFit(merged_data,
  formula = ~ (x1+x2+x3),
  family = 1,
  verbose = TRUE,
  method = "NR",
  n.iter = 100,
  tau = TRUE
  ) 


fit$res@model$coefficients
