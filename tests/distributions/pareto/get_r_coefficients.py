"""
R Script to get coefficients for Pareto distribution test.

To run this, you need R with gamlss and gamlss.dist packages installed.
Save this as a .R file and run in R, or use the R console.

```R
# Install packages if needed
# install.packages("gamlss")
# install.packages("gamlss.dist")

library("gamlss")
library("gamlss.dist")

# Load mtcars dataset
data(mtcars)

# Fit PARETO2 model
model <- gamlss(
    mpg ~ cyl + hp,
    sigma.formula = ~cyl + hp,
    family=PARETO2(),
    data=as.data.frame(mtcars)
)

# Get coefficients
cat("Mu coefficients:\n")
print(coef(model, "mu"))

cat("\nSigma coefficients:\n")
print(coef(model, "sigma"))
```

Expected usage in Python test:
After running the R code above, update test_coefs.py with the obtained coefficients.
"""
