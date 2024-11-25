install.packages("tidyverse")
install.packages("lm.beta")
library(tidyverse)
library(corrplot)
library(caret)
library(caTools)
library(olsrr)
library(rpart)
library(rpart.plot)

# URL: https://www.kaggle.com/datasets/niteshyadav3103/concrete-compressive-strength
concrete_df <- read.csv("C:/Users/nicho/Desktop/Concrete_Data.csv", stringsAsFactors = T)

str(concrete_df)

# Data cleaning (renaming variables)
concrete_df$Cement <- concrete_df$Cement..component.1..kg.in.a.m.3.mixture.
concrete_df$Cement..component.1..kg.in.a.m.3.mixture. <- NULL
concrete_df$BF.Slag <- concrete_df$Blast.Furnace.Slag..component.2..kg.in.a.m.3.mixture.
concrete_df$Blast.Furnace.Slag..component.2..kg.in.a.m.3.mixture. <- NULL
concrete_df$Fly.Ash <- concrete_df$Fly.Ash..component.3..kg.in.a.m.3.mixture.
concrete_df$Fly.Ash..component.3..kg.in.a.m.3.mixture. <- NULL
concrete_df$Water <- concrete_df$Water...component.4..kg.in.a.m.3.mixture.
concrete_df$Water...component.4..kg.in.a.m.3.mixture. <- NULL
concrete_df$Superplasticizer <- concrete_df$Superplasticizer..component.5..kg.in.a.m.3.mixture.
concrete_df$Superplasticizer..component.5..kg.in.a.m.3.mixture. <- NULL
concrete_df$Coarse.Agg <- concrete_df$Coarse.Aggregate...component.6..kg.in.a.m.3.mixture.
concrete_df$Coarse.Aggregate...component.6..kg.in.a.m.3.mixture. <- NULL
concrete_df$Fine.Agg<- concrete_df$Fine.Aggregate..component.7..kg.in.a.m.3.mixture.
concrete_df$Fine.Aggregate..component.7..kg.in.a.m.3.mixture. <- NULL
concrete_df$Age <- concrete_df$Age..day.
concrete_df$Age..day. <- NULL
concrete_df$Concrete.CS <- concrete_df$Concrete.compressive.strength.MPa..megapascals..
concrete_df$Concrete.compressive.strength.MPa..megapascals.. <- NULL

# Correlation
pearson_corr <- cor(concrete_df, method = "pearson")
pearson_corr
corrplot(pearson_corr, type = "upper", method = "pie")

# Scatterplot Matrix
pairs(~ . , panel=panel.smooth, data = concrete_df, main = "Scatterplot Matrix of Concrete Strength data")

set.seed(2024)
# Train-Test set
train <- sample.split(Y = concrete_df$Concrete.CS, SplitRatio = 0.7)
trainset <- subset(concrete_df, train==T)
testset <- subset(concrete_df, train==F)

summary(trainset$Concrete.CS)
summary(testset$Concrete.CS)

# Fitting linear model to trainset
model_train <- lm(Concrete.CS ~ ., data = trainset)
model <- ("Linear Regression")
RMSE_train <- round(sqrt(mean((trainset$Concrete.CS - predict(model_train))^2)))
RMSE_test <- round(sqrt(mean((testset$Concrete.CS - predict(model_train, newdata = testset))^2)))

# Forward Selection on trainset
FWDfit_p_train <- ols_step_forward_p(model_train, details = TRUE)

# Prediction on forward selection testset
FWD_testset_predictions <- predict(model_train, newdata = testset)

# Evaluation of testset
FWD_rmse_test <- sqrt(mean((testset$Concrete.CS - FWD_testset_predictions)^2))
# 10.01351

## Backward Elimination on trainset
BWDfit_p_train <- ols_step_backward_p(model_train, details = TRUE)

# Prediction on backward elimination testset
BWD_testset_predictions <- predict(model_train, newdata = testset)

# Evaluation of backward testset
BWD_rmse_test <- sqrt(mean((testset$Concrete.CS - BWD_testset_predictions)^2))
# 10.01351

### Best Subset on trainset
BSS_p_train <- ols_step_both_p(model_train, details = TRUE)

# Prediction on best subset testset
BSS_testset_predictions <- predict(model_train, newdata = testset)

# Evaluation of best subset testset
BSS_rmse_test <- sqrt(mean((testset$Concrete.CS - BSS_testset_predictions)^2))
# 10.01351

fit_all <- lm(Concrete.CS ~ ., data = concrete_df)

fit_start <- lm(Concrete.CS ~ 1, data = concrete_df)
summary(fit_start)

# Forward Selection Method
step(fit_all, direction = "forward")

# Backwards Elimination Method
step(fit_all, direction = "backward")

# Step-wise Regression Method
step(fit_start, direction = "both", scope = formula(fit_all))

########### CART ###########
model1 <- rpart(Concrete.CS ~ ., data = trainset, method = 'anova', cp = 0)
CV_error_cap <- model1$cptable[which.min(model1$cptable[,"xerror"]), "xerror"] +
                model1$cptable[which.min(model1$cptable[,"xerror"]), "xstd"]

i <- 1
while (model1$cptable[i,"xerror"] > CV_error_cap) {
  i <- i + 1
}

cp_table <- printcp(model1)
# Lowest xerror: 0.19938

plotcp(model1)

cp_opt = ifelse(i > 1, sqrt(model1$cptable[i, "CP"] * model1$cptable[i-1, "CP"]), 1)
# Optimal cp: 0.00217118417939809

cart_model_1SE <- prune(model1, cp = cp_opt)
printcp(cart_model_1SE)
plotcp(cart_model_1SE)

model <- c(model, "CART 1SE")
rpart.plot(cart_model_1SE, nn = T, main = "Optimal Tree", cex = 0.5, tweak = 0.3)

RMSE_train <- c(RMSE_train, round(sqrt(mean((trainset$Concrete.CS - predict(m_cart_1SE))^2))))
RMSE_test <- c(RMSE_test, round(sqrt(mean((testset$Concrete.CS - predict(m_cart_1SE, newdata = testset))^2))))

RF_model <- randomForest(Concrete.CS ~ . , data=trainset)

RF_model

# OOB RMSE
sqrt(RF_model$mse[RF_model$ntree])

plot(RF_model)

# Error stablised before 500 trees.

model <- c(model, "Random Forest")

RMSE_train <- c(RMSE_train, round(sqrt(mean((trainset$Concrete.CS - predict(RF_model, newdata = trainset))^2))))
RMSE_test <- c(RMSE_test, round(sqrt(mean((testset$Concrete.CS - predict(RF_model, newdata = testset))^2))))

results <- data.frame(model, RMSE_train, RMSE_test)
View(results)
# Random Forest has lowest testset error.