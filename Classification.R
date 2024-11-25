install.packages("tidyverse")
install.packages("olsrr")
install.packages("bootStepAIC")
install.packages("glmnet")
library(tidyverse)
library(olsrr)
library(nnet)
library(bootStepAIC)
library(glmnet)
library(caret)
library(caTools)
library(class)
library(pROC)

# URL: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
heart_df <- read.csv("C:/Users/nicho/Desktop/heart.csv", stringsAsFactors = T)

str(heart_df)
heart_df$HeartDisease <- factor(heart_df$HeartDisease)
summary(heart_df)

pairs(~ . , panel=panel.smooth, data = heart_df, main = "Scatterplot Matrix of Heart Disease Data")

log_model1 <- glm(HeartDisease ~ . , family = binomial, data = heart_df)
summary(log_model1)
# AIC = 626.19 

# Removed variables Age, RestingBP, RestingECG & MaxHR as they are not significant variables.
log_model2 <- glm(HeartDisease ~ Sex + ChestPainType + Cholesterol + FastingBS + 
                    ExerciseAngina + Oldpeak + ST_Slope, family = binomial, data = heart_df)
summary(log_model2)
# AIC = 621.61

bootstrap_model <- boot.stepAIC(log_model1, heart_df, B = 50)
bootstrap_model
# No even split between +ve and -ve in Coefficient sign
# AIC = 619.81

# Odds Ratio
OR_log_model2 <- exp(coef(log_model2))
OR_log_model2

# Odds Ratio Confidence Interval
OR_CI_log_model2 <- exp(confint(log_model2))
OR_CI_log_model2

predict1 <- predict(log_model2, type = 'response')
plot(x = heart_df$HeartDisease, y = predict1, main = 'Logistic Regression Probability of Heart Disease')

# Confusion Matrix with threshold = 0.5
threshold <- 0.5
HeartDisease_hat <- ifelse(predict1 > threshold, "Yes", "No")

table1 <- table(heart_df$HeartDisease, HeartDisease_hat)
table1
prop.table(table1)

# Overall Accuracy 
sum(diag(table1)) / sum(table1)
# 86.71%

set.seed(2024)
train <- sample.split(heart_df$HeartDisease, SplitRatio = 0.7)
trainset <- subset(heart_df, train == TRUE)
testset <- subset(heart_df, train == FALSE)

# Training data
x_train <- model.matrix(HeartDisease ~ . - 1, data = trainset)
y_train <- trainset$HeartDisease
dim(x_train)
length(y_train)

# Test data
x_test <- model.matrix(HeartDisease ~ . -1, data = testset)
y_test <- testset$HeartDisease
dim(x_test)
length(y_test)

# Fit the logistic regression model with Lasso regularization
lasso_model <- glmnet(x_train, y_train, family = "binomial", alpha = 1)
print(lasso_model)

cv_results <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
best_lambda <- cv_results$lambda.min
best_lambda
# Lambda = 0.004575469

# Prediction on test set
test_prediction <- predict(lasso_model, newx = x_test, s = best_lambda, type = "response")
test_prediction_hat <- ifelse(test_prediction > threshold, 1, 0)

test_accuracy <- sum(test_prediction_hat == y_test) / length(y_test)
test_accuracy
# Accuracy = 87.64%