# Regression: Concrete Compressive Strength Prediction

**Problem Description:** This showcase uses regression techniques to predict concrete compressive strength based on various components in the concrete mixture, such as cement, water, slag, and aggregates, as well as the age of the concrete.

**Introduction:** Concrete is a fundamental material in construction, essential in buildings, bridges, and even historical landmarks. The dataset consists of the dependent variable: concrete compressive strength, and eight independent variables: cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age. Compressive strength of concrete is a non-linear function of both the age and the constituent ingredients. During hydration, the mixture reacts with water, gaining strength rapidly at first, then gradually over time.

**Impact:** By using machine learning, the formulation of concrete can be optimized with data-driven insights, enhancing strength and durability, and overall performance. This approach not only reduces costs and minimizes environmental impact but also improves construction practices, going beyond traditional testing methods like destructive and non-destructive testing to provide more efficient and sustainable solutions.

---
```
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

# Summary of Dataset
summary(concrete_df)
```
![Dataset summary](https://github.com/user-attachments/assets/c1563b01-54fa-4782-b8a3-d583a9581ed3)

```
# Correlation Test
pearson_corr <- cor(concrete_df, method = "pearson")
pearson_corr
```
Correlation Matrix

![Correlation Matrix](https://github.com/user-attachments/assets/f3e1d479-5112-4a76-aef1-76b7d8357fc8)
```
corrplot(pearson_corr, type = "upper", method = "pie")
```
Correlation Plot

![Correlation Plot](https://github.com/user-attachments/assets/44a56548-07ac-4b85-992f-33c61efddd14)

```
# Scatterplot Matrix
pairs(~ . , panel=panel.smooth, data = concrete_df, main = "Scatterplot Matrix of Concrete Strength data")
```
Scatter Plot

![Scatterplot](https://github.com/user-attachments/assets/3e6defbd-4acf-4148-8fbb-84ccb9e9c90f)

## Linear Regression Base Model
```
linear_model_base <- lm(Concrete.CS ~ ., data = concrete_df)
summary(linear_model_base)
```
![Linear Regression Base Model](https://github.com/user-attachments/assets/0bd38008-22d8-4cec-93bd-9763c3ecf231)

```
# Calculate RMSE for the base model
predictions_base <- predict(linear_model_base, newdata = concrete_df)
rmse_base <- sqrt(mean((concrete_df$Concrete.CS - predictions_base)^2))
cat("RMSE (Base Model - No Train/Test Split):", rmse_base, "\n")
```

**R²**: 0.6155

R² represents the proportion of variance in the dependent variable that is explained by the independent variables. The model explains 61.55% of the variability in the dependent variable

## Linear Regression Model (with train/test split)
Using a Train-Test split of 70:30
```
set.seed(2024)

# Train-Test set
train <- sample.split(Y = concrete_df$Concrete.CS, SplitRatio = 0.7)
trainset <- subset(concrete_df, train==T)
testset <- subset(concrete_df, train==F)

# Fit Linear Model with Cross-Validation
cv_model <- train(Concrete.CS ~ ., data = trainset, method = "lm", trControl = train_control)

# RMSE from cross-validation
cv_rmse <- cv_model$results$RMSE
cv_r2 <- cv_model$results$Rsquared

# Feature Selection with K-Fold CV

# Forward Selection with K-Fold CV
FWD_model <- train(Concrete.CS ~ ., data = trainset, method = "leapForward", trControl = train_control)
FWD_rmse <- FWD_model$results$RMSE
FWD_r2 <- FWD_model$results$Rsquared

# Backward Elimination with K-Fold CV
BWD_model <- train(Concrete.CS ~ ., data = trainset, method = "leapBackward", trControl = train_control)
BWD_rmse <- BWD_model$results$RMSE
BWD_r2 <- BWD_model$results$Rsquared

# Best Subset Selection with K-Fold CV
BSS_model <- train(Concrete.CS ~ ., data = trainset, method = "leapSeq", trControl = train_control)
BSS_rmse <- BSS_model$results$RMSE
BSS_r2 <- BSS_model$results$Rsquared

# Print Results
cat("5-Fold Cross-Validation RMSE (Full Model):", mean(cv_rmse), "\n")
cat("5-Fold Cross-Validation R² (Full Model):", mean(cv_r2), "\n")
cat("5-Fold Cross-Validation RMSE (Forward Selection):", mean(FWD_rmse), "\n")
cat("5-Fold Cross-Validation R² (Forward Selection):", mean(FWD_r2), "\n")
cat("5-Fold Cross-Validation RMSE (Backward Elimination):", mean(BWD_rmse), "\n")
cat("5-Fold Cross-Validation R² (Backward Elimination):", mean(BWD_r2), "\n")
cat("5-Fold Cross-Validation RMSE (Best Subset Selection):", mean(BSS_rmse), "\n")
cat("5-Fold Cross-Validation R² (Best Subset Selection):", mean(BSS_r2), "\n")
```
| Model                  | CV RMSE  | Train RMSE | Test RMSE | CV R2   | 
|------------------------|----------|------------|-----------|---------|
| Base Model             | -        | 10.35361   | 9.86342   | -       |
| Full Model (5-Fold CV) | 10.68926 | 10.53155   | 10.01351  | 0.59834 |
| Forward Selection      | 12.68548 | 11.38453   | 10.81977  | 0.41306 |
| Backward Elimination   | 12.57990 | 11.23371   | 10.84777  | 0.42230 |
| Best Subset Selection  | 12.48967 | 11.23371   | 10.84777  | 0.43945 |

Root Mean Squared Error (RMSE) measures the magnitude of the errors made by the model.

Base Model:

- The Base Model without cross-validation has a Train RMSE of 10.35361 and a Test RMSE of 9.86342, which suggests that it performs slightly better on the test set compared to the Full Model.
- However, since it does not use cross-validation, its performance may not generalize well to unseen data.

Full Model (5-Fold CV):

- The Full Model with cross-validation has a CV RMSE of 10.68926 and a CV R² of 0.59834, indicating that it generalizes well across folds.
- The Train RMSE (10.53155) is close to the Test RMSE (10.01351), suggesting good generalization without significant overfitting.
- The Full Model outperforms all feature selection methods in terms of both CV RMSE and CV R².

Forward Selection:

- Forward Selection has the highest CV RMSE (12.68548) and the lowest CV R² (0.41306) among all models, indicating weaker predictive performance.
- The Train RMSE (11.38453) is higher than that of the Full Model, and the Test RMSE (10.81977) is also worse than the Full Model.
- This suggests that Forward Selection may have removed important predictors, leading to reduced accuracy.

Backward Elimination:

- Backward Elimination performs slightly better than Forward Selection but still worse than the Full Model.
- The CV RMSE is 12.57990, and CV R² is 0.42230, indicating moderate performance.
- The Train RMSE (11.23371) and Test RMSE (10.84777) are both higher than those of the Full Model, suggesting that this method also excluded useful predictors.

Best Subset Selection:

- Best Subset Selection has a slightly lower CV RMSE (12.48967) and higher CV R² (0.43945) compared to Forward and Backward methods but still underperforms compared to the Full Model.
- The Train RMSE (11.23371) and Test RMSE (10.84777) are identical to those of Backward Elimination, indicating similar predictor selection.

## Regularised Linear Regression
```
X_train <- as.matrix(trainset[, -which(names(trainset) == "Concrete.CS")])
Y_train <- trainset$Concrete.CS
X_test <- as.matrix(testset[, -which(names(testset) == "Concrete.CS")])
Y_test <- testset$Concrete.CS
```
### Lasso Regularisation
```
# Fit Lasso model (with cross-validation to find optimal lambda)
lasso_model <- cv.glmnet(X_train, Y_train, alpha = 1)

# Get the best lambda from cross-validation
best_lambda_lasso <- lasso_model$lambda.min
cat("Best lambda for Lasso:", best_lambda_lasso, "\n") # Best lambda for Lasso: 0.007082498 

# Predictions on training set
lasso_predictions_train <- predict(lasso_model, s = best_lambda_lasso, newx = X_train)
lasso_rmse_train <- sqrt(mean((Y_train - lasso_predictions_train)^2)) # Train RMSE
cat("Lasso RMSE (Train):", lasso_rmse_train, "\n") # Lasso RMSE (Train): 10.53208 

# Calculate Test RMSE
lasso_predictions_test <- predict(lasso_model, s = best_lambda_lasso, newx = X_test)
lasso_rmse_test <- sqrt(mean((Y_test - lasso_predictions_test)^2)) # Test RMSE
cat("Lasso RMSE (Test):", lasso_rmse_test, "\n") # Lasso RMSE (Test): 10.01103 
```
### Ridge regression
```
ridge_model <- cv.glmnet(X_train, Y_train, alpha = 0)  # alpha = 0 for Ridge

# Get the best lambda from cross-validation
best_lambda_ridge <- ridge_model$lambda.min
cat("Best lambda for Ridge:", best_lambda_ridge, "\n") # Best lambda for Ridge: 0.8334769 

# Predictions on training set
ridge_predictions_train <- predict(ridge_model, s = best_lambda_ridge, newx = X_train)
ridge_rmse_train <- sqrt(mean((Y_train - ridge_predictions_train)^2)) # Train RMSE
cat("Ridge RMSE (Train):", ridge_rmse_train, "\n") # Ridge RMSE (Train): 10.66311 

# Calculate Test RMSE
ridge_predictions_test <- predict(ridge_model, s = best_lambda_ridge, newx = X_test)
ridge_rmse_test <- sqrt(mean((Y_test - ridge_predictions_test)^2)) # Test RMSE
cat("Ridge RMSE (Test):", ridge_rmse_test, "\n") # Ridge RMSE (Test): 10.09583 
```
| Model                  | Train RMSE | Test RMSE | 
|------------------------|------------|-----------|
| Base Model             | 10.35361   | 9.86342   |
| Full Model (5-Fold CV) | 10.53155   | 10.01351  |
| Forward Selection      | 11.38453   | 10.81977  |
| Backward Elimination   | 11.23371   | 10.84777  |
| Best Subset Selection  | 11.23371   | 10.84777  |
| Lasso                  | 10.53208   | 10.01103  |
| Ridge                  | 10.66311   | 10.09583  |

Lasso Regression:

- The Test RMSE for Lasso (10.01103) is slightly lower than that of the Full Model (10.01351), indicating a marginal improvement in generalization.
- Since Lasso performs feature selection by shrinking some coefficients to zero, it suggests that a few variables might be less important. However, the improvement is negligible.

Ridge Regression:

- The Test RMSE for Ridge (10.09583) is slightly higher than that of the Full Model (10.01351), indicating a small reduction in generalization performance.
- Ridge shrinks coefficients without excluding them, suggesting that all variables contribute to the model.

## Comparing Model Selection Methods using Akaike Information Criterion (AIC)
```
fit_all <- lm(Concrete.CS ~ ., data = concrete_df)

fit_start <- lm(Concrete.CS ~ 1, data = concrete_df)
summary(fit_start)
```
![image](https://github.com/user-attachments/assets/c9b15add-7889-4d0d-895f-c5d285aa2734)

```
# Forward Selection Method
step(fit_all, direction = "forward")
```

![Forward Selection AIC](https://github.com/user-attachments/assets/67e4370b-eaaa-42ad-ace1-7aff4f485db4)

```
# Backwards Elimination Method
step(fit_all, direction = "backward")
```
![Backwards Elimination AIC](https://github.com/user-attachments/assets/8290fd0a-d5a1-46fa-8ec7-8cf3e8556a18)

```
# Step-wise Regression Method
step(fit_start, direction = "both", scope = formula(fit_all))
```
![Step-wise AIC part 1](https://github.com/user-attachments/assets/a02a67df-9c29-4ffc-bb82-55f7bbed014f)
![Step-wise AIC part 2](https://github.com/user-attachments/assets/04acbf94-afff-4db8-996b-c6ab53e2c3ac)
![Step-wise AIC part 3](https://github.com/user-attachments/assets/6758ede2-6952-4587-a8dc-a1707805594e)
![Step-wise AIC part 4](https://github.com/user-attachments/assets/4c0b6e7f-3023-4fca-b1e9-529e1aaeaffd)

| Method                | AIC     |
|-----------------------|---------|
| Forward Selection     | 4832.91 |
| Backwards Elimination | 4832.91 |
| Stepwise Regression   | 5801.45 |

Evaluation: Forward Selection and Backwards Elimination produced the same AIC (4832.91), indicating they selected the same set of variables. While Stepwise Regression has a higher AIC (5801.45), suggesting overfitting due to unnecessary complexity. This could be due to multicollinearity or the inclusion of less significant predictors.

## Classification And Regression Tree (CART)
### Building & Optimizing CART Tree
```
# Determining Cross-Validation (CV) Error Cap
model1 <- rpart(Concrete.CS ~ ., data = trainset, method = 'anova', cp = 0)

# Print Complexity Parameter (CP) Table
CP_table <- printcp(model1)
```
![CP Table part 1](https://github.com/user-attachments/assets/b0d241da-69ea-4390-b9f1-4069a186aa88)
![CP Table part 2](https://github.com/user-attachments/assets/b4d4af98-4ae9-44d9-94b3-9971b4bf266c)
![CP Table part 3](https://github.com/user-attachments/assets/f1f7c973-ad54-48e4-b32c-f065104192ae)

```
# Selecting the last row index
last_row <- nrow(model1$cptable)

# Computing the cross-validation error cap
CV_error_cap <- model1$cptable[last_row, "xerror"] + model1$cptable[last_row, "xstd"]
```
CV Error Cap: 0.205117709852608

```
# Finding the optimal Complexity Parameter (CP)
i <- 1
while (i < last_row && model1$cptable[i, "xerror"] > cv_error_cap) {
  i <- i + 1
}

plotcp(model1)
```
CP Plot

![CP Plot](https://github.com/user-attachments/assets/b14d2706-2eff-47bb-adb7-aa78d4b53946)
i = 35

```
# Finding Optimal CP
cp_opt <- ifelse(i > 1, sqrt(model1$cptable[i, "CP"] * model1$cptable[i - 1, "CP"]), 1)
```

Optimal CP: 0.002079368441975

```
# Pruning CART Model
cart_model_1SE <- prune(model1, cp = cp_opt)

# Print CP Table
printcp(cart_model_1SE)
```
![CART model part 1](https://github.com/user-attachments/assets/8b6f442b-1f36-4d64-8c48-537ca9f8092c)
![CART model part w](https://github.com/user-attachments/assets/b6806c72-f6fa-46ff-acf7-3c7569e00a20)

```
# Plot Pruned CP Plot
plotcp(cart_model_1SE)
```
![image](https://github.com/user-attachments/assets/8ad1b9dc-252d-4e08-931b-9d34f6a26c9f)

```
# Visualizing Pruned Decision Tree
model <- c(model, "CART 1SE")
rpart.plot(cart_model_1SE, nn = T, main = "Optimal Tree", cex = 0.4, tweak = 0.95, lwd = 3, faclen = 0)

RMSE_train <- c(RMSE_train, round(sqrt(mean((trainset$Concrete.CS - predict(cart_model_1SE))^2))))
RMSE_test <- c(RMSE_test, round(sqrt(mean((testset$Concrete.CS - predict(cart_model_1SE, newdata = testset))^2))))
```
![image](https://github.com/user-attachments/assets/1f45af7b-c88a-41b6-9296-61620b1f86f0)

### Random Forest
```
# Training the Random Forest Model
RF_model <- randomForest(Concrete.CS ~ . , data=trainset)

# Display Model Summary
RF_model
```
![image](https://github.com/user-attachments/assets/e1cb26b3-ff35-4a2c-89c7-9c84649855f1)

A Random Forest analysis was performed on the training set of the Concrete Compressive Strength dataset using 500 trees. The model resulted in a mean squared residual of 32.5234 and explained 88.17% of the variance.

```
# Computing Out-Of-Bag (OOB) RMSE
sqrt(RF_model$mse[RF_model$ntree])
```
OOB RMSE is an estimate of the model’s predictive error, computed from the bootstrap sampling used in Random Forest.

```
plot(RF_model)
```
![image](https://github.com/user-attachments/assets/4aff08ea-4317-45dd-b57d-13bacd352d04)

Error stablised before 500 trees.

```
model <- c(model, "Random Forest")

RMSE_train <- c(RMSE_train, round(sqrt(mean((trainset$Concrete.CS - predict(RF_model, newdata = trainset))^2))))
RMSE_test <- c(RMSE_test, round(sqrt(mean((testset$Concrete.CS - predict(RF_model, newdata = testset))^2))))

results <- data.frame(model, RMSE_train, RMSE_test)
View(results)
```
![image](https://github.com/user-attachments/assets/70c758dd-1e3c-489a-81e8-179293723702)

Random Forest has lowest testset error.

The Root Mean Square Error (RMSE) for both the training and test sets is highest for Linear Regression. CART 1SE has a lower RMSE than Linear Regression but higher than Random Forest. Random Forest achieves the lowest RMSE among the three models, indicating the best predictive performance. Therefore, Random Forest is the most suitable model as it produces the lowest error.
