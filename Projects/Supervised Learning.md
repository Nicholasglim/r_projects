# Supervised Learning

## 1. Regression: Concrete Compressive Strength Prediction

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

## Linear Regression Model
Using a Train-Test split of 70:30
```
set.seed(2024)
# Train-Test set
train <- sample.split(Y = concrete_df$Concrete.CS, SplitRatio = 0.7)
trainset <- subset(concrete_df, train==T)
testset <- subset(concrete_df, train==F)

summary(trainset$Concrete.CS)
summary(testset$Concrete.CS)
```
Summary of trainset (top) & testset (bottom)

![Summary of Train/Test sets](https://github.com/user-attachments/assets/6b28ddf1-0863-4b27-a0b9-9edd4769186a)

```
# Fitting Linear RTegression Model to Trainset
model_train <- lm(Concrete.CS ~ ., data = trainset)
model <- ("Linear Regression")

# Evaluation of Linear Regression Model Trainset
RMSE_train <- round(sqrt(mean((trainset$Concrete.CS - predict(model_train))^2)))

# Evaluation of Linear Regression Model Testset
RMSE_test <- round(sqrt(mean((testset$Concrete.CS - predict(model_train, newdata = testset))^2)))
```
Root Mean Squared Error (RMSE)

| Model Version | Train RMSE | Test RMSE |
|---------------|------------|-----------|
| Initial Model | 10         | 11        |

Evaluation: A higher RMSE in the test set compared to the training set suggests potential overfitting, as the model performs better on the training data than on unseen data. However, if the difference is small, it may still indicate reasonable generalization ability.

## Comparing Subset Selection Method for Prediction using RMSE
### Forward Selection Method
```
# Forward Selection on Trainset
FWDfit_p_train <- ols_step_forward_p(model_train, details = TRUE)
```
![Forward Selection part 1](https://github.com/user-attachments/assets/2c5731b1-fd4a-4699-a57b-957a2cd82495)
![Forward Selection part 2](https://github.com/user-attachments/assets/02e94efc-2396-4964-91c9-d9435d370cd0)
![Forward Selection part 3](https://github.com/user-attachments/assets/282e9068-eea9-4a42-acb1-015dce0a6035)
![Forward Selection part 4](https://github.com/user-attachments/assets/bc9a01a3-02c7-48ae-b3ab-d35a0fc8f20c)
![Forward Selection part 5](https://github.com/user-attachments/assets/c3432f6c-4fa4-4d29-a323-1206c8d8120e)

```
# Prediction on Forward Selection Testset
FWD_testset_predictions <- predict(model_train, newdata = testset)

# Evaluation of Forward Selection Testset
FWD_rmse_test <- sqrt(mean((testset$Concrete.CS - FWD_testset_predictions)^2))
```
| Model Version           | Train RMSE | Test RMSE |
|-------------------------|------------|-----------|
| Initial Model           | 10         | 11        |
| After Forward Selection | 10         | 10.0135   |

Evaluation: After Forward Selection, the test RMSE reduced from 11 to 10.0135, indicating improved generalization by eliminating unnecessary features. Additionally, the new test RMSE is much closer to the training RMSE, suggesting a more balanced model with reduced overfitting. Overall, forward selection enhanced model performance by improving accuracy on the test set while maintaining a good fit on the training data, making the model more robust and generalizable.

### Backwards Elimination Method
```
# Backward Elimination on Trainset
BWDfit_p_train <- ols_step_backward_p(model_train, details = TRUE)
```
![Backwards Elimination](https://github.com/user-attachments/assets/6dc9766c-f4bf-4d49-b7bd-d23b2cdfea92)

```
# Prediction on Backward Elimination Testset
BWD_testset_predictions <- predict(model_train, newdata = testset)

# Evaluation of Backward Elimination Testset
BWD_rmse_test <- sqrt(mean((testset$Concrete.CS - BWD_testset_predictions)^2))
```

| Model Version               | Train RMSE | Test RMSE |
|-----------------------------|------------|-----------|
| Initial Model               | 10         | 11        |
| After Forward Selection     | 10         | 10.0135   |
| After Backwards Elimination | 10         | 10.0135   |

Evaluation: Both subset selection methods performed equally well in terms of Test RMSE.

### Best Subset Method
```
BSS_p_train <- ols_step_both_p(model_train, details = TRUE)
```
![Best Subset part 1](https://github.com/user-attachments/assets/20109a0e-a9a9-464c-ae2d-4b3ea122ed4b)
![Best Subset part 2](https://github.com/user-attachments/assets/70fbba18-bf45-4c04-9614-2a85ac74165d)

```
# Prediction on Best Subset Testset
BSS_testset_predictions <- predict(model_train, newdata = testset)

# Evaluation of Best Subset Testset
BSS_rmse_test <- sqrt(mean((testset$Concrete.CS - BSS_testset_predictions)^2))
```

| Model Version               | Train RMSE | Test RMSE |
|-----------------------------|------------|-----------|
| Initial Model               | 10         | 11        |
| After Forward Selection     | 10         | 10.0135   |
| After Backwards Elimination | 10         | 10.0135   |
| After Best Subset           | 10         | 10.0135   |

Evaluation: After using these 3 subset selection methods, all 3 have very similar predictive performance. None had a significant advantage in preictive accuracy.

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
