# Regression: Concrete Compressive Strength Prediction

**Problem Description:** This showcase uses regression techniques to predict concrete compressive strength based on various components in the concrete mixture, such as cement, water, slag, and aggregates, as well as the age of the concrete.

**Introduction:** Concrete is a fundamental material in construction, essential in buildings, bridges, and even historical landmarks. The dataset consists of the dependent variable: concrete compressive strength, and eight independent variables: cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age. Compressive strength of concrete is a non-linear function of both the age and the constituent ingredients. During hydration, the mixture reacts with water, gaining strength rapidly at first, then gradually over time.

**Impact:** By using machine learning, the formulation of concrete can be optimized with data-driven insights, enhancing strength and durability, and overall performance. This approach not only reduces costs and minimizes environmental impact but also improves construction practices, going beyond traditional testing methods like destructive and non-destructive testing to provide more efficient and sustainable solutions.

---
```
install.packages("rBayesianOptimization")
library(rBayesianOptimization)
library(randomForest)
library(caret)
library(tidyverse)
library(corrplot)

# Load Data
concrete_df <- read.csv("https://raw.githubusercontent.com/Nicholasglim/r_project/main/Datasets/Concrete_Data.csv", stringsAsFactors = TRUE)

# Rename Columns
colnames(concrete_df) <- c("Cement", "BF.Slag", "Fly.Ash", "Water", "Superplasticizer", "Coarse.Agg", "Fine.Agg", "Age", "Concrete.CS")
```

## Data Exploration
```
# Correlation Matrix
pearson_corr <- cor(concrete_df, method = "pearson")
corrplot(pearson_corr, type = "upper", method = "pie", main = "Correlation Matrix")
```
Correlation Plot

![Correlation Plot](https://github.com/user-attachments/assets/989166b7-84dd-4783-b200-4ae1f03124b1)

```
# Scatterplot Matrix
panel.smooth <- function(x, y) {
  points(x, y)
  abline(lm(y ~ x), col = "blue")
}
pairs(~ . , panel=panel.smooth, data = concrete_df, main = "Scatterplot Matrix of Concrete Strength data")
```
Scatter Plot

![Scatterplot](https://github.com/user-attachments/assets/83c631d1-2bec-45d8-8656-adc54d75e765)


## Finding the Strongest Concrete mix
```
best_index <- which.max(concrete_df$Concrete.CS)  # Find row with max strength
best_mix <- concrete_df[best_index, ]  # Extract best mix
print(best_mix)
```

|        | Cement | BF.Slag | Fly.Ash | Water | Superplaticizer | Coarse.Agg | Fine.Agg | Age | Concrete.CS |
|--------|--------|---------|---------|-------|-----------------|------------|----------|-----|-------------|
| 182    | 389.9  | 189     |       0 | 145.9 |              22 |      944.7 |    755.8 |  91 |        82.6 |

## Train Random Forest Model
```
set.seed(2025)
train_index <- createDataPartition(concrete_df$Concrete.CS, p = 0.7, list = FALSE)
trainset <- concrete_df[train_index, ]
testset <- concrete_df[-train_index, ]

# Find optimised mtry to avoid weak models/overfitting
tune_results <- tuneRF(trainset[,-9], trainset$Concrete.CS, stepFactor=2, improve=0.07, trace=FALSE)
print(tune_results)
```
OOB (Out-Of-Bag) Error Rate vs. mtry Line Graph

| mtry | OOBError |
|------|----------|
| 1    | 58.70796 |
| 2    | 34.91938 |
| 4    | 25.43088 |
| 8    | 25.58883 |

![Line Graph](https://github.com/user-attachments/assets/33917b65-a26a-4d8b-8598-9d6a296d045b)

Interpretation: mtry = 8 has a higher OOB error than mtry = 4. Thus, mtry = 4 is chosen. A lower OOB error indicates how well the Random Forest model generalizes to unseen data.

```
# Manually set best_mtry based on the output of tuneRF
best_mtry <- 4

# Random Forest
rf_model <- randomForest(Concrete.CS ~ ., data = trainset, ntree = 500, mtry = best_mtry)

# Model Performance
rf_predictions <- predict(rf_model, newdata = testset)
rf_rmse <- sqrt(mean((testset$Concrete.CS - rf_predictions)^2))
cat("Random Forest RMSE on Test Data:", rf_rmse, "\n")
```

Random Forest RMSE on Test Data: 5.676416

```
# Calculate R2 for evaluation
actuals <- testset$Concrete.CS
predictions <- predict(rf_model, newdata = testset)
RF_R2 <- 1 - sum((actuals - predictions)^2) / sum((actuals - mean(actuals))^2)
cat("Random Forest R-squared (R2) value:", RF_R2, "\n")
```

Random Forest R-squared (R2) value: 0.8845187

## Find the Best Material Mix using Bayesian Optimization
```
BayesSearch <- BayesianOptimization(
  FUN = function(Cement, BF.Slag, Fly.Ash, Water, Superplasticizer, Coarse.Agg, Fine.Agg, Age) {
    new_data <- data.frame(
      Cement = Cement, BF.Slag = BF.Slag, Fly.Ash = Fly.Ash, Water = Water,
      Superplasticizer = Superplasticizer, Coarse.Agg = Coarse.Agg, Fine.Agg = Fine.Agg, Age = Age
    )
    pred_strength <- predict(rf_model, new_data)
    return(list(Score = pred_strength))  # Maximize predicted compressive strength
  },
  bounds = list(
    Cement = range(concrete_df$Cement),
    BF.Slag = range(concrete_df$BF.Slag),
    Fly.Ash = range(concrete_df$Fly.Ash),
    Water = range(concrete_df$Water),
    Superplasticizer = range(concrete_df$Superplasticizer),
    Coarse.Agg = range(concrete_df$Coarse.Agg),
    Fine.Agg = range(concrete_df$Fine.Agg),
    Age = range(concrete_df$Age)
  ),
  init_points = 10,  # Start with 10 random points
  n_iter = 20,       # Perform 20 optimization steps
  verbose = TRUE
)

# Extract the best mix from Bayesian Optimization
best_predicted_mix <- BayesSearch$Best_Par
print(best_predicted_mix)
```
Best Parameters Found:

Cement = 540.0000 kg/m^3 

BF.Slag = 359.4000 kg/m^3 

Fly.Ash = 1.592888 kg/m^3 

Water = 121.8000 kg/m^3 

Superplasticizer = 30.44815 kg/m^3 

Coarse.Agg = 801.0000 kg/m^3 

Fine.Agg = 665.4771 kg/m^3 

Age = 223.6235 days

Predicted Compressive Strength = 71.33776 MPa, megapascals

Note: Only 20 iterations have been carried out, due to harware limitations.

## Visualization
```
# Convert from wide to long format
best_mix_long <- pivot_longer(best_mix, cols = -Concrete.CS, names_to = "Component", values_to = "Value")

# Convert best predicted mix to data frame format
best_predicted_df <- data.frame(
  Component = names(best_predicted_mix),
  Value = as.numeric(best_predicted_mix)
)

# Add a Predicted_CS column
best_predicted_df$Predicted_CS <- predict(rf_model, newdata = as.data.frame(t(best_predicted_mix)))

# Changing best_predicted_df to best_predicted_long
best_predicted_long <- best_predicted_df

best_mix_long$Type <- "Best Actual"
best_predicted_long$Type <- "Best Predicted"

combined_data <- bind_rows(best_mix_long, best_predicted_long)

# Plot the best material mix
ggplot(combined_data, aes(x = Component, y = Value, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Comparison of Best Actual vs Predicted Mix",
       x = "Concrete Component", y = "Amount (kg/m³)",
       fill = "Mix Type") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

Best Actual vs Predicted Mix Grouped Bar Chart

![Bar Chart](https://github.com/user-attachments/assets/2350311a-2705-4cda-8174-74cd4838031b)

**Component Comparison**

Age: The model suggests more than double the curing time than in Best Actual Mix.

BF.Slag: The model suggests almost twice the amount of Blast Furnance Slag, compared to the Best Actual Mix.

Cement: The model suggests a higher amount of Cement than the Best Actual Mix.

Coarse.Agg: The model suggests a large decrease in the amount of Coarse Aggregate compared to the Best Actual Mix.

Fine.Agg: The model suggests a decrease in Fine Aggregate as well, but not as drastic as the decrease in Coarse Aggregate.

Fly.Ash: The model almost eliminates Fly Ash.

Superplasticizer: The model recommends a slightly higher amount of Superplasticizer.

Water: The model suggests a decrease in the amount of Water than in the Best Actual Mix.

| Component        | Best Actual | Best Predicted |
|------------------|-------------|----------------|
| Cement           | 389.9       |	540           |
| BF.Slag          | 189         |	359.4         |
| Fly.Ash          | 0           | 1.592888       |
| Water            | 145.9       | 121.8          |
| Superplasticizer | 22	         | 30.448149      |
| Coarse.Agg       | 944.7       | 801            |
| Fine.Agg         | 755.8       | 665.477096     |
| Age              | 91	         | 223.623501     |
| Concrete.CS	     | 82.6	       | 71.33776       |
