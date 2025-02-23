# Regression: Concrete Compressive Strength Prediction

**Problem Description:** This showcase uses regression techniques to predict concrete compressive strength based on various components in the concrete mixture, such as cement, water, slag, and aggregates, as well as the age of the concrete.

**Introduction:** Concrete is a fundamental material in construction, essential in buildings, bridges, and even historical landmarks. The dataset consists of the dependent variable: concrete compressive strength, and eight independent variables: cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age. Compressive strength of concrete is a non-linear function of both the age and the constituent ingredients. During hydration, the mixture reacts with water, gaining strength rapidly at first, then gradually over time.

**Impact:** By using machine learning, the formulation of concrete can be optimized with data-driven insights, enhancing strength and durability, and overall performance. This approach not only reduces costs and minimizes environmental impact but also improves construction practices, going beyond traditional testing methods like destructive and non-destructive testing to provide more efficient and sustainable solutions.

---
```
# Install required packages
install.packages("rBayesianOptimization")
install.packages("randomForest")
install.packages("caret")
install.packages("tidyverse")
install.packages("corrplot")

library(rBayesianOptimization)
library(randomForest)
library(caret)
library(tidyverse)
library(corrplot)

# Load Data
concrete_df <- read.csv("https://raw.githubusercontent.com/Nicholasglim/r_project/main/Datasets/Concrete_Data.csv", stringsAsFactors = TRUE)

# Rename Columns
colnames(concrete_df) <- c("Cement", "BF.Slag", "Fly.Ash", "Water", "Superplasticizer", "Coarse.Agg", "Fine.Agg", "Age", "Concrete.CS")

# Remove rows where Age < 28 days (As Concrete reaches 99% strength at 28 days)
concrete_df <- concrete_df %>% filter(Age >= 28)
```

## Data Exploration
```
# Correlation Matrix
pearson_corr <- cor(concrete_df, method = "pearson")
corrplot(pearson_corr, type = "upper", method = "pie", main = "Correlation Matrix")
```
Correlation Plot

![Correlation Plot](https://github.com/user-attachments/assets/af40012c-7055-4c4b-b97e-ff1d0bba2ff2)

```
# Scatterplot Matrix
panel.smooth <- function(x, y) {
  points(x, y)
  abline(lm(y ~ x), col = "blue")
}
pairs(~ . , panel=panel.smooth, data = concrete_df, main = "Scatterplot Matrix of Concrete Strength data")
```
Scatter Plot

![Scatter Plot](https://github.com/user-attachments/assets/56c91282-e6f5-4c6e-b7b8-3ba19411f9f9)

## Finding the Strongest Concrete mix
```
best_index <- which.max(concrete_df$Concrete.CS)  # Find row with max strength
best_mix <- concrete_df[best_index, ]  # Extract best mix
print(best_mix)
```

|        | Cement | BF.Slag | Fly.Ash | Water | Superplaticizer | Coarse.Agg | Fine.Agg | Age | Concrete.CS |
|--------|--------|---------|---------|-------|-----------------|------------|----------|-----|-------------|
| 125    | 389.9  | 189     |       0 | 145.9 |              22 |      944.7 |    755.8 |  91 |        82.6 |

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
| 1    | 49.64414 |
| 2    | 34.80255 |
| 4    | 32.25818 |
| 8    | 30.92297 |

![Line Graph](https://github.com/user-attachments/assets/baa29fc6-655c-4d27-8f28-2f9403f4fab3)

Interpretation: A lower OOB error indicates how well the Random Forest model generalizes to unseen data. Thus, mtry = 8 is chosen.

```
# Manually set best_mtry based on the output of tuneRF
best_mtry <- 8

# Random Forest
rf_model <- randomForest(Concrete.CS ~ ., data = trainset, ntree = 500, mtry = best_mtry)

# Model Performance
rf_predictions <- predict(rf_model, newdata = testset)
rf_rmse <- sqrt(mean((testset$Concrete.CS - rf_predictions)^2))
cat("Random Forest RMSE on Test Data:", rf_rmse, "\n")
```

Random Forest RMSE on Test Data: 5.172754

```
# Calculate R2 for evaluation
actuals <- testset$Concrete.CS
predictions <- predict(rf_model, newdata = testset)
RF_R2 <- 1 - sum((actuals - predictions)^2) / sum((actuals - mean(actuals))^2)
cat("Random Forest R-squared (R2) value:", RF_R2, "\n")
```

Random Forest R-squared (R2) value: 0.8910326

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

| Component        | Best Actual Mix | Bayesian Predicted Mix |
|------------------|-----------------|------------------------|
| Cement           | 389.9 kg/m^3    | 509.2 kg/m^3           |
| BF.Slag          | 189 kg/m^3      | 359.4 kg/m^3           |
| Fly.Ash          | 0 kg/m^3        | 1.59 kg/m^3            |
| Water            | 145.9 kg/m^3    | 121.8 kg/m^3           |
| Superplasticizer | 22	kg/m^3       | 32.2 kg/m^3            |
| Coarse.Agg       | 944.7 kg/m^3    | 982.9 kg/m^3           |
| Fine.Agg         | 755.8 kg/m^3    | 594 kg/m^3             |
| Age              | 91	days         | 188 days               |
| Concrete.CS	     | 82.6 mPa	       | 74.06 mPa              |

## Visualization
```
# Convert bayesian_optimised_df to wide format
bayesian_optimised_wide <- bayesian_optimised_df %>%
  select(Component, Value) %>%
  pivot_wider(names_from = Component, values_from = Value)

# Add Type label
bayesian_optimised_wide$Type <- "Bayesian Optimized Mix"

# Ensure best_actual_mix_df is structured similarly
best_actual_mix_df$Type <- "Best Actual Mix"

# Combine both datasets
comparison_df <- bind_rows(best_actual_mix_df, bayesian_optimised_wide)

# Reshape data for ggplot
comparison_df_long <- comparison_df %>%
  select(-Concrete.CS) %>%  # Remove strength column
  pivot_longer(cols = -Type, names_to = "Component", values_to = "Amount")

# Create bar chart with value labels
ggplot(comparison_df_long, aes(x = Component, y = Amount, fill = Type)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  geom_text(aes(label = round(Amount, 1)),  # Round values for readability
            position = position_dodge(width = 0.8), 
            vjust = -0.3, 
            size = 5, 
            fontface = "bold") +  # Make numbers bold
  labs(title = "Comparison of Best Actual Mix vs Bayesian Optimized Mix",
       x = "Concrete Mix Component",
       y = "Amount (kg/m³)",
       fill = "Mix Type") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

Best Actual vs Predicted Mix Grouped Bar Chart

![Bar Chart](https://github.com/user-attachments/assets/a1f993d7-ea85-4836-912e-a23a70a2f376)

**Component Comparison**

Age: The Bayesian Optimized Mix suggests a significantly longer curing time (188 days) compared to the Best Actual Mix (91 days).

BF.Slag (Blast Furnace Slag): The optimized mix recommends a considerably higher amount of BF.Slag (359.4 kg/m³) compared to the best actual mix (189 kg/m³).

Cement: The Bayesian Optimization suggests a higher quantity of Cement (509.2 kg/m³) than the Best Actual Mix (389.9 kg/m³).

Coarse.Agg (Coarse Aggregate): The optimized mix indicates a slightly higher amount of Coarse Aggregate at 982.9 kg/m³, compared to the best actual mix at 944.7 kg/m³.

Fine.Agg (Fine Aggregate): The optimized mix uses a smaller amount of Fine Aggregate (594 kg/m³) compared to the Best Actual Mix (755.8 kg/m³).

Fly.Ash: The Bayesian Optimized Mix proposes a slight amount of Fly Ash (1.59 kg/m³), while the Best Actual Mix has 0 kg/m³.

Superplasticizer: The optimized mix requires a slightly larger amount of Superplasticizer (32.2 kg/m³) than the Best Actual Mix (22 kg/m³).

Water: The optimized mix recommends slightly less water (121.8 kg/m³) compared to the best actual mix (145.9 kg/m³).
