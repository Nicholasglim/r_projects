# Proposing Data-Driven Concrete Design Mix Ratios Using Regression

**Problem Description:**
This project applies regression techniques to analyze concrete mix components and proposes a data-driven mix ratio based on observed relationships in the dataset. As there are no predefined design mixes for M30 and above, the goal is to identify suitable ratio for cement : fine aggregates : coarse aggregates, and other materials that can achieve desired compressive strength benchmarks.

**Introduction:**
Concrete is a fundamental construction material used in buildings, bridges, and historical landmarks. The dataset includes the dependent variable—concrete compressive strength—and eight independent variables: cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age. Compressive strength is a non-linear function of both age and constituent materials. By analyzing the dataset, mix ratios can be proposed that aligns with strength development trends observed in real-world data.

**Impact:**
This approach provides an additional way to determine concrete design mixes without relying solely on established empirical methods. By utilizing machine learning and regression analysis, new mix ratios can be proposed based on actual data rather than predefined standards. This can help in exploring innovative design mixes, improving material efficiency, and offering insights into how different mix components contribute to strength development.

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

# Filter data to include only ages 7 days and above
concrete_df <- concrete_df %>% filter(Age >= 7)
```

The age filter is set at 7 days and above, as compressive strength tests are commonly conducted on days 7 and 28.

## Data Exploration
```
# Correlation Matrix
pearson_corr <- cor(concrete_df, method = "pearson")
corrplot(pearson_corr, type = "upper", method = "pie", main = "Correlation Matrix")
```
Correlation Plot

![Correlation Plot](https://github.com/user-attachments/assets/4b6cbfab-37cb-48dc-8c79-0d85d5875161)

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

```
# Target Strengths
target_strengths <- c(30, 40, 45, 50, 55, 60)  # MPa

# Function to Run Bayesian Optimization for a given target strength
optimize_concrete_mix <- function(target_strength) {
  
  # Split data into train and test
  set.seed(2025)
  train_index <- createDataPartition(concrete_df$Concrete.CS, p = 0.7, list = FALSE)
  trainset <- concrete_df[train_index, ]
  testset <- concrete_df[-train_index, ]
  
  # Train Random Forest Models for 7-day and 28-day strength
  rf_model_7d <- randomForest(
    Concrete.CS ~ Cement + BF.Slag + Fly.Ash + Water + Superplasticizer + Coarse.Agg + Fine.Agg + Age,
    data = trainset %>% filter(Age == 7)
  )
  
  rf_model_28d <- randomForest(
    Concrete.CS ~ Cement + BF.Slag + Fly.Ash + Water + Superplasticizer + Coarse.Agg + Fine.Agg + Age,
    data = trainset %>% filter(Age == 28)
  )
  
  # Define Bayesian Optimization Function
  Bayesian_Objective <- function(Cement, BF.Slag, Fly.Ash, Water, Superplasticizer, Coarse.Agg, Fine.Agg) {
    new_data_7d <- data.frame(Cement, BF.Slag, Fly.Ash, Water, Superplasticizer, Coarse.Agg, Fine.Agg, Age = 7)
    new_data_28d <- data.frame(Cement, BF.Slag, Fly.Ash, Water, Superplasticizer, Coarse.Agg, Fine.Agg, Age = 28)
    
    # Predict the strength for 7-day and 28-day models
    pred_7d <- predict(rf_model_7d, new_data_7d)
    pred_28d <- predict(rf_model_28d, new_data_28d)
    
    # Define objective function to meet strength criteria
    target_7d <- 0.65 * target_strength # 65% of target at 7 days
    penalty <- abs(pred_28d - target_strength) + abs(pred_7d - target_7d) # Penalize deviation from target strengths
    return(list(Score = -penalty)) # Minimize penalty
  }
  
  # Run Bayesian Optimization
  BayesSearch <- BayesianOptimization(
    FUN = Bayesian_Objective,
    bounds = list(
      Cement = range(concrete_df$Cement),
      BF.Slag = range(concrete_df$BF.Slag),
      Fly.Ash = range(concrete_df$Fly.Ash),
      Water = range(concrete_df$Water),
      Superplasticizer = range(concrete_df$Superplasticizer),
      Coarse.Agg = range(concrete_df$Coarse.Agg),
      Fine.Agg = range(concrete_df$Fine.Agg)
    ),
    init_points = 10,  # Number of random exploration points
    n_iter = 20,       # Number of iterations
    verbose = FALSE # Suppress verbose output within the loop
  )
  
  # Extract and round optimal mix values
  optimal_mix_rounded <- round(BayesSearch$Best_Par, 2)
  
  # Extract Cement, Fine Aggregate, and Coarse Aggregate
  cement <- optimal_mix_rounded["Cement"]
  fine_agg <- optimal_mix_rounded["Fine.Agg"]
  coarse_agg <- optimal_mix_rounded["Coarse.Agg"]
  
  # Calculate and print ratio
  ratio <- c(cement, fine_agg, coarse_agg) / min(cement, fine_agg, coarse_agg)
  cat("For Target Strength ", target_strength, " MPa: Ratio of Cement: Fine Aggregate: Coarse Aggregate is ",
      paste(round(ratio, 2), collapse = " : "), "\n")
  
  return(optimal_mix_rounded) # Return the optimal mix
}

# Loop through target strengths and optimize
optimal_mixes <- list() # Store optimal mixes for each target strength
for (strength in target_strengths) {
  cat("Optimizing for target strength:", strength, "MPa\n")
  optimal_mixes[[as.character(strength)]] <- optimize_concrete_mix(strength)
}

print(optimal_mixes)  # Print the list of optimal mixes
```

## Optimal Mix

|        | Cement | BF.Slag | Fly.Ash | Water  | Superplaticizer | Coarse.Agg | Fine.Agg | Penalty Value |
|--------|--------|---------|---------|--------|-----------------|------------|----------|---------------|
| M30    | 102.00 | 359.40  |   83.10 | 247.00 |            0.00 |    1145.00 |   852.07 | -1.93         |
| M40    | 102.00 | 342.16  |    0.00 | 158.85 |            0.00 |     983.48 |   751.98 | -1.16         |
| M45    | 347.79 | 359.40  |    0.00 | 202.27 |            0.00 |     801.00 |   620.31 | -3.47         |
| M50    | 256.17 | 338.02  |    0.00 | 121.80 |           12.23 |    1143.02 |   796.36 | -2.84         |
| M55    | 540.00 | 359.40  |    0.00 | 191.56 |           29.24 |    1118.39 |   857.53 | -2.66         |
| M60    | 474.03 | 30.56   |    0.00 | 150.27 |            3.37 |    1043.88 |   992.60 | -2.13         |

**Bayesian optimization was conducted to achieve 65% of the target compressive strength at 7 days and 99% at 28 days.**

The penalty values in the table indicate the deviation of the predicted strengths at 7 and 28 days from the expected 65% and 99% of the target strength, respectively. Since Bayesian optimization aims to minimize this penalty, a lower value suggests that the predictions are closer to the target.

### Interpretation of Penalty Values:

- M40 has the lowest penalty (-1.16)

This indicates that the optimized mix for 40 MPa is the closest to meeting the target strengths at both 7 and 28 days. The mix likely has a well-balanced proportion of materials leading to minimal deviation.

- M45 has the highest penalty (-3.47)

This suggests a more significant deviation from the expected strength values. The mix may require adjustments, such as increasing Cement or optimizing Water and Superplasticizer content, to better match the target.

- M50, M55, and M60 have moderate penalties (-2.84, -2.66, -2.12)

These penalties indicate some deviation but are better than M45. The higher grades (M55 and M60) likely require fine-tuning, particularly in water-cement ratios or superplasticizer usage, to minimize the gap.

- M30 has a penalty of -1.93

While relatively low, it suggests a slight deviation from the target strengths. The Fly Ash content may be influencing early strength development.

## Design Mix Proportion Ratio

|     | Cement | Fine.Agg | Coarse.Agg |
|-----|--------|----------|------------|
| M30 | 1      | 8.35     | 11.23      |
| M40 | 1      | 7.37     | 9.64       |
| M45 | 1      | 1.78     | 2.3        |
| M50 | 1      | 3.11     | 4.46       |
| M55 | 1      | 1.59     | 2.07       |
| M60 | 1      | 2.09     | 2.2        |

---
# Conclusion & Disclaimer
The results obtained in this project are based on statistical modeling and machine learning optimization techniques. While Bayesian Optimization provides insights into potential mix proportions for achieving target concrete compressive strengths, **these findings should not be used as actual design mix recommendations for construction**. Real-world concrete performance is influenced by various external factors, including temperature, humidity, curing conditions, and material variability, which are not fully accounted for in this dataset.

Further experimental validation and field testing under diverse environmental conditions are necessary to refine these predictions. For practical applications in structural engineering, mix designs should always be developed in accordance with established industry standards, building codes, and laboratory testing protocols to ensure safety, durability, and compliance with regulatory requirements.
