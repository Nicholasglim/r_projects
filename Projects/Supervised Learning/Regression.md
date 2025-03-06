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

|        | Cement | BF.Slag | Fly.Ash | Water  | Superplaticizer | Coarse.Agg | Fine.Agg | Penalty Value |
|--------|--------|---------|---------|--------|-----------------|------------|----------|---------------|
| M30    | 389.9  | 189     |    0.00 | 145.9  |              22 |      944.7 |    755.8 | -1.93         |
| M40    | 102.00 | 342.16  |    0.00 | 158.85 |            0.00 |     983.48 |   751.98 | -1.16         |
| M45    | 347.79 | 359.40  |    0.00 | 202.27 |            0.00 |     801.00 |   620.31 | -3.47         |
| M50    | 256.17 | 338.02  |    0.00 | 121.80 |           12.23 |    1143.02 |   796.36 | -2.84         |
| M55    | 540.00 | 359.40  |    0.00 | 191.56 |           29.24 |    1118.39 |   857.53 | -2.66         |
| M60    | 474.03 | 30.56   |    0.00 | 150.27 |            3.37 |    1043.88 |   992.60 | -2.13         |

|     | Cement | Fine.Agg | Coarse.Agg |
|-----|--------|----------|------------|
| M30 | 1      | 8.35     | 11.23      |
| M40 | 1      | 7.37     | 9.64       |
| M45 | 1      | 1.78     | 2.3        |
| M50 | 1      | 3.11     | 4.46       |
| M55 | 1      | 1.59     | 2.07       |
| M60 | 1      | 2.09     | 2.2        |
