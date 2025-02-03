# This project showcases both **Unsupervised** & **Supervised** Machine Learning techniques, including Regression & Classification.

## 1. Unsupervised Learning: Abalone Age Prediction

**Problem Description:** This showcase uses unsupervised machine learning to estimate the age of abalones by counting shell rings, similar to tree rings. Each ring typically represents one year of growth, but additional environmental data could further refine predictions.

**Introduction:** Abalones are marine mollusks valued for their meat and iridescent shells, and they contribute to ocean ecosystems by improving water quality through filter feeding. The dataset consists of eight independent variables: sex, length, diameter, height, whole weight, shucked weight, viscera weight, and shell weight, with the dependent variable being the number of rings (age).

**Impact:** Machine learning, helps regulate sustainable harvesting, supporting conservation efforts and preventing overexploitation, similar to practices in lobster fishing where size and weight are used to determine harvest eligibility.

---
```
# Summary of dataset
summary(abalone_df)
```
![Summary of dataset](https://github.com/user-attachments/assets/65ad76b6-b4c9-4d84-9c56-6e78983c3c89)
```
# Selecting all continuous variables
abalone <- select(abalone_df, c(2,3,4,5,6,7,8))

# Checking for Principal Component Analysis (PCA) eligibility
cor(abalone)
```
Correlation Matrix

![Correlation matrix](https://github.com/user-attachments/assets/05e783c8-630e-40bb-a11a-eafca41576a7)
```
# Finding mean correlation
mean(cor(abalone))
```
0.907 > 0.3
The dataset used in Abalone dataset which seeks to identify the age of abalones by the number of rings using physical measurements. With a mean correlation of 0.9, the abalone dataset is suitable to use PCA for unsupervised learning.

## Principal Component Analysis (PCA)
```
PCA <- princomp(abalone)

names(PCA)
```
7 loadings have been identified:

"sdev"     "loadings" "center"   "scale"    "n.obs"    "scores"   "call"  
```
# Summary of PCA
summary(PCA)
```
<img src="https://github.com/user-attachments/assets/bcaa735a-601f-432b-b610-68593b1573af" alt="image" width="1000"/>

The standard deviation of component 1 is larger as it encompasses the majority of the variance in the dataset at 58.15.
```
# Loadings of each Principal Components (PC) in matrix form
PCA$loadings
```
![image](https://github.com/user-attachments/assets/c3a899e4-901f-48ea-8933-590e86667993)

```
# Checking correlation between all Principal Components
PC <- PCA$scores
cor(PC)
```
All Correlations are close to zero, meainng the 7 PCs are uncorrelated to each other, and each captured a unique aspect of the data's variance without redundancy.
![image](https://github.com/user-attachments/assets/3ca458a4-9135-408d-b96a-9d2df09a2b2b)

```
# Scree plot
fviz_eig(PCA, addlabels = TRUE)
```
Scree Plot

![Scree Plot](https://github.com/user-attachments/assets/6ea1dd54-530e-45a7-b3c9-8be3110d70a3)

Figure 2 show that component 1 explains the majority of the variance in the dataset, 97.41%, with component 2 explaining 1.14% of the variance.

## In this analysis, components 1 and 2 is used with a cumulative variance explained of 98.55%. 
```
# Biplot
fviz_pca_biplot(PCA, label = "var", habillage = abalone_df$Rings, col.var = "black")
```
### PCA loading & Biplot

<div style="display: flex; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/c53d7810-6b2b-42bf-9b7f-d80ebae3747e" alt="image1" width="300"/>
  <img src="https://github.com/user-attachments/assets/8277d02e-ed6f-44b4-b955-9ec3fd2474fc" alt="image2" width="600"/>
</div>
PCA loading (left) and biplot (right)
 
According to PCA loading, component 1 captures 84.3% of the variable (whole weight), indicating a very strong influence. Whilst component 2 captures a strong negative influence of shucked weight and a moderate positive influence of shell weight. In the biplot, the majority of rings of the abalone measured lies between 7 to 10.

## K-Means Clustering
```
set.seed(2024)

abalone_df <- read.csv("C:/Users/nicho/Desktop/abalone.csv")

abalone <- select(abalone_df, c(2,3,4,5,6,7,8))

# Scale data
abalone_scale <- scale(abalone)

# WSS (Within Sum Squares) plot
fviz_nbclust(abalone_scale, kmeans, method = "wss") +
  labs(subtitle = "Within groups sum of squares")
```
WSS Plot

![WSS Plot](https://github.com/user-attachments/assets/fc7fc1c9-04ee-454b-92b8-40bfc050edee)

According to the WSS plot, there is a diminishing return after cluster 3. Therefore, the optimal number of clusters is 3.

```
# K-means cluster
KM <- kmeans(abalone_scale, centers = 3, nstart = 100)
print(KM)
```
K-Means Clustering analysis

![image](https://github.com/user-attachments/assets/8b68e1e0-d1f8-42dd-ad92-b3f073ec547d)

### Cluster sizes: 
```
Cluster 1: 1,764 
Cluster 2: 1,194 
Cluster 3: 1,219
```
### Within-Cluster Sum of Squares (WSS):
```
Cluster 1: 1,920.078
Cluster 2: 3,311.914
Cluster 3: 1,790.464
```
WSS shows the internal variability of each cluster, with cluster 2 having more variability than the other two clusters.

### Between-Cluster Sum of Squares / Total Sum of Squares (BSS / TSS) Ratio:

The ratio of 76.0% means that 76% of the total variance in the data is explained by the clustering, which indicates a reasonably good separation between clusters.

```
# Cluster plot
KM_cluster <- KM$cluster
rownames(abalone_scale) <- paste(abalone_df$Rings, 1:dim(abalone)[1], sep = "_")
autoplot(KM, abalone, frame = TRUE)
```
Cluster plot

![image](https://github.com/user-attachments/assets/761532a4-5f9b-4358-b410-5e33a3973e3b)

Analysing the 3 clusters, we can observe that cluster 1 (Red)'s mean ranges around 0. Cluster 2 ranges around 1 and cluster 3 ranges around -1. From this information, we can identify that the clusters are well segregated.

Cluster 1 (red) has a moderate number of outliers, primarily along the y-axis, indicating some variability in PC2. Cluster 2 (green) has a large number of data points with a wider spread in both PC1 and PC2, showing more pronounced outliers compared to the other clusters. Cluster 3 (blue) has the fewest outliers, though its outliers appear more extreme in distance from the cluster's main concentration, particularly along the y-axis. Overall, Cluster 3 appears more compact, while Cluster 2 shows the greatest spread and variability.

*******************************************************************************************************************************************************************
# Supervised Learning

## 2. Regression: Concrete Compressive Strength Prediction

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

# Summary of dataset
summary(concrete_df)
```
![Dataset summary](https://github.com/user-attachments/assets/c1563b01-54fa-4782-b8a3-d583a9581ed3)

```
# Correlation test
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

### Linear Regression Method
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
# Fitting linear model to trainset
model_train <- lm(Concrete.CS ~ ., data = trainset)
model <- ("Linear Regression")
RMSE_train <- round(sqrt(mean((trainset$Concrete.CS - predict(model_train))^2)))
RMSE_test <- round(sqrt(mean((testset$Concrete.CS - predict(model_train, newdata = testset))^2)))
```
Root Mean Squared Error (RMSE)

| Model Version | Train RMSE | Test RMSE |
|---------------|------------|-----------|
| Initial Model | 10         | 11        |

Evaluation: A higher RMSE in the test set compared to the training set suggests potential overfitting, as the model performs better on the training data than on unseen data. However, if the difference is small, it may still indicate reasonable generalization ability.

# Subset Selection: Identifying the Best Method for Prediction, based on Statistical Significance (p-value)
## Forward Selection Method
```
# Forward Selection on trainset
FWDfit_p_train <- ols_step_forward_p(model_train, details = TRUE)
```
![Forward Selection part 1](https://github.com/user-attachments/assets/2c5731b1-fd4a-4699-a57b-957a2cd82495)
![Forward Selection part 2](https://github.com/user-attachments/assets/02e94efc-2396-4964-91c9-d9435d370cd0)
![Forward Selection part 3](https://github.com/user-attachments/assets/282e9068-eea9-4a42-acb1-015dce0a6035)
![Forward Selection part 4](https://github.com/user-attachments/assets/bc9a01a3-02c7-48ae-b3ab-d35a0fc8f20c)
![Forward Selection part 5](https://github.com/user-attachments/assets/c3432f6c-4fa4-4d29-a323-1206c8d8120e)

```
# Prediction on forward selection testset
FWD_testset_predictions <- predict(model_train, newdata = testset)

# Evaluation of testset
FWD_rmse_test <- sqrt(mean((testset$Concrete.CS - FWD_testset_predictions)^2))
```
| Model Version           | Train RMSE | Test RMSE |
|-------------------------|------------|-----------|
| Initial Model           | 10         | 11        |
| After Forward Selection | 10         | 10.0135   |

Evaluation: After Forward Selection, the test RMSE reduced from 11 to 10.0135, indicating improved generalization by eliminating unnecessary features. Additionally, the new test RMSE is much closer to the training RMSE, suggesting a more balanced model with reduced overfitting. Overall, forward selection enhanced model performance by improving accuracy on the test set while maintaining a good fit on the training data, making the model more robust and generalizable.

## Backwards Elimination Method
```
# Backward Elimination on trainset
BWDfit_p_train <- ols_step_backward_p(model_train, details = TRUE)
```
![Backwards Elimination](https://github.com/user-attachments/assets/6dc9766c-f4bf-4d49-b7bd-d23b2cdfea92)

```
# Prediction on backward elimination testset
BWD_testset_predictions <- predict(model_train, newdata = testset)

# Evaluation of backward testset
BWD_rmse_test <- sqrt(mean((testset$Concrete.CS - BWD_testset_predictions)^2))
```

| Model Version               | Train RMSE | Test RMSE |
|-----------------------------|------------|-----------|
| Initial Model               | 10         | 11        |
| After Forward Selection     | 10         | 10.0135   |
| After Backwards Elimination | 10         | 10.0135   |

Evaluation: Both subset selection methods performed equally well in terms of Test RMSE.

## Best Subset Method
```
BSS_p_train <- ols_step_both_p(model_train, details = TRUE)
```
![Best Subset part 1](https://github.com/user-attachments/assets/20109a0e-a9a9-464c-ae2d-4b3ea122ed4b)
![Best Subset part 2](https://github.com/user-attachments/assets/70fbba18-bf45-4c04-9614-2a85ac74165d)

```
# Prediction on best subset testset
BSS_testset_predictions <- predict(model_train, newdata = testset)

# Evaluation of best subset testset
BSS_rmse_test <- sqrt(mean((testset$Concrete.CS - BSS_testset_predictions)^2))
```

| Model Version               | Train RMSE | Test RMSE |
|-----------------------------|------------|-----------|
| Initial Model               | 10         | 11        |
| After Forward Selection     | 10         | 10.0135   |
| After Backwards Elimination | 10         | 10.0135   |
| After Best Subset           | 10         | 10.0135   |

Evaluation: After using these 3 subset selection methods, all 3 have very similar predictive performance. None had a significant advantage in preictive accuracy.

# Comparing Model Selection Methods, based on Akaike Information Criterion (AIC)
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

# Classification And Regression Tree (CART)
## Building & Optimizing CART Tree
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
```
![image](https://github.com/user-attachments/assets/1f45af7b-c88a-41b6-9296-61620b1f86f0)

```
RMSE_train <- c(RMSE_train, round(sqrt(mean((trainset$Concrete.CS - predict(cart_model_1SE))^2))))
RMSE_test <- c(RMSE_test, round(sqrt(mean((testset$Concrete.CS - predict(cart_model_1SE, newdata = testset))^2))))
```

# Random Forest
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

## 3. Classification: Heart Disease Prediction

**Problem Description:** This showcase uses classification methods to predict the likelihood of heart disease based on various health indicators, such as age, cholesterol levels, blood pressure, and other key factors

**Introduction:** Cardiovascular disease is the leading cause of death globally, with nearly 1/3 of all deaths attributed to it. The dataset includes 11 independent variables: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise angina, old peak, and ST slope, with the dependent variable being the presence or absence of heart disease.

**Impact:** Through machine learning algorithms, early detection of heart disease can be streamlined, helping doctors and healthcare professionals identify at-risk patients and take timely preventive or remedial action. This could enhance healthcare outcomes and reduce the overall burden of cardiovascular disease.

### Logistic Regression
![image](https://github.com/user-attachments/assets/256a3083-ba5d-4411-ba86-81e6c37ad727)
![image](https://github.com/user-attachments/assets/c6cd38f1-b666-4f8b-ad9e-552d035eab2e)
**Fig.19 Logistic Regression model 1 (left) & Logistic Regression model 2 with manual removal (right)**
```
Logistic Regression 2 have variables (Age, RestingBP, RestingECGST, & MaxHR) removed due to p-value >0.05.
```
### Bootstrap
![image](https://github.com/user-attachments/assets/459858e8-9eed-48a8-a607-2ce5eb8a52e5)

**Fig.20 Bootstrap Logistic Regression Model**

According to the logistic regression with manual removal of non-statistically significant variables of age, restingBP, restingECG & maxHR in fig.20 (right), the Akaike Information Criterion (AIC) is 621.61. However, the bootstrap logistic regression removed restingECG, restingBP and maxHR, which was ¾ of the variables manually removed in the non-bootstrap logistic regression, with an AIC of 619.81. It appears that the variable age, deemed non-statistically significant in the standard logistic regression contributed to a better fitting model in the bootstrap logistic regression model.

![image](https://github.com/user-attachments/assets/756c6594-82e0-40b7-99e4-0e09c994e8d7)
![image](https://github.com/user-attachments/assets/f9861bc9-b8f3-44d2-9b31-2d3657409b29)

**Fig.21 Visualisation (left) and details (right) of confusion matrix**

Confusion matrix accuracy is 86.71%. The percentage of false positives is 13.29%.

### Lasso Regression Model

Using a Train-Test set split of 70:30 of the dataset.

```
   Df  %Dev   Lambda
1   0  0.00 0.301000
2   1  4.53 0.274300
3   1  8.29 0.249900
4   1 11.45 0.227700
5   1 14.09 0.207500
6   2 17.05 0.189100
7   2 19.73 0.172300
8   2 22.02 0.157000
9   2 23.98 0.143000
10  2 25.66 0.130300
11  3 27.19 0.118700
12  4 28.98 0.108200
13  6 30.77 0.098580
14  8 32.85 0.089820
15 10 34.99 0.081840
16 10 36.97 0.074570
17 10 38.70 0.067940
18 11 40.28 0.061910
19 11 41.67 0.056410
20 11 42.88 0.051400
21 11 43.95 0.046830
22 11 44.88 0.042670
23 11 45.69 0.038880
24 11 46.40 0.035430
25 12 47.11 0.032280
26 12 47.77 0.029410
27 12 48.34 0.026800
28 12 48.84 0.024420
29 12 49.27 0.022250
30 12 49.64 0.020270
31 12 49.96 0.018470
32 12 50.24 0.016830
33 13 50.50 0.015340
34 13 50.74 0.013970
35 13 50.93 0.012730
36 13 51.11 0.011600
37 13 51.25 0.010570
38 14 51.38 0.009631
39 14 51.49 0.008775
40 14 51.58 0.007996
41 14 51.66 0.007285
42 15 51.73 0.006638
43 15 51.80 0.006049
44 15 51.85 0.005511
45 15 51.90 0.005022
46 15 51.95 0.004575
47 15 51.98 0.004169
48 15 52.01 0.003799
49 15 52.04 0.003461
50 15 52.06 0.003154
51 15 52.07 0.002874
52 15 52.09 0.002618
53 16 52.10 0.002386
54 16 52.12 0.002174
55 16 52.13 0.001981
56 16 52.14 0.001805
57 16 52.14 0.001644
58 16 52.15 0.001498
59 16 52.15 0.001365
60 16 52.16 0.001244
61 16 52.16 0.001133
62 16 52.17 0.001033
63 16 52.17 0.000941
64 16 52.17 0.000857
65 16 52.17 0.000781
66 16 52.17 0.000712
67 16 52.17 0.000649
68 16 52.18 0.000591
69 16 52.18 0.000538
```
**Fig.22 Lasso Regularised Model**

After doing model selection, the logistic regression model is fitted with lasso regularisation, the degree of freedom is 16. The regularisation path included a range of lambda values, and the selected lambda of **0.004575469** minimises the cross-validated error. The model explained 52.18% of the deviance in the training data. Hence, a lambda of 0.0004575469 will result in a test set accuracy of 87.64%.

### Results
![image](https://github.com/user-attachments/assets/c86f7ab4-cbe9-4878-b8bd-39212ab03678)

**Table 1**

In the 1st standard logistic regression, the AIC is 626.19. The 2nd standard logistic regression with manual removal of non-statistically significant variables AIC is 621.61. While the bootstrap logistic regression AIC is 619.81. As the bootstrap model has the lowest AIC, it is considered the best of the 3 models. Providing better trade-off between goodness of fit and complexity.

The accuracy from the confusion matrix from the 2nd standard logistic regression with manual removal is 86.71%. However, the testset with lasso regularization accuracy is 87.64. Meaning the lasso regularization is the better choice in predicting new unseen data.
