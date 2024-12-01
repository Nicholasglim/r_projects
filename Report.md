# This project showcases both Unsupervised & Supervised Machine Learning techniques, including Regression & Classification.

The first showcase uses Unsupervised Machine Learning techniques to determinine the age of abalones by counting the number of rings, similar to how tree rings indicate age. Each ring typically represents one year. However, due to the smaller size of abalone rings, they must be stained with dye to make them visible under a microscope. However, further information such as weather patterns and location may be needed to obtain more accurate results

The second showcase uses regression techniques to evaluate concrete compressive strength under various combinations of concrete mixtures.

The third showcase uses classification methods to detect the presence of heart disease, using 11 indicators.

---
List of datasets along with description
---
Dataset: https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset

### Introduction
Abalone is a type of marine mollusk that are prized for their meat and iridescent shells in many cultures. Additionally, as a marine mollusk, abalones play a vital role of cleaning the ocean and improving water quality through the process of filter feeding.

The dataset consists of the Dependent Variable of number of rings (indicating age) and 8 Independent Variables of sex, length, diameter, height, whole weight, shucked weight, viscera weight and shell weight.

Through the analysis of machine learning, a regulation about abalone harvesting can be backed up by statistics and data to ensure the sustainability of abalone and prevent overexploitation. Similarly to lobster fishing, the dimensions and weight is used to assess suitability of harvest.

### Principal Component Analysis (PCA)
The dataset used is Abalone dataset which seeks to identify the age of abalones by the number of rings using physical measurements. With a mean correlation of 0.9, the abalone dataset is suitable to use PCA for unsupervised learning.
<img src="https://github.com/user-attachments/assets/bcaa735a-601f-432b-b610-68593b1573af" alt="image" width="1000"/>
**Fig.1 Principal Component Analysis**

### Scree plot
![image](https://github.com/user-attachments/assets/6ea1dd54-530e-45a7-b3c9-8be3110d70a3)

**Fig.2 Scree plot**

Figures 1 and 2 show that component 1 explains the majority of the variance in the dataset, 97.41%, with component 2 explaining 1.14% of the variance. Additionally, the standard deviation of component 1 is larger as it encompasses the majority of the variance in the dataset at 58.15.

In this analysis, components 1 and 2 is used with a cumulative variance explained of 98.55%. 

### PCA loading & Biplot
<div style="display: flex; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/c53d7810-6b2b-42bf-9b7f-d80ebae3747e" alt="image1" width="300"/>
  <img src="https://github.com/user-attachments/assets/8277d02e-ed6f-44b4-b955-9ec3fd2474fc" alt="image2" width="600"/>
</div>

**Fig. 3 PCA loading (left) and biplot (right)**
 
According to PCA loading, component 1 captures 84.3% of the variable (whole weight), indicating a very strong influence. Whilst component 2 captures a strong negative influence of shucked weight and a moderate positive influence of shell weight. In the biplot, the majority of rings of the abalone measured lies between 7 to 10.

### K-Means Clustering
![image](https://github.com/user-attachments/assets/fc7fc1c9-04ee-454b-92b8-40bfc050edee)

**Fig. 4 Within Sum Squares (WSS) plot**

According to the WSS plot in figure 5, there is a diminishing return after cluster 3. Therefore, the optimal number of clusters is 3.

![image](https://github.com/user-attachments/assets/67577741-f7f2-4e0b-b3ac-c51685d0c739)
<img src="https://github.com/user-attachments/assets/b729ea0a-39db-4eec-9fc6-56876774be26" alt="image" width="360"/>

**Fig. 5 K-Means Clustering analysis**

Analysing the 3 clusters in figure 6, we can observe that cluster 1 (Red)'s mean ranges around 0. Cluster 2 ranges around 1 and cluster 3 ranges around -1. From this information, we can identify that the clusters are well segregated.

#### Cluster sizes: 
```
Cluster 1: 1,764 
Cluster 2: 1,194 
Cluster 3: 1,219
```
#### Within-Cluster Sum of Squares (WSS):
```
Cluster 1: 1,920.078
Cluster 2: 3,311.914
Cluster 3: 1,790.464
```
WSS shows the internal variability of each cluster, with cluster 2 having more variability than the other two clusters.

#### Between-Cluster Sum of Squares / Total Sum of Squares (BSS / TSS) Ratio:

The ratio of 76.0% means that 76% of the total variance in the data is explained by the clustering, which indicates a reasonably good separation between clusters.

### Cluster plot
![image](https://github.com/user-attachments/assets/761532a4-5f9b-4358-b410-5e33a3973e3b)

**Fig. 6 Cluster plot**

In Figure 6, the visualizations of clusters 1, 2, and 3 show that the distribution of data points is more spread along the x-axis (PC1, accounting for 97.41% of the variance) than the y-axis (PC2, representing 1.14% of the variance).

Cluster 1 (red) has a moderate number of outliers, primarily along the y-axis, indicating some variability in PC2. Cluster 2 (green) has a large number of data points with a wider spread in both PC1 and PC2, showing more pronounced outliers compared to the other clusters. Cluster 3 (blue) has the fewest outliers, though its outliers appear more extreme in distance from the cluster's main concentration, particularly along the y-axis. Overall, Cluster 3 appears more compact, while Cluster 2 shows the greatest spread and variability.

*******************************************************************************************************************************************************************
# Supervised Learning

## Regression
Dataset: https://www.kaggle.com/datasets/niteshyadav3103/concrete-compressive-strength

### Introduction
Concrete is a ubiquitous building material and a cornerstone of construction for millennia. From foundational structures to towering skyscrapers, bridges and even the Colosseum of ancient Rome. Concrete’s versatility and reliability has been an indispensable component of construction, facilitating the advancement of civilisations.

The dataset consists of the Dependent Variable of concrete compressive strength and 8 Independent Variables of cement, blast furnace slag, fly ash, water, superplasticizer, coarse and fine aggregate, and age.

Compressive strength of concrete is expressed in a non-linear function of age and constituent ingredients. Through the process of hydration, the mixture of materials reacts with water to rapidly gain strength at the initial stage. As the curing process continues, strength increases at a gradual rate and slows down overtime. 

Through the use of Machine Learning algorithms, the formulation of concrete can be improved and optimised further with data and statistics, other than using current destructive and non-destructive tests. Different formulation of material and proportion can be assessed, which can minimise costs and environmental impacts. 

### Visualisations
#### Pearson Correlation
![image](https://github.com/user-attachments/assets/d051f6b7-1a33-42c4-a553-3755eac8cf04)

**Fig.7 Pearson Correlation Plot**

#### Scatterplot Matrix
![image](https://github.com/user-attachments/assets/ebb50658-f966-442e-8bdb-e7b12c8b2a61)

**Fig.8 Scatterplot Matrix of concrete strength dataset**

### Linear Regression Method
Using a Train-Test split of 70:30
<div style="display: flex; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/71690310-5a12-4c5d-96b6-64ff2b3d5cea" alt="image1" width="500"/>
  <img src="https://github.com/user-attachments/assets/e0bcf14c-aa84-448d-b900-327916f77917" alt="image2" width="300"/>
  </div>
  
**Fig.9 Linear Regression Model (left), summary of trainset (right-top) & testset (right-bottom)**

According to the results of the linear regression model, the adjusted R-squared value is 0.6155, explaining 61.55% of the variance in the dependent variable (CCS) with the independent variables. Along with summary values of the testset with higher median, mean and max but lower min than the trainset.

### Forward Selection & Backward Elimination
<div style="display: flex; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/db67c097-3e9d-49dd-ae3f-c32468d78f9b" alt="image1" width="500"/>
  <img src="https://github.com/user-attachments/assets/2255c5c2-fa75-4fbb-af81-6bd80d6d8c66" alt="image2" width="500"/>
</div>

**Fig.10 Trainset analysis of Forward Selection (left) & Backwards Elimination (right)**

### Stepwise Regression
  ![image](https://github.com/user-attachments/assets/78763cf4-0f32-4dfc-9ee2-f264fdcbe67d)
  
**Fig.11 Stepwise regression Trainset analysis**

We note that the forward selection and backwards elimination methods produced identical values (R, R-Squared, RMSE, etc.) While the stepwise regression method’s values had very slight differences with the other 2 methods.

Using RMSE values for comparison, the model's evaluation across the three training sets yielded identical results of 10.01351 for all variable selection methods: Forward, Backward, and Stepwise.

### Model Selection
![image](https://github.com/user-attachments/assets/ef5d1d53-b292-4063-91ec-f57a51ea21db)

**Fig.12 Forward Model Selection**

In figure 12, the Akaike Information Criterion (AIC) of forward model selection is 4832.91 with all the independent variables used. 

Similarly, in the backwards elimination and best subset analysis, did not remove any independent variables but instead retained all variables. It is noted that all 3 methods produced the same AIC and performed very similarly.

### Classification And Regression Tree (CART)

#### Maximal Tree

![image](https://github.com/user-attachments/assets/40dc1079-4443-406c-adcc-9812e77a3856)

**Fig. 13 CART Maximal Tree with 57 decision nodes**

#### CP (Complexity Parameter) Plot
![image](https://github.com/user-attachments/assets/3d1f483d-c218-45a1-8742-2375c8683252)

**Fig. 14 CP plot of maximal tree**

<div style="display: flex; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/bf096b60-95af-40fb-a761-66d0bc11887a" alt="image1" width="750"/>
  <img src="https://github.com/user-attachments/assets/3cf205c6-c8b2-4e03-b16e-53a51b71a5bd" alt="image2" width="250"/>
</div>

**Fig. 15 Optimal CART 1SE model via CP plot (left) and corresponding values (right)**

![image](https://github.com/user-attachments/assets/feba2c39-f9f0-4313-8591-d748dd1ed852)
![image](https://github.com/user-attachments/assets/e0f1434a-1b44-4f61-aed5-b29ce817b6d7)

**Fig. 16 Optimal tree**

Using the optimal CP of 0.002225818664705837, the variables used in the optimal CART trees were all the independent variables except for the variable, fly ash. In the optimal tree, more than over half were pruned with 33 nodes. With a xerror and xstd of 0.22101 and 0.015606 respectively, the 1SE (Standard Error) is 0.236616.

### Random Forest

A random forest analysis is used on the trainset of the concrete compressive strength dataset, with 500 trees. Resulting in a mean of squared residuals of 32.19245 and explaining 88.29% of the variance.

![image](https://github.com/user-attachments/assets/2730cffc-017d-4103-9a8e-703fb243fb70)

**Fig.17 OOB error graph**

### Results

![image](https://github.com/user-attachments/assets/11e53160-185f-45e8-98c4-c2467de0ac23)

**Fig.18 RMSE for LR, CART & RF**

As shown in fig.18, the Root Mean Square Error (RMSE) of the trainset and testset for linear regression are the largest, RMSE for random forest are the lowest and RMSE for CART lies in the middle of the two other models. Thus, random forest is the best model to use as it possesses the lowest error.

## Classification
Dataset: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

### Introduction
Cardiovascular disease is the leading cause of death, accounting for nearly 1/3 of all deaths worldwide. A further 1/3 of cardiovascular deaths occur in people under 70 years old. Lifestyle factors such as an abundance of unhealthy food and general lack of exercise has contributed significantly to the development of cardiovascular disease. 

The dataset consists of the Dependent Variable of heart disease and 11 Independent Variables of age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting electrocardiogram, max heart rate, exercise angina, old peak and ST slope.

Through the use of Machine Learning algorithms, procedures for early detection of cardiovascular disease for doctors and nurses will be streamlined and remedial action can be taken.

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
