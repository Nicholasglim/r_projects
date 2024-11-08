# Unsupervised learning
Dataset: https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset

### Introduction
Abalone is a type of marine mollusk that are prized for their meat and iridescent shells in many cultures. Additionally, as a marine mollusk, abalones play a vital role of cleaning the ocean and improving water quality through the process of filter feeding.

The dataset consists of the dependent variable of rings and independent variable of sex, length, diameter, height, whole weight, shucked weight, viscera weight and shell weight.

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

Analysing the 3 clusters in figure 6, we can observe that cluster 1’s mean ranges around 0. Cluster 2 ranges around 1 and cluster 3 ranges around -1. From this information, we can identify that the clusters are well segregated. As shown in figure 7, the visualisations of clusters 1, 2 and 3 are more represented in the x-axis than in y-axis. There were some slight outliers in clusters 1 on the y-axis. Cluster 2 has a moderate number of outliers. Cluster 3 has a large number of outliers.

# Supervised Learning

## Regression
Dataset: https://www.kaggle.com/datasets/niteshyadav3103/concrete-compressive-strength

### Introduction
Concrete is a ubiquitous building material and a cornerstone of construction for millennia. From foundational structures to towering skyscrapers, bridges and even the Colosseum of ancient Rome. Concrete’s versatility and reliability has been an indispensable component of construction, facilitating the advancement of civilisations.

The dataset consists of the dependent variable of concrete compressive strength and 8 independent variables of cement, blast furnace slag, fly ash, water, superplasticizer, coarse and fine aggregate, and age.

Compressive strength of concrete is expressed in a non-linear function of age and constituent ingredients. Through the process of hydration, the mixture of materials reacts with water to rapidly gain strength at the initial stage. As the curing process continues, strength increases at a gradual rate and slows down overtime. 

Through the use of Machine Learning algorithms, the formulation of concrete can be improved and optimised further with data and statistics, other than using current destructive and non-destructive tests. Different formulation of material and proportion can be assessed, which can minimise costs and environmental impacts. 

### Visualisations
#### Pearson Correlation
![image](https://github.com/user-attachments/assets/d051f6b7-1a33-42c4-a553-3755eac8cf04)

**Fig.7 Pearson Correlation Plot**

#### Scatterplot Matrix
![image](https://github.com/user-attachments/assets/ebb50658-f966-442e-8bdb-e7b12c8b2a61)

**Fig.8 Scatterplot Matrix of concrete strength dataset**

### Linear Regression
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


## Classification
Dataset: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

### Introduction
Cardiovascular disease is the leading cause of death, accounting for nearly 1/3 of all deaths worldwide. A further 1/3 of cardiovascular deaths occur in people under 70 years old. Lifestyle factors such as an abundance of unhealthy food and general lack of exercise has contributed significantly to the development of cardiovascular disease. 

The dataset consists of the dependent variable of heart disease and 11 independent variables of age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting electrocardiogram, max heart rate, exercise angina, old peak and ST slope.

Through the use of Machine Learning algorithms, procedures for early detection of cardiovascular disease for doctors and nurses will be streamlined and remedial action can be taken.
