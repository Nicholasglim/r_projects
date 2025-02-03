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
```

## Principal Component Analysis (PCA)
```
# Checking PCA Eligibility
cor(abalone)
```
Correlation Matrix

![Correlation matrix](https://github.com/user-attachments/assets/05e783c8-630e-40bb-a11a-eafca41576a7)
```
# Finding mean correlation
mean(cor(abalone))
```
0.907 > 0.3 (Indicates PCA is suitable)

The dataset used in Abalone dataset which seeks to identify the age of abalones by the number of rings using physical measurements. With a mean correlation of 0.9, the abalone dataset is suitable to use PCA for unsupervised learning.

## Performing Principal Component Analysis
```
PCA <- princomp(abalone)

# Exploring PCA output loadings
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
# Principal Component Loadings
PCA$loadings
```
![image](https://github.com/user-attachments/assets/c3a899e4-901f-48ea-8933-590e86667993)

```
# Checking correlation between all Principal Components
PC <- PCA$scores
cor(PC)
```
All Correlations are close to zero, meaning the 7 PCs are uncorrelated to each other, and each captured a unique aspect of the data's variance without redundancy.
![image](https://github.com/user-attachments/assets/3ca458a4-9135-408d-b96a-9d2df09a2b2b)

```
# Scree plot
fviz_eig(PCA, addlabels = TRUE)
```
Scree Plot

![Scree Plot](https://github.com/user-attachments/assets/6ea1dd54-530e-45a7-b3c9-8be3110d70a3)

The Scree Plot above show that component 1 explains the majority of the variance in the dataset, 97.41%, with component 2 explaining 1.14% of the variance.

### In this analysis, components 1 and 2 is used with a cumulative variance explained of 98.55%. 
```
# PCA Biplot
fviz_pca_biplot(PCA, label = "var", habillage = abalone_df$Rings, col.var = "black")
```
PCA Loading & Biplot

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

Cluster sizes: 
```
Cluster 1: 1,764 
Cluster 2: 1,194 
Cluster 3: 1,219
```
Within-Cluster Sum of Squares (WSS):
```
Cluster 1: 1,920.078
Cluster 2: 3,311.914
Cluster 3: 1,790.464
```
WSS shows the internal variability of each cluster, with cluster 2 having more variability than the other two clusters.

Between-Cluster Sum of Squares / Total Sum of Squares (BSS / TSS) Ratio:

The ratio of 76.0% means that 76% of the total variance in the data is explained by the clustering, which indicates a reasonably good separation between clusters.

```
# Cluster plot
KM_cluster <- KM$cluster
rownames(abalone_scale) <- paste(abalone_df$Rings, 1:dim(abalone)[1], sep = "_")
autoplot(KM, abalone, frame = TRUE)
```
Cluster plot

![image](https://github.com/user-attachments/assets/761532a4-5f9b-4358-b410-5e33a3973e3b)

Analysing the 3 clusters, we can observe that cluster 1 (Red)'s mean ranges around 0. Cluster 2 ranges around +1 and cluster 3 ranges around -1. From this information, we can identify that the clusters are well segregated.

Cluster 1 (red) has a moderate number of outliers, primarily along the y-axis, indicating some variability in PC2. Cluster 2 (green) has a large number of data points with a wider spread in both PC1 and PC2, showing more pronounced outliers compared to the other clusters. Cluster 3 (blue) has the fewest outliers, though its outliers appear more extreme in distance from the cluster's main concentration, particularly along the y-axis. Overall, Cluster 3 appears more compact, while Cluster 2 shows the greatest spread and variability.

*******************************************************************************************************************************************************************
