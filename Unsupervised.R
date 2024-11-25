library(factoextra)
library(dplyr)
library(stats)
library(ggplot2)
library(ggfortify)

# URL: https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset
abalone_df <- read.csv("C:/Users/nicho/Desktop/abalone.csv")

summary(abalone_df)

abalone <- select(abalone_df, c(2,3,4,5,6,7,8))

############ PCA ##############
# PCA eligibility
cor(abalone)
mean(cor(abalone))
# 0.907 > 0.3

# Principal Component Analysis
PCA <- princomp(abalone)

names(PCA)
summary(PCA)

# PC loadings
PCA$loadings

# Principal components
PC <- PCA$scores
View(PC)
cor(PC)
# All correlation are almost 0, all principal components are independent

# Scree plot
fviz_eig(PCA, addlabels = TRUE)
# Bar 1 captured the majority of the total variance in the dataset. While the other comps 
#contribute relatively small amounts of variance.

# Biplot
fviz_pca_biplot(PCA, label = "var", habillage = abalone_df$Rings, col.var = "black")

############ K-means cluster ##############
set.seed(2024)

abalone_df <- read.csv("C:/Users/nicho/Desktop/abalone.csv")

abalone <- select(abalone_df, c(2,3,4,5,6,7,8))

# Scale data
abalone_scale <- scale(abalone)

# WSS (Within Sum Squares) plot
fviz_nbclust(abalone_scale, kmeans, method = "wss") +
  labs(subtitle = "Within groups sum of squares")

# Kmeans cluster
KM <- kmeans(abalone_scale, centers = 3, nstart = 100)
print(KM)

# Cluster plot
KM_cluster <- KM$cluster
rownames(abalone_scale) <- paste(abalone_df$Rings, 1:dim(abalone)[1], sep = "_")
fviz_cluster(list(data=abalone_scale, cluster = KM_cluster))
autoplot(KM, abalone, frame = TRUE)

# Cluster centres
KM$centers