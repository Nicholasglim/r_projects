# Classification: Heart Disease Prediction

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
