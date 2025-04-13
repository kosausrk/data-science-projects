# collection of data science projects

## I: titanic data EDA analysis 

Visualize Data

![data1](part1/data1.png)

![data2](part1/data2.png)

![data3](part1/data3.png)

![data4](part1/data4.png)





## II: Sentiment Analysis on Movie Reviews 
This project uses a Naive Bayes classifier on a shortened IMDb dataset (100 rows) to classify reviews as positive or negative. It loads data from part2/data/movie-data1.csv, preprocesses text using CountVectorizer, trains the model, and visualizes results with sentiment charts, word clouds, and a confusion matrix.


Visualize Data 

![data1](part2/data1.png)

![data2](part2/data2.png)

![data3](part2/data3.png)

![data4](part2/data4.png)

Citation: Stanford AI Large Movie Review Dataset https://ai.stanford.edu/~amaas/data/sentiment/ 

@InProceedings{maas-EtAl:2011:ACL-HLT2011,
author = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
title = {Learning Word Vectors for Sentiment Analysis},
booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
month = {June},
year = {2011},
address = {Portland, Oregon, USA},
publisher = {Association for Computational Linguistics},
pages = {142--150},
url = {http://www.aclweb.org/anthology/P11-1015}
}


## III: Predicting House Prices with Machine Learning
This project uses a Random Forest Regressor to predict house prices using the Ames Housing dataset (via Kaggle https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset?select=AmesHousing.csv). The data is preprocessed by filling missing values, encoding categorical features, and visualizing key insights. A regression model is trained, evaluated using RMSE and R² scores, and feature importances are visualized.

Visualize Data

![data1](part3/data1.png)

![data2](part3/data2.png)

![data3](part3/data3.png)



## IV: Customer Segmentation with K-Means Clustering

This project uses K-Means clustering to segment customers based on Age, Annual Income, and Spending Score. After performing exploratory data analysis (EDA), we preprocess the data with scaling and apply K-Means to discover underlying customer groups. The elbow method is used to determine the optimal number of clusters. PCA is then applied to reduce the data to two dimensions for visualizing the clusters effectively. This kind of segmentation can help businesses better understand their audience and tailor marketing strategies accordingly.

Visualize Data

**Gender Distribution**  
![data1](part4/data1.png)

**Spending Score vs. Annual Income**  
![data2](part4/data2.png)

**Elbow Method to Find Optimal Clusters**  
![data3](part4/data3.png)

**Customer Segments Visualized with PCA**  
![data4](part4/data4.png)

**Spending Score by Cluster**  
![data5](part4/data5.png)








