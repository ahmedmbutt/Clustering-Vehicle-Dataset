# Vehicle-Clustering

![cars_clus.csv](https://github.com/ahmedmbutt/Clustering-Vehicle-Dataset/assets/81696588/a5f73a16-9b78-44b4-aae8-93b6f63d8606)

# Data Preparation

The dataset contains missing values, such as empty cells and cells containing '$null$' values, and the dataset also contains attributes with wrong data format, such as numerical attributes wrongly defined as object type.

```
df.dropna(axis=0, inplace=True)
numeric_columns = ['sales','resale','type','price','engine_s','horsepow','wheelbas','width','length','curb_wgt','fuel_cap','mpg','lnsales']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)
```

To handle wrong data format, we converted the relevant attributes to numerical type. And to handle missing values, we dropped the rows containing missing string values and for missing numerical values we replaced it with the respective column's median value.

# Data Analysis

```
feature_set = df[['type','price','engine_s','horsepow','wheelbas','width','length','curb_wgt','fuel_cap','mpg']]
```

We selected the important features and dropped the rest. Features such as 'manufact', 'model', 'sales', 'resale', and 'partition' were dropped as they were considered irrelevant to compare for newly developed vehicle prototypes. While features such as 'type', 'price', 'engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', and 'mpg' were kept as they were considered important to identify the primary competitors for newly developed vehicle prototypes.


# Data Modeling

```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
feature_mtx = scaler.fit_transform(feature_set)
```

We standardize the feature set by applying Z-score normalization so that we can give equal considerations for each feature.

# Model Selection and Building

We selected Agglomerative Hierarchical Clustering to identify clusters of vehicles that possess unique characteristics. We used the 'euclidean' metric and 'ward' criterion for linkage.

```
clustering = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
clustering.fit_predict(feature_set)
```

The optimal number of clusters for the above model were chosen by plotting the dendrogram. From the dendrogram we concluded the optimal number of clusters to be 2.

```
Z = linkage(feature_mtx, 'ward')
dn = hierarchy.dendrogram(Z)
```

![image](https://github.com/ahmedmbutt/Clustering-Vehicle-Dataset/assets/81696588/bcdd6036-7cfe-4dcd-bee1-322e2959b4c7)


# Model Evaluation

We finally plotted our clusters using a scatter plot with varying colors to identify different clusters. Before plotting the scatter plot, we reduced our dataset to two features using Feature Agglomeration so we can easily plot the clusters on a 2-D axis.

```
agglo = FeatureAgglomeration(n_clusters=2, metric='euclidean', linkage='ward')
X_reduced = agglo.fit_transform(feature_set)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=df.clusters)
```

![image](https://github.com/ahmedmbutt/Clustering-Vehicle-Dataset/assets/81696588/6753449a-1c30-4b24-80c0-c1fa3ef46376)
