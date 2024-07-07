---
layout: post
title: The "You Are What You Eat" Customer Segmentation
image: "/posts/clustering-title-img.png"
tags: [Customer Segmentation, Machine Learning, Clustering, Python]
---

In this project, I used k-means clustering to segment the customer base to increase business understanding and enhance the relevancy of targeted messaging & customer communications.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
- [01. Data Overview](#data-overview)
- [02. K-Means](#kmeans-title)
    - [Concept Overview](#kmeans-overview)
    - [Data Preprocessing](#kmeans-preprocessing)
    - [Finding A Good Value For K](#kmeans-k-value)
    - [Model Fitting](#kmeans-model-fitting)
    - [Appending Clusters To Customers](#kmeans-append-clusters)
    - [Segment Profiling](#kmeans-cluster-profiling)
- [03. Application](#kmeans-application)
- [04. Growth & Next Steps](#growth-next-steps)

___
<br>
# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

The Senior Management team from a supermarket chain wanted to know how customers were shopping, and how lifestyle choices might affect the food areas customers were shopping into, or more interestingly, not shopping into.

I was asked to use data, and Machine Learning to help segment up their customers based on their engagement with each of the major food categories - aiding business understanding of the customer base, and enhancing the relevancy of targeted messaging & customer communications.

<br>
### Actions <a name="overview-actions"></a>

I first needed to compile the necessary data from several tables in the database, namely the *transactions* table, and the *product_areas* table.  I joined together the relevant information using Pandas, and then aggregated the transactional data across product areas, from the most recent six months to a customer level. The final data for clustering was, for each customer, the percentage of sales allocated to each product area.

As a starting point, I tested & applied k-means clustering for this task. I needed to apply some data pre-processing and, most importantly feature scaling to ensure all variables existed on the same scale - a very important consideration for distance-based algorithms such as k-means.

As k-means is an *unsupervised learning* approach, in other words, there are no labels - I used a process known as *Within Cluster Sum of Squares (WCSS)* to understand what a "good" number of clusters or segments was.

Based upon this, I applied the k-means algorithm onto the product area data, appended the clusters to my customer base, and then profile the resulting customer segments to understand what the differentiating factors were!

<br>

### Results <a name="overview-results"></a>

Based upon iterative testing using WCSS, I settled on a customer segmentation with 3 clusters. These clusters ranged in size, with Cluster 0 accounting for 73.6% of the customer base, Cluster 2 accounting for 14.6%, and Cluster 1 accounting for 11.8%.

There were some extremely interesting findings from profiling the clusters.

For *Cluster 0*, I saw a significant portion of spend being allocated to each of the product areas - showing customers without any particular dietary preference.  

For *Cluster 1*, I saw quite high proportions of spending being allocated to Fruit & Vegetables, but very little to the Dairy & Meat product areas.  It could be hypothesized that these customers were following a vegan diet.  

Finally customers in *Cluster 2* spent significant portions of Dairy, Fruit & Vegetables, but very little in the Meat product area - so similarly, I would make an early hypothesis that these customers were more along the lines of those following a vegetarian diet.

To help embed this segmentation into the business, I proposed to call this the "You Are What You Eat" segmentation.

___
<br>
# Data Overview  <a name="data-overview"></a>

I was primarily looking to discover segments of customers based on their transactions within *food* based product areas so I would need to only select those.

In the code below, I:

* Imported the required Python packages & libraries
* Imported the tables from the database
* Merged the tables to tag on *product_area_name* which only exists in the *product_areas* table
* Dropped the non-food categories
* Aggregated the sales data for each product area, at the customer level
* Pivoted the data to get it into the right format for clustering
* Changed the values from raw dollars, into a percentage of spend for each customer (to ensure each customer is comparable)

```python
# Import required Python packages
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# Import tables from the database
transactions = ...
product_areas = ...

# Merge product_area_name on
transactions = pd.merge(transactions, product_areas, how = "inner", on = "product_area_id")

# Drop the non-food category
transactions.drop(transactions[transactions["product_area_name"] == "Non-Food"].index, inplace = True)

# Aggregate sales at customer level (by product area)
transaction_summary = transactions.groupby(["customer_id", "product_area_name"])["sales_cost"].sum().reset_index()

# Pivot data to place product areas as columns
transaction_summary_pivot = transactions.pivot_table(index = "customer_id",
                                                    columns = "product_area_name",
                                                    values = "sales_cost",
                                                    aggfunc = "sum",
                                                    fill_value = 0,
                                                    margins = True,
                                                    margins_name = "Total").rename_axis(None,axis = 1)

# Transform sales into % sales
transaction_summary_pivot = transaction_summary_pivot.div(transaction_summary_pivot["Total"], axis = 0)

# Drop the "total" column as we don't need that for clustering
data_for_clustering = transaction_summary_pivot.drop(["Total"], axis = 1)
```
<br>

After the data pre-processing using Pandas, I had a dataset for clustering that looks like the below sample:


| **customer_id** | **dairy** | **fruit** | **meat** | **vegetables** |
|---|---|---|---|---|
| 2 | 0.246 | 0.198 | 0.394 | 0.162  |
| 3 | 0.142 | 0.233 | 0.528 | 0.097  |
| 4 | 0.341 | 0.245 | 0.272 | 0.142  |
| 5 | 0.213 | 0.250 | 0.430 | 0.107  |
| 6 | 0.180 | 0.178 | 0.546 | 0.095  |
| 7 | 0.000 | 0.517 | 0.000 | 0.483  |

<br>
The data is at the customer level, and I had a column for each of the highest-level food product areas. Within each of those, I had the *percentage* of sales that each customer allocated to that product area over the past six months.

___
<br>
# K-Means <a name="kmeans-title"></a>

<br>
### Concept Overview <a name="kmeans-overview"></a>

K-Means is an *unsupervised learning* algorithm, meaning that it does not look to predict known labels or values, but instead looks to isolate patterns within unlabelled data.

The algorithm works in a way where it partitions data points into distinct groups (clusters) based on their *similarity* to each other.

This similarity is most often the Euclidean (straight-line) distance between data points in n-dimensional space.  Each variable that is included lies on one of the dimensions in space.

The number of distinct groups (clusters) is determined by the value that is set for "k".

The algorithm does this by iterating over four key steps, namely:

1. It selects "k" random points in space (these points are known as centroids)
2. It then assigns each of the data points to the nearest centroid (based on euclidean distance)
3. It then repositions the centroids to the *mean* dimension values of its cluster
4. It then reassigns each data point to the nearest centroid

Steps 3 & 4 continue to iterate until no data points are reassigned to a closer centroid.

<br>
### Data Preprocessing <a name="kmeans-preprocessing"></a>

There are three vital preprocessing steps for k-means, namely:

* Missing values in the data
* The effect of outliers
* Feature Scaling

<br>
#### Missing Values

Missing values can cause issues for k-means, as the algorithm won't know where to plot those data points along the dimension where the value is not present. If we have observations with missing values, the most common options are to either remove the observations or to use an imputer to fill in or to estimate what those values might be.

Here, I actually didn't suffer from missing values so I didn't need to deal with that here.

<br>
#### Outliers

As k-means is a distance-based algorithm, outliers can cause problems. The main issue we face is when we come to scale our input variables, a very important step for a distance-based algorithm.

We don’t want any variables to be “bunched up” due to a single outlier value, as this will make it hard to compare their values to the other input variables. We should always investigate outliers rigorously - however, in my case where I was dealing with percentages, I thankfully don't face this issue!

<br>
#### Feature Scaling

Again, as k-means is a distance-based algorithm, in other words, it is reliant on an understanding of how similar or different data points are across different dimensions in n-dimensional space, the application of Feature Scaling is extremely important.

Feature Scaling is where we force the values from different columns to exist on the same scale, to enhance the learning capabilities of the model. There are two common approaches to this, Standardisation, and Normalisation.

Standardization rescales data to have a mean of 0, and a standard deviation of 1 - meaning most data points will most often fall between values of around -4 and +4.

Normalization rescales data points so that they exist in a range between 0 and 1.

For k-means clustering, either approach is going to be *far better* than using no scaling at all.  Here, I applied normalization as this would ensure all variables would end up having the same range, fixed between 0 and 1, and therefore the k-means algorithm could judge each variable in the same context. Standardization *couldn* result in different ranges, variable to variable, and this was not so useful (although this isn't explicitly true in all scenarios).

Another reason for choosing Normalisation over Standardisation was that my scaled data would *all* exist between 0 and 1, and these would then be compatible with any categorical variables that I had encoded as 1’s and 0’s (although I didn't have any variables of this type in my task here).

In my specific task here, I was using percentages, so my values were _already_ spread between 0 and 1. I would still apply normalization for the following reasons. One of the product areas might commonly make up a large proportion of customer sales, and this might end up dominating the clustering space. If I normalize all of my variables, even product areas that made up smaller volumes will be spread proportionately between 0 and 1!

The below code, I used the in-built MinMaxScaler functionality from scikit-learn to apply Normalisation to all of my variables.  The reason I created a new object (here called data_for_clustering_scaled) was that I wanted to use the scaled data for clustering, but when profiling the clusters later on, I might want to use the actual percentages as this might make more intuitive business sense, so it's good to have both options available!

```python
# Create our scaler object
scale_norm = MinMaxScaler()

# Normalize the data
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering), columns = data_for_clustering.columns)
```

<br>
### Finding A Good Value For k <a name="kmeans-k-value"></a>

At this point here, my data was ready to be fed into the k-means clustering algorithm.  Before that, however, I wanted to understand what number of clusters I wanted the data split into.

In the world of unsupervised learning, there is no *right or wrong* value for this - it really depends on the data you are dealing with, as well as the unique scenario you're utilizing the algorithm for.  For this specific case, having a very high number of clusters might not be appropriate as it would be too hard for the business to understand the nuance of each in a way where they can apply the right strategies.

Finding the "right" value for k can feel more like art than science, but there are some data-driven approaches that can help us!  

The approach I utilized here, is known as *Within Cluster Sum of Squares (WCSS)* which measures the sum of the squared Euclidean distances that data points lie from their closest centroid. WCSS can help us understand the point where adding *more clusters* provides little extra benefit in terms of separating our data.

By default, the k-means algorithm within scikit-learn will use k = 8 meaning that it will look to split the data into eight distinct clusters. I wanted to find a better value that fitted my data, and my task!

In the code below, I tested multiple values for k, and plotted how this WCSS metric changed. As I increased the value for k (in other words, as I increased the number of centroids or clusters) the WCSS value would always decrease. However, these decreases would get smaller and smaller each time I added another centroid and I was looking for a point where this decrease was quite prominent *before* this point of diminishing returned.

```python
# Set up range for search and empty list to append wcss scores to
k_values = list(range(1,10))
wcss_list = []

# Loop through each possible value of k, fit to the data, and append the wcss score
for k in k_values:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(data_for_clustering_scaled)
    wcss_list.append(kmeans.inertia_)

# Plot wcss by k
plt.plot(k_values, wcss_list)
plt.title("Within Cluster Sum of Squares -  by k")
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.tight_layout()
plt.show()
```
<br>
That code gave me the below plot - which visualizes my results!

![alt text](/img/posts/kmeans-optimal-k-value-plot.png "K-Means Optimal k Value Plot")

<br>
Based on the shape of the above plot - there appeared to be an elbow at k = 3. Before that, I saw a significant drop in the WCSS score, but following decreases were much smaller, meaning this could be a point that suggested adding *more clusters* would provide little extra benefit in terms of separating my data. A small number of clusters can be beneficial when considering how easy it is for the business to focus on, and understand, each - so I would continue on, and fit my k-means clustering solution with k = 3.

<br>
### Model Fitting <a name="kmeans-model-fitting"></a>

The below code instantiated my k-means object using a value for k equal to 3. I then fitted this object to my scaled dataset to separate my data into three distinct segments or clusters.

```python
# Instantiate our k-means object
kmeans = KMeans(n_clusters = 3, random_state = 42)

# Fit to our data
kmeans.fit(data_for_clustering_scaled)
```

<br>
### Append Clusters To Customers <a name="kmeans-append-clusters"></a>

With the k-means algorithm fitted to my data, I could append those clusters to my original dataset, meaning that each customer would be tagged with the cluster number that they most closely fitted into based on their sales data over each product area.

In the code below, I tagged this cluster number onto my original dataframe.

```python
# Add cluster labels to our original data
data_for_clustering["cluster"] = kmeans.labels_
```

<br>
### Cluster Profiling <a name="kmeans-cluster-profiling"></a>

Once I had my data separated into distinct clusters, my client needed to understand *what it was* that was driving the separation. This meant the business could understand the customers within each and the behaviors that made them unique.

<br>
#### Cluster Sizes

In the below code, I first assessed the number of customers that fell into each cluster.

```python
# Check cluster sizes
data_for_clustering["cluster"].value_counts(normalize=True)
```
<br>

Running that code showed us that the three clusters were different in size, with the following proportions:

* Cluster 0: **73.6%** of customers
* Cluster 2: **14.6%** of customers
* Cluster 1: **11.8%** of customers

Based on these results, it appeared we had a skew toward Cluster 0 with Cluster 1 & Cluster 2 being proportionally smaller. This wasn't right or wrong, it was simply showing up pockets of the customer base that were exhibiting different behaviors - and this was *exactly* what I wanted.

<br>
#### Cluster Attributes

To understand what these different behaviors or characteristics were, I analyzed the attributes of each cluster, in terms of the variables I fed into the k-means algorithm.

```python
# Profile clusters (mean % sales for each product area)
cluster_summary = data_for_clustering.groupby("cluster")[["Dairy","Fruit","Meat","Vegetables"]].mean().reset_index()
```
<br>
That code resulted in the following table...

| **Cluster** | **Dairy** | **Fruit** | **Meat** | **Vegetables** |
|---|---|---|---|---|
| 0 | 22.1% | 26.5% | 37.7% | 13.8%  |
| 1 | 0.2% | 63.8% | 0.4% | 35.6%  |
| 2 | 36.4% | 39.4% | 2.9% | 21.3%  |

<br>
For *Cluster 0* I saw a reasonably significant portion of spend being allocated to each of the product areas. For *Cluster 1*, I saw quite high proportions of spending being allocated to Fruit & Vegetables, but very little to the Dairy & Meat product areas. It could be hypothesized that these customers were following a vegan diet. Finally customers in *Cluster 2* spend, on average, significant portions of Dairy, Fruit & Vegetables, but very little in the Meat product area - so similarly, I would make an early hypothesis that these customers were more along the lines of those following a vegetarian diet - very interesting!

___
<br>
# Application <a name="kmeans-application"></a>

Even though this was a simple solution, based on high-level product areas it will help leaders in the business, and category managers gain a clearer understanding of the customer base.

Tracking these clusters over time would allow the client to more quickly react to dietary trends, and adjust their messaging and inventory accordingly.

Based upon these clusters, the client will be able to target customers more accurately - promoting products & discounts to customers that are truly relevant to them - overall enabling a more customer-focused communication strategy.

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

It would be interesting to run this clustering/segmentation at a lower level of product areas, so rather than just the four areas of Meat, Dairy, Fruit, and Vegetables - clustering spend across the sub-categories *below* those categories.  This would mean we could create more specific clusters, and get an even more granular understanding of dietary preferences within the customer base.

Here I just focused on variables that were linked directly to sales - it could be interesting to also include customer metrics such as distance to store, gender, etc to give an even more well-rounded customer segmentation.

It would be useful to test other clustering approaches such as hierarchical clustering or DBSCAN to compare the results.
