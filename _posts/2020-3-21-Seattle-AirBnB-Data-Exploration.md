![jpg]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/room.jpg)

## Introduction

In this article, we are going to look at Seattle AirBnB data (located [here](https://www.kaggle.com/airbnb/seattle/data)), and answer the following renters' questions.   

  * What are the top 4 months with the highest occupancy rate in Seattle?
  * How to identify the different groups of Seattle AirBnB rental listings?  What are their characteristics?
  * What are the characteristics of the listings with the highest average rental price?

## Top 4 months with the highest occupancy rate in Seattle

![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_21_0.png)

From the above plot, we can see that Janurary, July, August and Feburary in 2016 in that order have the highest occupancy rates.  
If you are a renter and you want to avoid crowds, you should skip these months.   However, the highest occupancy rate is 44.59%, this means that you should be able to secure a rental place in Seattle anytime of the year.

## Clusters of Seattle AirBnB listings

![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Clusters.png)

Above is the map of the 15 clusters using KMeans Clustering, and the following is some statistics on quantitative features of the clusters:

![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Capture_stats.png)

The top 3 clusters with the highest number of listings are cluster 14, 1 and 7.  Let us look at them in some details.

### Cluster 14

![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_142_0.png)

We call cluster 14 the high availablity group with host response time within a few hours, and low occupancy rate.   

The features that are positively associated with this cluster are availability_60, availability_90, availability_30, host_response_time_within a few hours and availability_365.   

The negatively associated features are  host_response_time_within an hour, "missing" reviews_per_month,
"missing" review_scores_rating, accommodates and occupancy_rate.


### Cluster 1

![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_146_0.png)

Cluster 1 is the high occupancy rate group with high host response/acceptance rate.   

The features that are positively associated with this cluster are occupancy_rate, review_scores_rating, host_response_rate, host_acceptance_rate and host_response_time_within a day.   

The negatively associated features are review_scores_rating_missing, availability_365,
availability_30, availability_90 and availability_60.   

We can see that clusters 14 and 1 are quite the opposite to each other.  Renters looking at cluster 14 listings should have an easier time in securing a rental.

### Cluster 7

![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_150_0.png)

Cluster 7 has lots of reviews, can be instantly booked with moderate cancellation policy, and lower cleaning fees.   

The features that are positively associated with this cluster are reviews_per_month, host_response_time_within an hour, number_of_reviews, instant_bookable_t and cancellation_policy_moderate.   

The negatively associated features are  cleaning_fee, "missing" reviews_per_month,
"missing" review_scores_rating, number of bedrooms and "missing" host_acceptance_rate.   


## Characteristics of the listings with the highest average rental price

Cluster 3 is the group that have the highest average rental price.

![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/cluster3.png)

Cluster 3 is all about the rental place - number of bedrooms, how many people it can accomodate, number of beds/bathrooms, and how many guests are included in the rental price.   

The features that are positively associated with this cluster are bedrooms, accommodates, beds, bathrooms and guests_included.   

The negatively associated features are cancellation_policy_moderate, "unknown" host_response_time, "missing" host_response_rate, "missing" cleaning_fee, and private room.   

## Conclusions

Janurary, July, August and Feburary in 2016 in that order have the highest occupancy rates. If you are a renter and you want to avoid crowds, you should skip these months.   However, the highest occupancy rate is 44.59%, this means that you should be able to secure a rental place in Seattle anytime of the year.   

Cluster 14 is the high availablity group with host response time within a few hours, and low occupancy rate.   

Cluster 1 is the high occupancy rate group with high host response/acceptance rate.   

We can see that clusters 14 and 1 are quite the opposite to each other. Renters looking at cluster 14 listings should have an easier time in securing a rental.

Cluster 7 has lots of reviews, can be instantly booked with moderate cancellation policy, and lower cleaning fees.

Cluster 3 is all about the rental place - number of bedrooms, how many people it can accomodate, number of beds/bathrooms, and how many guests are included in the rental price.

##Acknowledgement

This dataset is part of Airbnb Inside, and the original source can be found [here](http://insideairbnb.com/get-the-data.html).  The dataset is made available under a [Creative Commons CC0 1.0 Universal (CC0 1.0) "Public Domain Dedication"](http://creativecommons.org/publicdomain/zero/1.0/) license.

