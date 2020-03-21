## Introduction

In this article, we are going to look at Seattle AirBnB data (located [here](https://www.kaggle.com/airbnb/seattle/data)), and answer some renters' questions.

### Business Understanding

As Seattle renters, we would like to answer the following questions in this article:

  * What are the top 4 months with the highest occupancy rate in Seattle?
  * How to identify the different groups of Seattle AirBnB rental listings?  What are their characteristics?
  * What are the characteristics of the listings with the highest mean rental price?

### Data Understanding

  First, we load the dataset showing the rental availablity on a certain date for specific listing.   

  Following is the first 5 rows of this data:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241032</td>
      <td>2016-01-04</td>
      <td>t</td>
      <td>$85.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>241032</td>
      <td>2016-01-05</td>
      <td>t</td>
      <td>$85.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>241032</td>
      <td>2016-01-06</td>
      <td>f</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>241032</td>
      <td>2016-01-07</td>
      <td>f</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>241032</td>
      <td>2016-01-08</td>
      <td>f</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We split the date column into two new columns - 'month' and 'year'.  Following are examples from these computations:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
      <th>year</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241032</td>
      <td>2016-01-04</td>
      <td>t</td>
      <td>$85.00</td>
      <td>2016</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>241032</td>
      <td>2016-01-05</td>
      <td>t</td>
      <td>$85.00</td>
      <td>2016</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>241032</td>
      <td>2016-01-06</td>
      <td>f</td>
      <td>NaN</td>
      <td>2016</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>241032</td>
      <td>2016-01-07</td>
      <td>f</td>
      <td>NaN</td>
      <td>2016</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>241032</td>
      <td>2016-01-08</td>
      <td>f</td>
      <td>NaN</td>
      <td>2016</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We want to look at data within one year.  From the following result, we observed that this dataset has both 2016 and 2017 data.  There is a large portion of the 2016 data compared to 2017.  Thus, we should just keep the 2016 data.




    2016    1385934
    2017       7636
    Name: year, dtype: int64



Compute the monthly vacancies for all listings:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month</th>
      <th>vacancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>59239</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>73321</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>83938</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>76037</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>79971</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>77244</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>74222</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>76347</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>77246</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>82438</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>81780</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>87061</td>
    </tr>
  </tbody>
</table>
</div>



Compute the monthly occupancies for all listings:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month</th>
      <th>occupancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>47665</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>37401</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>34420</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>38503</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>38387</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>37296</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>44136</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>42011</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>37294</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>35920</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>32760</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>31297</td>
    </tr>
  </tbody>
</table>
</div>



Merged the two pieces of data together:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month</th>
      <th>vacancy</th>
      <th>occupancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>59239</td>
      <td>47665</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>73321</td>
      <td>37401</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>83938</td>
      <td>34420</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>76037</td>
      <td>38503</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>79971</td>
      <td>38387</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>77244</td>
      <td>37296</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>74222</td>
      <td>44136</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>76347</td>
      <td>42011</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>77246</td>
      <td>37294</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>82438</td>
      <td>35920</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>81780</td>
      <td>32760</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>87061</td>
      <td>31297</td>
    </tr>
  </tbody>
</table>
</div>



Calculate the monthly vacancy rates and occupancy rates:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month</th>
      <th>vacancy</th>
      <th>occupancy</th>
      <th>vacancy_rate</th>
      <th>occupancy_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>59239</td>
      <td>47665</td>
      <td>0.554133</td>
      <td>0.445867</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>74222</td>
      <td>44136</td>
      <td>0.627097</td>
      <td>0.372903</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>76347</td>
      <td>42011</td>
      <td>0.645051</td>
      <td>0.354949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>73321</td>
      <td>37401</td>
      <td>0.662208</td>
      <td>0.337792</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>76037</td>
      <td>38503</td>
      <td>0.663847</td>
      <td>0.336153</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>77244</td>
      <td>37296</td>
      <td>0.674384</td>
      <td>0.325616</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>77246</td>
      <td>37294</td>
      <td>0.674402</td>
      <td>0.325598</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>79971</td>
      <td>38387</td>
      <td>0.675670</td>
      <td>0.324330</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>82438</td>
      <td>35920</td>
      <td>0.696514</td>
      <td>0.303486</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>83938</td>
      <td>34420</td>
      <td>0.709187</td>
      <td>0.290813</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>81780</td>
      <td>32760</td>
      <td>0.713986</td>
      <td>0.286014</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>87061</td>
      <td>31297</td>
      <td>0.735573</td>
      <td>0.264427</td>
    </tr>
  </tbody>
</table>
</div>



Plot the above data for visualization:


![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_21_0.png)


From the above plot, we can see that Janurary, July, August and Feburary in 2016 in that order have the highest occupancy rates.  Therefore, you may want to avoid these months if you are the renter.

Now, we want to compute the occupancy rate of each listing and include them in the next dataset.   Following is number of days when the place is vacant for each listing:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>vacancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3335</td>
      <td>307</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4291</td>
      <td>363</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5682</td>
      <td>308</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6606</td>
      <td>363</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7369</td>
      <td>53</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>10331249</td>
      <td>352</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>10332096</td>
      <td>363</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>10334184</td>
      <td>359</td>
    </tr>
    <tr>
      <th>3687</th>
      <td>10339145</td>
      <td>363</td>
    </tr>
    <tr>
      <th>3688</th>
      <td>10340165</td>
      <td>356</td>
    </tr>
  </tbody>
</table>
<p>3689 rows × 2 columns</p>
</div>



Below is number of days when the place is occupied for each listing:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>occupancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3335</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5682</td>
      <td>55</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7369</td>
      <td>310</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9460</td>
      <td>306</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9531</td>
      <td>185</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3116</th>
      <td>10319529</td>
      <td>362</td>
    </tr>
    <tr>
      <th>3117</th>
      <td>10331249</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>10334184</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3119</th>
      <td>10339144</td>
      <td>363</td>
    </tr>
    <tr>
      <th>3120</th>
      <td>10340165</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>3121 rows × 2 columns</p>
</div>



Combining these two pieces of data, we get:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>vacancy</th>
      <th>occupancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3335</td>
      <td>307</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4291</td>
      <td>363</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5682</td>
      <td>308</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6606</td>
      <td>363</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7369</td>
      <td>53</td>
      <td>310.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>10331249</td>
      <td>352</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>10332096</td>
      <td>363</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>10334184</td>
      <td>359</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3687</th>
      <td>10339145</td>
      <td>363</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3688</th>
      <td>10340165</td>
      <td>356</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
<p>3689 rows × 3 columns</p>
</div>



Note that some entries in the columns 'vacancy' and	'occupancy' may have missing values.  We replace these missing values with zeros.  Now, this dataset looks better.




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>vacancy</th>
      <th>occupancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3335</td>
      <td>307</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4291</td>
      <td>363</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5682</td>
      <td>308</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6606</td>
      <td>363</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7369</td>
      <td>53</td>
      <td>310.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>10331249</td>
      <td>352</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>10332096</td>
      <td>363</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>10334184</td>
      <td>359</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3687</th>
      <td>10339145</td>
      <td>363</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3688</th>
      <td>10340165</td>
      <td>356</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
<p>3689 rows × 3 columns</p>
</div>



Now, we compute the occupancy rate and remove irrelevant columns.  We will keep this data for later.




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>occupancy_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3335</td>
      <td>0.154270</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4291</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5682</td>
      <td>0.151515</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6606</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7369</td>
      <td>0.853994</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>10331249</td>
      <td>0.030303</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>10332096</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3686</th>
      <td>10334184</td>
      <td>0.011019</td>
    </tr>
    <tr>
      <th>3687</th>
      <td>10339145</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3688</th>
      <td>10340165</td>
      <td>0.019284</td>
    </tr>
  </tbody>
</table>
<p>3689 rows × 2 columns</p>
</div>



Read in the data for all the listings.

Now, insert the occupancy rate into this dataset:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>241032</td>
      <td>953595</td>
    </tr>
    <tr>
      <th>listing_url</th>
      <td>https://www.airbnb.com/rooms/241032</td>
      <td>https://www.airbnb.com/rooms/953595</td>
    </tr>
    <tr>
      <th>scrape_id</th>
      <td>20160104002432</td>
      <td>20160104002432</td>
    </tr>
    <tr>
      <th>last_scraped</th>
      <td>2016-01-04</td>
      <td>2016-01-04</td>
    </tr>
    <tr>
      <th>name</th>
      <td>Stylish Queen Anne Apartment</td>
      <td>Bright &amp; Airy Queen Anne Apartment</td>
    </tr>
    <tr>
      <th>summary</th>
      <td>NaN</td>
      <td>Chemically sensitive? We've removed the irrita...</td>
    </tr>
    <tr>
      <th>space</th>
      <td>Make your self at home in this charming one-be...</td>
      <td>Beautiful, hypoallergenic apartment in an extr...</td>
    </tr>
    <tr>
      <th>description</th>
      <td>Make your self at home in this charming one-be...</td>
      <td>Chemically sensitive? We've removed the irrita...</td>
    </tr>
    <tr>
      <th>experiences_offered</th>
      <td>none</td>
      <td>none</td>
    </tr>
    <tr>
      <th>neighborhood_overview</th>
      <td>NaN</td>
      <td>Queen Anne is a wonderful, truly functional vi...</td>
    </tr>
    <tr>
      <th>notes</th>
      <td>NaN</td>
      <td>What's up with the free pillows?  Our home was...</td>
    </tr>
    <tr>
      <th>transit</th>
      <td>NaN</td>
      <td>Convenient bus stops are just down the block, ...</td>
    </tr>
    <tr>
      <th>thumbnail_url</th>
      <td>NaN</td>
      <td>https://a0.muscache.com/ac/pictures/14409893/f...</td>
    </tr>
    <tr>
      <th>medium_url</th>
      <td>NaN</td>
      <td>https://a0.muscache.com/im/pictures/14409893/f...</td>
    </tr>
    <tr>
      <th>picture_url</th>
      <td>https://a1.muscache.com/ac/pictures/67560560/c...</td>
      <td>https://a0.muscache.com/ac/pictures/14409893/f...</td>
    </tr>
    <tr>
      <th>xl_picture_url</th>
      <td>NaN</td>
      <td>https://a0.muscache.com/ac/pictures/14409893/f...</td>
    </tr>
    <tr>
      <th>host_id</th>
      <td>956883</td>
      <td>5177328</td>
    </tr>
    <tr>
      <th>host_url</th>
      <td>https://www.airbnb.com/users/show/956883</td>
      <td>https://www.airbnb.com/users/show/5177328</td>
    </tr>
    <tr>
      <th>host_name</th>
      <td>Maija</td>
      <td>Andrea</td>
    </tr>
    <tr>
      <th>host_since</th>
      <td>2011-08-11</td>
      <td>2013-02-21</td>
    </tr>
    <tr>
      <th>host_location</th>
      <td>Seattle, Washington, United States</td>
      <td>Seattle, Washington, United States</td>
    </tr>
    <tr>
      <th>host_about</th>
      <td>I am an artist, interior designer, and run a s...</td>
      <td>Living east coast/left coast/overseas.  Time i...</td>
    </tr>
    <tr>
      <th>host_response_time</th>
      <td>within a few hours</td>
      <td>within an hour</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>96%</td>
      <td>98%</td>
    </tr>
    <tr>
      <th>host_acceptance_rate</th>
      <td>100%</td>
      <td>100%</td>
    </tr>
    <tr>
      <th>host_is_superhost</th>
      <td>f</td>
      <td>t</td>
    </tr>
    <tr>
      <th>host_thumbnail_url</th>
      <td>https://a0.muscache.com/ac/users/956883/profil...</td>
      <td>https://a0.muscache.com/ac/users/5177328/profi...</td>
    </tr>
    <tr>
      <th>host_picture_url</th>
      <td>https://a0.muscache.com/ac/users/956883/profil...</td>
      <td>https://a0.muscache.com/ac/users/5177328/profi...</td>
    </tr>
    <tr>
      <th>host_neighbourhood</th>
      <td>Queen Anne</td>
      <td>Queen Anne</td>
    </tr>
    <tr>
      <th>host_listings_count</th>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>host_total_listings_count</th>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>host_verifications</th>
      <td>['email', 'phone', 'reviews', 'kba']</td>
      <td>['email', 'phone', 'facebook', 'linkedin', 're...</td>
    </tr>
    <tr>
      <th>host_has_profile_pic</th>
      <td>t</td>
      <td>t</td>
    </tr>
    <tr>
      <th>host_identity_verified</th>
      <td>t</td>
      <td>t</td>
    </tr>
    <tr>
      <th>street</th>
      <td>Gilman Dr W, Seattle, WA 98119, United States</td>
      <td>7th Avenue West, Seattle, WA 98119, United States</td>
    </tr>
    <tr>
      <th>neighbourhood</th>
      <td>Queen Anne</td>
      <td>Queen Anne</td>
    </tr>
    <tr>
      <th>neighbourhood_cleansed</th>
      <td>West Queen Anne</td>
      <td>West Queen Anne</td>
    </tr>
    <tr>
      <th>neighbourhood_group_cleansed</th>
      <td>Queen Anne</td>
      <td>Queen Anne</td>
    </tr>
    <tr>
      <th>city</th>
      <td>Seattle</td>
      <td>Seattle</td>
    </tr>
    <tr>
      <th>state</th>
      <td>WA</td>
      <td>WA</td>
    </tr>
    <tr>
      <th>zipcode</th>
      <td>98119</td>
      <td>98119</td>
    </tr>
    <tr>
      <th>market</th>
      <td>Seattle</td>
      <td>Seattle</td>
    </tr>
    <tr>
      <th>smart_location</th>
      <td>Seattle, WA</td>
      <td>Seattle, WA</td>
    </tr>
    <tr>
      <th>country_code</th>
      <td>US</td>
      <td>US</td>
    </tr>
    <tr>
      <th>country</th>
      <td>United States</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>47.6363</td>
      <td>47.6391</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>-122.371</td>
      <td>-122.366</td>
    </tr>
    <tr>
      <th>is_location_exact</th>
      <td>t</td>
      <td>t</td>
    </tr>
    <tr>
      <th>property_type</th>
      <td>Apartment</td>
      <td>Apartment</td>
    </tr>
    <tr>
      <th>room_type</th>
      <td>Entire home/apt</td>
      <td>Entire home/apt</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>bed_type</th>
      <td>Real Bed</td>
      <td>Real Bed</td>
    </tr>
    <tr>
      <th>amenities</th>
      <td>{TV,"Cable TV",Internet,"Wireless Internet","A...</td>
      <td>{TV,Internet,"Wireless Internet",Kitchen,"Free...</td>
    </tr>
    <tr>
      <th>square_feet</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>price</th>
      <td>$85.00</td>
      <td>$150.00</td>
    </tr>
    <tr>
      <th>weekly_price</th>
      <td>NaN</td>
      <td>$1,000.00</td>
    </tr>
    <tr>
      <th>monthly_price</th>
      <td>NaN</td>
      <td>$3,000.00</td>
    </tr>
    <tr>
      <th>security_deposit</th>
      <td>NaN</td>
      <td>$100.00</td>
    </tr>
    <tr>
      <th>cleaning_fee</th>
      <td>NaN</td>
      <td>$40.00</td>
    </tr>
    <tr>
      <th>guests_included</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>extra_people</th>
      <td>$5.00</td>
      <td>$0.00</td>
    </tr>
    <tr>
      <th>minimum_nights</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>maximum_nights</th>
      <td>365</td>
      <td>90</td>
    </tr>
    <tr>
      <th>calendar_updated</th>
      <td>4 weeks ago</td>
      <td>today</td>
    </tr>
    <tr>
      <th>has_availability</th>
      <td>t</td>
      <td>t</td>
    </tr>
    <tr>
      <th>availability_30</th>
      <td>14</td>
      <td>13</td>
    </tr>
    <tr>
      <th>availability_60</th>
      <td>41</td>
      <td>13</td>
    </tr>
    <tr>
      <th>availability_90</th>
      <td>71</td>
      <td>16</td>
    </tr>
    <tr>
      <th>availability_365</th>
      <td>346</td>
      <td>291</td>
    </tr>
    <tr>
      <th>calendar_last_scraped</th>
      <td>2016-01-04</td>
      <td>2016-01-04</td>
    </tr>
    <tr>
      <th>number_of_reviews</th>
      <td>207</td>
      <td>43</td>
    </tr>
    <tr>
      <th>first_review</th>
      <td>2011-11-01</td>
      <td>2013-08-19</td>
    </tr>
    <tr>
      <th>last_review</th>
      <td>2016-01-02</td>
      <td>2015-12-29</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>95</td>
      <td>96</td>
    </tr>
    <tr>
      <th>review_scores_accuracy</th>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>review_scores_cleanliness</th>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>review_scores_checkin</th>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>review_scores_communication</th>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>review_scores_location</th>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>review_scores_value</th>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>requires_license</th>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>license</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>jurisdiction_names</th>
      <td>WASHINGTON</td>
      <td>WASHINGTON</td>
    </tr>
    <tr>
      <th>instant_bookable</th>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>cancellation_policy</th>
      <td>moderate</td>
      <td>strict</td>
    </tr>
    <tr>
      <th>require_guest_profile_picture</th>
      <td>f</td>
      <td>t</td>
    </tr>
    <tr>
      <th>require_guest_phone_verification</th>
      <td>f</td>
      <td>t</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count</th>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>reviews_per_month</th>
      <td>4.07</td>
      <td>1.48</td>
    </tr>
    <tr>
      <th>occupancy_rate</th>
      <td>0.0523416</td>
      <td>0.203857</td>
    </tr>
  </tbody>
</table>
</div>



### Prepare Data


Now, let us look at how many columns have missing values.  The following shows the count and percentage of missing values in each column.




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_nan_count</th>
      <th>col_nan_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>beds</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>property_type</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>host_identity_verified</th>
      <td>2</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>host_has_profile_pic</th>
      <td>2</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>host_total_listings_count</th>
      <td>2</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>host_name</th>
      <td>2</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>host_since</th>
      <td>2</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>host_listings_count</th>
      <td>2</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>host_picture_url</th>
      <td>2</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>host_thumbnail_url</th>
      <td>2</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>host_is_superhost</th>
      <td>2</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>6</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>zipcode</th>
      <td>7</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>host_location</th>
      <td>8</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>16</td>
      <td>0.42</td>
    </tr>
    <tr>
      <th>summary</th>
      <td>177</td>
      <td>4.64</td>
    </tr>
    <tr>
      <th>host_neighbourhood</th>
      <td>300</td>
      <td>7.86</td>
    </tr>
    <tr>
      <th>xl_picture_url</th>
      <td>320</td>
      <td>8.38</td>
    </tr>
    <tr>
      <th>medium_url</th>
      <td>320</td>
      <td>8.38</td>
    </tr>
    <tr>
      <th>thumbnail_url</th>
      <td>320</td>
      <td>8.38</td>
    </tr>
    <tr>
      <th>neighbourhood</th>
      <td>416</td>
      <td>10.90</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>523</td>
      <td>13.70</td>
    </tr>
    <tr>
      <th>host_response_time</th>
      <td>523</td>
      <td>13.70</td>
    </tr>
    <tr>
      <th>space</th>
      <td>569</td>
      <td>14.90</td>
    </tr>
    <tr>
      <th>last_review</th>
      <td>627</td>
      <td>16.42</td>
    </tr>
    <tr>
      <th>first_review</th>
      <td>627</td>
      <td>16.42</td>
    </tr>
    <tr>
      <th>reviews_per_month</th>
      <td>627</td>
      <td>16.42</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>647</td>
      <td>16.95</td>
    </tr>
    <tr>
      <th>review_scores_communication</th>
      <td>651</td>
      <td>17.05</td>
    </tr>
    <tr>
      <th>review_scores_cleanliness</th>
      <td>653</td>
      <td>17.10</td>
    </tr>
    <tr>
      <th>review_scores_location</th>
      <td>655</td>
      <td>17.16</td>
    </tr>
    <tr>
      <th>review_scores_value</th>
      <td>656</td>
      <td>17.18</td>
    </tr>
    <tr>
      <th>review_scores_checkin</th>
      <td>658</td>
      <td>17.23</td>
    </tr>
    <tr>
      <th>review_scores_accuracy</th>
      <td>658</td>
      <td>17.23</td>
    </tr>
    <tr>
      <th>host_acceptance_rate</th>
      <td>773</td>
      <td>20.25</td>
    </tr>
    <tr>
      <th>host_about</th>
      <td>859</td>
      <td>22.50</td>
    </tr>
    <tr>
      <th>transit</th>
      <td>934</td>
      <td>24.46</td>
    </tr>
    <tr>
      <th>cleaning_fee</th>
      <td>1030</td>
      <td>26.98</td>
    </tr>
    <tr>
      <th>neighborhood_overview</th>
      <td>1032</td>
      <td>27.03</td>
    </tr>
    <tr>
      <th>notes</th>
      <td>1606</td>
      <td>42.06</td>
    </tr>
    <tr>
      <th>weekly_price</th>
      <td>1809</td>
      <td>47.38</td>
    </tr>
    <tr>
      <th>security_deposit</th>
      <td>1952</td>
      <td>51.13</td>
    </tr>
    <tr>
      <th>monthly_price</th>
      <td>2301</td>
      <td>60.27</td>
    </tr>
    <tr>
      <th>square_feet</th>
      <td>3721</td>
      <td>97.46</td>
    </tr>
    <tr>
      <th>license</th>
      <td>3818</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>



Let us plot this data.  Note that the column with the highest percentage of missing values will be on top.


![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_50_0.png)


Following is the distribution of the number of columns that have missing values against the percentage of missing values.


![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_54_0.png)


Although the above plot is Multinomial Distribution, we will still use the mean + 1 std to remove outlier columns.  Therefore, we will remove any columns that have more than 40.25% missing values.  Following are these columns:





    ['notes',
     'weekly_price',
     'security_deposit',
     'monthly_price',
     'square_feet',
     'license']



Get the list of numeric features:

    29 columns have numeric values:





    ['id',
     'scrape_id',
     'host_id',
     'host_listings_count',
     'host_total_listings_count',
     'latitude',
     'longitude',
     'accommodates',
     'bathrooms',
     'bedrooms',
     'beds',
     'guests_included',
     'minimum_nights',
     'maximum_nights',
     'availability_30',
     'availability_60',
     'availability_90',
     'availability_365',
     'number_of_reviews',
     'review_scores_rating',
     'review_scores_accuracy',
     'review_scores_cleanliness',
     'review_scores_checkin',
     'review_scores_communication',
     'review_scores_location',
     'review_scores_value',
     'calculated_host_listings_count',
     'reviews_per_month',
     'occupancy_rate']



Find those numeric columns that we want to keep for modeling.  Following is the list of numeric columns that we want to use for the model.




    ['reviews_per_month',
     'availability_60',
     'occupancy_rate',
     'guests_included',
     'host_total_listings_count',
     'calculated_host_listings_count',
     'maximum_nights',
     'availability_30',
     'availability_90',
     'number_of_reviews',
     'bedrooms',
     'minimum_nights',
     'beds',
     'availability_365',
     'review_scores_rating',
     'accommodates',
     'bathrooms',
     'price',
     'extra_people',
     'cleaning_fee',
     'host_response_rate',
     'host_acceptance_rate']



Following is the list of string columns:

    53 columns have string values:





    ['listing_url',
     'last_scraped',
     'name',
     'summary',
     'space',
     'description',
     'experiences_offered',
     'neighborhood_overview',
     'transit',
     'thumbnail_url',
     'medium_url',
     'picture_url',
     'xl_picture_url',
     'host_url',
     'host_name',
     'host_since',
     'host_location',
     'host_about',
     'host_response_time',
     'host_is_superhost',
     'host_thumbnail_url',
     'host_picture_url',
     'host_neighbourhood',
     'host_verifications',
     'host_has_profile_pic',
     'host_identity_verified',
     'street',
     'neighbourhood',
     'neighbourhood_cleansed',
     'neighbourhood_group_cleansed',
     'city',
     'state',
     'zipcode',
     'market',
     'smart_location',
     'country_code',
     'country',
     'is_location_exact',
     'property_type',
     'room_type',
     'bed_type',
     'amenities',
     'calendar_updated',
     'has_availability',
     'calendar_last_scraped',
     'first_review',
     'last_review',
     'requires_license',
     'jurisdiction_names',
     'instant_bookable',
     'cancellation_policy',
     'require_guest_profile_picture',
     'require_guest_phone_verification']



This is the list of potential categorical features for our model:




    ['host_identity_verified',
     'host_response_time',
     'property_type',
     'host_verifications',
     'amenities',
     'zipcode',
     'require_guest_profile_picture',
     'requires_license',
     'host_is_superhost',
     'cancellation_policy',
     'require_guest_phone_verification',
     'room_type',
     'jurisdiction_names',
     'instant_bookable',
     'host_location',
     'bed_type',
     'experiences_offered',
     'neighbourhood_cleansed',
     'calendar_updated',
     'has_availability',
     'is_location_exact',
     'host_neighbourhood',
     'host_has_profile_pic']



We want to exclude any columns from above that have lists as values.  This is to simplify the model, as there are quite a number of unique members in these lists.  Thus, we exclude the following two columns:




    ['host_verifications', 'amenities']



    19 members in list - ["'email'", " 'phone'", " 'reviews'", " 'kba'", " 'facebook'", " 'linkedin'", " 'jumio'", " 'google'", "'phone'", " 'manual_offline'", " 'amex'", " 'manual_online'", " 'sent_id'", " 'photographer'", '', 'None', " 'weibo'", "'google'", "'reviews'"]
    42 members in list - ['TV', '"Cable TV"', 'Internet', '"Wireless Internet"', '"Air Conditioning"', 'Kitchen', 'Heating', '"Family/Kid Friendly"', 'Washer', 'Dryer', '"Free Parking on Premises"', '"Buzzer/Wireless Intercom"', '"Smoke Detector"', '"Carbon Monoxide Detector"', '"First Aid Kit"', '"Safety Card"', '"Fire Extinguisher"', 'Essentials', '"Pets Allowed"', '"Pets live on this property"', 'Dog(s)', 'Cat(s)', '"Hot Tub"', '"Indoor Fireplace"', 'Shampoo', 'Breakfast', '"24-Hour Check-in"', 'Hangers', '"Hair Dryer"', 'Iron', '"Laptop Friendly Workspace"', '"Suitable for Events"', '"Elevator in Building"', '"Lock on Bedroom Door"', '"Wheelchair Accessible"', 'Gym', '', 'Pool', '"Smoking Allowed"', '"Other pet(s)"', 'Doorman', '"Washer / Dryer"']


Following is the list of potential categorical features for our model:




    ['host_identity_verified',
     'host_response_time',
     'jurisdiction_names',
     'property_type',
     'instant_bookable',
     'host_location',
     'bed_type',
     'zipcode',
     'experiences_offered',
     'require_guest_profile_picture',
     'neighbourhood_cleansed',
     'calendar_updated',
     'has_availability',
     'is_location_exact',
     'requires_license',
     'host_neighbourhood',
     'host_is_superhost',
     'cancellation_policy',
     'host_has_profile_pic',
     'require_guest_phone_verification',
     'room_type']



Plot the distribution of categorical feature's count of unique values.  The ones on the top will have higher numbers of unique values.




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff68b321630>




![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_85_1.png)


From the plot, we will remove any non-numeric features with more than 20 unique values.  We also removed any columns that have only 1 unique value since it does not provide net gain of information


    9 columns to be removed = ['jurisdiction_names', 'requires_license', 'experiences_offered', 'has_availability', 'zipcode', 'calendar_updated', 'neighbourhood_cleansed', 'host_neighbourhood', 'host_location']


Following is the list of numeric columns that have missing values:

    column reviews_per_month has 627 row(s) of missing value(s):
    column host_total_listings_count has 2 row(s) of missing value(s):
    column bedrooms has 6 row(s) of missing value(s):
    column beds has 1 row(s) of missing value(s):
    column review_scores_rating has 647 row(s) of missing value(s):
    column bathrooms has 16 row(s) of missing value(s):
    column cleaning_fee has 1030 row(s) of missing value(s):
    column host_response_rate has 523 row(s) of missing value(s):
    column host_acceptance_rate has 773 row(s) of missing value(s):


It is difficult to impute a value for the missing values in the columns beds, bedrooms, and bathrooms, as they should be important features for our model.  But, since there are only relatively few rows missing values in these columns, we can delete these rows.

Columns reviews_per_month, review_scores_rating, cleaning_fee, host_response_rate and host_acceptance_rate has a large number of missing values.  We will create a new column for each of these columns to indicate which row has missing values in these columns.

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      if sys.path[0] == '':


Below is the list of categorical features that have missing values:

    column host_identity_verified has 2 row(s) of missing value(s):
    column host_response_time has 521 row(s) of missing value(s):
    column property_type has 1 row(s) of missing value(s):
    column host_is_superhost has 2 row(s) of missing value(s):
    column host_has_profile_pic has 2 row(s) of missing value(s):


Column property_type seems to be an important feature, and there is only 1 row having missing values.  We will remove this row.

Next, we will replace missing values with appropriate values:

Now, verify that we do not have any missing values for all the features for our model:

    
    
    There are total of 3795 row(s) and 39 column(s) in this dataset
    
    List of column(s):
    Index(['reviews_per_month', 'availability_60', 'occupancy_rate',
           'guests_included', 'host_total_listings_count',
           'calculated_host_listings_count', 'maximum_nights', 'availability_30',
           'availability_90', 'number_of_reviews', 'bedrooms', 'minimum_nights',
           'beds', 'availability_365', 'review_scores_rating', 'accommodates',
           'bathrooms', 'price', 'extra_people', 'cleaning_fee',
           'host_response_rate', 'host_acceptance_rate', 'host_identity_verified',
           'host_response_time', 'require_guest_profile_picture', 'property_type',
           'instant_bookable', 'is_location_exact', 'bed_type',
           'host_is_superhost', 'cancellation_policy', 'host_has_profile_pic',
           'require_guest_phone_verification', 'room_type',
           'reviews_per_month_missing', 'review_scores_rating_missing',
           'cleaning_fee_missing', 'host_response_rate_missing',
           'host_acceptance_rate_missing'],
          dtype='object')
    
    
    0 column(s) have missing values:


### Model Data

Now, we perform on-hot encoding for all categorical features.  The final dataset is as follows:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>reviews_per_month</th>
      <td>4.070000</td>
      <td>1.480000</td>
      <td>1.150000</td>
      <td>0.02000</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>availability_60</th>
      <td>41.000000</td>
      <td>13.000000</td>
      <td>6.000000</td>
      <td>0.00000</td>
      <td>60.00</td>
    </tr>
    <tr>
      <th>occupancy_rate</th>
      <td>0.052342</td>
      <td>0.203857</td>
      <td>0.399449</td>
      <td>0.61157</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>guests_included</th>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>1.00000</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>host_total_listings_count</th>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>1.00000</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count</th>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>1.00000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>maximum_nights</th>
      <td>365.000000</td>
      <td>90.000000</td>
      <td>30.000000</td>
      <td>1125.00000</td>
      <td>1125.00</td>
    </tr>
    <tr>
      <th>availability_30</th>
      <td>14.000000</td>
      <td>13.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>30.00</td>
    </tr>
    <tr>
      <th>availability_90</th>
      <td>71.000000</td>
      <td>16.000000</td>
      <td>17.000000</td>
      <td>0.00000</td>
      <td>90.00</td>
    </tr>
    <tr>
      <th>number_of_reviews</th>
      <td>207.000000</td>
      <td>43.000000</td>
      <td>20.000000</td>
      <td>0.00000</td>
      <td>38.00</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>0.00000</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>minimum_nights</th>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>1.00000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>2.00000</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>availability_365</th>
      <td>346.000000</td>
      <td>291.000000</td>
      <td>220.000000</td>
      <td>143.00000</td>
      <td>365.00</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>95.000000</td>
      <td>96.000000</td>
      <td>97.000000</td>
      <td>20.00000</td>
      <td>92.00</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>11.000000</td>
      <td>3.00000</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.500000</td>
      <td>1.00000</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>price</th>
      <td>85.000000</td>
      <td>150.000000</td>
      <td>975.000000</td>
      <td>100.00000</td>
      <td>450.00</td>
    </tr>
    <tr>
      <th>extra_people</th>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.00000</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>cleaning_fee</th>
      <td>5.000000</td>
      <td>40.000000</td>
      <td>300.000000</td>
      <td>5.00000</td>
      <td>125.00</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>0.960000</td>
      <td>0.980000</td>
      <td>0.670000</td>
      <td>0.17000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>host_acceptance_rate</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>reviews_per_month_missing</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>review_scores_rating_missing</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>cleaning_fee_missing</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>host_response_rate_missing</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>host_acceptance_rate_missing</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>host_identity_verified_t</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>host_response_time_unknown</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>host_response_time_within a day</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>host_response_time_within a few hours</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>host_response_time_within an hour</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>require_guest_profile_picture_t</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Bed &amp; Breakfast</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Boat</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Bungalow</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Cabin</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Camper/RV</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Chalet</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Condominium</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Dorm</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_House</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>property_type_Loft</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Other</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Tent</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Townhouse</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Treehouse</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>property_type_Yurt</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>instant_bookable_t</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>is_location_exact_t</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>bed_type_Couch</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>bed_type_Futon</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>bed_type_Pull-out Sofa</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>bed_type_Real Bed</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>host_is_superhost_t</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>cancellation_policy_moderate</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>cancellation_policy_strict</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>host_has_profile_pic_t</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>require_guest_phone_verification_t</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>room_type_Private room</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>room_type_Shared room</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



Our purpose is to find groups of listings that are similar to each other.  Thus, the unsupervised learning method such as KMeans clustering would be a good choice here.   

Perform KMeans clustering for k from 1 to 30 to decide which value of K is a good choice.

    Clustering done for k = 1, with SSE 227700.0
    Clustering done for k = 2, with SSE 209295.51990079944
    Clustering done for k = 3, with SSE 199175.9833609662
    Clustering done for k = 4, with SSE 191219.18249103503
    Clustering done for k = 5, with SSE 185598.15897467945
    Clustering done for k = 6, with SSE 180608.94345094438
    Clustering done for k = 7, with SSE 175343.93582336744
    Clustering done for k = 8, with SSE 171610.8923381806
    Clustering done for k = 9, with SSE 168854.56026872498
    Clustering done for k = 10, with SSE 165646.85787909233
    Clustering done for k = 11, with SSE 161283.1235498798
    Clustering done for k = 12, with SSE 157472.02682657097
    Clustering done for k = 13, with SSE 153654.9888203994
    Clustering done for k = 14, with SSE 150061.55056951637
    Clustering done for k = 15, with SSE 148212.15986698648
    Clustering done for k = 16, with SSE 144659.42698716748
    Clustering done for k = 17, with SSE 139670.2207141205
    Clustering done for k = 18, with SSE 137974.02490163816
    Clustering done for k = 19, with SSE 135880.0604575394
    Clustering done for k = 20, with SSE 132000.8061167076
    Clustering done for k = 21, with SSE 128488.29774471794
    Clustering done for k = 22, with SSE 124679.38562879313
    Clustering done for k = 23, with SSE 122540.16646940258
    Clustering done for k = 24, with SSE 118803.52371723476
    Clustering done for k = 25, with SSE 117086.20454151196
    Clustering done for k = 26, with SSE 113805.93992263741
    Clustering done for k = 27, with SSE 110041.6418389026
    Clustering done for k = 28, with SSE 108277.20423259994
    Clustering done for k = 29, with SSE 105879.50405920515
    Clustering done for k = 30, with SSE 102667.39488414857


Plot of Sum of Square Error (SSE) against k:


![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_118_0.png)


From the above plot, it is difficult to decide on the value of k since SSE is always decreasing.  Let plot the relative SSE against K.


![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_120_0.png)


From the above plot, it seems when k = 15 is a good choice.

Add these cluster labels, price, longitude and latitude information to the final dataset:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>reviews_per_month</th>
      <td>4.07</td>
      <td>1.48</td>
      <td>1.15</td>
      <td>0.02</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>availability_60</th>
      <td>41</td>
      <td>13</td>
      <td>6</td>
      <td>0</td>
      <td>60</td>
    </tr>
    <tr>
      <th>occupancy_rate</th>
      <td>0.0523416</td>
      <td>0.203857</td>
      <td>0.399449</td>
      <td>0.61157</td>
      <td>0</td>
    </tr>
    <tr>
      <th>guests_included</th>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>host_total_listings_count</th>
      <td>3</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>calculated_host_listings_count</th>
      <td>2</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>maximum_nights</th>
      <td>365</td>
      <td>90</td>
      <td>30</td>
      <td>1125</td>
      <td>1125</td>
    </tr>
    <tr>
      <th>availability_30</th>
      <td>14</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>availability_90</th>
      <td>71</td>
      <td>16</td>
      <td>17</td>
      <td>0</td>
      <td>90</td>
    </tr>
    <tr>
      <th>number_of_reviews</th>
      <td>207</td>
      <td>43</td>
      <td>20</td>
      <td>0</td>
      <td>38</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>minimum_nights</th>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>availability_365</th>
      <td>346</td>
      <td>291</td>
      <td>220</td>
      <td>143</td>
      <td>365</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>95</td>
      <td>96</td>
      <td>97</td>
      <td>20</td>
      <td>92</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>4</td>
      <td>4</td>
      <td>11</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>1</td>
      <td>1</td>
      <td>4.5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>extra_people</th>
      <td>5</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>cleaning_fee</th>
      <td>5</td>
      <td>40</td>
      <td>300</td>
      <td>5</td>
      <td>125</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>0.96</td>
      <td>0.98</td>
      <td>0.67</td>
      <td>0.17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>host_acceptance_rate</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>host_identity_verified</th>
      <td>t</td>
      <td>t</td>
      <td>t</td>
      <td>t</td>
      <td>t</td>
    </tr>
    <tr>
      <th>host_response_time</th>
      <td>within a few hours</td>
      <td>within an hour</td>
      <td>within a few hours</td>
      <td>unknown</td>
      <td>within an hour</td>
    </tr>
    <tr>
      <th>require_guest_profile_picture</th>
      <td>f</td>
      <td>t</td>
      <td>f</td>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>property_type</th>
      <td>Apartment</td>
      <td>Apartment</td>
      <td>House</td>
      <td>Apartment</td>
      <td>House</td>
    </tr>
    <tr>
      <th>instant_bookable</th>
      <td>f</td>
      <td>f</td>
      <td>f</td>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>is_location_exact</th>
      <td>t</td>
      <td>t</td>
      <td>t</td>
      <td>t</td>
      <td>t</td>
    </tr>
    <tr>
      <th>bed_type</th>
      <td>Real Bed</td>
      <td>Real Bed</td>
      <td>Real Bed</td>
      <td>Real Bed</td>
      <td>Real Bed</td>
    </tr>
    <tr>
      <th>host_is_superhost</th>
      <td>f</td>
      <td>t</td>
      <td>f</td>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>cancellation_policy</th>
      <td>moderate</td>
      <td>strict</td>
      <td>strict</td>
      <td>flexible</td>
      <td>strict</td>
    </tr>
    <tr>
      <th>host_has_profile_pic</th>
      <td>t</td>
      <td>t</td>
      <td>t</td>
      <td>t</td>
      <td>t</td>
    </tr>
    <tr>
      <th>require_guest_phone_verification</th>
      <td>f</td>
      <td>t</td>
      <td>f</td>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>room_type</th>
      <td>Entire home/apt</td>
      <td>Entire home/apt</td>
      <td>Entire home/apt</td>
      <td>Entire home/apt</td>
      <td>Entire home/apt</td>
    </tr>
    <tr>
      <th>reviews_per_month_missing</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>review_scores_rating_missing</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cleaning_fee_missing</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>host_response_rate_missing</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>host_acceptance_rate_missing</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Cluster Labels</th>
      <td>7</td>
      <td>5</td>
      <td>3</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>47.6363</td>
      <td>47.6391</td>
      <td>47.6297</td>
      <td>47.6385</td>
      <td>47.6329</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>-122.371</td>
      <td>-122.366</td>
      <td>-122.369</td>
      <td>-122.369</td>
      <td>-122.372</td>
    </tr>
    <tr>
      <th>neighbourhood_cleansed</th>
      <td>West Queen Anne</td>
      <td>West Queen Anne</td>
      <td>West Queen Anne</td>
      <td>West Queen Anne</td>
      <td>West Queen Anne</td>
    </tr>
    <tr>
      <th>price</th>
      <td>85</td>
      <td>150</td>
      <td>975</td>
      <td>100</td>
      <td>450</td>
    </tr>
  </tbody>
</table>
</div>


#### Examine Clusters

Now, let us look at the characteristics of the 3 clusters with the most listings.

The following table shows the distribution of numeric features:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>14</th>
      <th>1</th>
      <th>7</th>
      <th>4</th>
      <th>0</th>
      <th>10</th>
      <th>5</th>
      <th>3</th>
      <th>6</th>
      <th>2</th>
      <th>12</th>
      <th>9</th>
      <th>13</th>
      <th>11</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cluster Labels</th>
      <td>14.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>12.000000</td>
      <td>9.000000</td>
      <td>13.000000</td>
      <td>11.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>0.977636</td>
      <td>1.231986</td>
      <td>0.906810</td>
      <td>1.380042</td>
      <td>1.260989</td>
      <td>1.428082</td>
      <td>1.298932</td>
      <td>3.360169</td>
      <td>1.198953</td>
      <td>0.851240</td>
      <td>1.870370</td>
      <td>1.076923</td>
      <td>0.692308</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>1.241214</td>
      <td>1.550088</td>
      <td>1.340502</td>
      <td>2.063694</td>
      <td>1.620879</td>
      <td>1.664384</td>
      <td>1.811388</td>
      <td>4.296610</td>
      <td>1.544503</td>
      <td>1.074380</td>
      <td>2.629630</td>
      <td>1.153846</td>
      <td>1.692308</td>
      <td>1.400000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>1.138978</td>
      <td>1.157293</td>
      <td>1.074373</td>
      <td>1.169851</td>
      <td>1.217033</td>
      <td>1.275685</td>
      <td>1.302491</td>
      <td>2.544492</td>
      <td>1.219895</td>
      <td>1.033058</td>
      <td>1.703704</td>
      <td>1.000000</td>
      <td>0.846154</td>
      <td>0.500000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>guests_included</th>
      <td>1.279553</td>
      <td>1.550088</td>
      <td>1.458781</td>
      <td>1.923567</td>
      <td>1.376374</td>
      <td>1.582192</td>
      <td>1.594306</td>
      <td>4.377119</td>
      <td>1.340314</td>
      <td>1.330579</td>
      <td>1.018519</td>
      <td>1.000000</td>
      <td>1.307692</td>
      <td>1.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>2.337061</td>
      <td>3.154657</td>
      <td>2.629032</td>
      <td>4.044586</td>
      <td>3.019231</td>
      <td>3.215753</td>
      <td>3.669039</td>
      <td>7.923729</td>
      <td>2.869110</td>
      <td>1.983471</td>
      <td>5.314815</td>
      <td>1.769231</td>
      <td>2.615385</td>
      <td>2.200000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>occupancy_rate</th>
      <td>0.114994</td>
      <td>0.530765</td>
      <td>0.311176</td>
      <td>0.206922</td>
      <td>0.341113</td>
      <td>0.320371</td>
      <td>0.272335</td>
      <td>0.301338</td>
      <td>0.362165</td>
      <td>0.278487</td>
      <td>0.061473</td>
      <td>0.331638</td>
      <td>0.279085</td>
      <td>0.322865</td>
      <td>0.512397</td>
    </tr>
    <tr>
      <th>price</th>
      <td>91.795527</td>
      <td>114.891037</td>
      <td>87.960573</td>
      <td>142.178344</td>
      <td>114.758242</td>
      <td>153.523973</td>
      <td>146.765125</td>
      <td>302.902542</td>
      <td>140.984293</td>
      <td>72.495868</td>
      <td>171.074074</td>
      <td>64.307692</td>
      <td>120.461538</td>
      <td>54.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>extra_people</th>
      <td>8.266773</td>
      <td>8.847100</td>
      <td>10.736559</td>
      <td>15.687898</td>
      <td>7.664835</td>
      <td>11.222603</td>
      <td>11.946619</td>
      <td>21.923729</td>
      <td>5.015707</td>
      <td>7.264463</td>
      <td>0.277778</td>
      <td>26.153846</td>
      <td>4.615385</td>
      <td>29.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>cleaning_fee</th>
      <td>25.349840</td>
      <td>34.857645</td>
      <td>25.718638</td>
      <td>70.874735</td>
      <td>34.434066</td>
      <td>44.452055</td>
      <td>74.779359</td>
      <td>134.983051</td>
      <td>32.193717</td>
      <td>16.256198</td>
      <td>106.240741</td>
      <td>17.000000</td>
      <td>18.076923</td>
      <td>5.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>cleaning_fee_missing</th>
      <td>0.376997</td>
      <td>0.281195</td>
      <td>0.299283</td>
      <td>0.027601</td>
      <td>0.373626</td>
      <td>0.301370</td>
      <td>0.110320</td>
      <td>0.042373</td>
      <td>0.539267</td>
      <td>0.487603</td>
      <td>0.037037</td>
      <td>0.615385</td>
      <td>0.384615</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>0.921326</td>
      <td>0.947223</td>
      <td>0.987634</td>
      <td>0.955096</td>
      <td>0.923242</td>
      <td>0.170000</td>
      <td>0.954128</td>
      <td>0.931059</td>
      <td>0.170838</td>
      <td>0.783636</td>
      <td>0.985370</td>
      <td>0.890769</td>
      <td>0.808462</td>
      <td>0.336000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_response_rate_missing</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.033898</td>
      <td>0.994764</td>
      <td>0.190083</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.230769</td>
      <td>0.800000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>host_acceptance_rate</th>
      <td>0.916933</td>
      <td>0.908612</td>
      <td>0.996416</td>
      <td>0.923567</td>
      <td>0.782967</td>
      <td>0.051370</td>
      <td>0.971530</td>
      <td>0.864407</td>
      <td>0.005236</td>
      <td>0.727273</td>
      <td>1.000000</td>
      <td>0.846154</td>
      <td>0.692308</td>
      <td>0.200000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_acceptance_rate_missing</th>
      <td>0.083067</td>
      <td>0.091388</td>
      <td>0.003584</td>
      <td>0.076433</td>
      <td>0.217033</td>
      <td>0.948630</td>
      <td>0.028470</td>
      <td>0.135593</td>
      <td>0.994764</td>
      <td>0.264463</td>
      <td>0.000000</td>
      <td>0.153846</td>
      <td>0.307692</td>
      <td>0.800000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>93.164537</td>
      <td>94.797891</td>
      <td>95.548387</td>
      <td>94.424628</td>
      <td>20.000000</td>
      <td>94.308219</td>
      <td>88.291815</td>
      <td>88.004237</td>
      <td>20.000000</td>
      <td>82.983471</td>
      <td>79.074074</td>
      <td>66.384615</td>
      <td>78.230769</td>
      <td>79.600000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>review_scores_rating_missing</th>
      <td>0.001597</td>
      <td>0.001757</td>
      <td>0.000000</td>
      <td>0.004246</td>
      <td>1.000000</td>
      <td>0.003425</td>
      <td>0.081851</td>
      <td>0.088983</td>
      <td>1.000000</td>
      <td>0.157025</td>
      <td>0.185185</td>
      <td>0.384615</td>
      <td>0.230769</td>
      <td>0.200000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Cluster_count</th>
      <td>626.000000</td>
      <td>569.000000</td>
      <td>558.000000</td>
      <td>471.000000</td>
      <td>364.000000</td>
      <td>292.000000</td>
      <td>281.000000</td>
      <td>236.000000</td>
      <td>191.000000</td>
      <td>121.000000</td>
      <td>54.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



From the above table, the top 3 clusters with the highest number of listings are cluster 14, 1 and 7.  Let us look at them in some details.

##### Cluster 14




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bedrooms</th>
      <td>626.0</td>
      <td>0.977636</td>
      <td>0.399373</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>626.0</td>
      <td>1.241214</td>
      <td>0.506875</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>626.0</td>
      <td>1.138978</td>
      <td>0.478178</td>
      <td>0.50</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>guests_included</th>
      <td>626.0</td>
      <td>1.279553</td>
      <td>0.598770</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>2.00000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>626.0</td>
      <td>2.337061</td>
      <td>1.050623</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.000000</td>
      <td>3.00000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>occupancy_rate</th>
      <td>626.0</td>
      <td>0.114994</td>
      <td>0.225399</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.011019</td>
      <td>0.07438</td>
      <td>0.818182</td>
    </tr>
    <tr>
      <th>price</th>
      <td>626.0</td>
      <td>91.795527</td>
      <td>45.824532</td>
      <td>27.00</td>
      <td>60.0</td>
      <td>85.000000</td>
      <td>109.75000</td>
      <td>600.000000</td>
    </tr>
    <tr>
      <th>extra_people</th>
      <td>626.0</td>
      <td>8.266773</td>
      <td>12.553912</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>15.00000</td>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>cleaning_fee</th>
      <td>626.0</td>
      <td>25.349840</td>
      <td>24.766312</td>
      <td>5.00</td>
      <td>5.0</td>
      <td>20.000000</td>
      <td>35.00000</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>cleaning_fee_missing</th>
      <td>626.0</td>
      <td>0.376997</td>
      <td>0.485022</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>626.0</td>
      <td>0.921326</td>
      <td>0.141557</td>
      <td>0.17</td>
      <td>0.9</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_response_rate_missing</th>
      <td>626.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>host_acceptance_rate</th>
      <td>626.0</td>
      <td>0.916933</td>
      <td>0.276204</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_acceptance_rate_missing</th>
      <td>626.0</td>
      <td>0.083067</td>
      <td>0.276204</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>626.0</td>
      <td>93.164537</td>
      <td>8.528358</td>
      <td>20.00</td>
      <td>91.0</td>
      <td>95.000000</td>
      <td>100.00000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>review_scores_rating_missing</th>
      <td>626.0</td>
      <td>0.001597</td>
      <td>0.039968</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_142_0.png)


          weight                              attribute
    1   0.737396                        availability_60
    8   0.725958                        availability_90
    7   0.715200                        availability_30
    29  0.643970  host_response_time_within a few hours
    13  0.620320                       availability_365
    30 -0.435392      host_response_time_within an hour
    21 -0.442752              reviews_per_month_missing
    22 -0.446956           review_scores_rating_missing
    15 -0.510896                           accommodates
    2  -0.553153                         occupancy_rate


We call cluster 14 the high availablity group with host response time within a few hours, and low occupancy rate.   

The features that are positively associated with this cluster are availability_60, availability_90, availability_30, host_response_time_within a few hours and availability_365.   

The negatively associated features are  host_response_time_within an hour, "missing" reviews_per_month,
"missing" review_scores_rating, accommodates and occupancy_rate.

##### Cluster 1





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bedrooms</th>
      <td>569.0</td>
      <td>1.231986</td>
      <td>0.730832</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>569.0</td>
      <td>1.550088</td>
      <td>0.758546</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>569.0</td>
      <td>1.157293</td>
      <td>0.379195</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.500000</td>
    </tr>
    <tr>
      <th>guests_included</th>
      <td>569.0</td>
      <td>1.550088</td>
      <td>0.942747</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>569.0</td>
      <td>3.154657</td>
      <td>1.442800</td>
      <td>1.00</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>occupancy_rate</th>
      <td>569.0</td>
      <td>0.530765</td>
      <td>0.306193</td>
      <td>0.00</td>
      <td>0.247934</td>
      <td>0.584022</td>
      <td>0.812672</td>
      <td>0.997245</td>
    </tr>
    <tr>
      <th>price</th>
      <td>569.0</td>
      <td>114.891037</td>
      <td>70.190401</td>
      <td>25.00</td>
      <td>75.000000</td>
      <td>100.000000</td>
      <td>140.000000</td>
      <td>999.000000</td>
    </tr>
    <tr>
      <th>extra_people</th>
      <td>569.0</td>
      <td>8.847100</td>
      <td>13.405098</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>cleaning_fee</th>
      <td>569.0</td>
      <td>34.857645</td>
      <td>32.537072</td>
      <td>5.00</td>
      <td>5.000000</td>
      <td>25.000000</td>
      <td>50.000000</td>
      <td>250.000000</td>
    </tr>
    <tr>
      <th>cleaning_fee_missing</th>
      <td>569.0</td>
      <td>0.281195</td>
      <td>0.449978</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>569.0</td>
      <td>0.947223</td>
      <td>0.127615</td>
      <td>0.17</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_response_rate_missing</th>
      <td>569.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>host_acceptance_rate</th>
      <td>569.0</td>
      <td>0.908612</td>
      <td>0.288414</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_acceptance_rate_missing</th>
      <td>569.0</td>
      <td>0.091388</td>
      <td>0.288414</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>569.0</td>
      <td>94.797891</td>
      <td>7.383336</td>
      <td>20.00</td>
      <td>93.000000</td>
      <td>97.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>review_scores_rating_missing</th>
      <td>569.0</td>
      <td>0.001757</td>
      <td>0.041922</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_146_0.png)


          weight                        attribute
    2   0.712161                   occupancy_rate
    14  0.449710             review_scores_rating
    19  0.362529               host_response_rate
    20  0.277401             host_acceptance_rate
    28  0.255124  host_response_time_within a day
    22 -0.446542     review_scores_rating_missing
    13 -0.776582                 availability_365
    7  -1.115668                  availability_30
    8  -1.171230                  availability_90
    1  -1.208166                  availability_60


Cluster 1 is the high occupancy rate group with high host response/acceptance rate.   

The features that are positively associated with this cluster are occupancy_rate, review_scores_rating, host_response_rate, host_acceptance_rate and host_response_time_within a day.   

The negatively associated features are  review_scores_rating_missing, availability_365,
availability_30, availability_90 and availability_60.   

We can see that clusters 14 and 1 are quite the opposite to each other.  Renters looking at cluster 14 listings should have an easier time in securing a rental.

##### Cluster 7




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bedrooms</th>
      <td>558.0</td>
      <td>0.906810</td>
      <td>0.524369</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>558.0</td>
      <td>1.340502</td>
      <td>0.613002</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>558.0</td>
      <td>1.074373</td>
      <td>0.256560</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.500000</td>
    </tr>
    <tr>
      <th>guests_included</th>
      <td>558.0</td>
      <td>1.458781</td>
      <td>0.779624</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>558.0</td>
      <td>2.629032</td>
      <td>1.119606</td>
      <td>1.0</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>occupancy_rate</th>
      <td>558.0</td>
      <td>0.311176</td>
      <td>0.316974</td>
      <td>0.0</td>
      <td>0.035813</td>
      <td>0.123967</td>
      <td>0.588843</td>
      <td>0.991736</td>
    </tr>
    <tr>
      <th>price</th>
      <td>558.0</td>
      <td>87.960573</td>
      <td>35.911999</td>
      <td>25.0</td>
      <td>65.000000</td>
      <td>84.000000</td>
      <td>102.250000</td>
      <td>300.000000</td>
    </tr>
    <tr>
      <th>extra_people</th>
      <td>558.0</td>
      <td>10.736559</td>
      <td>13.642412</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>20.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>cleaning_fee</th>
      <td>558.0</td>
      <td>25.718638</td>
      <td>21.890163</td>
      <td>5.0</td>
      <td>5.000000</td>
      <td>20.000000</td>
      <td>40.000000</td>
      <td>120.000000</td>
    </tr>
    <tr>
      <th>cleaning_fee_missing</th>
      <td>558.0</td>
      <td>0.299283</td>
      <td>0.458355</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>558.0</td>
      <td>0.987634</td>
      <td>0.036130</td>
      <td>0.7</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_response_rate_missing</th>
      <td>558.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>host_acceptance_rate</th>
      <td>558.0</td>
      <td>0.996416</td>
      <td>0.059815</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_acceptance_rate_missing</th>
      <td>558.0</td>
      <td>0.003584</td>
      <td>0.059815</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>558.0</td>
      <td>95.548387</td>
      <td>4.157188</td>
      <td>70.0</td>
      <td>94.000000</td>
      <td>97.000000</td>
      <td>98.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>review_scores_rating_missing</th>
      <td>558.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_150_0.png)


          weight                          attribute
    0   1.408224                  reviews_per_month
    30  0.967491  host_response_time_within an hour
    9   0.885768                  number_of_reviews
    47  0.875980                 instant_bookable_t
    54  0.644546       cancellation_policy_moderate
    18 -0.429873                       cleaning_fee
    21 -0.442752          reviews_per_month_missing
    22 -0.451238       review_scores_rating_missing
    10 -0.457133                           bedrooms
    25 -0.495644       host_acceptance_rate_missing


Cluster 7 has lots of reviews, can be instantly booked with moderate cancellation policy, and lower cleaning fees.   

The features that are positively associated with this cluster are reviews_per_month, host_response_time_within an hour, number_of_reviews, instant_bookable_t and cancellation_policy_moderate.   

The negatively associated features are  cleaning_fee, "missing" reviews_per_month,
"missing" review_scores_rating, number of bedrooms and "missing" host_acceptance_rate.   


##### Cluster 3

Cluster 3 is the group that have the highest average rental price.




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bedrooms</th>
      <td>236.0</td>
      <td>3.360169</td>
      <td>0.955133</td>
      <td>1.00</td>
      <td>3.00000</td>
      <td>3.00000</td>
      <td>4.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>236.0</td>
      <td>4.296610</td>
      <td>1.674893</td>
      <td>1.00</td>
      <td>3.00000</td>
      <td>4.00000</td>
      <td>5.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>236.0</td>
      <td>2.544492</td>
      <td>0.920876</td>
      <td>1.00</td>
      <td>2.00000</td>
      <td>2.50000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>guests_included</th>
      <td>236.0</td>
      <td>4.377119</td>
      <td>2.610483</td>
      <td>0.00</td>
      <td>1.00000</td>
      <td>4.00000</td>
      <td>6.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>236.0</td>
      <td>7.923729</td>
      <td>2.357990</td>
      <td>2.00</td>
      <td>6.00000</td>
      <td>8.00000</td>
      <td>9.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>occupancy_rate</th>
      <td>236.0</td>
      <td>0.301338</td>
      <td>0.316280</td>
      <td>0.00</td>
      <td>0.02135</td>
      <td>0.15978</td>
      <td>0.555096</td>
      <td>0.988981</td>
    </tr>
    <tr>
      <th>price</th>
      <td>236.0</td>
      <td>302.902542</td>
      <td>148.149533</td>
      <td>41.00</td>
      <td>200.00000</td>
      <td>267.00000</td>
      <td>350.000000</td>
      <td>975.000000</td>
    </tr>
    <tr>
      <th>extra_people</th>
      <td>236.0</td>
      <td>21.923729</td>
      <td>26.094642</td>
      <td>0.00</td>
      <td>0.00000</td>
      <td>20.00000</td>
      <td>26.000000</td>
      <td>250.000000</td>
    </tr>
    <tr>
      <th>cleaning_fee</th>
      <td>236.0</td>
      <td>134.983051</td>
      <td>66.683499</td>
      <td>5.00</td>
      <td>100.00000</td>
      <td>125.00000</td>
      <td>175.000000</td>
      <td>300.000000</td>
    </tr>
    <tr>
      <th>cleaning_fee_missing</th>
      <td>236.0</td>
      <td>0.042373</td>
      <td>0.201867</td>
      <td>0.00</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>236.0</td>
      <td>0.931059</td>
      <td>0.173552</td>
      <td>0.17</td>
      <td>0.96750</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_response_rate_missing</th>
      <td>236.0</td>
      <td>0.033898</td>
      <td>0.181352</td>
      <td>0.00</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_acceptance_rate</th>
      <td>236.0</td>
      <td>0.864407</td>
      <td>0.343084</td>
      <td>0.00</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>host_acceptance_rate_missing</th>
      <td>236.0</td>
      <td>0.135593</td>
      <td>0.343084</td>
      <td>0.00</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>236.0</td>
      <td>88.004237</td>
      <td>22.375043</td>
      <td>20.00</td>
      <td>90.75000</td>
      <td>96.00000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>review_scores_rating_missing</th>
      <td>236.0</td>
      <td>0.088983</td>
      <td>0.285324</td>
      <td>0.00</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png]({{ site.baseurl }}/images/Looking_at_Seattle_AirBnB_files/Looking_at_Seattle_AirBnB_154_0.png)


          weight                          attribute
    0   1.408224                  reviews_per_month
    30  0.967491  host_response_time_within an hour
    9   0.885768                  number_of_reviews
    47  0.875980                 instant_bookable_t
    54  0.644546       cancellation_policy_moderate
    18 -0.429873                       cleaning_fee
    21 -0.442752          reviews_per_month_missing
    22 -0.451238       review_scores_rating_missing
    10 -0.457133                           bedrooms
    25 -0.495644       host_acceptance_rate_missing


Cluster 3 is very similar to cluster 7 except that it has higher cleaning fees.   

The features that are positively associated with this cluster are reviews_per_month, host_response_time_within an hour, number_of_reviews, instant_bookable_t and cancellation_policy_moderate.   

The negatively associated features are  cleaning_fee, "missing" reviews_per_month,
"missing" review_scores_rating, number of bedrooms and "missing" host_acceptance_rate.   

#### Conclusions

Janurary, July, August and Feburary in 2016 in that order have the highest occupancy rates. Therefore, you may want to avoid these months if you are renter.   

Cluster 14 is the high availablity group with host response time within a few hours, and low occupancy rate.   

Cluster 1 is the high occupancy rate group with high host response/acceptance rate.   

We can see that clusters 14 and 1 are quite the opposite to each other. Renters looking at cluster 14 listings should have an easier time in securing a rental.

Cluster 7 has lots of reviews, can be instantly booked with moderate cancellation policy, and lower cleaning fees.

Cluster 3 is the group of listings that has the highest mean rental price.  Cluster 3 is very similar to Cluster 7 except that it has higher cleaning fees.

