# Implicit Modular Hybrid Recommender System with Doc2Vec and Read Enhanced ALS Modules
This is the private repository made by:
- [Lennert Aerts](https://www.linkedin.com/in/lennert-aerts-962a10b3/)
- [Arno Claes](https://www.linkedin.com/in/arno-claes-948994133/)
- [Dana de Leeuw](https://www.linkedin.com/in/dana-de-leeuw/)
- [Erine de Leeuw](https://www.linkedin.com/in/erine-de-leeuw-89a446b6/)

This readme will explain first the main algorithms and their inputs/outputs. Next the side-calculations, such as bias, are introduced.

## The main algorithm
The way from raw data to the final recommendations need multiple data-cleaning, preprocessing, calculation and post-processing steps.
The raw input data consists out of 3 types:

1) URL list     
2) Click-data   
3) Event-data

Mind that all the data given below is completely fictional, as the real data is protected for privacy reasons.

The programs used in the main algorithm are, in order:  
[Item Similarity](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ItemSimilarity.ipynb)  
[Item Popularity](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ItemPopularity.ipynb)  
[Data Cleaner](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/DataCleaner.ipynb)  
[Time on Page](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/TimeOnPage.ipynb)   
[View Read](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ViewRead.ipynb)


### 1. Content Based

The content based part consists out of gathering article data, creating the similarity matrix and calculating the popularity of each article.

#### Gathering Data
The URL list has the following lay-out:

| URL | TAG |
| :-------------: | :-------------: |
| url_1  | tag_1  |
| url_2  | tag_1  |

The first step is to manually scrape the raw text from all the URLS and scrape the TITLE, READING_TIME and DATE (publish) from the pages using a HTML webscraper. All these steps were manually combined in an excel file. After the merging, the data will look as follows:

 ***clean_article_data.csv***

| URL | TITLE | READING_TIME | DATE | TAG | TEXT |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| url_1 | title_1 | x minuten | date_1 | tag_1 | "Etiam faucibus iaculis lorem sit..." |
| url_2 | title_1 | y minuten | date_2 | tag_2 | " Duis sagittis lobortis volutpat... " |

To actually make use of the READING_TIME, it is transformed to milliseconds in [ViewRead](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ViewRead.ipynb), although this parameter is not further used for the content-based part:
```python
def transform_ms(input): #Transform all the 'x minuten' to actual milliseconds
  output = int(re.sub("[^0-9.]", "",input)) * 60000 #transform required reading time to milliseconds
  return output
```
#### Creating Similarity Matrix
The main calculations of the Doc2Vec algorithm are done in [Item Similarity](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ItemSimilarity.ipynb), where the input is the **TEXT** column. In this script, the text is first tokenized and next these tokens are used to train the Doc2Vec. The Doc2Vec model also uses pre-embedded words ([Wikipedia-320](http://www.clips.uantwerpen.be/dutchembeddings/wikipedia-320.tar.gz), thanks to [Embedding_GitHub](https://github.com/clips/dutchembeddings)) to kick-start the training.           
*Note that these word-embeddings are Dutch words only.*

The output of the [Item Similarity](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ItemSimilarity.ipynb) algorithm, is a single, symmetric, square matrix with the dimension equal to the number of unique articles.

![Item Similarity Output](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Pics/Simmatrix.png "Ouput I-S")

 #### Popularity Score
 The last step in the content-based part, is the calculation of the popularity score. This score will be used to scale the similarity. When the Collaborative Filtering threshold is not met, the more popular items are upsized. When, on the contrary, the threshold is met, we aim to push less popular items: **novelty**.

The weight vector that is the output of [Item Popularity](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ItemPopularity.ipynb) has one weight per article. The input of this algorithm are the datafiles containing URL information and page-click-data. These have to be merged to be able to output the titles, which are not given in the page data. This merge step is not crucial for the good continuance of the algorithm, but provides insight to which articles are the most read.

The [Item Popularity](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ItemPopularity.ipynb) code has another side-function. It provides a dummy, the same as BiebYN, but now ArticleYN. By cross-referencing with the URL list, we are able to find out which page visits are on articles and which URLs in the Bieb are not articles. This is done in following code:

```python
#Put 0/1 encoding of page is also an article
df_page = df_page.assign(ArticleYN=df_page.URL.isin(df_urls.URL).astype(int))
```

The next step is counting all the occurrences of every article and providing a popularity weight based on this count. We use a sigmoid function with upper-bound set to 1, as explained in our paper. The output of these weights look as follows:

***Popularity_score.csv***

| TITLE | URL | COUNT | popularity_weight |
| :-------------: | :-------------: | :-------------: | :-------------: |
| title_1 | url_1 | 15132 | 0.986751 |
| title_2 | url_2 | 532 | 0.054351 |

 ### 2. Collaborative Filtering
 Our collaborative filtering method is an extension on the View Enhanced Matrix Factorization introduced by [Ding et al. (2018)](https://github.com/dingjingtao/View_enhanced_ALS). The main differences are:
 - Application to informative article database instead of e-commerce platform
 - Modification of the optimization algorithm, ensuring a more robust convergence
 - Different evaluation, using Self-Normalizing Inverse Propensity Score

To properly be able to use the data, first it has to be cleaned and some extra variables have to be added.

 #### Data cleaning
 In [Data Cleaner](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/DataCleaner.ipynb), the biggest of the data cleaning process is performed. The inputs are 3 raw data files, given above. The first step is to put all seperate click-stream files together and formatting all the variables to either *string, integer, datetime, etc.* .
 Another crucial step is removing all the duplicate URLs. Often two back-end URLs are given in the click-stream data, leading to the same webpage. Because the all the URLs were already scraped, it is possible to remove the duplicates using the **TITLE**. To properly clean-up this step, all the *NaNs* created in dropping duplicates are removed.
 The output of the [Data Cleaner](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/DataCleaner.ipynb) are two files:

1)  ***clean_page_data.csv***

| URL | clientid_hashed | visitid | visitstarttime | hitnumber | time | channelgrouping | browser | devicecategory | BiebYN |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| url_1 | 5sdf45sdf654sdf | 45654512 | 2018-09-13 22:10:20 | 5 | 568181 | Organic | Safari | desktop | 1 |
| url_2 | 789sd1fzxkj4fgh | 78974982 | 2018-11-22 19:55:58 | 1 | 0 | Email | Chrome | mobile | 0 |

 Where BiebYN is an indicator if the **URL** is part of the Bieb platform.

2)  ***clean_event_data.csv***

| URL | clientid_hashed | visitid | visitstarttime | hitnumber | eventcategory | eventlabel |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| url_1 | 5sdf45sdf654sdf | 45654512 | 2018-09-13 22:10:20 | 5 | timer | 30000 |
| url_2 | 789sd1fzxkj4fgh | 78974982 | 2018-11-22 19:55:58 | 1 | Scroll Depth | 50% |

#### Read or Not Read?

In this part we aim to find out which of the article visits are actually long enough, so the reader could have read the article. These calculations are split over two different programs: [Time on Page](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/TimeOnPage.ipynb) and [View Read](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ViewRead.ipynb). First, the time on page is calculated by making a copy of the clean_page_data and pasting it next to the original, shifted by one row.
In this manner it is very efficient to calculate the time spent on a page, by simply substracting the one **time** from the other, given the **clientid_hashed** and **visitid** are the same, which outputs in:

***time_delta.csv***

| URL | clientid_hashed | visitid | visitstarttime | hitnumber | time | channelgrouping | browser | devicecategory | BiebYN | delta_time |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| url_1 | 5sdf45sdf654sdf | 45654512 | 2018-09-13 22:10:20 | 0 | 0 | Organic | Safari | desktop | 1 | 568181 |
| url_2 | 5sdf45sdf654sdf | 45654512 | 2018-09-13 22:10:20 | 1 | 568181 | Organic | Safari | desktop | 0 | 0 |

This output is in turn, together with ***clean_event_data.csv*** and ***clean_article_data.csv***, the input of [View Read](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ViewRead.ipynb), which aims to predict whether a user has actually read the article. When a user has spent less than 50% of the estimated **READING_TIME**, the entry is labeled with **ReadYN** equal to zero. When the user has spent more than 125% of the estimated **READING_TIME**, the label is set to one.   
When the **delta_time** falls between these two values, the ***clean_event_data.csv*** is used to give more information on the page visit. When the **eventlabel** for the same **clientid_hashed**, **visitid** and **URL** contains *read end article*, *75%* or *50%* the **ReadYN** dummy is set to one. To provide next steps with proper weights, the **Confidence_level** variable is made and set to one, given the **READING_TIME** exceeds 125%. The other **Confidence_level** values are 0.9, 0.8 and 0.5, respectively.  

```python
#Define confidences
df_event_subset.loc[df_event_subset['eventlabel'] == 'read end article', 'Confidence_level'] = 0.9
df_event_subset.loc[df_event_subset['eventlabel'] == '100%', 'Confidence_level'] = 0.8
df_event_subset.loc[df_event_subset['eventlabel'] == '75%', 'Confidence_level'] = 0.5
```
```python
minread = 0.5 #minimum percentage of article reading time
maxread = 1.25 #maximum percentage of article reading time

#Condition 1
df_clean_page_data.loc[(df_clean_page_data['time_on_page'] < (df_clean_page_data["READING_TIME"])*minread) & (df_clean_page_data['time_on_page'] != -1), "ReadYN"] = 0

#Condition 2
df_clean_page_data.loc[(df_clean_page_data['time_on_page'] > (df_clean_page_data["READING_TIME"])*maxread), "ReadYN"] = 1
df_clean_page_data.loc[(df_clean_page_data['time_on_page'] > (df_clean_page_data["READING_TIME"])*maxread), "Confidence_level"] = 1

#Condition 3
(df_clean_page_data.loc[df_clean_page_data['eventlabel'].notnull() & ((df_clean_page_data['time_on_page'] == -1) | ((df_clean_page_data['time_on_page'] >= (df_clean_page_data["READING_TIME"])*minread)
                                                                      & (df_clean_page_data['time_on_page'] <= (df_clean_page_data["READING_TIME"])*maxread))), "ReadYN"]) = 1
```

The output of the [View Read](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ViewRead.ipynb) algorithm, are two dataframes:

<table>
<tr><th>***read_pairs.csv***   </th><th>***clicked_pairs.csv***  </th></tr>
<tr><td>

| URL | clientid_hashed | Confidence_level |
| :-------------: | :-------------: | :-------------: |
| url_1 | sdf52fgd5s | 0.9 |
| url_2 | adsf4561fg | 0.5 |

---
</td><td>

| URL | clientid_hashed | clicked |
| :-------------: | :-------------: | :-------------: |
| url_1 | qkdcv5923q | 1 |
| url_2 | opghfma179 | 1 |

---
</td></tr> </table>
