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

### 1. Content Based
The URL list has the following lay-out:

| URL | TAG |
| :-------------: | :-------------: |
| url_1  | tag_1  |
| url_2  | tag_1  |

The first step is to manually scrape the raw text from all the URLS and scrape the TITLE, READING_TIME and DATE (publish) from the pages using a HTML webscraper. All these steps were manually combined in an excel file. After the merging, the data will look as follows:

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

The main calculations of the Doc2Vec algorithm are done in [Item Similarity](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ItemSimilarity.ipynb), where the input is the **TEXT** column. In this script, the text is first tokenized and next these tokens are used to train the Doc2Vec. The Doc2Vec model also uses pre-embedded words ([Wikipedia-320](http://www.clips.uantwerpen.be/dutchembeddings/wikipedia-320.tar.gz), thanks to [Embedding_GitHub](https://github.com/clips/dutchembeddings)) to kick-start the training.           
*Note that these word-embeddings are Dutch words only.*

The output of the [Item Similarity](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/ItemSimilarity.ipynb) algorithm, is a single, symmetric, square matrix with the dimension equal to the number of unique articles. 


![Item Similarity Output](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Pics/Simmatrix.png "Ouput I-S")
 
 ### 2. Collaborative Filtering
 Our collaborative filtering method is an extension on the View Enhanced Matrix Factorization introduced by [Ding et al. (2018)](https://github.com/dingjingtao/View_enhanced_ALS). The main differences are:
 - Application to informative article database instead of e-commerce platform
 - Modification of the optimization algorithm, ensuring a more robust convergence
 - Different evaluation, using Self-Normalizing Inverse Propensity Score
 
 #### Data cleaning
 In [Data Cleaning](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/DataCleaner.ipynb), the biggest of the data cleaning process is performed. The inputs are 3 raw data files, given above. The first step is to put all seperate click-stream files together and formatting all the variables to either *string, integer, datetime, etc.* .
 Another crucial step is removing all the duplicate URLs. Often two back-end URLs are given in the click-stream data, leading to the same webpage. Because the all the URLs were already scraped, it is possible to remove the duplicates using the **TITLE**. To properly clean-up this step, all the NaNs created in dropping duplicates are removed.
 The output of [Data Cleaning](https://github.com/ArnoClaes/Hybrid-Content-Based-Read-Enhanced-ALS/blob/master/Algorithms/DataCleaner.ipynb) are two files:
 1) Clean Page Data
 
 2) Clean Event Data
 
| URL | clientid_hashed | visitid | visitstarttime | hitnumber | eventcategory | eventlabel |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| url_1 | 5sdf45sdf654sdf | 45654512 | 2018-09-13 22:10:20 | 5 | timer | 30000 |
| url_2 | 789sd1fzxkj4fgh | 78974982 | 2018-11-22 19:55:58 | 1 | Scroll Depth | 50% | 
