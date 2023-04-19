# CECS456Project - Group 3
## Group 3: Online Shoppers Intention
The goal is to explain what actions of customer browsing on an e-commerce website will contribute to that customer buying a product. You will be using clustering and classification algorithms to make predictive models around shoppers’ intentions. ￼  
1.	Perform exploratory data analysis and data pre-processing.
a.	There are missing values in the dataset. One way of handlining missing values is to perform imputation, but in certain cases we can also drop them. How you want to treat the missing values? Handle categorical variables using one-hot encoding. 
b.	Plot the histogram of the features and boxplot for outlier detection,  
c.	Plot the correlation matrix for the features. 
d.	Provide a list of top 10 most important features. 
2.	The dataset contains 18 features. Your task is to use Principal Component Analysis (reduce the number of variables of a data set). 
3.	Design a machine learning algorithm to identify the behavior of customers if they are going to purchase the product or not.
a.	Use k-nearest neighbor, Naïve Bayes, logistic regression, SVM, and Random Forest classification algorithm to create online shopper intention (target variable: Is_Revenue).
b.	Compare the performance metrics of the classification algorithms.
c.	Plot ROC graph for the algorithms 
4.	Suggested a few possible ways to attract more customers to finish with purchasing.

# Dataset--------- 
https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset
### Phiona Nicole Tumbaga 
### Alberto Perez 
### Joey Rice <3
### Eric Chhour
### Jonathan Santos
### Hadi Al Lawati
### Abdullah Al Nahwi


# Data Set Information:

The dataset consists of feature vectors belonging to 12,330 sessions.
The dataset was formed so that each session
would belong to a different user in a 1-year period to avoid
any tendency to a specific campaign, special day, user
profile, or period.


Attribute Information:

The dataset consists of 10 numerical and 8 categorical attributes.
The 'Revenue' attribute can be used as the class label.

"Administrative", "Administrative Duration", "Informational", "Informational Duration", "Product Related" and "Product Related Duration" represent the number of different types of pages visited by the visitor in that session and total time spent in each of these page categories. The values of these features are derived from the URL information of the pages visited by the user and updated in real time when a user takes an action, e.g. moving from one page to another. The "Bounce Rate", "Exit Rate" and "Page Value" features represent the metrics measured by "Google Analytics" for each page in the e-commerce site. The value of "Bounce Rate" feature for a web page refers to the percentage of visitors who enter the site from that page and then leave ("bounce") without triggering any other requests to the analytics server during that session. The value of "Exit Rate" feature for a specific web page is calculated as for all pageviews to the page, the percentage that were the last in the session. The "Page Value" feature represents the average value for a web page that a user visited before completing an e-commerce transaction. The "Special Day" feature indicates the closeness of the site visiting time to a specific special day (e.g. Mother’s Day, Valentine's Day) in which the sessions are more likely to be finalized with transaction. The value of this attribute is determined by considering the dynamics of e-commerce such as the duration between the order date and delivery date. For example, for Valentina’s day, this value takes a nonzero value between February 2 and February 12, zero before and after this date unless it is close to another special day, and its maximum value of 1 on February 8. The dataset also includes operating system, browser, region, traffic type, visitor type as returning or new visitor, a Boolean value indicating whether the date of the visit is weekend, and month of the year.