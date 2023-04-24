import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

# read in the dataset
df = pd.read_csv('online_shoppers_intention.csv')

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# drop all rows with missing values
df = df.dropna()

categorical_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType']
numerical_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                      'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
                      'SpecialDay', 'Weekend']

df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].apply(lambda x: 1 if x else 0)

imputer = SimpleImputer(strategy='median')
df[numerical_features] = imputer.fit_transform(df[numerical_features])

imputer = SimpleImputer(strategy='most_frequent')
df[categorical_features] = imputer.fit_transform(df[categorical_features])

onehotencoder = OneHotEncoder(handle_unknown='ignore')
df = pd.get_dummies(df, columns=categorical_features)

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

X = df.drop('Revenue', axis=1)
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# create histograms and boxplots for numerical features
for feature in numerical_features:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(data=df, x=feature, kde=True, ax=ax[0])
    sns.boxplot(data=df, x=feature, ax=ax[1])
    fig.suptitle(feature, fontsize=16)
    plt.show()


corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Matrix")
plt.show()

rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

feat_importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
top_10_features = feat_importances.nlargest(10)

print(top_10_features)


knn_clf = KNeighborsClassifier()
nb_clf = GaussianNB()
logreg_clf = LogisticRegression(random_state=42)
svm_clf = SVC(random_state=42, probability=True)
rf_clf = RandomForestClassifier(random_state=42)

classifiers = [
    (knn_clf, "K-Nearest Neighbor"),
    (nb_clf, "NaÃ¯ve Bayes"),
    (logreg_clf, "Logistic Regression"),
    (svm_clf, "Support Vector Machine"),
    (rf_clf, "Random Forest")
]

for clf, clf_name in classifiers:
    clf.fit(X_pca_train, y_train)
    y_pred = clf.predict(X_pca_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{clf_name} accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    y_pred_proba = clf.predict_proba(X_pca_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=clf_name)
    display.plot()
    plt.title(clf_name)
    plt.show()
    
    
# 1. Improve website user experience: Make sure that the website is easy to navigate, has a good layout, and loads quickly. This can increase the time customers spend on the website and improve the chances of making a purchase.
# 2. Personalized recommendations: Use machine learning algorithms to recommend products that are relevant to customers based on their browsing history, preferences, and other factors. Personalized recommendations can encourage customers to explore more products and make purchases.
# 3. Offer promotions and discounts: Attract customers with special offers, discounts, or exclusive deals, especially during holidays or special events. This can incentivize customers to make purchases.
# 4. Retargeting: Use retargeting strategies to remind customers of the products they've viewed or added to their carts but haven't purchased. This can be done through email campaigns or ads on other websites or social media platforms.
# 5. Improve customer support: Providing excellent customer support can increase customer trust and satisfaction, leading to a higher likelihood of making a purchase.