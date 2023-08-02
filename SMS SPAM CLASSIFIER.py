#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import pandas as pd

# Assuming the file uses "latin-1" encoding
df = pd.read_csv("spam.csv", encoding="latin-1")

# If the file uses "cp1252" encoding, use the following:
# df = pd.read_csv("spam.csv", encoding="cp1252")


# In[3]:


df.sample(5)


# In[4]:


df.shape

# Data cleaning
# EDA
# Text PreProcessing
# Model Building
# Evaluation
# Improvements
# Website
# Deployment
# # Data Cleaning

# In[5]:


df.info()


# we can see that in the column of unnamed 2,3 $ 4 many values are missing 
# so we remove the columns 


# In[6]:


# drop last three columns

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)


# In[7]:


df.head()


# In[8]:


# renaming the columns names 

df.rename(columns={'v1' : 'Target', 'v2' : 'Text'}, inplace=True)


# In[9]:


df.head()


# In[10]:


# label encoding

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[11]:


df['Target']=encoder.fit_transform(df['Target'])



# in this 'HAM' is assigned as 0 and "SPAM" is assigned as 1


# In[12]:


df.head()


# In[13]:


# check missing values

df.isnull().sum()


# In[14]:


# check for duplicate values

df.duplicated().sum()


# There are 403 duplicated values as we see. so we keep the 1st values and drop the other same values 


# In[15]:


df=df.drop_duplicates(keep='first')


# In[16]:


df.duplicated().sum()


# In[17]:


df.shape


# # EDA

# In[18]:


df.head()


# In[19]:


# we want the ratio of spam and ham 

df['Target'].value_counts()


# In[20]:


import matplotlib.pyplot as plt
plt.pie(df['Target'].value_counts(), labels=['ham','spam'], autopct="%0.2f")
plt.show()

 from the above pie chart we can roughly said that 88% sms are ham and 14% sms are spam
 so our data is imbalanced
# In[21]:


import nltk


# In[22]:


nltk.download('punkt')

# for better analysis we made 3 columns 
 1st columns is for no. of characters in the sms
 2nd columns is for no. of words in the sms
 3rd columns is for no. of sentence in the sms
# In[23]:


df['num_characters']=df['Text'].apply(len)


# In[24]:


df.head()


# In[25]:


# no. of words 

df['Text'].apply(lambda x:nltk.word_tokenize(x))


# In[26]:


# so words are come in the form of array now i have to just get the len 

df['num_words']=df['Text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[27]:


df.head()


# In[28]:


df['Text'].apply(lambda x:nltk.sent_tokenize(x))


# In[29]:


# so sentence  are come in the form of array now i have to just get the len 

df['num_sent']=df['Text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[30]:


df.head()


# In[31]:


df[['num_characters','num_words','num_sent']].describe()


# In[32]:


# describe function output for "HAM" messages

df[df['Target']==0][['num_characters','num_words','num_sent']].describe()


# In[33]:


# describe function output for "SPAM" messages

df[df['Target']==1][['num_characters','num_words','num_sent']].describe()


# In[34]:


import seaborn as sns


# In[35]:


plt.figure(figsize=(8,6))
sns.histplot(df[df['Target']==0]['num_characters'])
sns.histplot(df[df['Target']==1]['num_characters'], color='red')


# In[36]:


plt.figure(figsize=(8,6))
sns.histplot(df[df['Target']==0]['num_words'])
sns.histplot(df[df['Target']==1]['num_words'], color='red')


# In[37]:


plt.figure(figsize=(8,6))
sns.histplot(df[df['Target']==0]['num_sent'])
sns.histplot(df[df['Target']==1]['num_sent'], color='red')


# In[38]:


sns.pairplot(df,hue='Target')

# in data the outliers are present


# In[39]:


df.corr()


# In[40]:


sns.heatmap(df.corr(), annot=True)


# MULTICOLLINEARITY IS PRESENT BETWEEN NUM_WORDS AND NUM_CHARCTERS, NUM_WORDS AND NUM_SENT


# # TEXT PREPROCESSING
LOWER CASING
TOKENIZATION
REMOVING SPECIAL CHARACTERS
REMOVING STOP WORDS AND PUNCTUATION
STEMMINGThe Below code for lower casing, tokenization, removing special characters, removing stop words and punctuation, stemming
# In[41]:


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)


# In[47]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[48]:


df['Text'][10]


# In[43]:


from nltk.corpus import stopwords
import string


# In[44]:


stopwords.words('english')
string.punctuation


# In[46]:


from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

ps.stem('loving')


# In[49]:


df['tranform_text']=df['Text'].apply(transform_text)


# In[50]:


df.head()


# In[ ]:


# pip install wordcloud

Generate the most frequent words comes in the spam message. It shows as a wordcloud
# In[51]:


from wordcloud import WordCloud

wc=WordCloud(width=500, height=500, min_font_size=10, background_color='white')


# In[52]:


spam_wc= wc.generate(df[df['Target']==1]["tranform_text"].str.cat(sep=" "))


# In[53]:


plt.figure(figsize=(12,6))
plt.imshow(spam_wc)

Generate the most frequent words comes in the ham message. It shows as a wordcloud
# In[54]:


ham_wc= wc.generate(df[df['Target']==0]["tranform_text"].str.cat(sep=" "))


# In[55]:


plt.figure(figsize=(12,6))
plt.imshow(ham_wc)


# In[56]:


df.head()

Most common 30 words comes in the spam message 
# In[57]:


spam_corpus=[]
for msg in df[df['Target']==1]['tranform_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
        


# In[58]:


len(spam_corpus)


# In[59]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[60]:


ham_corpus=[]
for msg in df[df['Target']==0]['tranform_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
        


# In[61]:


len(ham_corpus)


# In[62]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# # Feature Engineering 
Convert the text into numbers means text vectorization by Bag of Words, TFidf and Word2vec 
# In[63]:


from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()


# In[64]:


X=cv.fit_transform(df["tranform_text"]).toarray()


# In[65]:


X.shape


# In[66]:


y=df["Target"].values


# In[67]:


y


# In[68]:


from sklearn.model_selection import train_test_split


# In[69]:


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=1)


# # Model Building 
First we do with Naive Bayes, it is a saying that in a textual data naive bayes gives better result. Along with that we will also try other algorithms like randomforest classifier and logistic regression
# In[70]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


# In[71]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[72]:


gnb.fit(X_train, y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

As it is a imbalanced data we know that accuracy score is a crush for imbalanced data
Here precision score is matter the most 
Precsion socre is very low not performing upto the marks 
# In[73]:


mnb.fit(X_train, y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

As it is a imbalanced data we know that accuracy score is a crush for imbalanced data
Here precision score is matter the most 
Precsion socre is 85% we want more precision so we check with the bernoulli
# In[74]:


bnb.fit(X_train, y_train)
y_pred3=bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))

Here precision score is matter the most 
Precsion socre is 94%, which is better than other two (gaussian and multinomial)So this precision we got from by using countvectorizer means bag of words lets try tfidf 
# # Feature Engineering 2

# TFIDF

# In[75]:


# from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer()


# In[76]:


# X=tf.fit_transform(df["tranform_text"]).toarray()


# In[77]:


# X.shape


# In[78]:


# y=df["Target"].values


# In[79]:


# y


# In[80]:


# from sklearn.model_selection import train_test_split


# In[81]:


# X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=1)


# # Model Building 2

# In[82]:


# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


# In[83]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[84]:


gnb.fit(X_train, y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

As it is a imbalanced data we know that accuracy score is a crush for imbalanced data
Here precision score is matter the most 
Precsion socre is very low not performing upto the marks 
# In[85]:


mnb.fit(X_train, y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

As we see that in the multinomial naive bayes uses the tfidf vectorizer, it's gives the precion of 1, and the false positive is 0. 
# In[86]:


bnb.fit(X_train, y_train)
y_pred3=bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))

So we choose mnb with Tfidf
# In[87]:


# pip install xgboost

#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn. ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier#SVC=SVC(kernel='sigmoid', gamma=1.0)
knc=KNeighborsClassifier()
mnb=MultinomialNB()
dtc=DecisionTreeClassifier(max_depth=5)
lrc=LogisticRegression(solver='liblinear', penalty='l1')
rfc=RandomForestClassifier(n_estimators=50, random_state=2)
abc=AdaBoostClassifier(n_estimators=50, random_state=2)
bc=BaggingClassifier(n_estimators=50, random_state=2)
etc=ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt=GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb=XGBClassifier(n_estimators=50, random_state=2)clfs= {
    'SVC' : SVC,
    'KN' : knc,
    'NB' : mnb,
    'DT' : dtc,
    'LR' : lrc,
    'RF' : rfc,
    'AdaBoost' : abc,
    'BgC' : bc,
    'ETC' : etc,
    'GBDT': gbdt,
    'xgb' : xgb
}def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test, y_pred)
    
    return accuracy, precision
# In[92]:


train_classifier(SVC, X_train,y_train, X_test, y_test)


# here we trained SVC model and our accuracy is 96% and precsion is 95% 

accuracy_scores=[]
precsion_scores=[]

for name, clf in clfs.items():
    
    current_accuracy, current_precision= train_classifier(clf, X_train, y_train, X_test, y_test)
    
    print("For", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)
    
    
    accuracy_scores.append(current_accuracy)
    precsion_scores.append(current_precision)
# In[95]:


performance_df=pd.DataFrame({"Algorithm": clfs.keys(), "Accuracy": accuracy_scores, "Precision": precsion_scores})


# In[96]:


performance_df.sort_values(by="Precision", ascending=False, inplace=True)


# In[97]:


performance_df


# # model Improvement
# change the max_features of the tfidf, we have check for 1000,1500,2000,2500,3000,3500 and we got best results on 3000
# In[126]:


from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer(max_features=3000)


# In[172]:


X=tf.fit_transform(df["tranform_text"]).toarray()

X


# In[100]:


y=df["Target"].values
y


# In[101]:


from sklearn.model_selection import train_test_split


# In[102]:


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=1)


# In[103]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


# In[104]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


# In[105]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[106]:


gnb.fit(X_train, y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))


# In[107]:


mnb.fit(X_train, y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))


# In[108]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn. ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[109]:


SVC=SVC(kernel='sigmoid', gamma=1.0)
knc=KNeighborsClassifier()
mnb=MultinomialNB()
dtc=DecisionTreeClassifier(max_depth=5)
lrc=LogisticRegression(solver='liblinear', penalty='l1')
rfc=RandomForestClassifier(n_estimators=50, random_state=2)
abc=AdaBoostClassifier(n_estimators=50, random_state=2)
bc=BaggingClassifier(n_estimators=50, random_state=2)
etc=ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt=GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb=XGBClassifier(n_estimators=50, random_state=2)


# In[110]:


clfs= {
    'SVC' : SVC,
    'KN' : knc,
    'NB' : mnb,
    'DT' : dtc,
    'LR' : lrc,
    'RF' : rfc,
    'AdaBoost' : abc,
    'BgC' : bc,
    'ETC' : etc,
    'GBDT': gbdt,
    'xgb' : xgb
}


# In[111]:


def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test, y_pred)
    
    return accuracy, precision


# In[112]:


train_classifier(SVC, X_train,y_train, X_test, y_test)


# In[113]:


accuracy_max3000=[]
precsion_max3000=[]

for name, clf in clfs.items():
    
    current_accuracy, current_precision= train_classifier(clf, X_train, y_train, X_test, y_test)
    
    print("For", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)
    
    
    accuracy_max3000.append(current_accuracy)
    precsion_max3000.append(current_precision)


# In[114]:


performance_df1=pd.DataFrame({"Algorithm": clfs.keys(), "Accuracy_3000": accuracy_max3000, "Precision_3000": precsion_max3000})


# In[115]:


performance_df1.sort_values(by="Precision_3000", ascending=False, inplace=True)


# In[ ]:


performance_df1


# In[ ]:


performance_df


# In[116]:


performance_df=performance_df.merge(performance_df1, on= "Algorithm")


# In[117]:


performance_df

We can do scaling giving the x ranges between 0 and 1 
# In[130]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()


# In[131]:


X=tf.fit_transform(df["tranform_text"]).toarray()


# In[132]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X=scaler.fit_transform(X)


# In[133]:


X


# In[134]:


y=df["Target"].values
y


# In[135]:


from sklearn.model_selection import train_test_split


# In[136]:


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=1)


# In[137]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


# In[138]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[139]:


mnb.fit(X_train, y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))


# In[140]:


gnb.fit(X_train, y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))


# In[141]:


bnb.fit(X_train, y_train)
y_pred3=bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))


# In[142]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn. ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[143]:


SVC=SVC(kernel='sigmoid', gamma=1.0)
knc=KNeighborsClassifier()
mnb=MultinomialNB()
dtc=DecisionTreeClassifier(max_depth=5)
lrc=LogisticRegression(solver='liblinear', penalty='l1')
rfc=RandomForestClassifier(n_estimators=50, random_state=2)
abc=AdaBoostClassifier(n_estimators=50, random_state=2)
bc=BaggingClassifier(n_estimators=50, random_state=2)
etc=ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt=GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb=XGBClassifier(n_estimators=50, random_state=2)


# In[144]:


clfs= {
    'SVC' : SVC,
    'KN' : knc,
    'NB' : mnb,
    'DT' : dtc,
    'LR' : lrc,
    'RF' : rfc,
    'AdaBoost' : abc,
    'BgC' : bc,
    'ETC' : etc,
    'GBDT': gbdt,
    'xgb' : xgb
}


# In[145]:


def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test, y_pred)
    
    return accuracy, precision


# In[146]:


train_classifier(SVC, X_train,y_train, X_test, y_test)


# In[147]:


accuracy_scores=[]
precsion_scores=[]

for name, clf in clfs.items():
    
    current_accuracy, current_precision= train_classifier(clf, X_train, y_train, X_test, y_test)
    
    print("For", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)
    
    
    accuracy_scores.append(current_accuracy)
    precsion_scores.append(current_precision)


# In[149]:


performance_df2=pd.DataFrame({"Algorithm": clfs.keys(), "Accuracy_scaling": accuracy_scores, "Precision_scaling": precsion_scores})


# In[150]:


performance_df2.sort_values(by="Precision_scaling", ascending=False, inplace=True)


# In[151]:


performance_df2


# In[ ]:


performance_df=performance_df.merge(performance_df2, on= "Algorithm")


# In[ ]:


performance_df

# taking the num_charcters columns that we made 
# In[152]:


from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer(max_features=3000)

X=tf.fit_transform(df["tranform_text"]).toarray()


# In[153]:


# appending the num_characters column to x 
X=np.hstack((X, df['num_characters'].values.reshape(-1,1)))


# In[154]:


X.shape


# In[155]:


y=df["Target"].values
y


# In[156]:


from sklearn.model_selection import train_test_split


# In[157]:


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=1)


# In[158]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


# In[159]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score


# In[160]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[161]:


gnb.fit(X_train, y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))


# In[162]:


mnb.fit(X_train, y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))


# In[163]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn. ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[164]:


SVC=SVC(kernel='sigmoid', gamma=1.0)
knc=KNeighborsClassifier()
mnb=MultinomialNB()
dtc=DecisionTreeClassifier(max_depth=5)
lrc=LogisticRegression(solver='liblinear', penalty='l1')
rfc=RandomForestClassifier(n_estimators=50, random_state=2)
abc=AdaBoostClassifier(n_estimators=50, random_state=2)
bc=BaggingClassifier(n_estimators=50, random_state=2)
etc=ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt=GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb=XGBClassifier(n_estimators=50, random_state=2)


# In[165]:


clfs= {
    'SVC' : SVC,
    'KN' : knc,
    'NB' : mnb,
    'DT' : dtc,
    'LR' : lrc,
    'RF' : rfc,
    'AdaBoost' : abc,
    'BgC' : bc,
    'ETC' : etc,
    'GBDT': gbdt,
    'xgb' : xgb
}


# In[166]:


def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test, y_pred)
    
    return accuracy, precision


# In[167]:


train_classifier(SVC, X_train,y_train, X_test, y_test)


# In[168]:


accuracy_scores=[]
precsion_scores=[]

for name, clf in clfs.items():
    
    current_accuracy, current_precision= train_classifier(clf, X_train, y_train, X_test, y_test)
    
    print("For", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)
    
    
    accuracy_scores.append(current_accuracy)
    precsion_scores.append(current_precision)


# In[169]:


performance_df3=pd.DataFrame({"Algorithm": clfs.keys(), "Accuracy_num_char": accuracy_scores, "Precision_num_char": precsion_scores})


# In[170]:


performance_df3.sort_values(by="Precision_num_char", ascending=False, inplace=True)


# In[171]:


performance_df3


# In[ ]:


performance_df=performance_df.merge(performance_df3, on= "Algorithm")


# In[ ]:


performance_df


# In[118]:


# voting classifier
from sklearn.svm import SVC
svc=SVC(kernel='sigmoid', gamma=1.0, probability=True)
mnb=MultinomialNB()
etc=ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[119]:


voting=VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)], voting='soft')


# In[120]:


voting.fit(X_train, y_train)


# In[121]:


y_pred=voting.predict(X_test)
print("Accuracy", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))


# In[122]:


# applying stacking

estimators=[('svm',svc), ('nb',mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[123]:


from sklearn.ensemble import StackingClassifier


# In[124]:


clf=StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[125]:


clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("Accuracy", accuracy_score(y_test,y_pred))
print("Precision", precision_score(y_test, y_pred))


# In[128]:


import pickle


# In[129]:


pickle.dump(tf, open("vectorizer.pkl", "wb"))
pickle.dump(mnb, open("model.pkl", "wb"))


# In[ ]:




