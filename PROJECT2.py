#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#read the data and display first 5 rows
data=pd.read_csv("data.csv")
data.head()
datacopy=data


# In[2]:


#The information contains the number of columns, column labels, column data types,memory usage.
data.info()


# In[3]:


malignant = datacopy.loc[ datacopy['diagnosis'] == 'M']
benign =  datacopy[ datacopy['diagnosis'] == 'B']

for column in  datacopy.columns:
    if column not in  ['id', 'diagnosis', 'Unnamed: 32']:
        plt.figure(figsize = (5, 5))

        sns.distplot(a = malignant[[column]], hist = False, color = 'red')
        sns.distplot(a = benign[[column]], hist = False, color = 'blue')
        
        plt.title(column + ' Benign (blue) vs. Malignant (red)')
        plt.show()


# In[4]:


#As seen from the data, we do not need id,unnamed for classification and diagnosis is the class label. So we can drop it.We store the class label is a variable for further steps.
y = data.diagnosis                         # M or B 
list = ['Unnamed: 32','id','diagnosis']
data=data.drop(list, axis=1)
data.head()


# In[5]:


#The describe() method returns description of the data in the DataFrame.
#count - The number of not-empty values.
#mean - The average (mean) value.
#std - The standard deviation.
#min - the minimum value.
#25% - The 25% percentile*.
#50% - The 50% percentile*.
#75% - The 75% percentile*.
#max - the maximum value.
data.describe()


# In[6]:


#This shows us the number of benign and malignant cases in the dataset.
ax = sns.countplot(y,label="Count")  
B, M = y.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)


# In[7]:


# Because differences between values of features are very high to observe on plot we can plot 10 features to observe better.
data_dia = y
x=data
data_n_2 = (data - data.mean()) / (data.std()) #standardisation

#Voilin plot is to visualize the distribution of numerical data and are especially useful when you want to make a comparison of distributions between multiple groups.

data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

#observations:
#in all features except fractal_dimension_mean, median of the Malignant and Benign looks like separated so it can be good for classification.
# median of the Malignant and Benign does not looks like separated so it does not gives good information for classification.


# In[8]:


data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)


# In[9]:


data = pd.concat([y,data_n_2.iloc[:,20:31]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

#observations:concavity_worst and concave point_worst looks like similar.


# In[10]:


#Box plot is graphical method of displaying variation in a set of data.

data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(15,10))
sns.boxplot(x = 'features', y= 'value', hue = 'diagnosis',data= data)
plt.xticks(rotation=45)


# In[11]:


data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(15,10))
sns.boxplot(x = 'features', y= 'value', hue = 'diagnosis',data= data)
plt.xticks(rotation=45)


# In[12]:


data = pd.concat([y,data_n_2.iloc[:,20:30]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(15,10))
sns.boxplot(x = 'features', y= 'value', hue = 'diagnosis',data= data)
plt.xticks(rotation=45)


# In[13]:


#A violin plot is more informative than a plain box plot.
#While a box plot only shows summary statistics such as mean/median and interquartile ranges, the violin plot shows the full distribution of the data. 
#The difference is particularly useful when the data distribution is multimodal (more than one peak). 

#observations made:
#1.in all features except fractal_dimension_mean, median of the Malignant and Benign looks like separated so it can be good for classification.
#2.median of the Malignant and Benign does not looks like separated so it does not gives good information for classification.
#3.concavity_worst and concave point_worst looks like similar.


# In[14]:


#As concavity_worst and concave point_worst looks like similar, we can use joint plot to see if they are really correlated.
sns.jointplot(x.loc[:,'concavity_worst'], x.loc[:,'concave points_worst'], kind="reg")


# In[15]:


#radius_worst, perimeter_worst and area_worst seem similar. To check if they are correlated we can use pairplot.
sns.set(style="white")
data = x.loc[:,['radius_worst','perimeter_worst','area_worst']]
corr = sns.PairGrid(data, diag_sharey=False)
corr.map_lower(sns.kdeplot, cmap="Blues_d")
corr.map_upper(plt.scatter)
corr.map_diag(sns.kdeplot, lw=3)
#radius_worst, perimeter_worst and area_worst are correlated as it can be seen pair grid plot.


# In[16]:


#To see all the correlations, heatmap can be used. 
#Correlation heatmaps can be used to find both linear and nonlinear relationships between variables.
#Correlation ranges from -1 to +1. Values closer to zero means there is no linear trend between the two variables. 
#The close to 1 the correlation is the more positively correlated they are.
plt.figure(figsize=(20,10))
sns.heatmap(x.corr().abs(), annot=True)


# In[17]:


#values closer to 1 show correlation.
#Therefore as seen from all the above plots, we can make the following observations:
#1.radius_mean, perimeter_mean and area_mean are correlated as they have value 1 shown on the heatmap
#2.Compactness_mean, concavity_mean and concave points_mean which have values 0.88 and 0.92 with respect to concavity_mean.
#3.radius_se, perimeter_se and area_se are correlated as perimeter_se and area_se have values 0.97 and 0.98 with respect to radius_se.
#4.radius_worst, perimeter_worst and area_worst are correlated as perimeter_worst and area_worst have values 0.98 and 0.99 with respect to radius_worst
#5.Compactness_worst, concavity_worst and concave points_worst  seem correlated as the values of concavity_worst and concave points_worst are 0.89 and 0.8 respectively.


# In[18]:


#to determine what features to select we can plot swarm plots.


# In[19]:


sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)


# In[20]:


sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)


# In[21]:


#we drop the correlated features
drop_list1 = ['perimeter_mean','area_mean','perimeter_se','area_se','radius_worst','perimeter_worst','compactness_mean','concavity_mean','concavity_worst', 'concave points_worst']
dataforFeatureSel= x.drop(drop_list1,axis = 1 )        
dataforFeatureSel.head()


# In[22]:


plt.figure(figsize=(20,10))
sns.heatmap(dataforFeatureSel.corr().abs(), annot=True)


# In[23]:


####################Feature selection with correlation and random forest classification###########
#importing libraries necessary for random forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score


# In[24]:


#we split the data into training and testing data set( 70/30)
# to avoid overfitting we select features from training set.
x_train, x_test, y_train, y_test = train_test_split(dataforFeatureSel, y, test_size=0.3, random_state=42)


# In[25]:


#Fit the model on our data set
selFeatures = RandomForestClassifier(random_state=43)      
selFeatures=selFeatures.fit(x_train,y_train) 


# In[26]:


# NOw after fitting the model, we will test the accuracy
ac = accuracy_score(y_test,selFeatures.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,selFeatures.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")


# In[27]:


# here, the accuracy is 96 percent


# In[28]:


#######################Recursive feature elimination with cross validation and random forest classification#####################
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
#5-fold cross-validation is usually performed so we set cv=5
selFeatures2= RandomForestClassifier() 
selectingfeat = RFECV(estimator=selFeatures2, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
selectingfeat = selectingfeat .fit(x_train, y_train)

print('Optimal number of features :', selectingfeat.n_features_)
print('Best features :', x_train.columns[selectingfeat.support_])


# In[29]:


FeaturesSelected=['radius_mean', 'texture_mean', 'smoothness_mean', 'concave points_mean','fractal_dimension_mean', 'radius_se', 'texture_se', 'smoothness_se', 'concavity_se', 'concave points_se', 'fractal_dimension_se','texture_worst', 'area_worst', 'smoothness_worst', 'compactness_worst','symmetry_worst']


# In[30]:


import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(selectingfeat.grid_scores_) + 1),selectingfeat.grid_scores_)
plt.show()


# In[31]:


############################Tree based feature selection and random forest classification######################################


# In[32]:


selFeatures3 = RandomForestClassifier()      
selFeatures3 = selFeatures3 .fit(x_train,y_train)
importances = selFeatures3.feature_importances_
std = np.std([tree.feature_importances_ for tree in selFeatures3.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()


# In[33]:



drop_list2= ['fractal_dimension_se','texture_se','symmetry_mean','fractal_dimension_mean']
FinalFeaturesSelected= dataforFeatureSel.drop(drop_list2,axis = 1 )        
FinalFeaturesSelected.head()


# In[34]:


plt.figure(figsize=(16,10))
sns.heatmap(FinalFeaturesSelected.corr().abs(), annot=True)


# In[35]:


from sklearn.metrics import confusion_matrix, classification_report


# In[36]:


from sklearn import svm                                 #importing library for svm
SelectionOfFeatures= svm.SVC().fit(x_train, y_train)    # fitting the model
SELECTEDFEATURES=SelectionOfFeatures.predict(x_test)    
print(confusion_matrix(y_test,SELECTEDFEATURES))
print(classification_report(y_test,SELECTEDFEATURES))


# In[37]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=6)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3) 
clf.fit(x_train, y_train)  
print(clf.score(x_test, y_test))


# In[ ]:





# In[ ]:




