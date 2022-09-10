#!/usr/bin/env python
# coding: utf-8

# # Price Prediction for Pre-Owned Cars

# In[1]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd 
import plotly.express as px


# In[2]:


df = pd.read_csv("CAR DETAILS FROM CAR DEKHO (1).csv")


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.head()


# # There is no null values

# In[7]:


df.name.unique


# In[8]:


df = df.drop(["name"], axis = 1)


# In[9]:


columns = {'year':1,'selling_price':2,'km_driven':3}
plt.figure(figsize = (10,30))
for var, i in columns.items():
    plt.subplot(8,4,i)
    plt.boxplot(df[var], whis = 1.5)
    plt.title(var)
plt.show()


# In[10]:


import plotly.express as px
px.scatter(df, x = "km_driven", y = "selling_price",color = "seller_type",title = "radius_mean vs texture_mean vs diagnosis ")


# In[11]:


df.seller_type.unique()


# In[12]:


df.seller_type = df.seller_type.replace("Individual",1)
df.seller_type = df.seller_type.replace("Dealer",0)
df.seller_type = df.seller_type.replace("Trustmark Dealer",2)


# In[13]:


df.transmission.unique()


# In[14]:


df.transmission = df.transmission.replace("Manual",1)
df.transmission = df.transmission.replace("Automatic",0)


# In[15]:


df.fuel.unique()


# In[16]:


df.fuel = df.fuel.replace("Petrol",0)
df.fuel = df.fuel.replace("Diesel",1)
df.fuel = df.fuel.replace("CNG",2)
df.fuel =df.fuel.replace("LPG",3)
df.fuel =df.fuel.replace("Electric",4)


# ## Petrol = 0
# ## Diesel = 1
# ## CNG = 1
# ## LPG = 3
# ## Electric = 4

# In[17]:


df.owner.value_counts()


# In[18]:


df.owner = df.owner.replace("First Owner",0)
df.owner = df.owner.replace("Second Owner",1)
df.owner = df.owner.replace("Fourth & Above Owner",2)
df.owner = df.owner.replace("Fourth & Above Owner",3)
df.owner = df.owner.replace("Test Drive Car",4)


# In[19]:


df.head()


# In[20]:


df["current_year"]=2022
df["num_year"]=df["current_year"]-df["year"]


# In[21]:


df.head(5)


# In[22]:


import seaborn as sns
sns.pairplot(df)


# In[23]:





columns = {'year':1,'selling_price':2,'km_driven':3}

plt.figure(figsize = (10,30))
for var, i in columns.items():
    plt.subplot(5,4,i)
    plt.boxplot(df[var], whis = 1.5)
    plt.title(var)
plt.show()


# In[24]:


#There are outlayers in few varibales


# In[25]:


df.year.unique()


# In[26]:


import seaborn as sns
sns.pairplot(df, x_vars=['year','km_driven'], y_vars='selling_price',height=7, aspect=0.7, kind='scatter')


# In[27]:


plt.hist(df.km_driven,10)


# In[28]:


sns.heatmap(df.corr(),annot=True)


# In[29]:


import numpy as np
def outliers(a):
    a = sorted(a)
    q1 = np.percentile(a, 25)
    q3 = np.percentile(a, 75)
    # print(q1, q3)
    IQR = q3-q1
    Lower_limit = q1-(1.5*IQR)
    upper_limit = q3+(1.5*IQR)
    a = np.where(a < Lower_limit, Lower_limit,a )
    a = np.where(a > upper_limit, upper_limit,a )
    print("Lower_limit","=",Lower_limit)
    print("upper_limit","=",upper_limit)


# In[30]:


df.head(2)


# In[31]:


outliers(df.selling_price)
sns.boxplot(df.selling_price)


# In[32]:


Lower_limit = -378125.625
upper_limit = 1186875.375
df.selling_price = np.where(df.selling_price < Lower_limit, Lower_limit,df.selling_price )
df.selling_price = np.where(df.selling_price > upper_limit, upper_limit,df.selling_price )
sns.boxplot(df.selling_price)


# In[33]:


outliers(df.km_driven)
sns.boxplot(df.km_driven)


# In[34]:


Lower_limit = -47500.0
upper_limit = 172500.0
df.km_driven = np.where(df.km_driven < Lower_limit, Lower_limit,df.km_driven )
df.km_driven = np.where(df.km_driven > upper_limit, upper_limit,df.km_driven )
sns.boxplot(df.km_driven)


# In[35]:


outliers(df.num_year)
sns.boxplot(df.num_year)


# In[36]:


Lower_limit = -1.5
upper_limit = 18.5
df.num_year = np.where(df.year < Lower_limit, Lower_limit,df.num_year )
df.num_year = np.where(df.num_year > upper_limit, upper_limit,df.num_year )
sns.boxplot(df.num_year)


# In[37]:


outliers(df.year)
sns.boxplot(df.year)


# In[38]:


Lower_limit = 2003.5
upper_limit = 2023.5
df.year = np.where(df.year < Lower_limit, Lower_limit,df.year )
df.year = np.where(df.year > upper_limit, upper_limit,df.year )
sns.boxplot(df.year)


# In[39]:


df.info()


# In[40]:


df.fuel.unique()


# In[41]:


import seaborn as sns
sns.pairplot(df, x_vars=["km_driven",'num_year'], y_vars='selling_price',height=7, aspect=0.7, kind='scatter')


# In[42]:


sns.heatmap(df.corr(),annot=True)


# In[43]:



columns = {'year':1,'selling_price':2,'km_driven':3}

plt.figure(figsize = (10,30))
for var, i in columns.items():
    plt.subplot(5,4,i)
    plt.boxplot(df[var], whis = 1.5)
    plt.title(var)
plt.show()


# In[44]:


df.info()


# ### Adding feature number of km driven per year

# In[68]:


df1 = df


# In[70]:


df1


# In[71]:


df1["Km_per_year"] = df1.km_driven/df.num_year


# In[72]:


df1


# In[85]:


df = df1


# In[84]:


df.fuel.unique()


# In[83]:


df.owner.unique()


# In[88]:


df.owner = df.owner.replace(["Third Owner"],3)


# In[89]:


df.Km_per_year.describe()


# # Model Building

# In[91]:


import statsmodels.formula.api as smf

model = smf.ols(formula = "selling_price ~  km_driven + C(fuel) + C(seller_type) + (transmission) + owner + num_year + Km_per_year",data = df).fit()
print(model.summary())


# In[46]:


from sklearn.model_selection import train_test_split


# In[100]:


X_train, X_test, y_train, y_test = train_test_split(df[["km_driven", "fuel", "seller_type","transmission","num_year","owner" ]],df.selling_price,train_size=0.75)


# In[101]:


X_train


# In[102]:


from sklearn.linear_model import LinearRegression
la = LinearRegression()
la.fit(X_train,y_train)


# In[103]:


y_test.head(5)


# In[104]:


ya = la.predict(X_test)
ya


# In[105]:


#model 2

print("Coefficient", la.coef_ , "intercept", la.intercept_)

print("The R_square is: ", round(la.score (X_train,y_train),3))


# In[106]:


from sklearn.ensemble import RandomForestRegressor

RandomForest_model=RandomForestRegressor(n_estimators=200,max_depth=10)
RandomForest_model.fit(X_train,y_train)
accuracy=RandomForest_model.score(X_test,y_test)
accuracy


# In[107]:


sns.regplot(x=y_test,y= ya,ci=None,color='r');
plt.scatter(y_test, ya)
plt.show()


# # RandomForest Model gives us 67% accuracy

# # End
