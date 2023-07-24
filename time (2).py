#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


path=r'C:\Users\saisr\Downloads\yahoo_stock.csv'
df = pd.read_csv(path)
df.head()


# In[5]:


df.shape
df.describe


# In[20]:


sns.kdeplot(data = df[['High', 'Low', 'Open', 'Close']])


# In[6]:


plt.figure(figsize=(12,7))
sns.histplot(data=df, x='Open', kde=True)


# In[7]:


plt.figure(figsize=(12,7))
sns.histplot(data=df, x='Close', bins=20, kde=True)
plt.xlabel('Closeing Price')
plt.ylabel('Frequency')
plt.title('Distribution of Yahoo Stock Closing Price')
plt.show()


# In[8]:


df['Year'] = pd.to_datetime(df['Date']).dt.year
plt.figure(figsize=(12,7))
sns.boxplot(data=df, x='Year', y='Close')
plt.xlabel('Year')
plt.ylabel('Closeing Price')
plt.title('Distribution of Yahoo Stock Closing Price by Year')

plt.show()


# In[9]:


x = df[['High', 'Low', 'Open', 'Volume']].values
y = df['Close'].values


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[11]:


model = LinearRegression()
model.fit(x_train, y_train)


# In[12]:


y_pred = model.predict(x_test)


# In[22]:


plt.figure(figsize=(12, 7))
plt.style.use('dark_background')
plt.title('Linear Regression', color="yellow")
plt.scatter(y_pred, y_test, color="#FFFF")
plt.scatter(y_test, y_test, color="red")
plt.plot(y_test, y_test, color="yellow")
plt.legend(["Predicted_Close", "Actual_Close", "Regression Line"], loc="lower right", facecolor='green', labelcolor='white')

plt.xlabel('Predicted Close Price')
plt.ylabel('Actual Close Price')

plt.show()


# In[14]:


plt.figure(figsize=(12, 7))
plt.plot(y_test, color='red', label='Actual')
plt.plot(y_pred, color='white', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Linear Regresion Actual vs Predicted(Close Price)')
plt.legend()

plt.show()


# In[15]:


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error(MSE):", mse)

#Calculate RMSE
rmse = np.sqrt(mse)
print("Root Mean Squared Error(RMSE):", rmse)

#Calculate R2 Score
r2 = r2_score(y_test, y_pred)
print("R-Squared Score:", r2)


# In[16]:


#created df w/the predicted values
prediction_df = pd.DataFrame({'Predicted': y_pred})

#save df as a csv file
prediction_df.to_csv('Prediction.csv', index=False)

print("CSV file 'Prediction.csv' successfully saved!!!")


# In[32]:


X=[2,3,45,6,6]
Y=[3,4,5,6,7,7]
Z=[3,4,5,6,7,7100]


# In[36]:


plt.hist(X,Y,Z)
plt.title('TIMES')


# In[ ]:





# In[ ]:





# In[ ]:




