import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/home/shubhayu/Developer/Ads_Sales/Advertising.csv')
print(df.head())



df3=plt.scatter( df['TV'],  y=df['Sales'])
plt.show()
print(df3)


# TRAIN_TEST-SPLIT


# from sklearn.model_selection import train_test_split

# X = df.iloc[:, :-1]   # sab columns except last one
# y = df.iloc[:, -1]    # sirf last column

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
from sklearn.model_selection import train_test_split
X = df.iloc[:,:-1]
y= df.iloc[:, -1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)  
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# LINEAR REGRESSION 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# NOW TIME TO TRAIN IN THIS FUNC - 'LINEAR REGRESSION' A PREBUILT OBJECT IS MADE NAME - 'FIT' NOW PASS 80% DATA FROM MY DATASET 

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print(y_pred)

# Metrices R2_score 
from sklearn.metrics import r2_score,mean_absolute_error
acc = mean_absolute_error(y_test,y_pred) #MAE = \frac{1}{n} \sum_{i=1}^{n} |y_{true} - y_{pred}|
# 	•	Ye prediction aur actual values ke beech ka average absolute difference nikalta hai.
# 	•	Kam value ka matlab model better hai.
# 	•	Units same hoti hain jo target variable ki hoti hain (yaha Sales ka unit).

# Example:
# Actual Sales = [10, 20, 30]
# Predicted Sales = [12, 18, 33]
# Differences = [2, 2, 3] → MAE = (2+2+3)/3 = 2.33



print(acc) # acc - 1.274826210954934
r2_score(y_test, y_pred)


# Prediction function
def predict_sales(tv_budget, radio_budget, newspaper_budget):
    features = np.array([[tv_budget, radio_budget, newspaper_budget]])
    result = lr.predict(features)   # ye array return karega
    return result[0]                # single value nikalo

# Pehli row ka data check karna
print(df.iloc[0])

# Predict karna first row ke budget ke liye
tv_budget = 230.1
radio_budget = 37.8
newspaper_budget = 69.2

sales = predict_sales(tv_budget, radio_budget, newspaper_budget)
print(f"Predicted Sales: {sales}") #Predicted Sales: 21.372540280396883

print(lr.coef_) #[0.05450927 0.10094536 0.00433665]
print(lr.intercept_) #4.714126402214127