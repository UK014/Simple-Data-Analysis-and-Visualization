import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


sales = pd.read_csv("sales_data.csv")

##Prediction Part
sales_data = pd.DataFrame({
    'Year': [2011, 2012, 2013, 2014, 2015, 2016],
    'Order_Quantity': [sales.loc[sales['Year'] == 2011,'Order_Quantity'].sum(), sales.loc[sales['Year'] == 2012,'Order_Quantity'].sum(), sales.loc[sales['Year'] == 2013,'Order_Quantity'].sum(), sales.loc[sales['Year'] == 2014,'Order_Quantity'].sum(), sales.loc[sales['Year'] == 2015,'Order_Quantity'].sum(), sales.loc[sales['Year'] == 2016,'Order_Quantity'].sum()]
})
X = sales_data[['Year']]
y = sales_data['Order_Quantity']


model = LinearRegression()
model.fit(X, y)
print('Intercept:', model.intercept_)
print('Slope:', model.coef_[0])
predicted_num_sales = model.predict([[2017]])
print('Predicted number of sales for 2017:', predicted_num_sales[0])

##Visualization Part
sales.groupby(sales['Year']).agg({'Order_Quantity': 'sum'}).plot(kind='bar')
plt.title("Yearly Sales")
plt.show()

sales['Age_Group'].value_counts().plot(kind='pie',figsize=(6,6))
plt.title("Age Group")
plt.show()

sales.groupby(sales['Age_Group']).agg({'Order_Quantity': 'sum'}).plot(kind='bar')
plt.title("Sales by Age Group")
plt.show()

sales.loc[sales['State'] == 'Kentucky'].groupby(sales['Year']).agg({'Order_Quantity': 'sum'}).plot(kind='bar')
plt.title("Yearly Sales in The Kentucky")
plt.show()
