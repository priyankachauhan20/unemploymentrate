import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.dates as matdates
#plt.style.use('ggplot')
import math

dataset = pd.read_excel('US umemployment rate 1948-2017.xlsx')
dataset.head()
print(dataset.head())

dataset = dataset.rename(columns={'Value':'Unemployment Rate'})
print(dataset.head())

date=dataset.set_index('Year').groupby('Year').mean()
print(date.head())

daa = pd.DataFrame(dataset['Year'].unique(), columns=['Year'])
print(daa.head())

sum=0
avg=[]
n=0
for x in range(len(daa)):
    for y in range(n,len(dataset)):
        if(dataset['Year'][y] == daa['Year'][x]):
            sum += dataset['Unemployment Rate'][y]
        else:
            avg.append(sum/12)
            n=y
            sum=0
            break
        if(y == 839):
            avg.append((sum/12))
            print(avg[0:5])
            
daa['Unemployment Rate'] = pd.DataFrame(avg, columns=['Unemployment Rate'])
daa['Unemployment Rate'] = daa['Unemployment Rate'].round(3)
print(daa.head())     

fig,cn = plt.subplots(figsize=(15,5))
cn.plot(daa['Year'], daa['Unemployment Rate'])
cn.locator_params(nbins=70, axis='x')
fig.autofmt_xdate()
plt.title('US Unemployment Rate from 1948 to 2017')
plt.show()


daa['Unemployment Rate'] = np.log(daa['Unemployment Rate'])
print(daa.head())

daa_set=daa['Unemployment Rate'].values
print(len(daa_set))

training_set = daa_set[:50]
X_train = []
y_train = []
for i in range(30, len(training_set)):
    X_train.append(training_set[i-30:i])
    y_train.append(training_set[i])
    
test_set = daa_set[20:] 
X_test = []
y_test = daa_set[50:]
for i in range(30, 50):
    X_test.append(training_set[i-30:i])    
    
from sklearn.linear_model import LinearRegression
at = LinearRegression()
print(at.fit(X_train, y_train))


p_at = at.predict(X_test)

for i in range(20):
    y_test[i] = math.exp(y_test[i])
    p_at[i] = math.exp(p_at[i])
    
L20y = daa['Year'][50:]
fig,a = plt.subplots(figsize=(15,5))
one, = a.plot(L20y, y_test, color='red')
two, = a.plot(L20y, p_at, color='blue')
plt.legend([one,two],['Original','Predicted'])
a.locator_params(nbins=20, axis='x')

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=1)
print(knn.fit(X_train, y_train))

p_n = knn.predict(X_test)
for i in range(20):
    p_n[i] = math.exp(p_n[i])
fig,a = plt.subplots(figsize=(15,5))
one, = a.plot(L20y, y_test, color='red')
two, = a.plot(L20y, p_n, color='blue')
plt.legend([one,two],['Original','Predicted'])
a.locator_params(nbins=20, axis='x')

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
print(tree.fit(X_train, y_train))


p_t = tree.predict(X_test)
for i in range(20):
    p_t[i] = math.exp(p_t[i])
fig,a = plt.subplots(figsize=(15,5))
one, = a.plot(L20y, y_test, color='red')
two, = a.plot(L20y, p_t, color='blue')
plt.legend([one,two],['Original','Predicted'])
a.locator_params(nbins=20, axis='x')



from sklearn.ensemble import RandomForestRegressor
cs = RandomForestRegressor(n_jobs=100)
print(cs.fit(X_train, y_train))

p_cs = cs.predict(X_test)
for i in range(20):
    p_cs[i] = math.exp(p_cs[i])
fig,a = plt.subplots(figsize=(15,5))
one, = a.plot(L20y, y_test, color='red')
two, = a.plot(L20y, p_cs, color='blue')
plt.legend([one,two],['Original','Predicted'])
a.locator_params(nbins=20, axis='x')



from sklearn.svm import SVR

s= SVR()
print(s.fit(X_train, y_train))

p_s = s.predict(X_test)
for i in range(20):
    p_s[i] = math.exp(p_s[i])
fig,a = plt.subplots(figsize=(15,5))
one, = a.plot(L20y, y_test, color='red')
two, = a.plot(L20y, p_s, color='blue')
plt.legend([one,two],['Original','Predicted'])
a.locator_params(nbins=20, axis='x')    


fig,a = plt.subplots(figsize=(15,5))
b, = a.plot(L20y, y_test, color='red')
c, = a.plot(L20y, p_at, color='blue')
d, = a.plot(L20y, p_n, color='green')
e, = a.plot(L20y, p_t, color='black')
f, = a.plot(L20y, p_cs, color='orange')
g, = a.plot(L20y, p_s, color='brown')
plt.legend([b,c,d,e,f,g],['Original','Linear Regression', 'KNN', 'Tree', 'Randon Forest', 'SVM'])
