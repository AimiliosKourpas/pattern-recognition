#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Dense, Activation
from keras.models import Sequential
import numpy as np
import seaborn as sns



print('Libraries have been loaded')


dataset = pd.read_csv('housing.csv') # Load the database
dataset.head() #with this function i can see the first 5 records of housing csv (rows)

dataset.describe() #with this function i can see the statistical information of housing csv (columns)


#THEMA 1 

dataset.info() #with this function i can see the information of housing csv (columns)
X = dataset[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms', 
 'population','households','median_income','ocean_proximity']] 
z = dataset[['median_house_value']]

categorical = [col for col in X.columns if X[col].dtype=='object'] #categorical attributes
numerical = [col for col in X.columns if X[col].dtype!='object'] #numerical attributes



#Thema 1 deutero erwtima

scaler = MinMaxScaler() #we use MinMaxScaler to scale the numerical attributes
X_scaled = pd.DataFrame(scaler.fit_transform(X[numerical]),columns=numerical) #we use fit_transform to fit the scaler to the data and then transform it
X_temp = X.drop(numerical, axis=1) #we drop the numerical attributes from X 
X = pd.concat([X_temp, X_scaled], axis=1) #we concatenate the scaled numerical attributes with the categorical attributes
X.head() #we print the first 5 records of X

#do the same for the z
z = pd.DataFrame(scaler.fit_transform(y),columns=y.columns) #we use fit_transform to fit the scaler to the data and then transform it
z.head() #we print the first 5 records of z

X.describe() #we print the statistical information of X
z.describe()  #we print the statistical information of z
 
#Thema 1 trito erwtima

#we use one-hot encoding to face the categorical attribute 'ocean_proximity'
oc_prox = X['ocean_proximity'].unique() #we get the unique values of the categorical attribute 'ocean_proximity'
encoder = OneHotEncoder(handle_unknown='ignore',sparse=False) #sparse=False?
X_enc = pd.DataFrame(encoder.fit_transform(X[categorical]),columns=oc_prox) #we use fit_transform to fit the encoder to the data and then transform it
X_temp = X.drop(categorical,axis=1) #we drop the categorical attributes from X
X = pd.concat([X_temp,X_enc],axis=1) #we concatenate the encoded categorical attributes with the numerical attributes
X.head() #we print the first 5 records of X

#Thema 1 tetarto erwtima 


lost = dataset.isnull().sum() #we use isnull() to check if there are missing values and then we sum them
print(lost) #we print the missing values 
lost.plot.bar()

imputer = SimpleImputer(strategy="median") #we use SimpleImputer to impute the missing values
X_imp = pd.DataFrame(imputer.fit_transform(X[numerical]),columns=numerical) #we use fit_transform to fit the imputer to the data and then transform it
X_temp = X.drop(numerical,axis=1)  #we drop the numerical attributes from X
X = pd.concat([X_temp,X_imp],axis=1) #we concatenate the imputed numerical attributes with the categorical attributes
print(X.isnull().sum()) #we print the missing values of X 

##THEMA 2 

#1d histogram 

# longitude
sns.histplot(data[data.columns[0]],bins=50,kde=True,lw=2)
#latitude'
sns.histplot(data[data.columns[1]],bins=50,kde=True,lw=2)
#'housing_median_age'
sns.histplot(data[data.columns[2]],bins=50,kde=True,lw=2)
#'total_rooms'
sns.histplot(data[data.columns[3]],bins=50,kde=True,lw=2)
#'total_bedrooms'
sns.histplot(data[data.columns[4]],bins=50,kde=True,lw=2)
#'population'
sns.histplot(data[data.columns[5]],bins=50,kde=True,lw=2)
#'median_income'
sns.histplot(data[data.columns[7]],bins=50,kde=True,lw=2)
#'median_house_value'
sns.histplot(data[data.columns[8]],bins=50,kde=True,lw=2)
#'<1H OCEAN'
sns.histplot(data[data.columns[9]],bins=50,kde=True,lw=2)
#'INLAND'
sns.histplot(data[data.columns[10]],bins=50,kde=True,lw=2)
#'ISLAND'
sns.histplot(data[data.columns[11]],bins=50,kde=True,lw=2)
#'NEAR BAY'
sns.histplot(data[data.columns[12]],bins=50,kde=True,lw=2)
#'NEAR OCEAN'
sns.histplot(data[data.columns[13]],bins=50,kde=True,lw=2)


#2d histogram 

#we use the function boxplot to see the outliers of the numerical attributes 
dataset.plot(kind="scatter",x="longitude",y="median_house_value")  
dataset.plot(kind="scatter",x="latitude",y="median_house_value") 
dataset.plot(kind="scatter",x="housing_median_age",y="median_house_value")
dataset.plot(kind="scatter",x="total_rooms",y="median_house_value")
dataset.plot(kind="scatter",x="total_bedrooms",y="median_house_value")
dataset.plot(kind="scatter",x="population",y="median_house_value")
dataset.plot(kind="scatter",x="households",y="median_house_value")
dataset.plot(kind="scatter",x="median_income",y="median_house_value")
dataset.plot(kind="scatter",x="ocean_proximity",y="median_house_value")
plt.show()

#The map of California

#the relation of longitude and latitude. Then, show in the same graph which houses are located near the ocean.
sns.scatterplot(x=data['longitude'],y=data['latitude'],hue=data['NEAR OCEAN'])


#houses cost along each area of Callifornia.
sns.scatterplot(x=data['longitude'],y=data['latitude'],hue=data['median_house_value'])



#THEMA 3 

#2) Last square 

def least_squares_train(X, y): #X is the matrix of the attributes and y is the vector of the target attribute
    mul1 = X.T.dot(X) #we multiply the transpose of X with X
    inv1 = np.linalg.pinv(mul1) #we calculate the pseudo inverse of mul1 
    mul2 = X.T.dot(y) #we multiply the transpose of X with y 
    theta = np.matmul(inv1, mul2) #we multiply inv1 with mul2
    return theta #we return theta 

def least_squares_predict(X, w): #X is the matrix of the attributes and w is the vector of the weights
    return np.matmul(X, w) #we multiply X with w 

kf = KFold(n_splits=10) #we use KFold to split the data into 10 folds 
for k, (train_index, test_index) in enumerate(kf.split(X)): #we use enumerate to get the index of the folds
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] #we split the data into train and test
    y_train, y_test = y.iloc[train_index], y.iloc[test_index] #we split the target attribute into train and test 
    w = least_squares_train(X_train.to_numpy(), y_train.to_numpy()) #we use least_squares_train to get the weights
    pred = least_squares_predict(X_test.to_numpy(), w) #we use least_squares_predict to get the predictions
    mse = mean_squared_error(y_test.to_numpy(), pred) #we use mean_squared_error to get the mse
    mae = mean_absolute_error(y_test.to_numpy(), pred) #we use mean_absolute_error to get the mae
    print(f"Fold {k + 1} - MSE: {mse}")  #we print the mse and mae of each fold
    print(f"Fold {k + 1} - MAE: {mae}")  #we print the mse and mae of each fold
    print("\n") #we print a new line



# 3)Νευρωνικό δίκτυο


for fold, (train_index, test_index) in enumerate(kf.split(X)): #we use enumerate to get the index of the folds
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] #we split the data into train and test
    z_train, z_test = z.iloc[train_index], z.iloc[test_index] #we split the target attribute into train and test
    
    model = Sequential() #we use the Sequential model
    model.add(Dense(13, activation = 'relu', input_dim = 13)) #we add the input layer
    model.add(Dense(units = 13, activation = 'relu')) # Hidden layer 1
    model.add(Dense(units = 13, activation = 'relu')) # Hidden layer 2 
    model.add(Dense(units = 1)) # Output layer
    model.compile(optimizer = 'adam', loss = 'mean_squared_error') #we compile the model
    
    model.fit(X_train, z_train, batch_size = 10, epochs = 5) #we fit the model
    z_pred = model.predict(X_test) #we predict the target attribute
    
    mse = mean_squared_error(y_pred, y_test) #we use mean_squared_error to get the mse
    mae = mean_absolute_error(y_pred, y_test)   #we use mean_absolute_error to get the mae
    
    print(f'MSE for Fold Number: {fold + 1}, {mse}') #we print the mse and mae of each fold
    print(f'MAE for Fold Number: {fold + 1}, {mae}') #we print the mse and mae of each fold
    print("\n") #we print a new line


#The end of the code 