import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
class Moon:
    def __init__(self,data,data_origin,column=[]):
        self.data = data
        self.data_origin = data_origin
        self.column = []
    def columns(self):
        a = []
        for i in self.data.columns:
            a.append(i)
        return a
    def corr_all(self):
        corr = self.data.select_dtypes(include=['float64']).corr()
        sns.heatmap(data=corr)
        return corr
    def corr(self,x,y):
        data = self.data.dropna()
        db_corr = self.data.select_dtypes(include=['float64'])
        corr_num = db_corr.corr()[x][y]
        sns.relplot(data=db_corr,x=x,y=y)
        print(f"correlation factor bewteen {x} and {y} is {corr_num}")
        return corr_num
    def plot(self,x,y):
        sns.relplot(data=self.data,x=x,y=y)
    def drop(self,x):
        if x not in self.column and x in self.columns():
            self.column.append(x)
        self.data = self.data_origin.drop(columns=self.column)
    def recover(self,x):
        if x in self.column and x in self.data_origin.columns:
            self.column.remove(x)
        self.data = self.data_origin.drop(columns=self.column)
    def complete_recover(self):
        self.data = self.data_origin
        self.column = []
    def means(self,x):
        print(f"mean of {x} is {self.data[x].mean()}")
        return self.data[x].mean()
    def mean_groupby(self,x):
        return self.data.groupby(x).mean()
    def std(self,x):
        print(f"standard deviation of {x} is {self.data[x].std()}")
    def std_groupby(self,x):
        return self.data.groupby(x).std()
    def specific_moon(self,x):
        return self.data[x]
    def locate(self,x):
        return self.data.loc[x]
    def locate_singal(self,x,y):
        return self.data.loc[x][y]
    def power(self,x,y,z):
        if z not in self.data_origin.columns and x in self.columns():
            x_sq = np.power(self.data[x],y)
            self.data_origin[z] = x_sq
            self.data[z] = x_sq
        elif z in self.data_origin.columns:
            x_sq = np.power(self.data[x],y)
            self.data[z] = x_sq
    def day2s(self,x,y):
        if y not in self.data_origin.columns and x in self.columns():
            x_sq = self.data[x]*24*60*60
            self.data_origin[y] = x_sq
            self.data[y] = x_sq
        elif y in self.data_origin.columns:
            x_sq = self.data[x]*24*60*60
            self.data[y] = x_sq
    def km2m(self,x,y):
        if y not in self.data_origin.columns and x in self.columns():
            x_sq = self.data[x]*1000
            self.data_origin[y] = x_sq
            self.data[y] = x_sq
        elif y in self.data_origin.columns:
            x_sq = self.data[x]*1000
            self.data[y] = x_sq
    def model_construction(self,x,y):
        # we expect the intercept to be zero
        model = linear_model.LinearRegression(fit_intercept=True)
        
        X = self.data[[x]]
        Y = self.data[y]
        # 42 is the answer to everything, alos here split dataser into training set and testing set
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.42, random_state=42)
        #train model 
        model.fit(x_train,y_train)
        #gives prediction
        pred = model.predict(x_test)
        #generate scores
        print(f"The R2 score is: {r2_score(y_test,pred)}")
        print(f"The RMSE score is: {mean_squared_error(y_test,pred, squared=False)}")
        return model
        