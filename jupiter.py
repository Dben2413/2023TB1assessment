import seaborn as sns
class Moon:
    def __init__(self,data,data_origin):
        self.data = data
        self.data_origin = data_origin
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
    def drop(self,x):
        self.data = self.data_origin.drop(columns=[x])
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