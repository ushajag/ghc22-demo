import pandas

data=pandas.read_csv("../dataset/cancer-data.txt",header=None)
data.columns=['id','p1','p2','p3','p4','p5','p6','p7','p8','p9','Class']
print(data.head)

data.drop(['id'],axis=1,inplace=True)
data.replace('?',5,inplace=True)

xdata=data.drop(['Class'],axis=1)
ydata=data['Class']

from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(xdata,ydata,test_size=0.1)

from sklearn.linear_model import LogisticRegression
alg=LogisticRegression()

#train the model
alg.fit(xtr,ytr)


#accuracy
accuracy=alg.score(xts,yts)
print(accuracy)

