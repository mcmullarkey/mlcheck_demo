import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,roc_curve,auc

df = pd.read_csv("fil.csv")
df.drop(df.columns[[0]],axis=1,inplace=True)


y = df[df.columns[[-1]]]
x = df.drop(df.columns[[4097]],axis=1,inplace=False)

print(y.head())
print(x.head())






import pywt
n_channels = 5
ll=[]
for a in range(0,500):
	ll = ll + [pywt.wavedec(x.iloc[a], pywt.Wavelet("db1"), level=n_channels-1)]


	
	
	
	
	

			   
print(ll[0][0])



def mean_signal(m):
    return np.mean(m)

def std_signal(m):
    return np.std(m)

def mean_square_signal(m):
    return np.mean(m ** 2)

def abs_diffs_signal(m):
    return np.sum(np.abs(np.diff(m)))

def skew_signal(m):
    return scipy.stats.skew(m)
	




l = []
for a in range(0,500):
	mean = []
	std = []
	max = []
	mean_square = []
	abs_diff = []
	skew = []
	min = []
	for b in range(0,5):
		mean = mean + [mean_signal(np.array(ll[a][b]))]
		std = std + [std_signal(np.array(ll[a][b]))]
		max = max + [np.array(ll[a][b]).max()]
		mean_square= mean_square + [mean_square_signal(np.array(ll[a][b]))]
		abs_diff = abs_diff + [abs_diffs_signal(np.array(ll[a][b]))]
		skew = skew + [skew_signal(np.array(ll[a][b]))]
		min = min + [np.array(ll[a][b]).min()]
		
	
	
	l= l + [mean+std+max+min+mean_square+abs_diff+skew]
    

features = pd.DataFrame(l)
print(features.head())







x_train,x_test,y_train,y_test = train_test_split(features,y,test_size=0.4,random_state=4)


classifier_list=[]
accuracy_list=[]
precision_list=[]
f1_list=[]
recall_list=[]
auc_list=[]


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=13)
clf.fit(x_train,y_train)
y_pre = clf.predict(x_test)

classifier_list.append("Knn")
accuracy_list.append(accuracy_score(y_test,y_pre))
precision_list.append(precision_score(y_test,y_pre))
f1_list.append(f1_score(y_test,y_pre))
recall_list.append(recall_score(y_test,y_pre))

fpr,tpr,t = roc_curve(y_test,y_pre)
roc_auc = auc(fpr,tpr)
auc_list.append(roc_auc)

plt.title('ROC - Knn ')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


ac=[]
y_pre = clf.predict(np.array(features.iloc[0:100]))
ac = ac + [accuracy_score(y.iloc[0:100],y_pre)]
y_pre = clf.predict(np.array(features.iloc[100:200]))
ac = ac + [accuracy_score(y.iloc[100:200],y_pre)]
y_pre = clf.predict(np.array(features.iloc[200:300]))
ac = ac + [accuracy_score(y.iloc[200:300],y_pre)]
y_pre = clf.predict(np.array(features.iloc[300:400]))
ac = ac + [accuracy_score(y.iloc[300:400],y_pre)]
y_pre = clf.predict(np.array(features.iloc[400:500]))
ac = ac + [accuracy_score(y.iloc[400:500],y_pre)]


ac_list = ac_list + [ac]

objects = ("Z","O","N","F","S")
y_pos = np.arange(len(objects))
plt.bar(y_pos,ac,align="center")
plt.xticks(y_pos,objects)
plt.ylabel("Accuarcy")
plt.title("knn")
plt.show()



from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x_train,y_train)
y_pre = clf.predict(x_test)

classifier_list.append("Logistic Regression")
accuracy_list.append(accuracy_score(y_test,y_pre))
precision_list.append(precision_score(y_test,y_pre))
f1_list.append(f1_score(y_test,y_pre))
recall_list.append(recall_score(y_test,y_pre))

fpr,tpr,t = roc_curve(y_test,y_pre)
roc_auc = auc(fpr,tpr)
auc_list.append(roc_auc)

plt.title('ROC - Logistic Regression')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


ac=[]
y_pre = clf.predict(np.array(features.iloc[0:100]))
ac = ac + [accuracy_score(y.iloc[0:100],y_pre)]
y_pre = clf.predict(np.array(features.iloc[100:200]))
ac = ac + [accuracy_score(y.iloc[100:200],y_pre)]
y_pre = clf.predict(np.array(features.iloc[200:300]))
ac = ac + [accuracy_score(y.iloc[200:300],y_pre)]
y_pre = clf.predict(np.array(features.iloc[300:400]))
ac = ac + [accuracy_score(y.iloc[300:400],y_pre)]
y_pre = clf.predict(np.array(features.iloc[400:500]))
ac = ac + [accuracy_score(y.iloc[400:500],y_pre)]


ac_list = ac_list + [ac]

objects = ("Z","O","N","F","S")
y_pos = np.arange(len(objects))
plt.bar(y_pos,ac,align="center")
plt.xticks(y_pos,objects)
plt.ylabel("Accuarcy")
plt.title("Logistic Regression")
plt.show()



from sklearn import svm
clf = svm.SVC(kernel="linear")
clf.fit(x_train,y_train)
y_pre = clf.predict(x_test)

classifier_list.append("SVM Linear")
accuracy_list.append(accuracy_score(y_test,y_pre))
precision_list.append(precision_score(y_test,y_pre))
f1_list.append(f1_score(y_test,y_pre))
recall_list.append(recall_score(y_test,y_pre))

fpr,tpr,t = roc_curve(y_test,y_pre)
roc_auc = auc(fpr,tpr)
auc_list.append(roc_auc)

plt.title('ROC - SVM LINEAR')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

ac=[]
y_pre = clf.predict(np.array(features.iloc[0:100]))
ac = ac + [accuracy_score(y.iloc[0:100],y_pre)]
y_pre = clf.predict(np.array(features.iloc[100:200]))
ac = ac + [accuracy_score(y.iloc[100:200],y_pre)]
y_pre = clf.predict(np.array(features.iloc[200:300]))
ac = ac + [accuracy_score(y.iloc[200:300],y_pre)]
y_pre = clf.predict(np.array(features.iloc[300:400]))
ac = ac + [accuracy_score(y.iloc[300:400],y_pre)]
y_pre = clf.predict(np.array(features.iloc[400:500]))
ac = ac + [accuracy_score(y.iloc[400:500],y_pre)]


ac_list = ac_list + [ac]

objects = ("Z","O","N","F","S")
y_pos = np.arange(len(objects))
plt.bar(y_pos,ac,align="center")
plt.xticks(y_pos,objects)
plt.ylabel("Accuarcy")
plt.title("SVM Linear")
plt.show()



clf = svm.SVC(kernel="poly")
clf.fit(x_train,y_train)
y_pre = clf.predict(x_test)

classifier_list.append("SVM POLY")
accuracy_list.append(accuracy_score(y_test,y_pre))
precision_list.append(precision_score(y_test,y_pre))
f1_list.append(f1_score(y_test,y_pre))
recall_list.append(recall_score(y_test,y_pre))

fpr,tpr,t = roc_curve(y_test,y_pre)
roc_auc = auc(fpr,tpr)
auc_list.append(roc_auc)

plt.title('ROC - SVM POLY')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



ac=[]
y_pre = clf.predict(np.array(features.iloc[0:100]))
ac = ac + [accuracy_score(y.iloc[0:100],y_pre)]
y_pre = clf.predict(np.array(features.iloc[100:200]))
ac = ac + [accuracy_score(y.iloc[100:200],y_pre)]
y_pre = clf.predict(np.array(features.iloc[200:300]))
ac = ac + [accuracy_score(y.iloc[200:300],y_pre)]
y_pre = clf.predict(np.array(features.iloc[300:400]))
ac = ac + [accuracy_score(y.iloc[300:400],y_pre)]
y_pre = clf.predict(np.array(features.iloc[400:500]))
ac = ac + [accuracy_score(y.iloc[400:500],y_pre)]



ac_list = ac_list + [ac]

objects = ("Z","O","N","F","S")
y_pos = np.arange(len(objects))
plt.bar(y_pos,ac,align="center")
plt.xticks(y_pos,objects)
plt.ylabel("Accuarcy")
plt.title("SVM poly")
plt.show()



from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train,y_train)
y_pre = clf.predict(x_test)
classifier_list.append("Naive Bayes")

accuracy_list.append(accuracy_score(y_test,y_pre))

precision_list.append(precision_score(y_test,y_pre))

f1_list.append(f1_score(y_test,y_pre))

recall_list.append(recall_score(y_test,y_pre))

fpr,tpr,t = roc_curve(y_test,y_pre)
roc_auc = auc(fpr,tpr)
auc_list.append(roc_auc)

plt.title('ROC - Gaussian NB')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


ac=[]
y_pre = clf.predict(np.array(features.iloc[0:100]))
ac = ac + [accuracy_score(y.iloc[0:100],y_pre)]
y_pre = clf.predict(np.array(features.iloc[100:200]))
ac = ac + [accuracy_score(y.iloc[100:200],y_pre)]
y_pre = clf.predict(np.array(features.iloc[200:300]))
ac = ac + [accuracy_score(y.iloc[200:300],y_pre)]
y_pre = clf.predict(np.array(features.iloc[300:400]))
ac = ac + [accuracy_score(y.iloc[300:400],y_pre)]
y_pre = clf.predict(np.array(features.iloc[400:500]))
ac = ac + [accuracy_score(y.iloc[400:500],y_pre)]


ac_list = ac_list + [ac]

objects = ("Z","O","N","F","S")
y_pos = np.arange(len(objects))
plt.bar(y_pos,ac,align="center")
plt.xticks(y_pos,objects)
plt.ylabel("Accuarcy")
plt.title("knn")
plt.show()


table={'Classifier' : classifier_list,'Accuracy':accuracy_list,'Precision':precision_list,'F1-score':f1_list,'Recall':recall_list,'AUC':auc_list}
table = pd.DataFrame(table,columns=['Classifier','Accuracy','Precision','F1-score','Recall','AUC'])

print(table)

print(ac_list)

table.to_csv("dwt.csv")