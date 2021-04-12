#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


# In[2]:


# dir = 'D:\\ML project data\\multimodal-deep-learning-disaster-response-mouzannar\\multimodal'


# In[3]:


# categories=['damaged_infrastructure','damaged_nature','fires','flood','human_damage','non_damage']

# data=[]

# for category in categories:
#     path=os.path.join(dir,category)  # this wil traverse to the given folders
#     label=categories.index(category) # thiss will define my folder names into 0.1.2 as follows
    
#     for img in os.listdir(path):  # this will keep the names of each image in the form of a list
#         imgpath= os.path.join(path,img) # this will give the ultimate path of every image
#         img_test=cv2.imread(imgpath,0)
#         try:
#             img_test=cv2.resize(img_test,(50,50)) # images have different sizes hence resizing to same size for all images
#             image=np.array(img_test).flatten() #images are 2d, flatten will convert them to 1d

#             data.append([image,label])
#         except Exception as e:
#             pass
        

# pick_in= open('data.pickle','wb')
# pickle.dump(data,pick_in)
# pick_in.close()
        
        


# In[4]:


pick_in= open('data.pickle','rb')
data=pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features=[]
labels=[]


# clf1=Perceptron(penalty='l2',eta0=1)
# clf2=LogisticRegression(penalty='none',C=10)
# clf3=SVC(C=100,kernel='rbf') # kernel='linear'
# clf4=KNeighborsClassifier(n_neighbors=30)
# clf5=GaussianNB()
# clf6=DecisionTreeClassifier(max_depth=10)
# clf7=RandomForestClassifier(n_estimators=100,max_depth=50)



for feature, label in data:
    features.append(feature)
    labels.append(label)
    
xtrain, xtest, ytrain, ytest= train_test_split(features, labels, test_size= 0.25, )

model_svc= SVC(C=1, kernel='poly', gamma= 'auto')
model_svc.fit(xtrain,ytrain)



prediction1svc= model_svc.predict(xtrain)
prediction2svc= model_svc.predict(xtest)

accuracy1svc= accuracy_score(ytrain, prediction1svc)
accuracy2svc= accuracy_score(ytest, prediction2svc)


categories=['damaged_infrastructure','damaged_nature','fires','flood','human_damage','non_damage']

print('Accuracy1svc: ', accuracy1svc)
print('Prediction1svc is: ', categories[prediction1svc[0]])

print('Accuracy2svc: ', accuracy2svc)
print('Prediction2svc is: ', categories[prediction2svc[0]])

myphoto=xtest[0].reshape(50,50)

plt.imshow(myphoto, cmap='gray')
plt.show()


# In[5]:


model_knn= KNeighborsClassifier(n_neighbors=3)
model_knn.fit(xtrain,ytrain)



prediction1knn= model_knn.predict(xtrain)
prediction2knn= model_knn.predict(xtest)

accuracy1knn= accuracy_score(ytrain, prediction1knn)
accuracy2knn= accuracy_score(ytest, prediction2knn)


categories=['damaged_infrastructure','damaged_nature','fires','flood','human_damage','non_damage']

print('Accuracy1knn: ', accuracy1knn)
print('Prediction1knn is: ', categories[prediction1knn[0]])

print('Accuracy2knn: ', accuracy2knn)
print('Prediction2knn is: ', categories[prediction2knn[0]])

myphoto=xtest[0].reshape(50,50)

plt.imshow(myphoto, cmap='gray')
plt.show()


# In[6]:


model_rdf=RandomForestClassifier(n_estimators=100,max_depth=5)
model_rdf.fit(xtrain,ytrain)


prediction1rdf= model_rdf.predict(xtrain)
prediction2rdf= model_rdf.predict(xtest)

accuracy1rdf= accuracy_score(ytrain, prediction1rdf)
accuracy2rdf= accuracy_score(ytest, prediction2rdf)


categories=['damaged_infrastructure','damaged_nature','fires','flood','human_damage','non_damage']

print('Accuracy1rdf: ', accuracy1rdf)
print('Prediction1rdf is: ', categories[prediction1rdf[0]])

print('Accuracy2rdf: ', accuracy2rdf)
print('Prediction2rdf is: ', categories[prediction2rdf[0]])

myphoto=xtest[0].reshape(50,50)

plt.imshow(myphoto, cmap='gray')
plt.show()


# In[7]:


param_knn = {
    'n_neighbors': list(range(1,10)),
    'leaf_size' : list(range(1,10)),
    'p': [1,2]
}
model_tuned_knn = RandomizedSearchCV(model_knn,param_knn,cv=3)
model_tuned_knn.fit(xtrain,ytrain)
print(model_tuned_knn.best_params_)
print('mean vald accuracy ',model_tuned_knn.best_score_)
model_tuned_knn = RandomizedSearchCV(model_knn,param_knn,cv=3)
model_tuned_knn.fit(xtest,ytest)
print(model_tuned_knn.best_params_)
print('mean vald accuracy ',model_tuned_knn.best_score_)


# In[8]:


param_svm = {
    'C': [0.05,0.1,0.5,1,2,3,5,10],
    'kernel' : ['linear','poly','rbf','sigmoid'],
    'gamma': ['scale','auto']
}
model_tuned_svm = RandomizedSearchCV(model_svc,param_svm,cv=3,random_state=42)
model_tuned_svm.fit(xtrain,ytrain)
print(model_tuned_svm.best_params_)
print('mean vald accuracy of SVM ',model_tuned_svm.best_score_)


# In[9]:


model_tuned_svm = RandomizedSearchCV(model_svc,param_svm,cv=3,random_state=42)
model_tuned_svm.fit(xtest,ytest)
print(model_tuned_svm.best_params_)
print('mean vald accuracy of SVM ',model_tuned_svm.best_score_)


# In[10]:


param_rfc = {
    'n_estimators': [10,50,100,150,200],
    'max_features' : ['auto', 'sqrt'],
    'max_depth': [5,10,15]
}
model_tuned_rfc = RandomizedSearchCV(model_rdf,param_rfc,cv=3,random_state=42)
model_tuned_rfc.fit(xtrain,ytrain)
print(model_tuned_rfc.best_params_)
print('mean vald accuracy of Random Forest ',model_tuned_rfc.best_score_)
model_tuned_rfc = RandomizedSearchCV(model_rdf,param_rfc,cv=3,random_state=42)
model_tuned_rfc.fit(xtest,ytest)
print(model_tuned_rfc.best_params_)
print('mean vald accuracy of Random Forest ',model_tuned_rfc.best_score_)


# In[13]:


from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
cr=classification_report(ytest,prediction2knn)
print(cr)
print("confusion matrix for KNN")
plot_confusion_matrix(model_knn,features,labels)


# In[15]:


from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
cr=classification_report(ytest,prediction2svc)
print(cr)
print("confusion matrix for SVC")
plot_confusion_matrix(model_svc,features,labels)


# In[14]:


from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
cr=classification_report(ytest,prediction2rdf)
print(cr)
print("confusion matrix for RDF")
plot_confusion_matrix(model_rdf,features,labels)


# In[ ]:




