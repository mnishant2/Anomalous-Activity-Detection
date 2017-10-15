
# coding: utf-8

# In[3]:

#import data
import numpy as np
import pandas
import cPickle
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# In[24]:

# load dataset--------------------------------------------------------------
f1=open('C:/Users/Asus/Downloads/Data/X_cent400.save','rb')
f2=open('C:/Users/Asus/Downloads/Data/X_pic400.save','rb')
f3=open('C:/Users/Asus/Downloads/Data/Y_label400.save','rb')
X_cent=cPickle.load(f1)
X_pic=cPickle.load(f2)
Y_label=cPickle.load(f3)
x=X_cent[:,0::10]
y=X_cent[:,1::10]

x=(x[:,1:]-x[:,:-1])**2
y=(y[:,1:]-y[:,:-1])**2

feature=x+y

#temp1=X_cent[:,:-2]
#temp2=X_cent[:,2:]
#temp3=(temp2-temp1)**
X = np.concatenate((feature,X_cent,X_pic), axis=1)
print X_cent


# In[25]:


Y = Y_label
number_of_class=7  #mention numner of classes
dim_h1=64            #num of neuron in hidden layer1
dim_h2=16            #num of neuron in hidden layer2


# In[39]:

print "Dataset Shape: ", X.shape
print "Number of Samples: ", len(X)
print "Number of Features: ", len(X[0])


# In[40]:

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[47]:

from sklearn.cross_validation import train_test_split

# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.1, random_state = 42, stratify=Y)
covamat= np.cov(X_train)
#X=X_train
#
#a=np.cov(X)
#u,s,v=np.linalg.svd(a)
#su=0
##s=np.asarray(s)
#
#tot= s.sum()
#for i in range(0,a.shape[0]-1):
#    su=su+s[i]
#    if su>0.99*tot :
#       break
#print(i+1)

pca = PCA(n_components=100)
pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)
f=open('C:/Users/Asus/Downloads/pcalik.save','wb')
cPickle.dump(pca,f)
f.close()
print "Training Samples: ", len(X_train)
print "Testing Samples: ", len(X_test)


# In[48]:

# define baseline model------------------------
def baseline_model():
   # create model
    model = Sequential()
    model.add(Dense(dim_h1, input_dim=len(X_train[0]), init='normal', activation='relu'))
      #    model.add(Dropout(0.5))
    model.add(Dense(dim_h2, init='normal', activation='relu'))
     #    model.add(Dropout(0.5))
    model.add(Dense(number_of_class, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[49]:

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)


# In[50]:

#Evaluate The Model with k-Fold Cross Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


# In[51]:

results = cross_val_score(estimator, X_train, Y_train, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[52]:

estimator.fit(X_train, Y_train)
print "Accuracy: {}%\n".format(estimator.score(X_test, Y_test) *100)

predictions = estimator.predict(X_test)
estimator.model.save('model_activity_400_7.h5')
#model_j=estimator.model.to_json()
#with open('model1.json','w') as json_file:
#    json_file.write(model_j)
#estimator.model.save_weights('weight1.h5')
print(predictions)
print(encoder.inverse_transform(predictions))

