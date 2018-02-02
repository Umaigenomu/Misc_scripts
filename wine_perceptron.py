import numpy as np
import pandas as pd

red = pd.read_csv("winequality/winequality-red.csv")
white = pd.read_csv("winequality/winequality-white.csv", sep =';')
red['type'] = 1
white['type'] = 0
wines=red.append(white,ignore_index=True) #one below the other

from sklearn.model_selection import train_test_split

#______________________________________________________Predicting Type_____________________________________
X=wines.ix[:,:11] #selecting every col except quality and type
y=np.ravel(wines.type) #flattening type col into ndarray
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#-------------- Standardizing ----------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#---------------------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12,activation='relu',input_shape=(11,)))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model2 = Sequential() #better overall scores, possibly more overfitting
model2.add(Dense(8,activation='relu',input_shape=(11,)))
model2.add(Dense(6,activation='relu'))
model2.add(Dense(6,activation='relu'))
model2.add(Dense(1,activation='sigmoid')) #has to be sigmoid in this case

#model.output_shape
#model.summary()
#model.get_config()
#model.get_weights()

#Like the logistic sigmoid, the tanh function is also sigmoidal (“s”-shaped), but instead outputs values that range (-1, 1). 
#Thus strongly negative inputs to the tanh will map to negative outputs. 
#Additionally, only zero-valued inputs are mapped to near-zero outputs.
#These properties make the network less likely to get “stuck” during training

#1 epoch = 1: pass over the entire dataset once 
# for every batch processed the model's parameters get recalculated (in this case every instance of the training set)
# verbose: logging; 1 = progress bar, 2 = one line per epoch
# monitoring the accuracy during the training was made possible by passing ['accuracy'] to the metrics argument.
#cross entropy is commonly used in binary classification. For Multi class, multi class cross entropy
model.compile(loss='binary_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size= 1, verbose = 1) 
'''
Some of the most popular optimization algorithms used are the Stochastic Gradient Descent (SGD), ADAM and RMSprop. 
Depending on whichever algorithm you choose, you’ll need to tune certain parameters, such as learning rate or momentum. 
The choice for a loss function depends on the task that you have at hand: for example, 
for a regression problem, you’ll usually use the Mean Squared Error (MSE). 
As you see in this example, you used binary_crossentropy for the binary classification problem of determining 
whether a wine is red or white. 
Lastly, with multi-class classification, you’ll make use of categorical_crossentropy.
'''

y_pred = model.predict(X_test)
class_result=y_pred.round()
score = model.evaluate(X_test, y_test, verbose=1)

#Precision is a measure of a classifier’s exactness. The higher the precision, the more accurate the classifier.
#Recall is a measure of a classifier’s completeness. The higher the recall, the more cases the classifier covers.
#The F1 Score or F-score is a weighted average of precision and recall.
#The Kappa or Cohen’s kappa is the classification accuracy normalized by the imbalance of the classes in the data.
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
confusion_matrix(y_test,class_result)
precision_score(y_test,class_result)
recall_score(y_test,class_result)
f1_score(y_test,class_result)
cohen_kappa_score(y_test,class_result)

#__________________________________Predicting Quality______________________________________________________________

y= wines.quality
X= wines.drop('quality',axis=1)
#-------------- Scaling----------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
#-----------------------K-fold validation----------------------------------------------------------------------------=



#white.info()
#red.isnull().sum()
#red.describe()

''' quality x sulphates for both types of wine
import matplotlib.pyplot as plt
fig, ax= plt.subplots(1,2)
ax[0].hist(red.quality,10,facecolor='red',alpha=0.5,label="Red Wine")
ax[1].hist(white.quality,10,facecolor='blue',alpha=0.5,label="White Wine")

ax[0].scatter(red.quality,red.sulphates,color='red')
ax[1].scatter(white.quality,white.sulphates,color='white',edgecolors='black')
ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Quality")
ax[1].set_xlabel("Quality")
ax[0].set_ylabel("Sulphates")
ax[1].set_ylabel("Sulphates")
ax[0].set_xlim([0,10])
ax[1].set_xlim([0,10])
ax[0].set_ylim([0,2.5])
ax[1].set_ylim([0,2.5])
plt.show()
'''
''' alcohol x volatile acidity for both types of wine
np.random.seed(570)
redlabels = np.unique(red.quality)
whitelabels = np.unique(white.quality)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
redcolors = np.random.rand(6,4)
whitecolors = np.append(redcolors, np.random.rand(1,4), axis=0)

for i in range(len(redcolors)):
    redy = red['alcohol'][red.quality == redlabels[i]]
    redx = red['volatile acidity'][red.quality == redlabels[i]]
    ax[0].scatter(redx, redy, c=redcolors[i])

for i in range(len(whitecolors)):
    whitey = white['alcohol'][white.quality == whitelabels[i]]
    whitex = white['volatile acidity'][white.quality == whitelabels[i]]
    ax[1].scatter(whitex, whitey, c=whitecolors[i])
ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))    
ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Volatile Acidity")
ax[0].set_ylabel("Alcohol")
ax[1].set_xlabel("Volatile Acidity")
ax[1].set_ylabel("Alcohol") 
fig.subplots_adjust(top=0.85, wspace=0.7)
plt.show()
'''

''' confusion matrix
import seaborn as sns 
corr = wines.corr()
sns.heatmap(corr)
plt.show()
'''









