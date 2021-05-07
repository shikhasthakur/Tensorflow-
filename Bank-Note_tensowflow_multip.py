#!/usr/bin/env python
# coding: utf-8

import pandas 
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv'
# load the dataset
df = pandas.read_csv(url, header=None)
# summarize shape
print(df.shape)


df.columns=['Variance','Skewness','Curtosis','Entropy','Label']


'''
# 1.3.5.11.
# Measures of Skewness and Kurtosis
# Skewness and Kurtosis	A fundamental task in many statistical analyses is to characterize the location and variability of a data set. A further characterization of the data includes skewness and kurtosis.
# Skewness is a measure of symmetry, or more precisely, the lack of symmetry. A distribution, or data set, is symmetric if it looks the same to the left and right of the center point.
# 
# Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. That is, data sets with high kurtosis tend to have heavy tails, or outliers. Data sets with low kurtosis tend to have light tails, or lack of outliers. A uniform distribution would be the extreme case.
# 
# The histogram is an effective graphical technique for showing both the skewness and kurtosis of data set.
# 
# 
# is a measure of the randomness in the information being processed. The higher the entropy, the harder it is to draw any conclusions from that information. Flipping a coin is an example of an action that provides information that is random. ... This is the essence of entropy.
# 
# variance is a measure of how spread out a data set is. It is calculated as the average squared deviation of each number from the mean of a data set

# In[4]:
'''

print(df.head())


df.info()


df.describe()


df.hist()
plt.show()
plt.rcParams["figure.figsize"]=20,20


X= df[['Variance','Skewness','Curtosis','Entropy']]
y= df[df.columns[4]]


import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


#Encoding the dependent varible
import numpy as np
encod= LabelEncoder()
encod.fit(y)
y=encod.transform(y)
n_labels =len(y)
unique_labels= len(np.unique(y)) #2 unique labels 
#print(unique_labels)

Y =np.zeros((n_labels,unique_labels)) #shape is the dimension of len of labels and unique labels 

Y[np.arange(n_labels),y] = 1
print(Y)

print(X.shape)
print(Y.shape)


X,Y = shuffle(X,Y, random_state =1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20, random_state=415)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


learn_rate = 0.3 #the rate in which the steps will be updated 
train_epoch= 100 #iterations
cost_hist = np.empty(shape=[1],dtype=float)#empty array for later use 
dimens= X.shape[1] #shape[1] here refers column
print(dimens)
classes = 2 #fake notes and real notes


#model_path= '/Users/shikhathakur/Desktop'
'''import tensorflow as tf
#no. of nuerons in hidden layers
nuerons_hid_1= 10
nuerons_hid_2= 10
nuerons_hid_3= 10
nuerons_hid_4= 10

x_ = tf.placeholder(float32,[None,dimens])
#w = tf.Variable(tf.zeros([dimens,classes]))
#b= tf.Variable(tf.zeros([classes]))
'''


conda install tensorflow


#no. of nuerons in hidden layers
nuerons_hid_1= 10
nuerons_hid_2= 10
nuerons_hid_3= 10
nuerons_hid_4= 10
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
x = tf.placeholder(tf.float32,[None,dimens])
w = tf.Variable(tf.zeros([dimens,classes]))
b= tf.Variable(tf.zeros([classes]))
y_ = tf.placeholder(tf.float32,[None,classes]) #real output (to compare later with the predicted output)


#conda install python=3.7

def multilayer_perc(x,weight,bias):
    #hidden layer with activaion function 
    lay_1 = tf.add(tf.matmul(x,weight['w1']),bias['b1'])  # k1 + k2*x
    lay_1 = tf.nn.sigmoid(lay_1)
    
    #lay_2 = tf.add(tf.matmul(lay_1,weight['w2']),bias['b2'])
    #lay_2 = tf.nn.sigmoid(lay_2)
    # if relu is used in layer 2 as activation function, the accuracy goes down so we 
    #have kept sigmoid function in layer 2 for 
   
    
    last_out_layer = tf.matmul(lay_1,weight['out']+bias['out'])
    return last_out_layer
    

weight = {'w1':tf.Variable(tf.truncated_normal([dimens,nuerons_hid_1])),
          #'w2':tf.Variable(tf.truncated_normal([nuerons_hid_1,nuerons_hid_2])),
          'out':tf.Variable(tf.truncated_normal([nuerons_hid_1,classes]))
}

bias={'b1':tf.Variable(tf.truncated_normal([nuerons_hid_1])),
      #'b2':tf.Variable(tf.truncated_normal([nuerons_hid_2])),
      'out':tf.truncated_normal([classes])
    
}


init=tf.global_variables_initializer()#initialising all the variables 
saver = tf.train.Saver()


#model call
y=multilayer_perc(x, weight,bias)


#cost function and model optimizer 
cost_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
training_mode = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost_fun)



session = tf.Session()
session.run(init)



#calculation of mean squared error and accuracy in each iteration or each epoch 
mse_page=[]#msehistroy
accuracy_page=[]#accuracy history


#Use feed_dict to feed values to TensorFlow placeholders so that you don't run
#into the error that says you must feed a value for placeholder tensors

for epoch in range (train_epoch):
    session.run(training_mode, feed_dict={x:X_train, y_ : Y_train})
    cost= session.run(cost_fun, feed_dict={x:X_train, y_:Y_train})
    cost_hist = np.append(cost_hist,cost)
    the_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy= tf.reduce_mean(tf.cast(the_prediction, tf.float32))
    print(accuracy)
    pred_y = session.run(y,feed_dict={x: X_test})
    mean_se = tf.reduce_mean(tf.square(pred_y- Y_test))
    mse =session.run(mean_se)
    mse_page.append(mse)
    accuracy = (session.run(accuracy,feed_dict={x:X_train, y_ : Y_train}))
    accuracy_page.append(accuracy)
    
    print('epoch :', epoch, '--', 'cost:',cost, '--','MSE:',mse, '--','Training Accu:',accuracy)
    
    
save_path = saver.save(session,model_path)
print('model saved in paath : %s' %save_path)


plt.plot(accuracy_page)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


# In[28]:


plt.plot(mse_page)
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()

the_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(the_prediction,tf.float32))
print('Accuracy on Test' , (session.run(accuracy,feed_dict= {x:X_test, y_ :Y_test})))



pred_y = session.run(y, feed_dict={x:X_test})
mse = tf.reduce_mean(tf.square(pred_y - Y_test))
print('MSE : %.4f' %session.run(mse))


'''
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
tf.compat.v1.placeholder('float', None)
x = tf.placeholder("float", None)
y = x * 2
print(y)'''



correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('Test accuracy',session.run(accuracy,feed_dict={x:X_test,y_:Y_test}))
pred_y = session.run(y,feed_dict={x:X_test})
mse = tf.reduce_mean(tf.square(pred_y-Y_test))
print('Mse:',session.run(mse))

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


y1 = df[df.columns[4]]
p=tf.argmax(y,-1)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#y_[1].shape
#if running the below loop will give you the predicted reslts and the actault results '''

#for i in range (700,723):
    #run_predic = session.run(p,feed_dict={x: X[i].reshape(1,4)})
    #run_acc = session.run(accuracy,feed_dict={x:X[i].reshape(1,4),y_:y_[i]})
    #print(rn_predic)



yy=session.run(p,feed_dict={x:X_test})
print(yy)


print(session.run(accuracy, feed_dict= {x:X_test, y_:Y_test}))



#########_____________________________________________END_____________________________________________________________
