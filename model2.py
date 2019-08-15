import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
#from NN import func

data= np.loadtxt('/Users/kanun/Desktop/ML_project/train_data.txt') # to load text saved usig num
test_data= np.loadtxt('/Users/kanun/Desktop/ML_project/test_data.txt')

"""splitting the data into train, validation and test"""

features= data[:,0:9]
Y_output  = data[:,9]

X_test= test_data[:,0:9]
y_test  = test_data[:,9]



"""split train, test data"""
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(features, Y_output, test_size= 0.1, random_state= 23) 

"""Hyperparameter"""
n_features = 9
n_output = 1
hidden_layer1 = 150
hidden_layer2 = 50
epochs = 1


X = tf.placeholder( dtype = tf.float32, shape = [None, n_features], name = "X" )
Y = tf.placeholder( dtype = tf.float32, name = "Y" )
keep_prob = tf.placeholder(dtype = tf.float32)


"""defining weights and bias for the input layer"""
w_hidden1= tf.Variable(tf.glorot_uniform_initializer()((n_features, hidden_layer1)), name = 'w_hidden1')
w_hidden2 = tf.Variable(tf.glorot_uniform_initializer()((hidden_layer1, hidden_layer2)), name = 'w_hidden2') #hidden-> output 
w_out = tf.Variable(tf.glorot_uniform_initializer()((hidden_layer2, n_output)), name = 'w_out1')

b_hidden1= tf.Variable(tf.zeros([hidden_layer1]), name = 'b_hidden1')
b_hidden2= tf.Variable(tf.zeros([hidden_layer2]), name = 'b_hidden2')
b_out = tf.Variable(tf.zeros([n_output]), name = 'b_out')   

""" first layer activation function operation"""
first_layer_input  = tf.add(tf.matmul(X, w_hidden1), b_hidden1) #multiplying features with weight and adding bias in first layer
first_layer_output = tf.nn.tanh(first_layer_input) # the obtained value is then sigmoid

second_layer_input  = tf.add(tf.matmul(first_layer_output, w_hidden2), b_hidden2) #multiplying features with weight and adding bias in first layer
second_layer_output = tf.nn.tanh(second_layer_input)

""" final layer operation"""
final_layer_input = tf.add(tf.matmul(second_layer_output, w_out), b_out, name = 'predict')
final_output = final_layer_input


"""Perdiction made by the network"""
Y_predicted = final_output

""" Defining the error function"""
cost = tf.reduce_mean(tf.square(Y-Y_predicted))

""" Defining the optimizer"""

optimizer = tf.train.AdamOptimizer(learning_rate = 0.003, beta1= 0.8, beta2=0.99, epsilon = 1e-3)
training = optimizer.minimize(cost)


""" Initialize the variables"""
init = tf.global_variables_initializer()
saver = tf.train.Saver()

value = []
error= []
error_test = []
error_val = []
with tf.Session() as sess:
    sess.run(init)

    for i in tqdm(range(epochs)):
        error_batch=[]
        for j in range(len(X_train)):

            _,predict,c = sess.run([training,Y_predicted,cost], feed_dict = {X:X_train[j:(j+1)], Y:y_train[j:(j+1)]})
            error_batch.append(c) #I am giving single value at a time so error is in a batch of length of X_train
            
            if i == epochs-1: # if I dont do this step then depending upon value of X, the prediced value will be added. doing this only last value of prediction will be appended 
                value.append(predict)
        
        
        error.append(np.mean(error_batch)) # I am taking mean of those batches to have a single value
        predict_val,c_val = sess.run([Y_predicted,cost], feed_dict = {X:X_validation, Y:y_validation}) #this is to predict and calculate error for validation set. there is no training calculated to not optimse the error again after having trained them
        error_val.append(c_val)

        predict_test,c1 = sess.run([Y_predicted,cost], feed_dict = {X:X_test, Y:y_test}) #this is to predict and calculate error for validation set. there is no training calculated to not optimse the error again after having trained them
        error_test.append(c1)  # testing error
    

    model2 = saver.save(sess, '/Users/kanun/Desktop/ML_project/python_code/new/new')



predicted_value = np.concatenate(value) #concatenate to make the result in a single arrray. np.array will result in a batch of X

train_predict = predicted_value
test_predict = predict_test
validation_predict = predict_val
"""
f= open("/Users/kanun/Desktop/ML_project/train.txt","w")
data = train_predict
np.savetxt(f, data,fmt='%.8f' )
f.close()

test_predict = predict_test
g= open("/Users/kanun/Desktop/ML_project/test.txt","w")
data = test_predict
np.savetxt(g, data,fmt='%.8f' )
g.close()
"""

error_train = np.array(error)
error_test = np.array(error_test)

size = 0.2





fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = fig.add_subplot(212)

ax1.scatter(y_train,predicted_value,size,color = 'grey',label="training data")
ax1.scatter(y_validation,predict_val,size,color = 'red', label = "validation data")
ax1.scatter(y_test,predict_test,size,color = 'blue', label = "test data")

ax1.plot([0,1.8],[0,1.8],c='k')

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
ax1.set_ylabel("$F_{xc}$ predicted", fontsize = 16)
ax1.set_xlabel("$F_{xc}$ actual", fontsize = 16)
#ax1.set_title("actual vs ML predicted")
#ax2.plot(range(epochs), error_train, color = 'red',label = 'train error')
#ax2.plot(range(epochs), error_test, color = 'blue', label = 'validation error' )
#ax2.set_xlabel("epochs")
#ax2.set_ylabel("MSE")
plt.axis('square')
ax1.tick_params(direction= 'in')
ax1.set_xlim(0,1.8)
ax1.set_ylim(0,1.8)
plt.tight_layout()
plt.legend()
plt.show()

    #model1 = saver.save(sess, '/Users/kanun/Desktop/ML_project/python_code/raw_data/raw_model')
    #model2 = saver.save(sess, '/Users/kanun/Desktop/ML_project/python_code/processed_data/processed_model')


    


    






        
        
        
        
        
            

                      

        
