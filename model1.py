import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
#from NN import func

data= np.loadtxt('/Users/kanun/Desktop/ML_project/out.txt') # to load text saved usig num


"""splitting the data into train, validation and test"""
features= data[:,0:4]
Y_output  = data[:,4]




"""split train, test data"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, Y_output, test_size= 0.2, random_state= 44) 

"""Hyperparameter"""
n_features = 4
n_output = 1
hidden_layer1 = 10
epochs = 30

X = tf.placeholder( dtype = tf.float32, shape = [None, n_features], name = "X" )
Y = tf.placeholder( dtype = tf.float32, name = "Y" )


"""defining weights and bias for the input layer"""
w_hidden= tf.Variable(tf.glorot_uniform_initializer()((n_features, hidden_layer1)), name = 'w_hidden')
w_out = tf.Variable(tf.glorot_uniform_initializer()((hidden_layer1, n_output)), name = 'w_out') #hidden-> output 

b_hidden= tf.Variable(tf.zeros([hidden_layer1]), name = 'b_hidden')
b_out = tf.Variable(tf.zeros([n_output]), name = 'b_out')   

""" first layer activation function operation"""
first_layer_input  = tf.add(tf.matmul(X, w_hidden), b_hidden) #multiplying features with weight and adding bias in first layer
first_layer_output = tf.nn.sigmoid(first_layer_input) # the obtained value is then sigmoid


""" final layer operation"""
final_layer_input = tf.add(tf.matmul(first_layer_output, w_out), b_out, name = 'predict')
final_output = final_layer_input


"""Perdiction made by the network"""
Y_predicted = final_output

""" Defining the error function"""
cost = tf.reduce_mean(tf.square(Y-Y_predicted))

""" Defining the optimizer"""

optimizer = tf.train.AdamOptimizer(learning_rate = 0.005, beta1= 0.8, beta2=0.999, epsilon = 1e-3)
training = optimizer.minimize(cost)


""" Initialize the variables"""
init = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(init)

    for i in tqdm(np.arange(epochs)):
        for j in range(len(X_train)):

            _,predict = sess.run([training,Y_predicted], feed_dict = {X:X_train[j:(j+1)], Y:y_train[j:(j+1)] })
    


    p=np.array(sess.run([Y_predicted], feed_dict  = {X:X_train, Y:y_train}))
    w_hidden = np.array(sess.run(w_hidden))
    w_out = np.array(sess.run(w_out))
    b_hidden = np.array(sess.run(b_hidden))
    b_out = np.array(sess.run(b_out))

    model = saver.save(sess, '/Users/kanun/Desktop/ML_project/python_code/my_model')
#plt.scatter(y_train, p.squeeze())
#plt.show()

    


    






        
        
        
        
        
        
            

                      

        
