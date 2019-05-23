import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
#from NN import func

data= np.loadtxt('/Users/kanun/Desktop/ML_project/out.txt') # to load text saved usig num

data1= np.loadtxt('/Users/kanun/Desktop/ML_project/he.txt')

"""splitting the data into train, validation and test"""
features= data[:,0:4]
Y_output  = data[:,4]

X_test1 = data1[:,0:4]
y_test1 = data1[:,4]



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
weights = {
    'w_hidden': tf.Variable(tf.glorot_uniform_initializer()((n_features, hidden_layer1))),
    'w_out' : tf.Variable(tf.glorot_uniform_initializer()((hidden_layer1, n_output))) #hidden-> output 
}

biases = {
    'b_hidden' : tf.Variable(tf.zeros([hidden_layer1])),
    'b_out': tf.Variable(tf.zeros([n_output])) #1->output
}    

""" first layer activation function operation"""
first_layer_input  = tf.add(tf.matmul(X, weights['w_hidden']), biases['b_hidden']) #multiplying features with weight and adding bias in first layer
first_layer_output = tf.nn.sigmoid(first_layer_input) # the obtained value is then sigmoid


""" final layer operation"""
final_layer_input = tf.add(tf.matmul(first_layer_output, weights['w_out']), biases['b_out'], name = 'output')
final_output = final_layer_input


"""Perdiction made by the network"""
Y_predicted = tf.add(final_output, 0.0, name = 'predict')

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
  

    #size = 0.4        
    #p = np.array(sess.run([Y_predicted], feed_dict = {X:X_train, Y:y_train}))
    
    w1 = np.array(sess.run([weights['w_hidden']]))
    w2 = np.array(sess.run([weights['w_out']]))
    b1 = np.array(sess.run([biases['b_hidden']]))
    b2 = np.array(sess.run([biases['b_out']]))
    

    
    model = saver.save(sess, 'my_model')

    #out =np.array(sess.run([Y_predicted], feed_dict  = {X:X_test1}))
    #F_xc= out.squeeze()

    #f= open("/Users/kanun/Desktop/ML_project/F_xc.txt","w")
    #np.savetxt(f, F_xc,fmt='%.8f' )
    #f.close()
    
    


    






        
        
        
        
        
        
            

                      

        
