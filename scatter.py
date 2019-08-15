import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

data= np.loadtxt('/Users/kanun/Desktop/ML_project/python_code/relu_tanh/data.txt')
features= data[:,0:9]
Y_output  = data[:,9]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, Y_output, test_size= 0.1, random_state= 97) 

y_train = np.array(y_train)
y_test = np.array(y_test)
y_total = np.concatenate((y_train, y_test))

train_predict= np.loadtxt('/Users/kanun/Desktop/ML_project/data.txt') 
test_predict= np.loadtxt('/Users/kanun/Desktop/ML_project/python_code/tanh/test.txt') 
predict_total = np.concatenate((train_predict, test_predict))

nn_inputs = np.append(train_predict, test_predict)
g= open("/Users/kanun/Desktop/ML_project/NN_inputs.txt","w")
data = nn_inputs
np.savetxt(g, data,fmt='%.8f' )
g.close()



"""
y_mean = np.mean(y_test)
SSE_actual = np.sum(np.square(y_test-y_mean))

SSE_predicted = np.sum(np.square(test_predict-y_mean))

SSE_residual = np.sum(np.square(y_test-test_predict))

r_square = 1-(SSE_residual/SSE_actual)

print(r_square)
"""
y_test = y_test
y_test_pred = test_predict

R2_test = r2_score(y_test, y_test_pred)

y_train = y_train
y_train_pred = train_predict
R2_train= r2_score(y_train, y_train_pred)

y_total = y_total
y_predict = predict_total
R2_total = r2_score(y_total, predict_total)

print(R2_train, R2_test,R2_total) 



fig = plt.figure()
ax1 = fig.add_subplot(111)
size = 0.2

ax1.scatter(y_train,train_predict,size,color = 'red',label="training data")
ax1.scatter(y_test,test_predict,size,color = 'blue', label = "test data")

ax1.plot([0,1.8],[0,1.8],c='k')

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
ax1.set_ylabel("$F_{xc}$ machine learned", fontsize = 16)
ax1.set_xlabel("$F_{xc}$ reference", fontsize = 16)
#ax1.set_title("actual vs ML predicted")
#ax2.plot(range(epochs), error_train, color = 'red',label = 'train error')
#ax2.plot(range(epochs), error_test, color = 'blue', label = 'validation error' )
#ax2.set_xlabel("epochs")
#ax2.set_ylabel("MSE")
plt.axis('square')
ax1.tick_params(direction= 'in')
ax1.set_xlim(0,1.8)
ax1.set_ylim(0,1.8)
plt.text(0.8,0.3, "$R^2 train$ = 0.9923",color = 'black', fontsize = 14)
plt.text(0.8,0.2, "$R^2 test$ = 0.9961",color = 'black', fontsize = 14)
plt.text(0.8,0.1, "$R^2 total$ = 0.9927",color = 'black', fontsize = 14)
plt.tight_layout()
plt.legend()
plt.show()


print(r_square)


