import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import ml_project 
import Densities
import SCAN

he = Densities.Atom("HE")


r, h = np.linspace(0.001, 10, 3000, retstep=True)

d0, d1, g0, g1, t0, t1, l0, l1 = he.get_densities(r)


""" making sure that the values of these parameters are not very very less"""
idxs = (d0+d1) > 1e-10
r = r[idxs]
d0 = d0[idxs]
d1 = d1[idxs]
g0 = g0[idxs]
g1 = g1[idxs]
t0 = t0[idxs]
t1 = t1[idxs]
l0 = l0[idxs]
l1 = l1[idxs]

radius=ml_project.Wigner_Seitz_radius(d0+d1)
zeta = ml_project.spin_polarization(d0,d1)
reduced_density = ml_project.reduced_density_gradient(g0+g1, d0+d1)
reduced_laplacian = ml_project.reduced_density_laplacian(l0+l1, d0+d1) 

radius = 1/(1+radius)
reduced_density = 1/(1+reduced_density)
reduced_laplacian = 1/(1+reduced_laplacian**2)

""" obtaining test data and e_x_uniform electron gas from the above import python file"""
X_test1= np.vstack([radius, zeta, reduced_density, reduced_laplacian]).T
uniform = ml_project.uniform(radius)

""" running tensor flow and obtaining F_xc for the trained model using a test set"""
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('/Users/kanun/Desktop/ML_project/python_code/my_model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()

    X1 = graph.get_tensor_by_name('X:0')
    feed_dict = {X1:X_test1}

    predict1 = graph.get_tensor_by_name('predict:0')
    Fxc_out= sess.run(predict1, feed_dict)

Exc_ML = 4*math.pi*h*Fxc_out.squeeze()*uniform*r**2*(d0+d1)
E_xc = np.sum(Exc_ML)

""" To get Exc_SCAN from the import files from above"""

ex = SCAN.getscan_x(SCAN.DEFAULT_X_PARAMS, d0, d1, g0, g1, t0, t1)
ec = SCAN.getscan_c(SCAN.DEFAULT_C_PARAMS, d0, d1, g0, g1, t0, t1, zeta)
e_xc = ex+ec
epp_xc_SCAN = 4*math.pi*h*e_xc*r**2
Exc_SCAN = np.sum(epp_xc_SCAN)

plt.plot(r,epp_xc_SCAN, color = 'red', label = 'Exc_SCAN')
plt.plot(r, Exc_ML, color = 'blue', label = 'Exc_ML')
print(Exc_SCAN, E_xc)
plt.legend()
plt.show()






    
    
    
    
    
    
    
    #out =np.array(sess.run([Y_predicted], feed_dict  = {X:X_test1}))
    #F_xc= out.squeeze()

    
    
