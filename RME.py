import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import ml_project 
import Densities
import SCAN

data_SCAN = []
data_ML = []
data_diff = []
atoms = ['LI','B', 'C', 'N','O','F','HE','NA','P','AR','K','CR','CU','CU+','AS','KR','XE','BE','CL','AG']
for i in atoms:
    atm= Densities.Atom(i)

    r, h = np.linspace(0.001, 10, 2000, retstep = True)

    d0, d1, g0, g1, t0, t1, l0, l1 = atm.get_densities(r)
    
    """ making sure that the values of these parameters are not very very less"""
    idxs = (d0+d1) > 1e-10
    r  = r[idxs]
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
    uniform = ml_project.uniform(radius)
    Ec_0 = SCAN.corgga_0(SCAN.DEFAULT_C_PARAMS, radius, reduced_density, zeta)
    Ec_1 = SCAN.corgga_1(SCAN.DEFAULT_C_PARAMS, radius,reduced_density , zeta)
    gx =   SCAN.get_gx(SCAN.DEFAULT_X_PARAMS,reduced_density**2 )
    g_constant = np.full_like(gx, 1.174)

    def h1(s):
        d1 = ((10*s**2)/81)/0.064
        d2 = 1+d1
        v1 = 0.064/d2
        v2 = 1.064-v1
        return v2
    h1 = h1(reduced_density)
        

    radius = 1/(1+radius)
    reduced_density = 1/(1+reduced_density)
    reduced_laplacian = np.tanh(reduced_laplacian)
    dx_zeta = 0.5*((1+zeta)**(4/3)+(1-zeta)**(4/3))

    
    value = np.vstack([radius, reduced_density, reduced_laplacian,dx_zeta, Ec_0, Ec_1, gx, g_constant, h1])
    inputs = value.T

    ex = SCAN.getscan_x(SCAN.DEFAULT_X_PARAMS, d0, d1, g0, g1, t0, t1)
    ec = SCAN.getscan_c(SCAN.DEFAULT_C_PARAMS, d0, d1, g0, g1, t0, t1, zeta)
    e_xc = ex+ec
    Fxc_SCAN = e_xc/(uniform*(d0+d1))

    epp_xc_SCAN = 4*math.pi*h*e_xc*r**2
    Exc_SCAN = np.sum(epp_xc_SCAN)
    data_SCAN.append(Exc_SCAN)

    with tf.Session() as sess:
        new_saver1 = tf.train.import_meta_graph('/Users/kanun/Desktop/ML_project/python_code/constraint/cons.meta')
        graph1=new_saver1.restore(sess, tf.train.latest_checkpoint('./constraint'))

        graph1 = tf.get_default_graph()

        X1 = graph1.get_tensor_by_name('X:0')
        feed_dict = {X1:inputs}

        predict1 = graph1.get_tensor_by_name('predict:0')
        Fxc_out= sess.run(predict1, feed_dict)

    Exc_ML = 4*math.pi*h*Fxc_out.squeeze()*uniform*r**2*(d0+d1)
    E_xc  = np.sum(Exc_ML)
    
    data_ML.append(E_xc)
   
    # data_diff.append(np.sum(4*math.pi*r**2*h*(Exc_ML - Fxc_SCAN))/abs(Exc_SCAN))
    data_diff.append(100*(abs(E_xc - Exc_SCAN))/abs(Exc_SCAN))
    print( i, data_diff[-1])


E_ml = np.array(data_ML)
E_scan = np.array(data_SCAN)

print("MAE: ", np.mean(data_diff))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(r,((Exc_ML/h)-(epp_xc_SCAN/h)))
ax2 = ax.twinx()
ax2.plot(r, 4*math.pi*r**2*(d0+d1),c='g')
plt.show()


  


