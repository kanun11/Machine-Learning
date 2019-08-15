import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import ml_project 
import Densities
import SCAN
import SCANL

atom = Densities.Atom("xe")

r, h = np.linspace(0.001, 10, 2000, retstep=True)

d0, d1, g0, g1, t0, t1, l0, l1 = atom.get_densities(r)


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
uniform = ml_project.uniform(radius)
Ec_0 = SCAN.corgga_0(SCAN.DEFAULT_C_PARAMS, radius, reduced_density, zeta)
Ec_1 = SCAN.corgga_1(SCAN.DEFAULT_C_PARAMS, radius,reduced_density , zeta)
gx =   SCAN.get_gx(SCAN.DEFAULT_X_PARAMS,reduced_density**2)
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


""" obtaining test data and e_x_uniform electron gas from the above import python file"""
X_test_processed= np.vstack([radius, reduced_density, reduced_laplacian,dx_zeta, Ec_0, Ec_1, gx, g_constant, h1]).T


""" running tensor flow and obtaining F_xc for the trained model using a test set"""
""" loading a saved model too here"""
fig = plt.figure()

ax1 = fig.add_subplot(111)

with tf.Session() as sess:
    new_saver1 = tf.train.import_meta_graph('/Users/kanun/Desktop/ML_project/python_code/new/new.meta')
    graph1=new_saver1.restore(sess, tf.train.latest_checkpoint('./new'))

    graph1 = tf.get_default_graph()

    X1 = graph1.get_tensor_by_name('X:0')
    feed_dict = {X1:X_test_processed}

    predict1 = graph1.get_tensor_by_name('predict:0')
    Fxc_out1= sess.run(predict1, feed_dict)
"""
f= open("/Users/kanun/Desktop/Fxc.txt","w")
data = Fxc_out1
np.savetxt(f, data,fmt='%.8f' )
f.close()
"""
Exc_ML1 = 4*math.pi*h*Fxc_out1.squeeze()*uniform*r**2*(d0+d1)
E_xc1 = np.sum(Exc_ML1)
"""
with tf.Session() as sess:
    new_saver2 = tf.train.import_meta_graph('/Users/kanun/Desktop/ML_project/python_code/relu_tanh/rt.meta')
    graph2=new_saver2.restore(sess, tf.train.latest_checkpoint('./relu_tanh'))

    graph2 = tf.get_default_graph()

    X1 = graph2.get_tensor_by_name('X:0')
    feed_dict = {X1:X_test_processed}

    predict2 = graph2.get_tensor_by_name('predict:0')
    Fxc_out2= sess.run(predict2, feed_dict)
Exc_ML2 = 4*math.pi*h*Fxc_out2.squeeze()*uniform*r**2*(d0+d1)
E_xc2 = np.sum(Exc_ML2)
"""
""" SCAN Fxc"""
ex = SCAN.getscan_x(SCAN.DEFAULT_X_PARAMS, d0, d1, g0, g1, t0, t1)
ec = SCAN.getscan_c(SCAN.DEFAULT_C_PARAMS, d0, d1, g0, g1, t0, t1, zeta)
e_xc = ex+ec
Fxc_SCAN = e_xc/(uniform*(d0+d1))

epp_xc_SCAN = 4*math.pi*h*e_xc*r**2
Exc_SCAN = np.sum(epp_xc_SCAN)

""" spin 0"""

radius_0=ml_project.Wigner_Seitz_radius(d0)
#zeta = ml_project.spin_polarization(d0)
reduced_density_0 = ml_project.reduced_density_gradient(g0, d0)
reduced_laplacian_0 = ml_project.reduced_density_laplacian(l0, d0) 
 
Ft0 = 1

def Ft2(s, q):
    Ft2 = ((5/27)*(s**2))+((20/9)*q)
    return Ft2
Ft2_spin0 = Ft2(reduced_density_0,reduced_laplacian_0)
   

def Ft4(s,q):
    Ft4 = ((8/81)*(q**2))-((1/9)*(s**2)*q)+((8/243)*(s**4))
    return Ft4
Ft4_spin0 = Ft4(reduced_density_0,reduced_laplacian_0)   

def Ftw(s):
    Ftw = (5/3)*(s**2)
    return Ftw
Ftw_spin0 = Ftw(reduced_density_0)    

def FtMGE4(ft0,ft2,ft4,ftw):
    numerator = ft0+ft2+ft4
    den1 = ft4/(1+ftw)
    den2 = den1**2
    den3 = 1+den2
    den4 = np.sqrt(den3)
    FtMGE4 = numerator/den4
    return FtMGE4
FtMGE4_spin0 = FtMGE4(Ft0, Ft2_spin0, Ft4_spin0, Ftw_spin0)    

def z_pc(ftmge4,ftw):
    z_pc = ftmge4-ftw
    return z_pc
z_pc_spin0 = z_pc(FtMGE4_spin0,Ftw_spin0)
z_pc_spin0 = np.around(z_pc_spin0,2)

def theta_pc(z_pc):
    a = 1.784720
    b = 0.258304
    theta_pc = np.zeros(z_pc.shape)
    idxs = (z_pc > 0)*(z_pc < a)
    theta_pc[idxs] = ((1+ np.exp(a/(a-z_pc[idxs])))/(np.exp(a/z_pc[idxs])+np.exp(a/(a-z_pc[idxs]))))**b
    theta_pc[z_pc >= a] = 1.0
    return theta_pc
theta_pc_spin0 = theta_pc(z_pc_spin0)

Ft_PC_spin0 = Ftw_spin0+z_pc_spin0*theta_pc_spin0
tau_UEG_spin0 = (3*(3*math.pi**2)**(2/3)*(d0)**(5/3))/10 
tau_d_spin0 = Ft_PC_spin0*tau_UEG_spin0

"""for spin 1"""

radius1=ml_project.Wigner_Seitz_radius(d1)
#zeta = ml_project.spin_polarization(d1)
reduced_density_1 = ml_project.reduced_density_gradient(g1, d1)
reduced_laplacian_1 = ml_project.reduced_density_laplacian(l1, d1) 
 
 
Ft0 = 1

def Ft2(s, q):
    Ft2 = ((5/27)*(s**2))+((20/9)*q)
    return Ft2
Ft2_spin1 = Ft2(reduced_density_1,reduced_laplacian_1)
   

def Ft4(s,q):
    Ft4 = ((8/81)*(q**2))-((1/9)*(s**2)*q)+((8/243)*(s**4))
    return Ft4
Ft4_spin1 = Ft4(reduced_density_1,reduced_laplacian_1)   

def Ftw(s):
    Ftw = (5/3)*(s**2)
    return Ftw
Ftw_spin1 = Ftw(reduced_density_1)    

def FtMGE4(ft0,ft2,ft4,ftw):
    numerator = ft0+ft2+ft4
    den1 = ft4/(1+ftw)
    den2 = den1**2
    den3 = 1+den2
    den4 = np.sqrt(den3)
    FtMGE4 = numerator/den4
    return FtMGE4
FtMGE4_spin1 = FtMGE4(Ft0, Ft2_spin1, Ft4_spin1, Ftw_spin1)    

def z_pc(ftmge4,ftw):
    z_pc = ftmge4-ftw
    return z_pc
z_pc_spin1 = z_pc(FtMGE4_spin1,Ftw_spin1)
z_pc_spin1 = np.around(z_pc_spin1,2)
def theta_pc(z_pc):
    a = 1.784720
    b = 0.258304
    theta_pc = np.zeros(z_pc_spin1.shape)
    idxs = (z_pc > 0)*(z_pc < a)
    theta_pc[idxs] = ((1+ np.exp(a/(a-z_pc[idxs])))/(np.exp(a/z_pc[idxs])+np.exp(a/(a-z_pc[idxs]))))**b
    theta_pc[z_pc >= a] = 1.0
    return theta_pc
theta_pc_spin1 = theta_pc(z_pc_spin1)

Ft_PC_spin1 = Ftw_spin1+z_pc_spin1*theta_pc_spin1
tau_UEG_spin1 = (3*(3*math.pi**2)**(2/3)*(d1)**(5/3))/10 
tau_d_spin1 = Ft_PC_spin1*tau_UEG_spin1
 
""" total density"""


radius=ml_project.Wigner_Seitz_radius(d0+d1)
zeta = ml_project.spin_polarization(d0,d1)
reduced_density = ml_project.reduced_density_gradient(g0+g1, d0+d1)
reduced_laplacian = ml_project.reduced_density_laplacian(l0+l1, d0+d1) 

Ft0 = 1

def Ft2(s, q):
    Ft2 = ((5/27)*(s**2))+((20/9)*q)
    return Ft2
Ft2 = Ft2(reduced_density,reduced_laplacian)
   

def Ft4(s,q):
    Ft4 = ((8/81)*(q**2))-((1/9)*(s**2)*q)+((8/243)*(s**4))
    return Ft4
Ft4 = Ft4(reduced_density,reduced_laplacian)   

def Ftw(s):
    Ftw = (5/3)*(s**2)
    return Ftw
Ftw = Ftw(reduced_density)    

def FtMGE4(ft0,ft2,ft4,ftw):
    numerator = ft0+ft4+ftw
    den1 = ft4/(1+ftw)
    den2 = den1**2
    den3 = 1+den2
    den4 = np.sqrt(den3)
    FtMGE4 = numerator/den4
    return FtMGE4
FtMGE4 = FtMGE4(Ft0, Ft2, Ft4, Ftw)    

def z_pc(ftmge4,ftw):
    z_pc = ftmge4-ftw
    return z_pc
z_pc = z_pc(FtMGE4,Ftw)
z_pc = np.around(z_pc,2)
def theta_pc(z_pc):
    a = 1.784720
    b = 0.258304
    theta_pc = np.zeros(z_pc.shape)
    idxs = (z_pc > 0)*(z_pc < a)
    theta_pc[idxs] = ((1+ np.exp(a/(a-z_pc[idxs])))/(np.exp(a/z_pc[idxs])+np.exp(a/(a-z_pc[idxs]))))**b
    theta_pc[z_pc >= a] = 1.0
    return theta_pc
theta_pc = theta_pc(z_pc)


Ft_PC = Ftw+z_pc*theta_pc
ds_zeta = (np.power(1.0 + zeta, 5.0/3.0) + np.power(1.0 - zeta, 5.0/3.0))/2.0
tau_UEG = (3*(3*math.pi**2)**(2/3)*(d0+d1)**(5/3))/10
tau_UEG = tau_UEG*ds_zeta
tau_d_total = Ft_PC*tau_UEG
 


ex = SCANL.getscan_x(SCANL.DEFAULT_X_PARAMS, d0, d1, g0, g1, tau_d_spin0, tau_d_spin1)
ec = SCANL.getscan_c(SCANL.DEFAULT_C_PARAMS, d0+d1, g0+g1, tau_d_total, zeta)
e_xc = ex+ec
Fxc_scanl = e_xc/(uniform*(d0+d1))

epp_xc_scanl = 4*math.pi*h*e_xc*r**2
Exc_scan_l = np.sum(epp_xc_scanl)


size = 0.4
ax1.plot(r,Fxc_SCAN, color = 'black', label = '$F_{xc}[SCAN]$')
ax1.plot(r, Fxc_out1.squeeze(),'--' ,color = 'red', label = '$F_{xc}[ML]$')
ax1.plot(r, Fxc_scanl,'--' ,color = 'blue', label = '$F_{xc}[SCANL]$')
#ax1.plot(r, Fxc_scanl,':' ,color = 'blue', label = '$F_{xc}[SCAN-L]$')
ax1.set_ylim([0,1.8])
ax1.set_xlim(0,9)
ax1.legend()
ax1.set_ylabel('$F_{xc}$',fontsize=12)
ax1.set_xlabel('r ($a_{0}$)',fontsize=12)
ax1.tick_params(direction= 'in')
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.legend(loc = 0, fontsize = 'small')
plt.text(0.8,0.7, "Silicon:",color = 'black', fontsize = 12)
plt.text(0.5, 0.6, "E${}_{xc}$[ML]:    "+"{:.6f} $E_h$" .format(E_xc1), color = 'black',fontsize = 12)
plt.text(0.5, 0.5, "E${}_{xc}$[SCAN]: "+"{:.6f} $E_h$ " .format(Exc_SCAN),color = 'black', fontsize = 12)
#plt.text(0.5, 0.3, "E${}_{xc}$[ML-relu_tanh]:    "+"{:.6f} $E_h$" .format(E_xc2), color = 'black',fontsize = 12)
plt.text(0.5, 0.3, "E${}_{xc}$[SCAN-L]: "+"{:.6f} $E_h$ " .format(Exc_scan_l),color = 'black', fontsize = 12)
plt.tight_layout()
plt.show()










    
    
    
    
    
    
    
    

    
    
