
import numpy as np
import math
import matplotlib.pyplot as plt
import ml_project 
import Densities
import SCAN

data = []
atoms = ['HE','LI','BE', 'B', 'C','O','N','F','NA','P','AR','K','CR','CU','CU+','AS','KR','AG','CL','MG']
test_set = ['NE', 'SI', 'XE']
for i in atoms:
    atm= Densities.Atom(i)

    r1, h = np.linspace(0.001, 1.0, 3000, retstep = True)
    r2, h = np.linspace(1, 4.0, 2000, retstep = True)
    r3, h = np.linspace(4.0, 10.0, 200, retstep = True)
    r = np.hstack((r1,r2,r3))
    
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

    
    """inputs for the NN"""
    radius=ml_project.Wigner_Seitz_radius(d0+d1)
    zeta = ml_project.spin_polarization(d0,d1)
    reduced_density = ml_project.reduced_density_gradient(g0+g1, d0+d1)      
    reduced_laplacian = ml_project.reduced_density_laplacian(l0+l1, d0+d1) 
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
    

    

    """ 
    this is to generate exchange and correlation part and add them to get exc from SCAN
    """
    ex = SCAN.getscan_x(SCAN.DEFAULT_X_PARAMS, d0, d1, g0, g1, t0, t1)
    ec = SCAN.getscan_c(SCAN.DEFAULT_C_PARAMS, d0, d1, g0, g1, t0, t1, zeta)
    

    e_xc = ex+ec
    e_xc = e_xc/(d0+d1)
   
    """ to find e_x_uniform electron gas"""
    numerator = -3*(9*math.pi/4)**(1/3)
    denominator = 4*math.pi*radius
    e_x_uni = numerator/denominator
    f_xc = e_xc/e_x_uni


    """preprocessing the data"""
    radius = 1/(1+radius)
    reduced_density = 1/(1+reduced_density)
    reduced_laplacian = np.tanh(reduced_laplacian)
    dx_zeta = 0.5*((1+zeta)**(4/3)+(1-zeta)**(4/3))
    

    

    #print(reduced_laplacian)
    #print(radius.shape, zeta.shape, reduced_density.shape, reduced_laplacian.shape, Ec_0.shape, Ec_1.shape,gx.shape,g_constant.shape,h1.shape)

    #value = np.vstack([radius, ds_zeta, reduced_density, reduced_laplacian, Ec_0, Ec_1, gx, g_constant, h1, f_xc])
    value = np.vstack([radius, reduced_density, reduced_laplacian,dx_zeta, Ec_0, Ec_1, gx, g_constant, h1,f_xc])
    value1 = value.T
    data.append(value1)

full_set = np.vstack(data)
f= open("/Users/kanun/Desktop/ML_project/train_data.txt","w")
data = full_set
np.savetxt(f, data,fmt='%.8f' )
f.close()


