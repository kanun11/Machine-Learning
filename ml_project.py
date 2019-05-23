import math
import numpy as np
import Densities 
import matplotlib.pyplot as plt
import SCAN



def Wigner_Seitz_radius(n): # n being the density
    numerator = 3
    denominator = (4*math.pi*n)
    r = (numerator/denominator)**(1/3)  
    return r
    
    
                
def spin_polarization(n1,n2): # n1 being up density and n2 being down
    numerator = n1-n2
    denominator = n1+n2
    zeta = numerator/denominator
    return zeta
    
def reduced_density_gradient(gradn, n): 
    numerator = abs(gradn)
    denominator = 2*(3*(math.pi)**2)**(1/3)*(n)**(4/3)
    S = numerator/denominator
    return S 

def reduced_density_laplacian(lapn, n):
    k2 = 4*(3*(math.pi)**2)**(2/3)
    numerator = lapn
    denominator = k2*(n)**(5/3)
    q= numerator/denominator
    return q

def alpha(n, gradn, tau):
    numerator2 = ((abs(gradn))**2)/8*n
    numerator1 = tau
    denominator =(3*(3*math.pi**2)**(2/3)*n**(5/3))/10 
    alpha = (numerator1-numerator2)/denominator
    return alpha  

def uniform(radius):
    numerator = -3*(9*math.pi/4)**(1/3)
    denominator = 4*math.pi*radius
    e_x_uni = numerator/denominator
    return e_x_uni

if __name__ == "__main__":
    """ 
    this section is to generate density, laplacian, gradient and kinetic energy density
    """
    he = Densities.Atom("HE")


    r, h = np.linspace(0.01, 10, 1000, retstep=True)

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



    """
    this secion gives the dimensionless quantities required while developing SCAN functional
    """
    radius=(Wigner_Seitz_radius(d0+d1))
    zeta = (spin_polarization(d0,d1))
    reduced_density = (reduced_density_gradient(g0+g1, d0+d1))
    reduced_laplacian = (reduced_density_laplacian(l0+l1, d0+d1))

    alpha = alpha(d0+d1, g0+g1,t0+t1)

    ex = SCAN.getscan_x(SCAN.DEFAULT_X_PARAMS, d0, d1, g0, g1, t0, t1)
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

    """numerator = -3*(3*(d0+d1)/math.pi)**(1/3)
    denominator = 4
    new = numerator/denominator
    """

    """enhancement factor"""
    f_xc = e_xc/e_x_uni


    """
    combining the four inputs required for the NN and saving the data in txt file for further use
    """
    #a= np.vstack([radius, zeta,reduced_density, reduced_laplacian,f_xc])

    """ to obtain a map from r,s,alpha to fxc from SCAN"""
    radius = 1/(1+radius)
    reduced_density = 1/(1+reduced_density)
    reduced_laplacian = 1/(1+reduced_laplacian**2)
    #reduced_laplacian1 = 1/(1+(math.exp)**reduced_laplacian)
    alpha = 1/(1+alpha)

    a= np.vstack([radius, zeta, reduced_density, reduced_laplacian, f_xc])
    #b= np.vstack([radius, zeta, reduced_density, reduced_laplacian1, f_xc])

    f= open("/Users/kanun/Desktop/ML_project/he.txt","w")
    g= open("/Users/kanun/Desktop/ML_project/uniform.txt","w")
    data = a.T
    #data1= b.T
    np.savetxt(f, data,fmt='%.8f' )
    np.savetxt(g, e_x_uni,fmt='%.8f' )

    #np.savetxt(g, data1,fmt='%.8f' )
    f.close()
    g.close()
    #g.close()








    #print("D0", np.sum(4*math.pi*r**2*d0*h))
    #print("D1", np.sum(4*math.pi*r**2*d1*h))


    #print("EX", np.sum(4*math.pi*r**2*ex*h))

    #plt.plot(r, 4*math.pi*r**2*d0)
    #plt.show()
