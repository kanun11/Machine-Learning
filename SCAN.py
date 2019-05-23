import numpy as np
from math import pi

DEFAULT_X_PARAMS = dict(
    K1 = 0.065,
    A1 = 4.9479,
    C1X = 0.667,
    C2X= 0.8,
    DX = 1.24,
    CXBX = 1.0,
    MUAK = 10.0/81.0,
    K0 = 0.1740,
    B1X = 0.156632,
    B2X = 0.12083,
    B3X = 0.5,
    B4X = 0.2218,
    AX = -0.7385587663820224058842300326808360
)

DEFAULT_C_PARAMS = dict(
    C1C = 0.64,
    C2C = 1.5,
    DC = 0.7,
    B1C = 0.0285764,
    B2C = 0.0889,
    B3C = 0.125541,
    KAILD = 0.12802585262625815,
    GAMMA = 0.031090690869654895034940863712730,
    BETA_MB = 0.066724550603149220,
    AFACTOR = 0.1,
    BFACTOR = 0.1778,
    BETA_RS_0 = 0.066725,  # Same as BETA_MB?
    C_TILDE = 1.467,
    P_TAU = 4.5,
    F0 = -0.9
)


def getscan_x(params, d0, d1, g0, g1, t0, t1, only_0=False, only_Fx=False):
    # First spin 0
    rho = 2*d0
    drho = 2*g0
    tauw = drho**2/rho/8.0
    tau_rho = 2*t0
    p = drho**2/(4*(3*pi**2)**(2.0/3.0)*rho**(8.0/3.0))
    tau_unif = 3.0/10.0*(3*pi**2)**(2.0/3.0)*rho**(5.0/3.0)
    alpha = (tau_rho - tauw)/tau_unif

    # construct LDA exchange energy density
    exunif_0 = params['AX']*rho**(1.0/3.0)
    exlda_0 = exunif_0*rho

    # and enhancement factor

    Fx0 = scanFx(params, p, alpha)

    Ex_0 = exlda_0*Fx0

    if only_0:
        return Ex_0

    # Now spin 1
    rho = 2*d1
    drho = 2*g1
    tauw = drho**2/rho/8.0
    tau_rho = 2*t1
    p = drho**2/(4*(3*pi**2)**(2.0/3.0)*rho**(8.0/3.0))
    tau_unif = 3.0/10.0*(3*pi**2)**(2.0/3.0)*rho**(5.0/3.0)
    alpha = (tau_rho - tauw)/tau_unif

    # construct LDA exchange energy density
    exunif_1 = params['AX']*rho**(1.0/3.0)
    exlda_1 = exunif_1*rho

    # and enhancement factor
    Fx1 = scanFx(params, p, alpha)
    Ex_1 = exlda_1*Fx1

    if only_Fx:
        return Fx0, Fx1

    return (Ex_0 + Ex_1)/2.0


def scanFx(params, p, alpha):
    p2 = p**2
    oma = 1.0 - alpha
    oma2 = oma**2

    # make HX0
    hx0 = 1.0 + params['K0']

    # make HX1
    cfb4 = params['MUAK']**2/params['K1'] - 0.112654

    if cfb4 > 0.0:
        wfac = cfb4*p2*np.exp(-cfb4*p/params['MUAK'])
    else:
        wfac = cfb4*p2*np.exp(cfb4*p/params['MUAK'])
    vfac = params['B1X']*p + params['B2X']*oma*np.exp(-params['B3X']*oma2)
    yfac = params['MUAK']*p + wfac + vfac**2
    hx1 = 1.0 + params['K1'] - params['K1']/(1.0 + yfac/params['K1'])

    # FA
    FA = np.zeros(alpha.shape)
    FA[alpha < 1.0] = np.exp(-params['C1X']*alpha[alpha < 1.0]/oma[alpha < 1.0])
    FA[alpha > 1.0] = -params['DX']*np.exp(params['C2X']/oma[alpha > 1.0])

    # gx
    p14 = p**(1.0/4.0)
    gx = np.ones(p.shape)
    gx[p > 0.0] = 1.0 - np.exp(-params['A1']/p14[p > 0.0])

    # Fx1
    Fx1 = hx1 + FA*(hx0 - hx1)

    # Fx
    return Fx1*gx

def getscan_c(params, d0, d1, g0, g1, t0, t1, zeta):
    """
    This follows the ugly Fortran of the original optimisation program. Sorry.

    Note: g0 and g1 are absolute value of gradients
    """
    dT = d0 + d1
    g00 = g0**2
    g11 = g1**2
    gT = (g0 + g1)
    gTT = gT**2
    gC = (gT - g00 - g11)/2.0
    tauw = gTT/(8*dT)
    tT = t0 + t1


    ds_zeta = (np.power(1.0 + zeta, 5.0/3.0) + np.power(1.0 - zeta, 5.0/3.0))/2.0
    dx_zeta = (np.power(1.0 + zeta, 4.0/3.0) + np.power(1.0 - zeta, 4.0/3.0))/2.0
    tau0 = 0.3*np.power(3*pi**2, 2.0/3.0)*np.power(dT, 5.0/3.0)*ds_zeta
    alpha = (tT - tauw)/tau0

    # Alpha interpolation Function
    f_alpha = np.zeros(alpha.shape)
    f_alpha[alpha < 1.0] = np.exp(params['C1C']*alpha[alpha < 1.0]/(alpha[alpha < 1.0] - 1.0))
    f_alpha[alpha > 1.0] = -params['DC']*np.exp(-params['C2C']/(alpha[alpha > 1.0] - 1.0))

    dthrd = np.exp(np.log(dT)*1.0/3.0)
    rs = (0.75/pi)**(1.0/3.0)/dthrd

    s = gT/(2.0*(3.0*pi**2)**(1.0/3.0)*np.power(dT, 4.0/3.0))

    eppgga0 = corgga_0(params, rs, s, zeta)
    eppgga1 = corgga_1(params, rs, s, zeta)

    epp = eppgga1 + f_alpha*(eppgga0 - eppgga1)

    return dT*epp


def corgga_0(params, rs, s, zeta):
    # _0 does not refer to spin in function name

    ax_lda = -3.0/(4.0*pi)*(9.0*pi/4.0)**(1.0/3.0)

    phi = (np.exp((2.0/3.0)*np.log(1.0 + zeta)) + np.exp((2.0/3.0)*np.log(1.0 - zeta)))/2.0
    afix_T = np.sqrt(pi/4.0)*np.power(9.0*pi/4.0, 1.0/6.0)

    sqrt_rs = np.sqrt(rs)
    f1 = 1.0 + params['B2C']*sqrt_rs + params['B3C']*rs
    ec0_lda = -params['B1C']/f1


    dx_zeta = (np.power(1.0 + zeta, 4.0/3.0) + np.power(1.0 - zeta, 4.0/3.0))/2.0
    gc_zeta = (2**(1.0/3.0) - dx_zeta)/(2**(1.0/3.0) - 1.0)  # This is different from published?!

    w0 = np.exp(-ec0_lda/params['B1C']) - 1.0

    gf_inf = 1.0/(1.0 + 4.0*params['KAILD']*s**2)**(1.0/4.0)

    hcore0 = 1.0 + w0*(1.0 - gf_inf)
    h0 = params['B1C']*np.log(hcore0)

    EPPGGA = (ec0_lda + h0)*gc_zeta

    return EPPGGA


def corgga_1(params, rs, s, zeta):

    dthrd = rs/(0.75/pi)**(1.0/3.0)
    phi = (np.exp((2.0/3.0)*np.log(1.0 + zeta)) + np.exp((2.0/3.0)*np.log(1.0 - zeta)))/2.0

    afix_T = np.sqrt(pi/4.0)*np.power(9.0*pi/4.0, 1.0/6.0)

    T = afix_T*s/np.sqrt(rs)/phi
    FK = (3.0*pi**2)**(1.0/3.0)*dthrd
    sk = np.sqrt(4.0*FK/pi)

    EC, H = corpbe_rtpss(rs, zeta, T, phi, sk)

    beta_num = 1.0 + params['AFACTOR']*rs
    beta_den = 1.0 + params['BFACTOR']*rs
    beta = params['BETA_MB']*beta_num/beta_den

    phi3 = phi**3
    pon = -EC/(phi3*params['GAMMA'])
    w = np.exp(pon) - 1

    A = beta/(params['GAMMA']*w)
    V = A*T**2

    f_g = 1.0/np.power(1.0 + 4.0*V, 0.25)

    hcore = 1.0 + w*(1.0 - f_g)
    ah = params['GAMMA']*phi**3
    H = ah*np.log(hcore)

    return EC+H


def corpbe_rtpss(rs, zeta, T, phi, sk):
    GAM = 0.51984209978974632953442121455650  # 2^(4/3)-2
    FZZ = 8.0/(9.0*GAM)
    gamma = 0.031090690869654895034940863712730  # (1-log(2))/pi^2
    bet_mb = 0.066724550603149220
    sqrt_rs = np.sqrt(rs)

    EU, EURS = gcor2(0.03109070, 0.213700, 7.59570, 3.58760, 1.63820, 0.492940, sqrt_rs)
    EP, EPRS = gcor2(0.015545350, 0.205480, 14.11890, 6.19770, 3.36620, 0.625170, sqrt_rs)
    ALFM, ALFRSM = gcor2(0.01688690, 0.111250, 10.3570, 3.62310, 0.880260, 0.496710, sqrt_rs)

    ALFC = -ALFM
    Z4 = zeta**4

    # LDA part of the energy
    F = ((1.0 + zeta)**(4.0/3.0) + (1.0 - zeta)**(4.0/3.0) - 2.0)/GAM
    EC = EU*(1.0 - F*zeta**4) + EP*F*Z4 - ALFM*F*(1.0 - Z4)/FZZ

    # PBE correction
    bet = bet_mb*(1.0 + 0.1*rs)/(1.0 + 0.1778*rs)

    delt = bet/gamma
    phi3 = phi**3
    pon = -EC/(phi3*gamma)
    B = delt/(np.exp(pon) - 1.0)
    B2 = B**2
    T2 = T**2
    T4 = T**4
    Q4 = 1.0 + B*T2
    Q5 = 1.0 + B*T2 + B2*T4
    H = phi3*(bet/delt)*np.log(1.0 + delt*Q4*T2/Q5)   # Non-local part of correlation

    return EC, H


def gcor2(A, A1, B1, B2, B3, B4, sqrtrs):
    Q0 = -2.0*A*(1.0 + A1*sqrtrs*sqrtrs)
    Q1 = 2.0*A*sqrtrs*(B1 + sqrtrs*(B2 + sqrtrs*(B3 + B4*sqrtrs)))
    Q2 = np.log(1.0 + 1.0/Q1)
    GG = Q0*Q2
    Q3 = A*(B1/sqrtrs + 2.0*B2 + sqrtrs*(3.0*B3 + 4.0*B4*sqrtrs))
    GGRS = -2.0*A*A1*Q2 - Q0*Q3/(Q1*(1.0 + Q1))
    return GG, GGRS


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Beginning XCFun style SCAN implementation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def getscan_x_both(params, d0, d1, g0, g1, t0, t1, weights):
    e0 = getscan_x(params, 2*d0, 2*g0, 2*t0, weights)
    e1 = getscan_x(params, 2*d1, 2*g1, 2*t1, weights)
    return 0.5*(e0 + e1)


def getscan_c_both(params, d0, d1, g0, g1, t0, t1, weights):
    zeta = (d0 - d1)/(d0 + d1)
    e0 = getscan_c(params, d0, d1, np.abs(g0), np.abs(g1), t0, t1, zeta, weights)
    return e0


def eps_c_0_high_dens_zeta_0(params, s):
    """
    Assuming zeta = 0
    """

    cx = -(3.0/(4.0*pi))*(9.0*pi/4.0)**(1.0/3.0)
    beta_inf = params['BETA_RS_0']*params['AFACTOR']/params['BFACTOR']
    chi_inf = (3.0*pi**2/16.0)**(2.0/3.0)*beta_inf/(cx - params['F0'])  # checked
    g_inf = 1.0/np.power(1.0 + 4*chi_inf*s**2, 1.0/4.0)
    return params['B1C']*np.log(1.0 - g_inf*np.expm1(1)/np.exp(1))


