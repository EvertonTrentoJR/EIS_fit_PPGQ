import numpy as np

def Transmission_lines (theta, args):
    return

def Longo2020(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R||C - C || (R - R||CPE) circuit
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta)*args[1] #scaling
    R1 = theta[:,0]
    tau1 = theta[:,1]
    R2 = theta[:,2]
    tau2 = theta[:,3]
    R3 = theta[:,4]
    tau3 = theta[:,5]
    n3 = theta[:,6]
    tau4 = theta[:,7]

    #impedance computation
    Z_b1 = R1/(1+1j*omega*tau1) #p(R1,C1) block
    Z_b2n = R2 + (R3/(1+(1j*omega*tau3)**n3)) #num of the p(C2, R2-p(R3, CPE)) block
    Z_b2d = 1 + 1j*omega*tau2 + (1j*omega*tau4)/(1 + (1j*omega*tau3)**n3) #den of the p(C2, R2-p(R3, CPE)) block
    Z_b2 = Z_b2n/Z_b2d #p(C2, R2-p(R3, CPE)) block

    return Z_b1 + Z_b2