import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# minimize
#     F = x[1]^2 + 4x[2]^2 -32x[2] + 64

# subject to:
#      x[1] + x[2] <= 7
#     -x[1] + 2x[2] <= 4
#      x[1] >= 0
#      x[2] >= 0
#      x[2] <= 4

# in matrix notation:
#     F = (1/2)*x.T*H*x + c*x + c0

# subject to:
#     Ax <= b

# where:
#     H = [[2, 0],
#          [0, 8]]

#     c = [0, -32]

#     c0 = 64

#     A = [[ 1, 1],
#          [-1, 2],
#          [-1, 0],
#          [0, -1],
#          [0,  1]]

#     b = [7,4,0,0,4]

H = np.array([[2.]])

c = np.array([0])

c0 = 0



# G
# [-5.75e-02]
# [-2.50e-03]
# [ 5.75e-02]
# [ 2.50e-03]
#
# h
# [ 1.31e+00]
# [ 3.35e-01]
# [-1.30e+00]
# [-3.29e-01]

#    -0.0575 x <= 1.31, => x >= -22.782608695652176
#    -0.00250 x <= 0.335 => x >= -134
#     0.0575 x <= -1.30 => x <= -22.608695652173914
#     0.00250 x <= -0.329 => x <= -131.6

#[-5.75e-02] x <= [ 1.69e+00] => x >= -29.391304347826086
#[-2.50e-03] x <= [ 4.20e-01]  => x >= -168.0
#[ 5.75e-02] x <= [-1.35e+00] => x  <= -23.47826086956522
#[ 2.50e-03] x <= [-1.10e-01]  => x <= -44.0




A = np.array(
    [[-5.75e-02], #x <= 1.31, =>
    [-2.50e-03], #x <= 0.335 =>
     [5.75e-02], #x <= -1.30 =>
     [2.50e-03]]) #x <= -0.329 =>

b = np.array([1.69e+00,
 4.20e-01,
-1.35e+00,
-1.10e-01])

x0 = np.random.randn(1)

def loss(x, sign=1.):
    #return np.linalg.norm(x)
    return sign * (0.5 * np.dot(x.T, np.dot(H, x))+ np.dot(c, x) + c0)

def jac(x, sign=1.):
    return sign * (np.dot(x.T, H) + c)

cons = {'type':'ineq',
        'fun':lambda x: b - np.dot(A,x),
        'jac':lambda x: -A}

opt = {'disp':False}

def solve():

    res_cons = optimize.minimize(loss, x0, jac=jac,constraints=cons,
                                 method='SLSQP', options=opt)

    #res_uncons = optimize.minimize(loss, x0, jac=jac, method='SLSQP',
    #                               options=opt)

    print('\nConstrained:')
    print(res_cons)

    print('\nUnconstrained:')
    #print (res_uncons)

    x1 = res_cons['x']
    f = res_cons['fun']

    #x1_unc, x2_unc = res_uncons['x']
    #f_unc = res_uncons['fun']

    # plotting
    # xgrid = np.mgrid[-2:4:0.1, 1.5:5.5:0.1]
    # xvec = xgrid.reshape(2, -1).T
    # F = np.vstack([loss(xi) for xi in xvec]).reshape(xgrid.shape[1:])
    #
    # ax = plt.axes(projection='3d')
    # #ax.hold(True)
    # ax.plot_surface(xgrid[0], xgrid[1], F, rstride=1, cstride=1,
    #                 cmap=plt.cm.jet, shade=True, alpha=0.9, linewidth=0)
    # ax.plot3D([x1],  [f], 'og', mec='w', label='Constrained minimum')
    # #ax.plot3D([x1_unc], [x2_unc], [f_unc], 'oy', mec='w',
    # #          label='Unconstrained minimum')
    # ax.legend(fancybox=True, numpoints=1)
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('F')

if __name__ == '__main__':
    solve()