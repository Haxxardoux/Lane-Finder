import numpy as np 
import random
import matplotlib.pyplot as plt
import time

def inverse_3x3(m):
    """Inverse 3x3 matrix. Manual implementation!
    Very basic benchmarks show it's ~3x faster than calling numpy inverse
    method. Nevertheless, I imagine that much better optimised version exist
    in the MKL or other library (using SIMD, AVX, and so on).
    I have no idea how far Numba+LLVM is able to go in terms of optimisation of
    this code.
    """
    mflat = m.reshape((m.size, ))
    minv = np.zeros_like(mflat)

    minv[0] = mflat[4] * mflat[8] - mflat[5] * mflat[7]
    minv[3] = -mflat[3] * mflat[8] + mflat[5] * mflat[6]
    minv[6] = mflat[3] * mflat[7] - mflat[4] * mflat[6]

    minv[1] = -mflat[1] * mflat[8] + mflat[2] * mflat[7]
    minv[4] = mflat[0] * mflat[8] - mflat[2] * mflat[6]
    minv[7] = -mflat[0] * mflat[7] + mflat[1] * mflat[6]

    minv[2] = mflat[1] * mflat[5] - mflat[2] * mflat[4]
    minv[5] = -mflat[0] * mflat[5] + mflat[2] * mflat[3]
    minv[8] = mflat[0] * mflat[4] - mflat[1] * mflat[3]

    det = mflat[0] * minv[0] + mflat[1] * minv[3] + mflat[2] * minv[6]
    # UGGGGGLLLLLLLLYYYYYYYYYY!
    if np.abs(det) <= _EPSILON:
        det = 1e-10

    det = 1.0 / det
    for i in range(9):
        minv[i] = minv[i] * det
    minv = minv.reshape((3, 3))
    return minv

timedelta = []
for i in range(100,1000000, 100):
    print(i)
    points = np.linspace(0 ,10, i)
    poly_points = np.polyval([1.1,2.2,1.3], points)
    rand_polypoints = np.array(poly_points)+random.randint(-5,5)
    t0 = time.time()
    fit = np.polyfit(points, rand_polypoints,2)
    t1 = time.time()
    # if i % 1000 == 0:   
    #     model_pts = np.polyval(fit, points)
    #     plt.plot(points, rand_polypoints)
    #     plt.plot(points, model_pts)
    #     plt.show()
    timedelta.append(t1-t0)

plt.plot(timedelta)
plt.show()


# from scipy.optimize import least_squares

# smootht = np.linspace(0, 6000, 100)

# #np.polyval(maybemodel, righty)

# def error(params):
#     return smootht - np.polyval(params, smootht)

# model = least_squares(error, [1,1,1])

# S = [0.038,0.194,.425,.626,1.253,2.500,3.740]
# rate = [0.050,0.127,0.094,0.2122,0.2729,0.2665,0.3317]
# iterations = 5
# rows = 7
# cols = 2

# B = np.matrix([[.9],[.2]]) # original guess for B
# print(B)

# Jf = np.zeros((rows,cols)) # Jacobian matrix from r
# r = np.zeros((rows,1)) #r equations


# def model(Vmax, Km, Sval):
#    return ((Vmax * Sval) / (Km + Sval))

# def residual(x,y,B1,B2):
#    return (y - ((B1*x)/(B2+x)))

# for _ in xrange(iterations):

#    sumOfResid=0
#    #calculate Jr and r for this iteration.
#    for j in xrange(rows):
#       r[j,0] = residual(S[j],rate[j],B[0],B[1])
#       sumOfResid += (r[j,0] * r[j,0])
#       Jf[j,0] = partialDerB1(B[1],S[j])
#       Jf[j,1] = partialDerB2(B[0],B[1],S[j])

#    Jft =  Jf.T
#    B -= np.dot(np.dot( inv(np.dot(Jft,Jf)),Jft),r)

#    print B 