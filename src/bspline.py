import numpy as np

def bspline(knots, degree):
    # make sure knots are in acesent order
    knots.sort()
    #
    n = knots.size
    assert n >= 2, 'At least include starting and ending points'
    assert degree in [0, 1, 2, 3], 'Only support upto cubic bspline'
    #
    l = []
    r = []
    f = []
    # first build the zero degree spline
    l_0 = [knots[j]   for j in range(n-1)]
    r_0 = [knots[j+1] for j in range(n-1)]
    f_0 = [lambda x, j=0: (x <= r_0[j]).astype(float)]
    for j in range(1, n-2):
        f_0.append(lambda x, j=j: (x > l_0[j])*(x <= r_0[j]).astype(float))
    f_0.append(lambda x, j=n-2: (x > l_0[j]).astype(float))
    #
    l.append(l_0)
    r.append(r_0)
    f.append(f_0)
    #
    # the rest of the degree
    for i in range(1, degree+1):
        l_i = [knots[0]]
        r_i = [knots[1]]
        f_i = [lambda x, i=i, j=0:
            f[i-1][ j ](x)*(x - r[i-1][ j ])/(l[i-1][ j ] - r[i-1][ j ])]
        #
        for j in range(1, i + n - 2):
            l_i.append(l[i-1][j-1])
            r_i.append(r[i-1][j])
            f_i.append(lambda x, i=i, j=j:
                f[i-1][j-1](x)*(x - l[i-1][j-1])/(r[i-1][j-1] - l[i-1][j-1]) + 
                f[i-1][ j ](x)*(x - r[i-1][ j ])/(l[i-1][ j ] - r[i-1][ j ]))
        #
        l_i.append(knots[-2])
        r_i.append(knots[-1])
        f_i.append(lambda x, i=i, j=i+n-2:
            f[i-1][j-1](x)*(x - l[i-1][j-1])/(r[i-1][j-1] - l[i-1][j-1]))
        #
        l.append(l_i)
        r.append(r_i)
        f.append(f_i)
    #
    H = np.eye(n-1)
    for i in range(degree):
        A = np.zeros((n+i-1, n+i))
        for j in range(n+i-1):
            A[j,j]   = -1.0/(r[i][j] - l[i][j])
            A[j,j+1] =  1.0/(r[i][j] - l[i][j])
        #
        H = H.dot(A)*(i+1)
    #
    return f, H

def dmatrix(x, knots, degree, l_linear=False, r_linear=False):

    if l_linear and r_linear:
        assert knots.size >= 4, 'not enough knots for both side linear'
        # extract the linear spline functions
        f, H = bspline(knots[1:-1], degree)
        f = f[-1]
        #
        # separate the x into inner and outer part
        l_id = np.where(x <= knots[1])[0]
        i_id = np.where((knots[1] < x) & (x < knots[-2]))[0]
        r_id = np.where(x >= knots[-2])[0]
        #
        if l_id.size > 0:
            l_x = x[l_id]
            l1 = degree*(l_x - knots[1])/(knots[1] - knots[2]) + 1.0
            l2 = degree*(l_x - knots[1])/(knots[2] - knots[1])
        else:
            l1 = np.array([])
            l2 = np.array([])
        #
        if i_id.size > 0:
            i_x = x[i_id]
            i0 = np.array([f[j](i_x) for j in range(len(f))]).T
        else:
            i0 = np.array([]).reshape(0, len(f))
        #
        if r_id.size > 0:
            r_x = x[r_id]
            r1 = degree*(r_x - knots[-2])/(knots[-2] - knots[-3]) + 1.0
            r2 = degree*(r_x - knots[-2])/(knots[-3] - knots[-2])
        else:
            r1 = np.array([])
            r2 = np.array([])
        #
        X = np.zeros((x.size, len(f)))
        X[l_id,0]  = l1
        X[l_id,1]  = l2
        X[i_id,:]  = i0
        X[r_id,-1] = r1
        X[r_id,-2] = r2
        #
        A = np.zeros((knots.size-1, X.shape[1]))
        A[0,0] = 1.0/(knots[0] - knots[1])
        A[0,1] = 1.0/(knots[1] - knots[0])
        A[1:-1,:] = H
        A[-1,-2] = 1.0/(knots[-2] - knots[-1])
        A[-1,-1] = 1.0/(knots[-1] - knots[-2])
    #
    elif l_linear:
        assert knots.size >= 3, 'not enough knots for left side linear'
        # extract the linear spline functions
        f, H = bspline(knots[1:], degree)
        f = f[-1]
        #
        l_id = np.where(x <= knots[1])[0]
        r_id = np.where(x >  knots[1])[0]
        #
        if l_id.size > 0:
            l_x = x[l_id]
            l1 = degree*(l_x - knots[1])/(knots[1] - knots[2]) + 1.0
            l2 = degree*(l_x - knots[1])/(knots[2] - knots[1])
        else:
            l1 = np.array([])
            l2 = np.array([])
        #
        if r_id.size > 0:
            r_x = x[r_id]
            r0 = np.array([f[j](r_x) for j in range(len(f))]).T
        else:
            r0 = np.array([]).reshape(0, len(f))
        #
        X = np.zeros((x.size, len(f)))
        X[l_id,0] = l1
        X[l_id,1] = l2
        X[r_id,:] = r0
        #
        A = np.zeros((knots.size-1, X.shape[1]))
        A[0,0] = 1.0/(knots[0] - knots[1])
        A[0,1] = 1.0/(knots[1] - knots[0])
        A[1:,:] = H
    #
    elif r_linear:
        assert knots.size >= 3, 'not enough knots for right side linear'
        # extract the linear spline functions
        f, H = bspline(knots[:-1], degree)
        f = f[-1]
        #
        l_id = np.where(x <  knots[-2])[0]
        r_id = np.where(x >= knots[-2])[0]
        #
        if l_id.size > 0:
            l_x = x[l_id]
            l0 = np.array([f[j](l_x) for j in range(len(f))]).T
        else:
            l0 = np.array([]).reshape(0, len(f))
        #
        if r_id.size > 0:
            r_x = x[r_id]
            r1 = degree*(r_x - knots[-2])/(knots[-2] - knots[-3]) + 1.0
            r2 = degree*(r_x - knots[-2])/(knots[-3] - knots[-2])
        else:
            r1 = np.array([])
            r2 = np.array([])
        #
        X = np.zeros((x.size, len(f)))
        X[l_id,:]  = l0
        X[r_id,-1] = r1
        X[r_id,-2] = r2
        #
        A = np.zeros((knots.size-1, X.shape[1]))
        A[:-1,:] = H
        A[-1,-2] = 1.0/(knots[-2] - knots[-1])
        A[-1,-1] = 1.0/(knots[-1] - knots[-2])
    #
    else:
        f, H = bspline(knots, degree)
        f = f[-1]
        #
        X = np.zeros((x.size, len(f)))
        for i in range(len(f)):
            X[:,i] = f[i](x)
        #
        A = H.copy()
    #
    return X, A