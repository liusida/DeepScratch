import numpy as np

def test_init():
    init(50)
    debug_show_all_variables()
#test_init()

def test_one_hot():
    Y = np.ones((5,1))
    Y = one_hot(Y, 10)
    assert(Y[0,0]==0)
    assert(Y[1,0]==1)
    assert(Y[2,1]==0)

def test_ReLU(ReLU):
    X = np.array([[1., 2., -2., -3.], [-2, -2, 1, 0.1]])
    Y = np.array([[1., 2., 0., 0.], [0, 0, 1, 0.1]])
    bias = np.sum(np.abs(Y - ReLU(X)))
    assert(bias<0.0001)
#test_ReLU()

def test_softmax(softmax):
    X = np.array([-3.44,1.16,-0.81,3.91])
    Y = np.array([0.0006, 0.0596, 0.0083, 0.9315])
    bias = np.sum(np.abs(Y - softmax(X)))
    assert(bias<0.0001)
    X = np.array([[1.,2,3,4],[2,2,3,4],[5,5,5,4]])
    Y = np.array([[ 0.01714783,  0.0452785 ,  0.10650698,  0.33333333],
       [ 0.04661262,  0.0452785 ,  0.10650698,  0.33333333],
       [ 0.93623955,  0.909443  ,  0.78698604,  0.33333333]])
    bias = np.sum(np.abs(Y - softmax(X)))
    assert(bias<1e-5)
#test_softmax()

def test_forward_propagation_each_layer(forward_propagation_each_layer):
    W = np.array([[-1.]])
    b = np.array([[0.]])
    A_prev = np.array([[2.]])
    Z, A = forward_propagation_each_layer(W, A_prev, b)
    assert(np.squeeze(Z)==-2)
    assert(np.squeeze(A)==0)
    
    A_prev = np.array([[-2,0.1,0],[1,0.2,0],[0,0.3,0],[-1,0.4,0],[2,-0.5,0]])
    W = np.array([[1.1,1,1,1,1],[-1,-1,0,-1,-1]])
    b = np.array([[-0.5],[-1.63]])
    Z, A = forward_propagation_each_layer(W, A_prev, b)
    result = -2*1.1+1*1+0-1*1+2*1-0.5
    assert(np.abs(Z[0,0]-result)<1e-5)
    assert(np.abs(Z[0,2]+0.5)<1e-5)
    #print(Z, "\n\n", A)
#test_forward_propagation_each_layer()

def test_loss(loss, softmax):
    Y = np.asarray([[0., 1., 1.], [1., 0., 0.]])
    aL = np.array([[.8, .9, .4], [.2, .1, .6]])
    assert(loss(aL, Y) - 0.87702971 < 1e-7)
    
    z = np.array([[1.,2,3,4],[2,2,3,4],[5,5,5,4]])
    y = np.array([[0.,0,1],[0,0,1],[0,0,1],[1,0,0]]).T
    s = softmax(z)
    bias = 0.37474097876709611 - loss(s, y)
    assert(np.abs(bias)<1e-5)
#test_loss()

def test_predict():
    Y = np.array([[0.9,0.01,0.01,0.01,0.02,0.01,0.01,0.01,0.01,0.01],
                  [0.01,0.01,0.01,0.02,0.9,0.01,0.01,0.01,0.01,0.01]
                 ]).T
    print(Y.shape)
    Y_result = predict(Y)
    print(Y_result)
    Z = np.array([0,4]).reshape(2,1)
    assert(np.sum(Y_result != Z)==0)
#test_predict()

def test_backpropagation_cost(backpropagate_cost):
    Y = np.asarray([[0., 1., 1.], [1., 0., 0.]])
    aL = np.array([[.8, .9, .4], [.2, .1, .6]])
    for i in range(1):
        dAL = backpropagate_cost(Y, aL)
        aL = aL - dAL
        #print(aL,"\n")
#test_backpropagation_cost()


def test_backpropagation_softmax(backpropagate_softmax, softmax, loss, backpropagate_cost):
    z = np.array([[1.,2,3,4],[2,2,3,4],[5,5,5,4]])
    y = np.array([[0.,0,1],[0,0,1],[0,0,1],[1,0,0]]).T
    a = softmax(z)
    l = loss(a, y)
    print("softmax + loss = ", l)
    da = backpropagate_cost(y, a)
    print("softmax = ", a)
    print("da = ", da)
    dz = backpropagate_softmax(a, da, y, z)
    result = np.array([[ 0.01714783,0.0452785,0.10650698,-0.66666667],
                     [ 0.04661262,0.0452785,0.10650698,0.33333333],
                     [-0.06376045,-0.090557,-0.21301396,0.33333333]])
    print("z=",z)
    print("dz=", dz)
    assert(np.sum(np.abs(result - dz))<1e-6)
#test_backpropagation_softmax()


def test_backpropagation_linear():
    """ TODO """
    A_prev = np.array([[1,2],[2,2],[3,2]])
    #W = 
    pass

def test_backpropagation_ReLU():
    """ TODO """
    pass