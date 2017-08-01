import caffe
import numpy as np


solver = caffe.SGDSolver('solver.prototxt')

caffe.set_mode_cpu()
#caffe.set_mode_gpu()
#caffe.set_device(0)
#solver.solve()

iter = solver.iter
#w_pre = np.copy(solver.net.params["conv1"][0].data)

while iter<10000:
    solver.step(1)
    iter = solver.iter
    #w_diff = np.copy(solver.net.params["conv1"][0].diff)
    cont = solver.net.blobs["cont"].data
    print("----------------------------------")
    print(cont)
    #loss = solver.net.blobs['loss'].data
    #print("loss: ", loss)

    # w_diff = abs(w_pre-w_cur)
    # print(np.max(w_diff), np.min(w_diff))

    """
    input_data = solver.net.blobs['data'].data
    loss = solver.net.blobs['loss'].data
    accuracy = solver.test_nets[0].blobs['accuracy'].data
    print 'iter:', iter, 'loss:', loss,'accuracy:',accuracy
    """
