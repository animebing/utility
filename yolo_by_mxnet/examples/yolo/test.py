import numpy as np
import mxnet as mx

data = mx.sym.Variable('data');
upsample = mx.sym.UpSampling(data, scale=2, sample_type='nearest', num_args=1)
exe = upsample.simple_bind(ctx=mx.gpu(0), data=(1, 1, 2, 2))
tmp = np.random.randn(1, 1, 2, 2)
print tmp
exe.arg_dict['data'][:] = mx.nd.array(tmp)
exe.forward()
print exe.outputs[0].asnumpy()
