'''
Created on Jun 1, 2015

@author: xujian
'''
import cPickle, gzip, numpy, theano,numpy
import theano.tensor as T

# Load the dataset
f = gzip.open('/home/xujian/Downloads/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
print f.seek(10)
f.close()

def shared_dataset(data_xy):
  """ Function that loads the dataset into shared variables
  The reason we store our dataset in shared variables is to allow
  Theano to copy it into the GPU memory (when code is run on GPU).
  Since copying data into the GPU is slow, copying a minibatch everytime
  is needed (the default behaviour if the data is not in a shared
  variable) would lead to a large decrease in performance.
  """
  data_x, data_y = data_xy
  shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
  shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))

  return shared_x, T.cast(shared_y, 'int32')

test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)

batch_size = 500    # size of the minibatch
# accessing the third minibatch of the training set

data  = train_set_x[2 * 500: 3 * 500]
label = train_set_y[2 * 500: 3 * 500]

# Minibatch Stochastic Gradient Descent

# assume loss is a symbolic description of the loss function given
# the symbolic variables params (shared variable), x_batch, y_batch;

# compute gradient of loss with respect to params
d_loss_wrt_params = T.grad(loss, params)

# compile the MSGD step into a theano function
updates = [(params, params - learning_rate * d_loss_wrt_params)]
MSGD = theano.function([x_batch,y_batch], loss, updates=updates)

for (x_batch, y_batch) in train_batches:
  # here x_batch and y_batch are elements of train_batches and
  # therefore numpy arrays; function MSGD also updates the params
  print('Current loss is ', MSGD(x_batch, y_batch))
  if stopping_condition_is_met:
    return params