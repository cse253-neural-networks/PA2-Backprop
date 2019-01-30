from neuralnet import *

config = {}
config['layer_specs'] = [784, 50, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
epsilon = 1e-2

training_file_name = 'MNIST_train.pkl'
images, labels = load_data(training_file_name)
n_sample = 20
images, labels = images[:n_sample], labels[:n_sample]  # choose a subset of training data

nnet = Neuralnetwork(config)
nnet.forward_pass(images, targets = labels)
nnet.backward_pass()

idx_w = {0: [(300, 10), (200, 20)],
         2: [(20, 8), (7, 5)]}  # index of weight to be checked

def check_error(diff, epsilon):
    if diff < epsilon**2:
        print('Correct!')
    else:
        print('Wrong!')

print('weight')
layer_no = 0
for layer_idx, idx_s in idx_w.items():
    layer_no += 1
    for idx in idx_s:
        i, j = idx
        nnet.layers[layer_idx].w[i][j] += epsilon
        loss_p = nnet.forward_pass(images, targets=labels)[0]

        nnet.layers[layer_idx].w[i][j] -= 2 * epsilon  # since we already added one epsilon
        loss_m = nnet.forward_pass(images, targets=labels)[0]

        nnet.layers[layer_idx].w[i][j] += epsilon  # recover original weight

        d_w = (loss_p - loss_m) / (2 * epsilon) * n_sample
        d_w_backprop = -nnet.layers[layer_idx].d_w[i][j]
        diff = abs(d_w - d_w_backprop)
        print('layer %d, index (%d, %d)' %(layer_no, i, j))
        print('numerical: %f' %d_w)
        print('backprop: %f' %d_w_backprop)
        print('difference: %f' %diff)
        check_error(diff, epsilon)
        print()


idx_b = {0: 10,
         2: 6}
print('bias')
layer_no = 0
for layer_idx, idx in idx_b.items():
    layer_no += 1
    nnet.layers[layer_idx].b[0][idx] += epsilon
    loss_p = nnet.forward_pass(images, targets=labels)[0]

    nnet.layers[layer_idx].b[0][idx] -= 2 * epsilon  # since we already added one epsilon
    loss_m = nnet.forward_pass(images, targets=labels)[0]

    nnet.layers[layer_idx].b[0][idx] += epsilon  # recover original bias

    d_b = (loss_p - loss_m) / (2 * epsilon) * n_sample
    d_b_backprop = -nnet.layers[layer_idx].d_b[0][idx]
    diff = abs(d_b - d_b_backprop)
    print('layer %d, index %d' % (layer_no, idx))
    print('numerical: %f' % d_b)
    print('backprop: %f' % d_b_backprop)
    print('difference: %f' % diff)
    check_error(diff, epsilon)
    print()
