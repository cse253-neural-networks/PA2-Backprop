import numpy as np
import pickle
from tqdm import tqdm

config = {}
config['layer_specs'] = [784, 50, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid'  # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001  # Learning rate of gradient descent algorithm


def softmax(x):
    """
    Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
    """
    exp_x = np.exp(x)
    sample_n = x.shape[0]
    output = exp_x / exp_x.sum(axis=1).reshape(sample_n, 1)
    return output


def load_data(fname):
    """
  Write code to read the data and return it as 2 numpy arrays.
  Make sure to convert labels to one hot encoded format.
  """
    images = []
    labels = []
    with open(fname, 'rb') as d:
        datum_s = pickle.load(d)
    for datum in datum_s:
        label_onehot = np.zeros(10)
        images.append(datum[:-1])
        label_onehot[int(datum[-1])] = 1
        labels.append(label_onehot)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


class Activation:
    def __init__(self, activation_type="sigmoid"):
        self.activation_type = activation_type
        self.x = None  # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.

    def forward_pass(self, a):
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward_pass(self, delta):
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
    Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
        self.x = x
        output = 1.0 / (1.0 + np.exp(-self.x))
        return output

    def tanh(self, x):
        """
    Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
    """
        self.x = x
        output = np.tanh(self.x)
        return output

    def ReLU(self, x):
        """
    Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
    """
        self.x = x
        output = np.maximum(0, self.x)
        return output

    def grad_sigmoid(self):
        """
    Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
        grad = np.divide(np.exp(-self.x), np.square(1.0 + np.exp(-self.x)))
        return grad

    def grad_tanh(self):
        """
    Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
    """
        grad = 1.0 - np.square(np.tanh(self.x))
        return grad

    def grad_ReLU(self):
        """
    Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
    """
        grad = 1.0 * (self.x > 0)
        return grad


class Layer():
    def __init__(self, in_units, out_units):
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)  # Weight matrix
        self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
        self.x = None  # Save the input to forward_pass in this
        self.a = None  # Save the output of forward pass in this (without activation)
        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def forward_pass(self, x):
        """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
        self.x = x
        self.a = np.dot(self.x, self.w) + self.b
        return self.a

    def backward_pass(self, delta):
        """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """
        self.d_w = np.dot(self.x.T, delta)  # note w += w as minus sign is absorbed(to meet checker.py)
        self.d_b = np.array([delta.sum(axis=0)])
        self.d_x = np.dot(delta, self.w.T)
        return self.d_x


class Neuralnetwork():
    def __init__(self, config):
        self.layers = []
        self.x = None  # Save the input to forward_pass in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward_pass in this variable
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def forward_pass(self, x, targets=None):
        """
    Write the code for forward pass through all layers of the model and return loss and predictions.
    If targets == None, loss should be None. If not, then return the loss computed.
    """
        self.x = x
        self.y = np.copy(x)
        for l in self.layers:
            self.y = l.forward_pass(self.y)
        # softmax regression for output
        self.y = softmax(self.y)

        self.targets = targets
        loss = self.loss_func(self.y, self.targets)
        return loss, self.y

    def loss_func(self, logits, targets):
        '''
      find cross entropy loss between logits and targets
      '''
        if targets is None:
            output = None
        else:
            output = np.sum(-targets * np.log(logits)) / len(logits)  # take the average
        return output

    def backward_pass(self):
        '''
      implement the backward pass for the whole network.
      hint - use previously built functions.
      '''
        delta = self.targets - self.y
        for l in self.layers[::-1]:
            delta = l.backward_pass(delta)


def get_accuracy(logits, target):
    """
  get the accuracy between predicted and target
  """
    predicted_idx = np.argmax(logits, axis=1).reshape(-1)
    target_idx = np.argmax(target, axis=1).reshape(-1)
    diff_idx = target_idx - predicted_idx  # compare the max index between predicted and target, and get the accuracy
    num_right = np.count_nonzero(diff_idx == 0)  # get number of right prediction
    num_tot = logits.shape[0]
    return num_right / num_tot


def trainer(model, X_train, y_train, X_valid, y_valid, config):
    """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
    # result for making the plot
    result = {
        'epoch': [],
        'train_err': [],
        'train_acc': [],
        'valid_err': [],
        'valid_acc': [],
        'weights': {},
        'bias': {},
        'weights_diff': {},
        'bias_diff': {}
    }

    if config['momentum']:
        print('Momentum is applied!')
    else:
        print('Momentum is NOT applied!')
        if config['momentum_gamma'] != 0:
            raise ValueError("Momentum gamma should be set to 0 since it's not used!")

    epoch_s = np.linspace(1, config['epochs'], config['epochs'])
    train_tot_idx = np.arange(0, len(X_train), dtype=int)
    for epoch in tqdm(epoch_s, desc='epoch'):
        valid_err, valid_logits = model.forward_pass(X_valid, y_valid)
        # early stop
        if len(result['valid_err']) > 0:
            if valid_err > result['valid_err'][-1]:
                # recover weights and bias from last time
                layer_no = 0
                for layer_idx, layer in enumerate(model.layers):
                    if isinstance(layer, Layer):
                        layer_no += 1
                        layer.w = result['weights'][layer_no]
                        layer.b = result['bias'][layer_no]

                # store early stop epoch number
                config['early_stop_epoch'] = epoch - 1  # since this epoch doesn't begin training yet

                print('Early stop!')
                break

        # we need to store a bunch of things before updating
        # store accuracy and loss error for validation data
        result['valid_err'].append(valid_err)
        result['valid_acc'].append(get_accuracy(valid_logits, y_valid))

        # store accuracy and loss error for train data
        train_err, train_logits = model.forward_pass(X_train, y_train)
        result['train_err'].append(train_err)
        result['train_acc'].append(get_accuracy(train_logits, y_train))

        # store weights and bias difference for momentum
        layer_no = 0
        for layer_idx, layer in enumerate(model.layers):
            if isinstance(layer, Layer):
                layer_no += 1
                if len(result['weights']):  # we update new weight/bias difference
                    result['weights_diff'][layer_no] = layer.w - result['weights'][layer_no]
                    result['bias_diff'][layer_no] = layer.b - result['bias'][layer_no]
                else:  # initialize it as 0
                    result['weights_diff'][layer_no] = np.zeros_like(layer.w)
                    result['bias_diff'][layer_no] = np.zeros_like(layer.b)

        # store weights and bias
        layer_no = 0
        for layer_idx, layer in enumerate(model.layers):
            if isinstance(layer, Layer):
                layer_no += 1
                result['weights'][layer_no] = layer.w
                result['bias'][layer_no] = layer.b

        # store epoch
        result['epoch'].append(epoch)

        # begin update
        np.random.seed(100)  #TODO set a fixed seed?
        np.random.shuffle(train_tot_idx)
        num_batch = int(len(train_tot_idx) / config['batch_size'])  # how many batch we have
        for i in range(num_batch):
            idx_begin, idx_end = i*config['batch_size'], (i+1)*config['batch_size']
            train_idx = train_tot_idx[idx_begin:idx_end]
            # input and target in this batch
            x_train_batch = X_train[train_idx]
            y_train_batch = y_train[train_idx]

            # begin training
            model.forward_pass(x_train_batch, y_train_batch)
            model.backward_pass()
            layer_no = 0
            for layer_idx, layer in enumerate(model.layers):
                if isinstance(layer, Layer):
                    layer_no += 1
                    layer.w += config['learning_rate'] * layer.d_w + config['momentum_gamma'] * result['weights_diff'][layer_no]
                    layer.b += config['learning_rate'] * layer.d_b + config['momentum_gamma'] * result['bias_diff'][layer_no]

    # save result about train and validation data
    pickle.dump(result, open('train_validation_result.pkl', 'wb'))


def test(model, X_test, y_test, config):
    """
    Write code to run the model on the data passed as input and return accuracy.
    """
    test_logits = model.forward_pass(X_test, y_test)[1]
    accuracy = get_accuracy(test_logits, y_test)
    print('Test accuracy is %.2f%%' %(accuracy*100))
    return accuracy


if __name__ == "__main__":
    train_data_fname = 'MNIST_train.pkl'
    valid_data_fname = 'MNIST_valid.pkl'
    test_data_fname = 'MNIST_test.pkl'

    ### Train the network ###
    model = Neuralnetwork(config)
    X_train, y_train = load_data(train_data_fname)
    X_valid, y_valid = load_data(valid_data_fname)
    X_test, y_test = load_data(test_data_fname)
    trainer(model, X_train, y_train, X_valid, y_valid, config)
    test_acc = test(model, X_test, y_test, config)
