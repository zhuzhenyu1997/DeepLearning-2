import torch
import math

class Linear(object):
    """
    Linear (Fully connected) layer: y = Wx + b.
    """
    def __init__(self, in_dim, out_dim, bias=True):
        """
        Initialize linear layer:
        Weight parameter is initialized by Normal distribution N(0, sqrt(out_dim)).
        bias parameter is initialized by zero vector.
        :param in_dim: input dimensions.
        :param out_dim: output dimensions.
        :param bias: if True, add bias item.
        """
        # initialize weight parameter
        self.parameters = [torch.empty(out_dim, in_dim)]
        self.parameters[0].normal_(0, math.sqrt(2 / out_dim))

        # initialize bias parameter
        if bias:
            self.parameters.append(torch.empty(out_dim))
            self.parameters[-1].zero_()

        self.bias = bias
        self.input = None
        self.gradwrtoutput = None

    def forward(self, input):
        """
        Compute Y = XW + b, X(N*n), W(n*m), b(m) , Y(N*m).
        :param input: X.
        :return: Y = XW + b.
        """
        self.input = input
        output = input.matmul(self.parameters[0].t())
        if self.bias:
            output += self.parameters[1]
        return output

    def backward(self, gradwrtoutput):
        """
        Compute gradient.
        :param gradwrtoutput: backward output from last layer, gradient w.r.t layer output.
        :return: gradient.
        """
        self.gradwrtoutput = gradwrtoutput
        return gradwrtoutput.matmul(self.parameters[0])

    def grad(self):
        """
        Compute gradient w.r.t each parameter, when we use mini-batch SGD, take the mean value.
        :return: gradient w.r.t each parameter.
        """
        grad_wrt_weights = self.gradwrtoutput.t().matmul(self.input)
        if self.bias:
            grad_wrt_biases = self.gradwrtoutput.sum(0)
            return [grad_wrt_weights, grad_wrt_biases]
        else:
            return grad_wrt_weights

    def param(self):
        """
        :return: all parameter in this layer (weight and bias).
        """
        return self.parameters

class Tanh(object):
    """
    Activation function: Tanh.
    """
    def forward(self, input):
        """
        forward: apply Tanh() to each element in input data.
        :param input: input data x.
        :return: Tanh(input).
        """
        self.input = input
        return torch.tanh(input)

    def backward(self, gradwrtoutput):
        """
        backward: apply gradient of Tanh from input data on gradwrtoutput.
        :param gradwrtoutput: output from last layer in backward path.
        :return: gradient of Tanh from input data on gradwrtoutput.
        """
        return gradwrtoutput * (1 - (self.input.tanh()) ** 2)

    def param(self):
        """
        No parameter.
        """
        return None

class ReLU(object):
    """
    Activation function: ReLU.
    """
    def forward(self, input):
        """
        forward: apply ReLU() to each element in input data.
        :param input: input data x.
        :return: ReLU(input).
        """
        self.input = input
        return input.relu()

    def backward(self, gradwrtoutput):
        """
        backward: apply gradient of ReLU from input data on gradwrtoutput.
        :param gradwrtoutput: output from last layer in backward path.
        :return: gradient of ReLU from input data on gradwrtoutput.
        """
        grad = self.input
        grad[grad > 0.0] = 1.0
        grad[grad <= 0.0] = 0.0
        return gradwrtoutput * grad

    def param(self):
        """
        No parameter.
        """
        return None

class Sequential(object):
    """
    A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """
    def __init__(self, *layers):
        """
        Create model.
        :param layers: define each layer in model in sequential.
        """
        self.model = []
        for layer in layers:
            self.model.extend(layer)

    def forward(self, input):
        """
        forward: propagate input in forward path in each layer sequentially.
        :param input: input data X.
        :return: output of model: Model(X).
        """
        output = input
        for layer in self.model:
            output = layer.forward(output)
        return output

    def backward(self, gradwrtoutput):
        output = gradwrtoutput
        for layer in reversed(self.model):
            output = layer.backward(output)

    def param(self):
        """
        Give all parameter in the model.
        :return: list of all parameter in model.
        """
        output = []
        for layer in self.model:
            if layer.param() is not None:
                output.extend(layer.param())
        return output

    def gard(self):
        """
        Compute the gradient of final loss w.r.t each parameter.
        :return: gradient w.r.t each parameter.
        """
        output = []
        for layer in self.model:
            if layer.param() is not None:
                output.extend(layer.grad())
        return output

    def update(self, parameters):
        """
        Update the parameter in model.
        """
        i = 0
        for layer in self.model:
            if layer.param() is not None:
                for t in range(len(layer.param())):
                    layer.parameters[t] = parameters[i]
                    i += 1

    def zero_grad(self):
        """
        Set the grad of all network parameters to 0.
        """
        for layer in self.model:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()

class LossMSE(object):
    """
    Compute the Mean Square Error loss.
    """
    def forward(self, pred, label):
        """
        Compute the Mean Square Error loss of pred and label.
        :param pred: prediction given by model.
        :param label: label converted by ground-truth target.
        :return: MSE Loss of pred and label.
        """
        self.pred = pred
        self.label = label
        return (pred - label).pow(2).sum(-1).mean()

    def backward(self):
        """
        Compute the gradient of  MSE Loss of pred and label.
        :return: gradient of  MSE Loss of pred and label.
        """
        return 2.0 * (self.pred - self.label) / self.pred.size(0)