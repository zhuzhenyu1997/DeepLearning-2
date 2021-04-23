import torch
import math

class Linear(object):
    def __init__(self, in_dim, out_dim, bias=True):
        self.parameters = [torch.empty(out_dim, in_dim)]
        self.parameters[0].normal_(0, math.sqrt(2 / out_dim))
        if bias:
            self.parameters.append(torch.empty(out_dim))
            self.parameters[-1].zero_()

        self.bias = bias
        self.input = None
        self.gradwrtoutput = None

    def forward(self, input):
        self.input = input
        output = input.matmul(self.parameters[0].t())
        if self.bias:
            output += self.parameters[1]
        return output

    def backward(self, gradwrtoutput):
        self.gradwrtoutput = gradwrtoutput
        return gradwrtoutput.matmul(self.parameters[0])

    def grad(self):
        grad_wrt_weights = self.gradwrtoutput.t().matmul(self.input)
        if self.bias:
            grad_wrt_biases = self.gradwrtoutput.sum(0)
            return [grad_wrt_weights, grad_wrt_biases]
        else:
            return grad_wrt_weights

    def param(self):
        return self.parameters

class Tanh(object):
    def forward(self, input):
        self.input = input
        return torch.tanh(input)

    def backward(self, gradwrtoutput):
        return gradwrtoutput * (1 - (self.input.tanh()) ** 2)

    def param(self):
        return None

class ReLU(object):
    def forward(self, input):
        self.input = input
        return input.relu()

    def backward(self, gradwrtoutput):
        grad = self.input
        grad[grad > 0.0] = 1.0
        grad[grad <= 0.0] = 0.0
        return gradwrtoutput * grad

    def param(self):
        return None

class Sequential(object):

    def __init__(self, *layers):
        self.model = []
        for layer in layers:
            self.model.extend(layer)

    def forward(self, input):
        output = input
        for layer in self.model:
            output = layer.forward(output)
        return output

    def backward(self, gradwrtoutput):
        output = gradwrtoutput
        for layer in reversed(self.model):
            output = layer.backward(output)

    def param(self):
        output = []
        for layer in self.model:
            if layer.param() is not None:
                output.extend(layer.param())
        return output

    def gard(self):
        output = []
        for layer in self.model:
            if layer.param() is not None:
                output.extend(layer.grad())
        return output

    def update(self, parameters):
        i = 0
        for layer in self.model:
            if layer.param() is not None:
                for t in range(len(layer.param())):
                    layer.parameters[t] = parameters[i]
                    i += 1

    def zero_grad(self):
        for layer in self.model:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()

class LossMSE(object):
    def forward(self, pred, label):
        self.pred = pred
        self.label = label
        return (pred - label).pow(2).sum(-1).mean()

    def backward(self):
        return 2.0 * (self.pred - self.label) / self.pred.size(0)