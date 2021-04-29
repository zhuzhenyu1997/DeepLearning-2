from model import *
from utils import *
import torch

class Solver(object):
    def __init__(self, config):
        # Read configs
        self.data_size = config.data_size
        self.batch_size = config.batch_size
        self.epoch_num = config.epoch_num
        self.lr = config.learning_rate
        self.log_path = config.log_path
        self.log_file = config.log_file
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.hidden_size = config.hidden_size
        self.loss = config.loss
        self.normalize = config.normalize
        print(config)

        # Set logs
        self.logs = open("{}/{}".format(self.log_path, self.log_file), mode='w')

        # Set loss
        if self.loss == 'MSE':
            self.criterion = LossMSE()

        # Generate data
        self.train_input, self.train_target = generate_data(self.data_size)
        self.test_input, self.test_target = generate_data(self.data_size)

        # Normalize data
        if self.normalize == True:
            mean, std = self.train_input.mean(), self.train_input.std()
            self.train_input.sub_(mean).div_(std)
            self.test_input.sub_(mean).div_(std)

        # Create model
        self.model = Sequential(
            (Linear(self.input_size, self.hidden_size), ReLU(), Linear(self.hidden_size, self.hidden_size), ReLU(),
             Linear(self.hidden_size, self.hidden_size), ReLU(), Linear(self.hidden_size, self.output_size), Tanh()))

    def train(self):
        # Turn off the auto_grad
        torch.set_grad_enabled(False)
        for epoch in range(self.epoch_num):

            for input, target in zip(self.train_input.split(self.batch_size), self.train_target.split(self.batch_size)):
                # forward
                pred = self.model.forward(input)
                labels = torch.ones(input.size(0), 2) * -1
                labels.scatter_(1, target.unsqueeze(1), 1)

                # mini-batch SGD
                self.criterion.forward(pred, labels)
                self.model.backward(self.criterion.backward())
                param = self.model.param()
                grad = self.model.gard()
                update_param = []
                for p, g in zip(param, grad):
                    update_param.append(p - self.lr * g)
                self.model.update(update_param)

            # Record train loss
            labels = torch.ones(self.train_input.size(0), 2) * -1
            labels.scatter_(1, self.train_target.unsqueeze(1), 1)
            pred = self.model.forward(self.train_input)
            loss = self.criterion.forward(pred, labels).item()
            print('Epoch %d: train loss = %.6f,' % (epoch + 1, loss), end=' ')
            self.logs.write('Epoch %d: train loss = %.6f, ' % (epoch + 1, loss))

            # Record test loss
            labels = torch.ones(self.test_input.size(0), 2) * -1
            labels.scatter_(1, self.test_target.unsqueeze(1), 1)
            pred = self.model.forward(self.test_input)
            loss = self.criterion.forward(pred, labels).item()
            print('test loss = %.6f,' % loss, end=' ')
            self.logs.write('test loss = %.6f, ' % loss)

            # Record train error rate
            train_num_errors = compute_num_errors(self.batch_size, self.model, self.train_input, self.train_target)
            print('train error rate = %.2f%%, ' % (100.0 * train_num_errors / self.data_size), end=' ')
            self.logs.write('train error rate = %.2f%%, ' % (100.0 * train_num_errors / self.data_size))

            # Record test error rate
            test_num_errors = compute_num_errors(self.batch_size, self.model, self.test_input, self.test_target)
            print('test error rate = %.2f%%.\n' % (100.0 * test_num_errors / self.data_size), end='')
            self.logs.write('test error rate = %.2f%%.\n' % (100.0 * test_num_errors / self.data_size))

        self.logs.close()
        print('Finish Training')