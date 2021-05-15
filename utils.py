from torch import empty
import math

def generate_data(data_size):
    # Generate uniform input
    # input = torch.Tensor(data_size, 2).uniform_(0, 1)
    input = empty(data_size, 2).uniform_(0, 1)
    # Compute target
    # target_ = torch.LongTensor([1 if (i - 0.5).pow(2).sum().item() < 1.0 / (2.0 * math.pi) else 0 for i in input])
    zeros = empty(data_size)
    zeros = zeros.zero_()
    ones = zeros + 1
    target = (input - 0.5).pow(2).sum(1)
    target = target.where(target >= 1.0 / (2.0 * math.pi), zeros)
    target = target.where(target < 1.0 / (2.0 * math.pi), ones)
    target = 1 - target
    target = target.long()
    return input, target

def compute_num_errors(batch_size, model, test_input, test_target):
    num_errors = 0
    for batch_input, batch_target in zip(test_input.split(batch_size), test_target.split(batch_size)):
        # Compute the output of network
        output = model.forward(batch_input)
        # Compute the predictions of network
        _, predicted_classes = output.max(1)
        # Counting the number of incorrect predictions
        for k in range(len(predicted_classes)):
            if batch_target[k] != predicted_classes[k]:
                num_errors = num_errors + 1
    return num_errors