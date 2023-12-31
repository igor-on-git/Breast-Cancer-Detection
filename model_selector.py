from utils import *
from torch import nn
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models


def model_selector(model_name):

    # default values
    batch_size = 128
    train_stop_criteria = 'valid loss'  # 'F1 score'
    train_stop_patience = 5

    if model_name == 'resnet152_v1':
        learn_rate = 0.0001
        dropout_prob = 0.3
        model = models.resnet152(weights="IMAGENET1K_V2")
        for param in model.parameters(): # Freeze parameters so we don't backprop through them
            param.requires_grad = False
        fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2048, 512)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(dropout_prob)),
            ('fc2', nn.Linear(512, 128)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(dropout_prob)),
            ('fc3', nn.Linear(128, 32)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(dropout_prob)),
            ('fc4', nn.Linear(32, 2)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.fc = fc
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()

    elif model_name == 'resnet152_v2':
        learn_rate = 0.0001
        dropout_prob = 0.3
        model = models.resnet152(weights="IMAGENET1K_V2")
        for param in model.parameters(): # Freeze parameters so we don't backprop through them
            param.requires_grad = False
        fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2048, 128)),
            ('relu', nn.ReLU()),
            ('norm1', nn.BatchNorm1d(128)),
            ('fc2', nn.Linear(128, 64)),
            ('relu', nn.ReLU()),
            ('norm2', nn.BatchNorm1d(64)),
            ('fc3', nn.Linear(64, 64)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(dropout_prob)),
            ('fc4', nn.Linear(64, 24)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(dropout_prob)),
            ('fc5', nn.Linear(24, 2)),
            ('relu', nn.ReLU()),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.fc = fc
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()

    elif model_name == 'resnet152_v3':
        learn_rate = 0.0001
        dropout_prob = 0.3
        model = models.resnet152(weights="IMAGENET1K_V2")
        for param in model.parameters(): # Freeze parameters so we don't backprop through them
            param.requires_grad = False
        fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2048, 128)),
            ('relu', nn.ReLU()),
            ('norm1', nn.BatchNorm1d(128)),
            ('drop', nn.Dropout(dropout_prob)),
            ('fc2', nn.Linear(128, 64)),
            ('relu', nn.ReLU()),
            ('norm2', nn.BatchNorm1d(64)),
            ('drop', nn.Dropout(dropout_prob)),
            ('fc3', nn.Linear(64, 64)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(dropout_prob)),
            ('fc4', nn.Linear(64, 24)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(dropout_prob)),
            ('fc5', nn.Linear(24, 2)),
            ('relu', nn.ReLU()),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.fc = fc
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()

    elif model_name == 'densenet121':

        learn_rate = 0.0001
        dropout_prob = 0.2
        model = models.densenet121(weights='IMAGENET1K_V1')
        for param in model.parameters(): # Freeze parameters so we don't backprop through them
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 256)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(dropout_prob)),
            ('fc2', nn.Linear(256, 2)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()

    elif model_name == 'IgorNetKaggle':

        dropout_prob = 0.3
        learn_rate = 0.0001

        model = IgorNetKaggle(dropout_prob)
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()


    elif model_name == 'IgorNet_v1': #no batch norm, dropout every second layer

        dropout_prob = 0.3
        learn_rate = 0.0001

        model = IgorNet_v1(dropout_prob)
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()

    elif model_name == 'IgorNet_v2': #no batch norm, dropout every layer

        dropout_prob = 0.3
        learn_rate = 0.0001

        model = IgorNet_v2(dropout_prob)
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()

    elif model_name == 'IgorNet_v3': #dropout every second layer batch norm every other layer

        dropout_prob = 0.3
        learn_rate = 0.0001

        model = IgorNet_v3(dropout_prob)
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()

    elif model_name == 'IgorNet_v4': #dropout every second layer

        dropout_prob = 0.3
        learn_rate = 0.0001

        model = IgorNet_v4(dropout_prob)
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()

    elif model_name == 'IgorNet_vold': #no batch norm, dropout every second layer

        dropout_prob = 0.3
        learn_rate = 0.0001

        model = IgorNet_vold(dropout_prob)
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        criterion = nn.NLLLoss()

    else:
        print('Error - wrong model specified')
        model = optimizer = criterion = 0

    os.makedirs('models/' + model_name, exist_ok=True)

    return model, optimizer, criterion, batch_size, learn_rate, train_stop_criteria, train_stop_patience


class IgorNet_v1(nn.Module):
    def __init__(self, dropout_prob):
        super(IgorNet_v1, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # input 3 x 50 x 50 output: 64 x 25 x 25
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 12 x 12
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 1024 x 6 x 6
            nn.Dropout2d(dropout_prob),

            nn.Flatten(),
            nn.Linear(1024 * 6 * 6, 2 * 1024),
            nn.ReLU(),
            nn.Linear(2 * 1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)


class IgorNet_v2(nn.Module):
    def __init__(self, dropout_prob):
        super(IgorNet_v2, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # input 3 x 50 x 50 output: 64 x 25 x 25
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 12 x 12
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 1024 x 6 x 6
            nn.Dropout2d(dropout_prob),

            nn.Flatten(),
            nn.Linear(1024 * 6 * 6, 2 * 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(2 * 1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)


class IgorNet_v3(nn.Module):
    def __init__(self, dropout_prob):
        super(IgorNet_v3, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # input 3 x 50 x 50 output: 64 x 25 x 25
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 12 x 12
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 1024 x 6 x 6
            nn.Dropout2d(dropout_prob),

            nn.Flatten(),
            nn.Linear(1024 * 6 * 6, 2 * 1024),
            nn.ReLU(),
            nn.BatchNorm1d(2 * 1024),
            nn.Linear(2 * 1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

class IgorNet_v4(nn.Module):
    def __init__(self, dropout_prob):
        super(IgorNet_v4, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # input 3 x 50 x 50 output: 64 x 25 x 25
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 12 x 12
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 1024 x 6 x 6
            nn.Dropout2d(dropout_prob),

            nn.Flatten(),
            nn.Linear(1024 * 6 * 6, 2 * 1024),
            nn.ReLU(),
            nn.Linear(2 * 1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

class IgorNet_vold(nn.Module):
    def __init__(self, dropout_prob):
        super(IgorNet_vold, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # input 3 x 50 x 50 output: 64 x 25 x 25
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # output: 256 x 12 x 12
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2, 2),  # output: 1024 x 6 x 6
            nn.Dropout2d(dropout_prob),

            nn.Flatten(),
            nn.Linear(1024 * 6 * 6, 2 * 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(2 * 1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)


class IgorNetKaggle(nn.Module):
    def __init__(self, dropout_prob):
        super(IgorNetKaggle, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # input 3 x 50 x 50
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # input 32 x 50 x 50 output: 32 x 25 x 25
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # input 64 x 25 x 25 output: 64 x 12 x 12
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 128), # input 128 x 12 x 12
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 2),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)
