#https://www.kaggle.com/code/ihsncnkz/logistic-regression-ann-and-cnn-with-pytorch
#https://www.kaggle.com/datasets/khotijahs1/digitrecognizer/code

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


matplotlib.use('TkAgg')


# Create Artificial Neural Network (ANN)
class ANNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()

        # linear function 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity1
        self.relu1 = nn.ReLU()

        # Linear fuction 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.tanh2 = nn.Tanh()

        # Linear Function 3
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearty 3
        self.elu3 = nn.ELU()

        # Linear Function 4 (readout)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear Function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear Function 2
        out = self.fc2(out)
        # Non-Linearity 2
        out = self.tanh2(out)

        # Linear Function 3
        out = self.fc3(out)
        # Non-linearity 3
        out = self.elu3(out)

        # Linear Function
        out = self.fc4(out)
        return out


if __name__ == '__main__':
    # Load data
    train = pd.read_csv(r"../kaggle/train.csv", dtype=np.float32)

    # I split the dataset into features(pixels) and labels(numbers from 0 to 9)
    y = train.label.values  # target
    x = train.loc[:, train.columns != "label"].values / 255  # normalization features

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # I create a Tensor from a numpy.ndarray.
    X_Train = torch.from_numpy(x_train)
    Y_Train = torch.from_numpy(y_train).type(torch.LongTensor)  # data type is long

    X_Test = torch.from_numpy(x_test)
    Y_Test = torch.from_numpy(y_test).type(torch.LongTensor)  # data type is long

    # I specify the batch size and the epoch.
    batch_size = 100
    n_iters = 10000
    num_epoch = n_iters / (len(x_train) / batch_size)
    num_epoch = int(num_epoch)


    #I specify data for Pytorch.
    train = torch.utils.data.TensorDataset(X_Train, Y_Train)
    test = torch.utils.data.TensorDataset(X_Test, Y_Test)

    # data loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)


    # Determine input_dim_ann, hidden_dim_ann, output_dim_ann
    input_dim_ann = 28 * 28
    hidden_dim_ann = 150
    output_dim_ann = 10

    # Create ann
    ANNmodel = ANNModel(input_dim_ann, hidden_dim_ann, output_dim_ann)

    # Cross Entropy Loss
    ANNerror = nn.CrossEntropyLoss()

    # SGD Optimizer
    learning_rate_ann = 0.2
    optimizer_ann = torch.optim.SGD(ANNmodel.parameters(), lr=learning_rate_ann)

    #I train the model with the dataset.I train the model with the dataset.
    count_ann = 0
    loss_list_ann = []
    iteration_list_ann = []
    accuracy_list_ann = []

    for epoch in range(num_epoch):
        for i, (images_ann, labels_ann) in enumerate(train_loader):

            train = Variable(images_ann.view(-1, 28 * 28))
            labels = Variable(labels_ann)

            # Clear Gradients
            optimizer_ann.zero_grad()

            # Forward Propagation
            outputs = ANNmodel(train)

            # Calculate softmax and ross entropy loss
            loss = ANNerror(outputs, labels)

            # Calculating gradients
            loss.backward()

            # Update parameters
            optimizer_ann.step()

            count_ann += 1

            if count_ann % 50 == 0:
                # Calculate Accuracy
                correct_ann = 0
                total_ann = 0

                # predict test dataset
                for images_ann, labels_ann in test_loader:
                    test = Variable(images_ann.view(-1, 28 * 28))

                    # Forward propagation
                    outputs = ANNmodel(test)

                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]

                    # Total number of labels
                    total_ann += len(labels_ann)

                    # Total correct predictions
                    correct_ann += (predicted == labels_ann).sum()

                accuracy_ann = 100 * correct_ann / float(total_ann)

                # store loss and iteration
                loss_list_ann.append(loss.data)
                iteration_list_ann.append(count_ann)
                accuracy_list_ann.append(accuracy_ann)
            if count_ann % 500 == 0:
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count_ann, loss.data, accuracy_ann))


    # I visualize the results of the model.
    '''plt.plot(iteration_list_ann, loss_list_ann)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("ANN: Loss vs Number of iteration")
    plt.show()

    # visualization accuracy
    plt.plot(iteration_list_ann, accuracy_list_ann, color="red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.title("ANN: Accuracy vs Number of iteration")
    plt.show()'''

    image_path = r"../frames/0.jpg"
    image = Image.open(image_path).convert("RGB")

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the required size
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # Normalize the image (mean values for ImageNet)
            std=[0.229, 0.224, 0.225]     # Standard deviation values for ImageNet
        )
    ])

    # Apply the transformations
    input_tensor = transform(image)

    test = Variable(input_tensor)
    outputs = ANNmodel(test)
    predicted = torch.max(outputs.data, 1)[1]
    print(predicted)


