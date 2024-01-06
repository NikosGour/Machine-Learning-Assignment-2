import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import torchvision.transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from MLProject2Dataset import MLProject2Dataset
from SimpleConvNN import SimpleConvNN
from ComplexConvNN import ComplexConvNN
from MLProject2DatasetBonus import MLProject2DatasetBonus
from ComplexConvNNBonus import ComplexConvNNBonus

import time

val_loss = []
train_loss = []
val_accuracy = []
train_accuracy = []


def train_net(model: nn.Module, trainloader: DataLoader, valloader: DataLoader = None, epochs: int = 10,
              optimizer: optim.Optimizer = None, loss: nn.modules.loss = None, device: str = 'cpu',
              print_period: int = 10) -> None:
    global val_loss, train_loss, val_accuracy, train_accuracy
    val_loss = []
    train_loss = []
    val_accuracy = []
    train_accuracy = []
    for epoch in range(epochs):
        total = 0
        correct = 0
        running_loss = 0.0
        for batch, (X, y) in enumerate(trainloader, 0):
            model.train()
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred = model(X)
            # If I don't cast y to long, I get an error
            # RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Char'
            # solution: https://stackoverflow.com/a/71126544/13250408
            current_loss = loss(y_pred, y.long())

            current_loss.backward()

            optimizer.step()

            total += y.size(0)
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            running_loss += current_loss.item()
            if batch % print_period == print_period - 1:
                avg_loss = running_loss / print_period
                avg_accuracy = correct / total * 100
                train_loss.append(avg_loss)
                train_accuracy.append(avg_accuracy)

                total_val = 0
                correct_val = 0
                running_loss_val = 0.0

                if valloader is not None:
                    model.eval()
                    with torch.no_grad():
                        for (X_val, y_val) in valloader:
                            X_val = X_val.to(device)
                            y_val = y_val.to(device)

                            y_pred_val = model(X_val)
                            running_loss_val += loss(y_pred_val, y_val.long()).item()
                            total_val += y_val.size(0)
                            correct_val += (y_pred_val.argmax(1) == y_val).type(torch.float).sum().item()

                    avg_loss_val = running_loss_val / len(valloader)
                    avg_accuracy_val = correct_val / total_val * 100
                    val_loss.append(avg_loss_val)
                    val_accuracy.append(avg_accuracy_val)
                    # writer.add_scalar('Loss/train', avg_loss, (epoch+1) * len(trainloader) + batch)

                    print(
                        f'[{epoch + 1}, {batch + 1}] loss: {avg_loss} | accuracy: {avg_accuracy:.2f}% | val_loss: {avg_loss_val} | val_accuracy: {avg_accuracy_val:.2f}%')
                else:
                    print(f'[{epoch + 1}, {batch + 1}] loss: {avg_loss} | accuracy: {avg_accuracy :.4f}%')
                running_loss = 0.0
                total = 0
                correct = 0


def train_bonus_net(model: ComplexConvNNBonus, trainloader: DataLoader, valloader: DataLoader = None, epochs: int = 10,
                    optimizer: optim.Optimizer = None, loss: nn.modules.loss = None, device: str = 'cpu',
                    print_period: int = 10) -> None:
    global val_loss, train_loss, val_accuracy, train_accuracy
    val_loss = []
    train_loss = []
    val_accuracy = []
    train_accuracy = []
    for epoch in range(epochs):
        total = 0
        correct = 0
        running_loss = 0.0
        for batch, (X, p, y) in enumerate(trainloader, 0):
            model.train()
            X = X.to(device)
            y = y.to(device)
            p = p.to(device)

            optimizer.zero_grad()

            y_pred = model(X, p)
            # If I don't cast y to long, I get an error
            # RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Char'
            # solution: https://stackoverflow.com/a/71126544/13250408
            current_loss = loss(y_pred, y.long())

            current_loss.backward()

            optimizer.step()

            total += y.size(0)
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            running_loss += current_loss.item()
            if batch % print_period == print_period - 1:
                avg_loss = running_loss / print_period
                avg_accuracy = correct / total * 100
                train_loss.append(avg_loss)
                train_accuracy.append(avg_accuracy)

                total_val = 0
                correct_val = 0
                running_loss_val = 0.0

                if valloader is not None:
                    model.eval()
                    with torch.no_grad():
                        for (X_val, p_val, y_val) in valloader:
                            X_val = X_val.to(device)
                            y_val = y_val.to(device)
                            p_val = p_val.to(device)

                            y_pred_val = model(X_val, p_val)
                            running_loss_val += loss(y_pred_val, y_val.long()).item()
                            total_val += y_val.size(0)
                            correct_val += (y_pred_val.argmax(1) == y_val).type(torch.float).sum().item()

                    avg_loss_val = running_loss_val / len(valloader)
                    avg_accuracy_val = correct_val / total_val * 100
                    val_loss.append(avg_loss_val)
                    val_accuracy.append(avg_accuracy_val)
                    # writer.add_scalar('Loss/train', avg_loss, (epoch+1) * len(trainloader) + batch)

                    print(
                        f'[{epoch + 1}, {batch + 1}] loss: {avg_loss} | accuracy: {avg_accuracy:.2f}% | val_loss: {avg_loss_val} | val_accuracy: {avg_accuracy_val:.2f}%')
                else:
                    print(f'[{epoch + 1}, {batch + 1}] loss: {avg_loss} | accuracy: {avg_accuracy :.4f}%')
                running_loss = 0.0
                total = 0
                correct = 0


def test_bonus_net(model: ComplexConvNNBonus, testloader: DataLoader, number_to_label_array: list[str],
                   loss: nn.modules.loss = None, device: str = 'cpu',
                   model_name="") -> None:
    model.eval()
    correct = 0
    loss_test = 0.0
    y_trues = []
    y_preds = []
    with torch.no_grad():
        for (X, p, y) in testloader:
            X = X.to(device)
            y = y.to(device)
            p = p.to(device)

            y_pred_test = model(X, p)
            y_trues.extend(y.cpu().numpy())
            y_preds.extend(y_pred_test.argmax(1).cpu().numpy())
            loss_test += loss(y_pred_test, y.long())
            correct += (y_pred_test.argmax(1) == y).type(torch.float).sum().item()

    print(f"Test accuracy: {correct / len(testloader.dataset) * 100:.2f}% | Test loss: {loss_test / len(testloader)}")
    conf_matrix = confusion_matrix(y_trues, y_preds)
    df_cm = pd.DataFrame(conf_matrix, index=number_to_label_array,
                         columns=number_to_label_array)
    # Plot the confusion matrix
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'results/confusion_matrices/{model_name}_confusion_matrix.png')
    plt.show()


def test_net(model: nn.Module, testloader: DataLoader, number_to_label_array: list[str],
             loss: nn.modules.loss = None, device: str = 'cpu',
             model_name="") -> None:
    model.eval()
    correct = 0
    loss_test = 0.0
    y_trues = []
    y_preds = []
    with torch.no_grad():
        for (X, y) in testloader:
            X = X.to(device)
            y = y.to(device)

            y_pred_test = model(X)
            y_trues.extend(y.cpu().numpy())
            y_preds.extend(y_pred_test.argmax(1).cpu().numpy())
            loss_test += loss(y_pred_test, y.long())
            correct += (y_pred_test.argmax(1) == y).type(torch.float).sum().item()

    print(f"Test accuracy: {correct / len(testloader.dataset) * 100:.2f}% | Test loss: {loss_test / len(testloader)}")
    conf_matrix = confusion_matrix(y_trues, y_preds)
    df_cm = pd.DataFrame(conf_matrix, index=number_to_label_array,
                         columns=number_to_label_array)
    # Plot the confusion matrix
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'results/confusion_matrices/{model_name}_confusion_matrix.png')
    plt.show()


def train_simple():
    print("Training Simple")
    # Creating image side sizes
    M = 50
    N = 62
    # Creating transforms
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((M, N), antialias=False),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Creating dataset
    dataset = MLProject2Dataset(data_dir='data', metadata_fname='metadata.csv', transform=transforms)
    # Splitting dataset into train, validation and test
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [.6, .1, .3],
                                                                             generator=torch.Generator().manual_seed(
                                                                                 42))

    # Creating dataloaders
    BATCH_SIZE = 32
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Creating model
    model = SimpleConvNN()
    model = model.to(device)
    # Printing model summary
    summary(model, (3, M, N))
    # Creating optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # Creating loss function
    loss = nn.CrossEntropyLoss()
    # Creating tensorboard writer
    writer = SummaryWriter("runs/MLProject2Simple")
    # Training model
    train_net(model, trainloader, valloader, epochs=20, optimizer=optimizer, loss=loss, device=device)
    number_to_label_array = testloader.dataset.dataset.number_to_label_array
    # Testing model
    test_net(model, testloader, number_to_label_array, loss=loss, device=device, model_name="simple")
    # Closing tensorboard writer
    writer.close()

    # Plotting losses
    x_axis = list(map(lambda x: x * 10, range(len(train_loss))))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_loss, label='Training Loss')
    plt.plot(x_axis, val_loss, label='Validation Loss')
    xticks = np.arange(x_axis[0], x_axis[-1] + 1, 200.0)
    plt.xticks([*xticks, x_axis[-1]])
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('results/losses/simple_loss.png')
    plt.show()

    # Plotting accuracies
    x_axis = list(map(lambda x: x * 10, range(len(train_accuracy))))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_accuracy, label='Training Accuracy')
    plt.plot(x_axis, val_accuracy, label='Validation Accuracy')
    xticks = np.arange(x_axis[0], x_axis[-1] + 1, 200.0)
    plt.xticks([*xticks, x_axis[-1]])
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('results/accuracies/simple_accuracy.png')
    plt.show()
    # Saving model
    torch.save(model.state_dict(), 'results/model_states/simple_model.pth')


def train_complex():
    print("Training Complex")
    # Creating image side sizes
    M = 100
    N = 125
    # Creating transforms
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((M, N), antialias=False),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Creating dataset
    dataset = MLProject2Dataset(data_dir='data', metadata_fname='metadata.csv', transform=transforms)
    # Splitting dataset into train, validation and test
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [.6, .1, .3],
                                                                             generator=torch.Generator().manual_seed(
                                                                                 42))
    # Creating dataloaders
    BATCH_SIZE = 32
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Creating model
    model = ComplexConvNN()
    model = model.to(device)
    # Printing model summary
    summary(model, (3, M, N))
    # Creating optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Creating loss function
    loss = nn.CrossEntropyLoss()
    # Creating tensorboard writer
    writer = SummaryWriter("runs/MLProject2Complex")
    # Training model
    train_net(model, trainloader, valloader, epochs=20, optimizer=optimizer, loss=loss, device=device)
    number_to_label_array = testloader.dataset.dataset.number_to_label_array
    # Testing model
    test_net(model, testloader, number_to_label_array, loss=loss, device=device, model_name="complex")
    # Closing tensorboard writer
    writer.close()

    # Plotting losses
    x_axis = list(map(lambda x: x * 10, range(len(train_loss))))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_loss, label='Training Loss')
    plt.plot(x_axis, val_loss, label='Validation Loss')
    xticks = np.arange(x_axis[0], x_axis[-1] + 1, 400.0)
    plt.xticks([*xticks, x_axis[-1]])
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('results/losses/complex_loss.png')
    plt.show()
    # Plotting accuracies
    x_axis = list(map(lambda x: x * 10, range(len(train_accuracy))))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_accuracy, label='Training Accuracy')
    plt.plot(x_axis, val_accuracy, label='Validation Accuracy')
    xticks = np.arange(x_axis[0], x_axis[-1] + 1, 200.0)
    plt.xticks([*xticks, x_axis[-1]])
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('results/accuracies/complex_accuracy.png')
    plt.show()
    # Saving model
    torch.save(model.state_dict(), 'results/model_states/complex_model.pth')


def train_resnet():
    print("Training ResNet")
    # Creating transforms
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        ,
    }

    # Creating dataset
    dataset = MLProject2Dataset(data_dir='data', metadata_fname='metadata.csv')
    # Splitting dataset into train, validation and test
    train_dataset, val_test_dataset = torch.utils.data.random_split(dataset, [.6, .4],
                                                                    generator=torch.Generator().manual_seed(42))
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [.25, .75],
                                                              generator=torch.Generator().manual_seed(42))

    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['val']

    # Creating dataloaders
    BATCH_SIZE = 32
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Creating model
    model = torchvision.models.resnet34(weights="DEFAULT")
    # Changing last layer to have 7 outputs instead of 1000
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 7)

    model = model.to(device)
    # Printing model summary
    summary(model, (3, 224, 224))
    # Creating optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Creating loss function
    loss = nn.CrossEntropyLoss()
    # Training model
    train_net(model, trainloader, valloader, epochs=5, optimizer=optimizer, loss=loss, device=device)
    number_to_label_array = testloader.dataset.dataset.dataset.number_to_label_array
    # Testing model
    test_net(model, testloader, number_to_label_array, loss=loss, device=device, model_name="resnet")

    # Plotting losses
    x_axis = list(map(lambda x: x * 10, range(len(train_loss))))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_loss, label='Training Loss')
    plt.plot(x_axis, val_loss, label='Validation Loss')
    xticks = np.arange(x_axis[0], x_axis[-1] + 1, 400.0)
    plt.xticks([*xticks, x_axis[-1]])
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('results/losses/resnet_loss.png')
    plt.show()

    # Plotting accuracies
    x_axis = list(map(lambda x: x * 10, range(len(train_accuracy))))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_accuracy, label='Training Accuracy')
    plt.plot(x_axis, val_accuracy, label='Validation Accuracy')
    xticks = np.arange(x_axis[0], x_axis[-1] + 1, 100.0)
    plt.xticks([*xticks, x_axis[-1]])
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('results/accuracies/resnet_accuracy.png')
    plt.show()

    # Saving model
    torch.save(model.state_dict(), 'results/model_states/resnet.pth')


def bonus():
    print("Training Bonus")
    # Creating image side sizes
    M = 100
    N = 125
    # Creating transforms
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((M, N), antialias=False),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Creating dataset
    dataset = MLProject2DatasetBonus(data_dir='data', metadata_fname='metadata.csv', transform=transforms)
    # Splitting dataset into train, validation and test
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [.6, .1, .3],
                                                                             generator=torch.Generator().manual_seed(
                                                                                 42))
    # Creating dataloaders
    BATCH_SIZE = 32
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Creating model
    model = ComplexConvNNBonus()
    model = model.to(device)
    # Printing model summary
    # summary(model, ((3, 100, 125),18))
    # Creating optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Creating loss function
    loss = nn.CrossEntropyLoss()
    # Creating tensorboard writer
    writer = SummaryWriter("runs/MLProject2Bonus")
    # Training model
    train_bonus_net(model, trainloader, valloader, epochs=20, optimizer=optimizer, loss=loss, device=device)
    number_to_label_array = testloader.dataset.dataset.number_to_label_array
    # Testing model
    test_bonus_net(model, testloader, number_to_label_array, loss=loss, device=device, model_name="bonus")
    # Closing tensorboard writer
    writer.close()

    # Plotting losses
    x_axis = list(map(lambda x: x * 10, range(len(train_loss))))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_loss, label='Training Loss')
    plt.plot(x_axis, val_loss, label='Validation Loss')
    xticks = np.arange(x_axis[0], x_axis[-1] + 1, 400.0)
    plt.xticks([*xticks, x_axis[-1]])
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('results/losses/bonus_loss.png')
    plt.show()
    # Plotting accuracies
    x_axis = list(map(lambda x: x * 10, range(len(train_accuracy))))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_accuracy, label='Training Accuracy')
    plt.plot(x_axis, val_accuracy, label='Validation Accuracy')
    xticks = np.arange(x_axis[0], x_axis[-1] + 1, 200.0)
    plt.xticks([*xticks, x_axis[-1]])
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('results/accuracies/bonus_accuracy.png')
    plt.show()
    # Saving model
    torch.save(model.state_dict(), 'results/model_states/bonus_model.pth')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    start = time.time()
    # train_simple()
    # train_complex()
    train_resnet()
    # bonus()
    print(f"Total time: {time.time() - start} seconds")
