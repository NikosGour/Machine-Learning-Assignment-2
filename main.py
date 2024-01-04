import torchvision.transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ComplexConvNN import ComplexConvNN
from MLProject2Dataset import MLProject2Dataset
from SimpleConvNN import SimpleConvNN
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

val_loss = []
train_loss = []
val_accuracy = []
train_accuracy = []
def train_net(model: nn.Module, trainloader: DataLoader, valloader: DataLoader = None, epochs: int = 10,
              optimizer: optim.Optimizer = None, loss: nn.modules.loss = None, device: str = 'cpu',
              print_period: int = 10) -> None:
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


def test_net(model: nn.Module, testloader: DataLoader, loss: nn.modules.loss = None, device: str = 'cpu') -> None:
    model.eval()
    correct = 0
    loss_test = 0.0
    with torch.no_grad():
        for (X, y) in testloader:
            X = X.to(device)
            y = y.to(device)

            y_pred_test = model(X)
            loss_test += loss(y_pred_test, y.long())
            correct += (y_pred_test.argmax(1) == y).type(torch.float).sum().item()

    print(f"Test accuracy: {correct / len(testloader.dataset) * 100:.2f}% | Test loss: {loss_test / len(testloader)}")

SimpleModel = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
if SimpleModel:
    M = 50
    N = 62
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((M, N),antialias=False),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = MLProject2Dataset(data_dir='data', metadata_fname='metadata.csv', transform=transforms)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [.6, .1, .3],
                                                                             generator=torch.Generator().manual_seed(42))

    BATCH_SIZE = 32
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleConvNN()
    model = model.to(device)
    summary(model, (3, M, N))
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()
    writer = SummaryWriter("runs/MLProject2Simple")
    train_net(model, trainloader,valloader,epochs=20, optimizer=optimizer, loss=loss, device=device)
    test_net(model, testloader, loss=loss, device=device)
    writer.close()

    fig, ax = plt.subplots()
    ax.plot(map(lambda x:x*10,range(len(train_loss))),train_loss, label='Training Loss')
    ax.plot(map(lambda x:x*10,range(len(train_loss))),val_loss, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    # show the figure
    fig.show()
    fig.savefig('loss.png')

    fig, ax = plt.subplots()
    ax.plot(train_accuracy, label='Training Accuracy')
    ax.plot(val_accuracy, label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    # show the figure
    fig.show()
    fig.savefig('accuracy.png')
    torch.save(model.state_dict(), 'simplemodel.pth')
else:
    M = 100
    N = 125
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((M, N),antialias=False),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = MLProject2Dataset(data_dir='data', metadata_fname='metadata.csv', transform=transforms)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [.6, .1, .3],
                                                                             generator=torch.Generator().manual_seed(42))

    BATCH_SIZE = 32
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ComplexConvNN()
    model = model.to(device)
    summary(model, (3, M, N))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()
    writer = SummaryWriter("runs/MLProject2Complex")
    train_net(model, trainloader,valloader,epochs=20, optimizer=optimizer, loss=loss, device=device)
    test_net(model, testloader, loss=loss, device=device)

    # train_loss = [x.item() for x in train_loss]
    # val_loss = [x.cpu() for x in val_loss]

    x_axis = list(map(lambda x: x * 10, range(len(train_loss))))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_loss, label='Training Loss')
    plt.plot(x_axis, val_loss, label='Validation Loss')
    xticks = np.arange(x_axis[0], x_axis[-1] + 1, 100.0)
    plt.xticks([*xticks, x_axis[-1]])
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()
    # show the figure

    plt.savefig('loss.png')

    # train_accuracy = [x.item() for x in train_accuracy]
    # val_accuracy = [x.cpu() for x in val_accuracy]

    fig, ax = plt.subplots()
    x_axis = list(map(lambda x:x*10,range(len(train_accuracy))))
    ax.plot(x_axis ,train_accuracy, label='Training Accuracy')
    ax.plot(x_axis ,val_accuracy, label='Validation Accuracy')
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    # show the figure
    fig.show()
    fig.savefig('accuracy.png')


    writer.close()
    torch.save(model.state_dict(), 'complexmodel.pth')

ResNet = False
if ResNet:
    import torchvision.transforms as transforms
    data_transforms = {
        'train': torchvision.transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )]),
        'val': torchvision.transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )])
        ,
    }

    dataset = MLProject2Dataset(data_dir='data', metadata_fname='metadata.csv')
    train_dataset, val_test_dataset = torch.utils.data.random_split(dataset, [.6, .4],
                                                                             generator=torch.Generator().manual_seed(42))
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [.25, .75],generator=torch.Generator().manual_seed(42))

    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['val']

    BATCH_SIZE = 32
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = torchvision.models.resnet34(weights="DEFAULT")
    model = model.to(device)
    summary(model, (3, 224, 224))
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)

    loss = nn.CrossEntropyLoss()
    train_net(model, trainloader,valloader,epochs=5, optimizer=optimizer, loss=loss, device=device)
    test_net(model, testloader, loss=loss, device=device)

    fig, ax = plt.subplots()

    ax.plot(train_loss, label='Training Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    # show the figure
    fig.show()
    fig.savefig('resnet_loss.png')

    fig, ax = plt.subplots()
    ax.plot(train_accuracy, label='Training Accuracy')
    ax.plot(val_accuracy, label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    # show the figure
    fig.show()
    fig.savefig('resnet_accuracy.png')