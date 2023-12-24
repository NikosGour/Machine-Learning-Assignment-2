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
            current_loss = loss(y_pred, y.long())

            current_loss.backward()

            optimizer.step()

            total += y.size(0)
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            running_loss += current_loss.item()
            if batch % print_period == print_period - 1:
                avg_loss = running_loss / print_period
                train_loss.append(avg_loss)
                train_accuracy.append(correct / total * 100)
                correct_val = 0
                loss_val = 0.0
                if valloader is not None:
                    model.eval()
                    with torch.no_grad():
                        for (X_val, y_val) in valloader:
                            X_val = X_val.to(device)
                            y_val = y_val.to(device)

                            y_pred_val = model(X_val)
                            loss_val += loss(y_pred_val, y_val.long())
                            correct_val += (y_pred_val.argmax(1) == y_val).type(torch.float).sum().item()

                    val_loss.append(loss_val/len(valloader.dataset))
                    val_accuracy.append(correct_val / len(valloader.dataset) * 100)
                    writer.add_scalar('Loss/train', avg_loss, (epoch+1) * len(trainloader) + batch)
                    print(
                        f'[{epoch + 1}, {batch + 1}] loss: {avg_loss} | accuracy: {correct / total * 100:.2f}% | val_loss: {loss_val/len(valloader.dataset)} | val_accuracy: {correct_val / len(valloader.dataset) * 100:.2f}%')
                else:
                    print(f'[{epoch + 1}, {batch + 1}] loss: {avg_loss} | accuracy: {correct / total * 100 :.4f}%')
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

    print(f"Test accuracy: {correct / len(testloader.dataset) * 100:.2f}% | Test loss: {loss_test / len(testloader.dataset)}")

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
    val_loss = [x.cpu() for x in val_loss]

    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Training Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    # show the figure
    fig.show()
    fig.savefig('loss.png')

    # train_accuracy = [x.item() for x in train_accuracy]
    # val_accuracy = [x.cpu() for x in val_accuracy]

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
    optimizer = optim.Adam(model.parameters(), lr=10**-3)
    loss = nn.CrossEntropyLoss()
    writer = SummaryWriter("runs/MLProject2Complex")
    train_net(model, trainloader,valloader,epochs=20, optimizer=optimizer, loss=loss, device=device)
    test_net(model, testloader, loss=loss, device=device)

    # train_loss = [x.item() for x in train_loss]
    val_loss = [x.cpu() for x in val_loss]

    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Training Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    # show the figure
    fig.show()
    fig.savefig('loss.png')

    # train_accuracy = [x.item() for x in train_accuracy]
    # val_accuracy = [x.cpu() for x in val_accuracy]

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


    writer.close()
    torch.save(model.state_dict(), 'complexmodel.pth')
