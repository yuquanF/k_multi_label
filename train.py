import torch
from torch import nn
from neau_hpc_utils.data import get_data_loader
from neau_hpc_utils.helper import get_current_time, Classification
from config import *
from model import MLC
from tqdm import tqdm


def one_hot(x, b_size):
    res = torch.zeros(b_size, num_classes)
    for idx, v in enumerate(x):
        res[idx][v] = 1

    return torch.Tensor(res)


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    test_loss, correct = 0, 0
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        b_size = len(y)
        y_one_hot = one_hot(y, b_size).to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y_one_hot)

        # save loss and correct per batch
        test_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_loss /= num_batches
    correct /= size
    print(f"Train: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")

    return test_loss, correct


def valid_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            b_size = len(y)
            y_one_hot = one_hot(y, b_size).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y_one_hot).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Valid: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct


if __name__ == '__main__':

    print('=============================loading model=============================')
    # loading model
    model = MLC(num_classes, device, device_id)

    print('=============================loading data=============================')
    # loading data
    train_loader = get_data_loader(annotations_file=train_annotations_file,
                                   size=img_size,
                                   batch_size=batch_size,
                                   shuffle=True)
    valid_loader = get_data_loader(annotations_file=valid_annotations_file,
                                   size=img_size,
                                   batch_size=batch_size)

    print('=============================training=============================')
    # training
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        # 1 train and valid
        train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer, device)
        val_loss, val_acc = valid_loop(valid_loader, model, loss_fn, device)

        # 2 adding log
        print('')
        log_text = "epoch:{} , acc:{:6.4f}, loss:{:6.4f}, val_acc:{:6.4f}, val_loss:{:6.4f}" \
            .format(t + 1, train_acc, train_loss, val_acc, val_loss)
        print('\n', log_text)
        log.add(log_text, train_acc, train_loss, val_acc, val_loss)

    print('=============================saving model parameters=============================')
    # saving model
    now = get_current_time()
    torch.save(model.state_dict(), f'./output/{now}_model.pth')
    print(f"Saved PyTorch Model State to {now}_model.pth")

    print('=============================saving log=============================')
    # saving log
    log.save(f'./output/{model_name}_{now}_log.csv')
