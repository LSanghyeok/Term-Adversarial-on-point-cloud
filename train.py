import torch
import time
import torch.nn.functional as F

from torch.optim import Adam


def run(train_loader, test_loader, model, epochs, batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay, device):

    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_test_acc=0
    
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        train(model, optimizer, train_loader, device)
        test_acc = test(model, test_loader, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        
        if best_test_acc<test_acc:
            best_test_acc=test_acc
            torch.save(model.state_dict(), "saves/{}_{}.pth".format(model._get_name(),epoch))
        print('Epoch: {:03d}, Test: {:.4f}, Best_Test: {:.4f}, Duration: {:.2f}'.format(
            epoch, test_acc, best_test_acc,t_end - t_start))
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']


def train(model, optimizer, train_loader, device):
    model.train()
    losses=0
    total = len(train_loader)
    interval = 100
    total = len(train_loader)//interval
    for i,data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        losses+=loss
        optimizer.step()
        
        if (i+1)%interval==0:
            print("{}/{} loss : {:.4f}".format(int((i+1)/interval), total, losses/interval))
            losses=0

def test(model, test_loader, device):
    model.eval()

    correct = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data.pos, data.batch).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    test_acc = correct / len(test_loader.dataset)

    return test_acc


