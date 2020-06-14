import torch
import time
import torch.nn.functional as F

from torch.optim import Adam
from attack import fgsm_attack

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

                
def train(model, optimizer, train_loader, eps,device):
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


def run_defense(train_loader, test_loader, model, model_a, epochs, batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay, eps,device):

    model = model.to(device)
    model_a = model_a.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_test_acc=0
    
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        train_defense(model, model_a, optimizer, train_loader, eps, device)
        test_cls_acc, test_d_acc = test_defense(model, model_a, test_loader, eps, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        
        if best_test_acc<test_cls_acc:
            best_test_acc=test_cls_acc
            torch.save(model.state_dict(), "saves/{}_{}.pth".format(model._get_name(),epoch))
        print('Epoch: {:03d}, Test_cls: {:.4f}, Test_D: {:.4f}, Best_Test_cls: {:.4f}, Duration: {:.2f}'.format(
            epoch, test_cls_acc, test_d_acc, best_test_acc,t_end - t_start))
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

def train_defense(model, model_a, optimizer, train_loader, eps, device):
    model.train()
    losses=0
    losses2=0
    losses3=0

    total = len(train_loader)
    interval = 50
    total = len(train_loader)//interval
    
    for i,data in enumerate(train_loader):
        data = data.to(device)

        #Generate attacked data
        data.pos.requires_grad = True
        output = model_a(data.pos, data.batch)
        pred = output.max(1)[1]
        keep = pred.eq(data.y)
        loss = F.nll_loss(output, data.y)
        model_a.zero_grad()
        loss.backward()
        data_grad = data.pos.grad.data
        data_grad = data_grad.reshape(len(data.y), -1, 3)
        data_grad[~keep] = 0 #initialize alreay incorrect answer's grad to 0
        data_grad = data_grad.reshape(-1, 3)
        perturbed_data = fgsm_attack(data.pos, eps, data_grad)
        
        batch = torch.cat([data.batch, data.batch+len(data.y)])
        pos = torch.cat([data.pos, perturbed_data.to(device)],0)
    
        #train classifier
        optimizer.zero_grad()
        cls_score, D_score = model(pos.detach(), batch)
        loss = F.nll_loss(cls_score, torch.cat([data.y,data.y]))
        losses+=loss
        loss.backward()
        optimizer.step()
        
        #train discirminator
        optimizer.zero_grad()
        model.features.eval()
        model.discriminator.train()
        _, D_score = model(pos.detach(), batch)
        target = torch.cat([torch.zeros(len(data.y), dtype=torch.long), torch.ones(len(data.y), dtype=torch.long)]).to(device)
        loss2 = F.nll_loss(D_score, target)
        loss2.backward()
        losses2+=loss2
        optimizer.step()
        
        #train Feature generator
        optimizer.zero_grad()
        model.features.train()
        model.discriminator.eval()
        _, D_score = model(pos, batch)
        loss3 = F.nll_loss(D_score, torch.zeros(2*len(data.y), dtype=torch.long).to(device))
        loss3.backward()
        losses3+=loss3
        optimizer.step()

        if (i+1)%interval==0:
            print("{}/{} cls_loss : {:.4f}, discriminator_loss : {:.4f}, feature_loss : {:.4f} ".format(int((i+1)/interval), total, losses/interval, losses2/interval,  losses3/interval))
            losses=0
            losses2=0
            losses3=0
            
def test_defense(model, model_a, test_loader, eps, device):
    model.eval()

    correct = 0
    correct2 = 0
    
    for i,data in enumerate(test_loader):
        data = data.to(device)

        #Generate attacked data
        data.pos.requires_grad = True
        output = model_a(data.pos, data.batch)
        pred = output.max(1)[1]
        keep = pred.eq(data.y)
        loss = F.nll_loss(output, data.y)
        model_a.zero_grad()
        loss.backward()
        data_grad = data.pos.grad.data
        data_grad = data_grad.reshape(len(data.y), -1, 3)
        data_grad[~keep] = 0 #initialize alreay incorrect answer's grad to 0
        data_grad = data_grad.reshape(-1, 3)
        perturbed_data = fgsm_attack(data.pos, eps, data_grad)
        
        batch = torch.cat([data.batch, data.batch+len(data.y)])
        pos = torch.cat([data.pos, perturbed_data.to(device)],0)
        
        pred_cls_score, pred_D_score = model(pos, batch)
        
        pred_cls = pred_cls_score[:len(data.y)].max(1)[1]
        correct += pred_cls.eq(data.y).sum().item()
        
        target = torch.cat([torch.zeros(len(data.y), dtype=torch.long), torch.ones(len(data.y), dtype=torch.long)]).to(device)
        
        pred_D = pred_D_score.max(1)[1]
        correct2 += pred_D.eq(target).sum().item()
        
    cls_acc = correct / len(test_loader.dataset)
    D_acc = correct2 / (2*len(test_loader.dataset))
    
    return cls_acc, D_acc