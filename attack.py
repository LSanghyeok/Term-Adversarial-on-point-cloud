import torch
import torch.nn.functional as F

def fgsm_attack(data, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_data = data + epsilon*sign_data_grad
    # Adding clipping to maintain [-1,1] range
    perturbed_data = torch.clamp(perturbed_data, -1, 1)
    # Return the perturbed image
    return perturbed_data

def attacktest(model_a, model_t, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []
    total = len(test_loader.dataset)
    
    model_a.to(device)
    model_t.to(device)
    
    
    for data in test_loader:
        data = data.to(device)
        
        # Set requires_grad attribute of tensor. Important for Attack
        data.pos.requires_grad = True

        # Forward pass the data through the model
        if model_a._get_name() == 'Defense_PointNet':
            output, _ = model_a(data.pos, data.batch)
        else:
            output = model_a(data.pos, data.batch)
        
        #이미 틀린 정답 제외
        pred = output.max(1)[1]
        keep = pred.eq(data.y)
        
        # Calculate the loss
        loss = F.nll_loss(output, data.y)

        # Zero all existing gradients
        model_a.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.pos.grad.data
        data_grad = data_grad.reshape(len(data.y), -1, 3)
        data_grad[~keep] = 0 #initialize alreay incorrect answer's grad to 0
        data_grad = data_grad.reshape(-1, 3)
        
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data.pos, epsilon, data_grad)

        # Re-classify the perturbed image
        if model_t._get_name() == 'Defense_PointNet':
            output, _ = model_t(perturbed_data, data.batch)
        else:
            output = model_t(perturbed_data, data.batch)
        
        # Check for success
        pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(total)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

#attack by d
def attacktest2(model_a, model_t, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []
    total = len(test_loader.dataset)
    
    model_a.to(device)
    model_t.to(device)
    
    
    for data in test_loader:
        data = data.to(device)
        
        # Set requires_grad attribute of tensor. Important for Attack
        data.pos.requires_grad = True

        # Forward pass the data through the model
        if model_a._get_name() == 'Defense_PointNet':
            output, _ = model_a(data.pos, data.batch)
        else:
            output = model_a(data.pos, data.batch)
        
        #이미 틀린 정답 제외
        pred = output.max(1)[1]
        keep = pred.eq(data.y)
        
        # Calculate the loss
        loss = F.nll_loss(output, data.y)

        # Zero all existing gradients
        model_a.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.pos.grad.data
        data_grad = data_grad.reshape(len(data.y), -1, 3)
        data_grad[~keep] = 0 #initialize alreay incorrect answer's grad to 0
        data_grad = data_grad.reshape(-1, 3)
        
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data.pos, epsilon, data_grad)

        # Re-classify the perturbed image
        if model_t._get_name() == 'Defense_PointNet':
            output, _ = model_t(perturbed_data, data.batch)
        else:
            output = model_t(perturbed_data, data.batch)
        
        # Check for success
        pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(total)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
