import time
import copy
import torch
from torch import nn
import numpy as np
import torchvision
from time import strftime, localtime
from torch.utils.tensorboard import SummaryWriter



def train_model(model, dataloaders, optimizer, device, num_epochs=25):
    lossFunc = nn.MSELoss()
    since = time.time()


    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_opt_wts = copy.deepcopy(optimizer.state_dict())
    best_acc = 0.0
    br = False

    for epoch in range(num_epochs):
        if br:
            break
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()  # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    cnt = 0

                    # Iterate over data.
                    for data in dataloaders[phase]:
                        input_tensor = data['input_tens'].float().to(device)
                        target_tensor = data['target'].float().to(device)



                        cnt += 1

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in trainsb
                        with torch.set_grad_enabled(phase == 'train'):
                            h0 = torch.zeros(1, 400, 10).to(device)
                            c0 = torch.zeros(1, 400, 10).to(device)
                            model_predict, _ = model.forward(input_tensor, (h0, c0))
                            loss = lossFunc(model_predict, target_tensor)

                        # print(loss.item())
                        if cnt % len(dataloaders[phase]) == 0:
                            print("predicted : {} and should have got : {}".format(model_predict[0].detach().cpu().numpy(), target_tensor[0].detach().cpu().numpy()))

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                            # print(loss.item())

                        # if phase == 'train':
                        #     counter += 4

                        running_loss += loss.item()*(len(input_tensor))

                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    print(epoch_loss)


                        # if phase == 'val' and epoch_acc >= best_acc:
                        #     best_acc = epoch_acc
                        #     best_model_wts = copy.deepcopy(model.state_dict())
                        #     best_opt_wts = copy.deepcopy(optimizer.state_dict())
                        # if phase == 'val':
                        #     val_acc_history.append(epoch_acc)
    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, val_acc_history
