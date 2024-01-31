import torch
import torch.utils.data 
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

### set a random seed for reproducibility (do not change this)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

### Set if you wish to use cuda or not
use_cuda_if_available = True

### Define the Convolutional Neural Network Model
class CNN(torch.nn.Module):
    def __init__(self, num_bins): 
        super(CNN, self).__init__()
        
        ### Initialize the various Network Layers
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(3, 20, stride=2, kernel_size=(8,8))  # Output channels: 20
        self.pool1 = torch.nn.MaxPool2d((2,3), stride=2)
        self.relu1 = torch.nn.PReLU()
        self.batchnorm1 = torch.nn.BatchNorm2d(20)
        self.dropout1 = torch.nn.Dropout(0.5)

        self.conv2 = torch.nn.Conv2d(20, 36, stride=1, kernel_size=(6,6))  # Output channels: 36
        self.pool2 = torch.nn.MaxPool2d((2,2), stride=1)
        self.relu2 = torch.nn.PReLU()
        self.batchnorm2 = torch.nn.BatchNorm2d(36)
        self.dropout2 = torch.nn.Dropout(0.4)

        self.conv3 = torch.nn.Conv2d(36, 9, stride=1, kernel_size=(4,4))  # Output channels: 9
        self.pool3 = torch.nn.MaxPool2d((6,6), stride=1)
        self.relu3 = torch.nn.PReLU()
        self.batchnorm3 = torch.nn.BatchNorm2d(9)  
        self.dropout3 = torch.nn.Dropout(0.3)

        # Size for the linear layer input after the conv and pool layers
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(360, 70)
        self.relu4 = torch.nn.PReLU()
        self.batchnorm4 = torch.nn.BatchNorm1d(70)
        self.dropout4 = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(70, num_bins)
        self.batchnorm5 = torch.nn.BatchNorm1d(num_bins)

        if torch.cuda.is_available():
            self = self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu4(x)
        x = self.batchnorm4(x)
        x = self.dropout4(x)
        x = self.linear2(x)
        x = self.batchnorm5(x)

        x = x.squeeze()  # (Batch_size x num_bins)
        return x

### Define the custom PyTorch dataloader for this assignment
class dataloader(torch.utils.data.Dataset):
    """Loads the KITTI Odometry Benchmark Dataset"""
    def __init__(self, matfile, binsize=20, mode='train'):
        self.data = sio.loadmat(matfile)
        
        self.images = self.data['images']
        self.mode = mode

        # Fill in this function if you wish to normalize the input
        # Data to zero mean.
        self.normalize_to_zero_mean()
        
        if self.mode != 'test':

            # Generate targets for images by 'digitizing' each azimuth 
            # angle into the appropriate bin (from 0 to num_bins)
            self.azimuth = self.data['azimuth']
            bin_edges = np.arange(-180,180+1,binsize)
            self.targets = (np.digitize(self.azimuth,bin_edges) -1).reshape((-1))

    def normalize_to_zero_mean(self):
        #take mean of all images for channel and zero-center
        for i in range(0,3):
            self.images[:, i, ...] = self.images[:, i, ...] - np.mean(self.images[:, i, ...])

    def __len__(self):
        return int(self.images.shape[0])
  
    def __getitem__(self, idx):
        if self.mode != 'test':
            return self.images[idx], self.targets[idx]    
        else:
            return self.images[idx]

if __name__ == "__main__": 
    '''
    Initialize the Network
    '''
    binsize=20 #degrees **set this to 20 for part 2**
    bin_edges = np.arange(-180,180+1,binsize)
    num_bins = bin_edges.shape[0] - 1
    cnn = CNN(num_bins) #Initialize our CNN Class
    
    '''
    Uncomment section to get a summary of the network (requires torchsummary to be installed):
        to install: pip install torchsummary
    '''
    #from torchsummary import summary
    #inputs = torch.zeros((1,3,68,224))
    #summary(cnn, input_size=(3, 68, 224))
    
    '''
    Training procedure
    '''
    initial_lr = 5e-4  # Slightly increased learning rate
    weight_decay = 2e-4  # Increased weight decay for stronger regularization

    CE_loss = torch.nn.CrossEntropyLoss(reduction='sum') #initialize our loss (specifying that the output as a sum of all sample losses)
    params = list(cnn.parameters())
    optimizer = torch.optim.Adam(params, lr=initial_lr, weight_decay=weight_decay)

    ### Initialize our dataloader for the training and validation set (specifying minibatch size of 128)
    dsets = {x: dataloader('sun-cnn_{}.mat'.format(x),binsize=binsize) for x in ['train', 'val']} 
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=128, shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    loss = {'train': [], 'val': []}
    top1err = {'train': [], 'val': []}
    top5err = {'train': [], 'val': []}

    early_stopping_patience = 30
    epochs_since_improvement = 0
    best_err = float('inf')  # Initialize best error as infinity
    best_epoch = 0  # To keep track of the best epoch

    ### Iterate through the data for the desired number of epochs
    # Track start time
    start_time = time.time()
    #20 -> 300 epochs
    for epoch in range(0,300):
        epoch_time = time.time()
        for mode in ['train', 'val']:    #iterate 
            epoch_loss=0
            top1_incorrect = 0
            top5_incorrect = 0
            if mode == 'train':
                cnn.train(True)    # Set model to training mode
            else:
                cnn.train(False)    # Set model to Evaluation mode
                cnn.eval()
            
            dset_size = dset_loaders[mode].dataset.__len__()
            for image, target in dset_loaders[mode]:    #Iterate through all data (each iteration loads a minibatch)
                
                # Cast to types and Load GPU if desired and available
                if use_cuda_if_available and torch.cuda.is_available():
                    image = image.cuda().type(torch.cuda.FloatTensor)
                    target = target.cuda().type(torch.cuda.LongTensor)
                else:
                    image = image.type(torch.FloatTensor)
                    target = target.type(torch.LongTensor)

                optimizer.zero_grad()    #zero the gradients of the cnn weights prior to backprop
                pred = cnn(image)   # Forward pass through the network
                minibatch_loss = CE_loss(pred, target)  #Compute the minibatch loss
                epoch_loss += minibatch_loss.item() #Add minibatch loss to the epoch loss 
                
                if mode == 'train': #only backprop through training loss and not validation loss       
                    minibatch_loss.backward()
                    optimizer.step()        
                        
                
                _, predicted = torch.max(pred.data, 1) #from the network output, get the class prediction
                top1_incorrect += (predicted != target).sum().item() #compute the Top 1 error rate
                
                top5_val, top5_idx = torch.topk(pred.data,5,dim=1)
                top5_incorrect += ((top5_idx != target.view((-1,1))).sum(dim=1) == 5).sum().item() #compute the top5 error rate
    
                
            loss[mode].append(epoch_loss/dset_size)
            top1err[mode].append(top1_incorrect/dset_size)
            top5err[mode].append(top5_incorrect/dset_size)
    
            print("{} Loss: {}".format(mode, loss[mode][epoch]))
            print("{} Top 1 Error: {}".format(mode, top1err[mode][epoch]))    
            print("{} Top 5 Error: {}".format(mode, top5err[mode][epoch])) 
            # Early stopping logic
            if mode == 'val':
                print("Completed Epoch {}".format(epoch))
                val_loss = loss['val'][epoch]

                # Check if the validation error has improved
                if top1err['val'][epoch] < best_err:
                    best_err = top1err['val'][epoch]
                    best_epoch = epoch
                    torch.save(cnn.state_dict(), 'best_model_{}.pth'.format(binsize))
                    epochs_since_improvement = 0
                    print("Validation error improved to {:.4f}".format(best_err))
                else:
                    epochs_since_improvement += 1
                    print("No improvement in validation error for {} epochs".format(epochs_since_improvement))

                if epochs_since_improvement >= early_stopping_patience:
                    print("Early stopping triggered after {} epochs without improvement".format(early_stopping_patience))
                    break  # Break from the 'val' loop

        # Time taken for one epoch
        print(f'Epoch {epoch} took {round(time.time() - epoch_time, 2)}s')

        if epochs_since_improvement >= early_stopping_patience:
            print("Training stopped early due to lack of improvement in validation error.")
            break  # Break from the epoch loop

    # Print final results outside the loop
    print("Training Complete")
    print("Lowest validation set error of {:.2f} at epoch {}".format(best_err, best_epoch))
    print(f"Total training took {round(time.time() - start_time, 2)}s")       
    '''
    Plotting
    '''        
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.grid()
    ax1.plot(loss['train'],linewidth=2)
    ax1.plot(loss['val'],linewidth=2)
    #ax1.legend(['Train', 'Val'],fontsize=12)
    ax1.legend(['Train', 'Val'])
    ax1.set_title('Objective', fontsize=18, color='black')
    ax1.set_xlabel('Epoch', fontsize=12)
    
    ax2.grid()
    ax2.plot(top1err['train'],linewidth=2)
    ax2.plot(top1err['val'],linewidth=2)
    ax2.legend(['Train', 'Val'])
    ax2.set_title('Top 1 Error', fontsize=18, color='black')
    ax2.set_xlabel('Epoch', fontsize=12)
    
    ax3.grid()
    ax3.plot(top5err['train'],linewidth=2)
    ax3.plot(top5err['val'],linewidth=2)
    ax3.legend(['Train', 'Val'])
    ax3.set_title('Top 5 Error', fontsize=18, color='black')
    ax3.set_xlabel('Epoch', fontsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig('net-train.pdf')
