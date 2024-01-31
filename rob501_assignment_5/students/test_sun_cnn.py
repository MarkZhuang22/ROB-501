import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sun_cnn_20 import CNN, dataloader  # Replace 'your_cnn_module' with the actual module where your CNN class is defined

if __name__ == '__main__':
    '''
    Initialize the Network
    '''
    binsize = 20  # degrees
    bin_edges = np.arange(-180, 180 + 1, binsize)
    num_bins = bin_edges.shape[0] - 1

    # Check for GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cnn = CNN(num_bins).to(device)  # Initialize our CNN Class and move it to the appropriate device
    cnn.load_state_dict(torch.load('best_model_{}.pth'.format(binsize), map_location=device))

    dsets = {
        x: dataloader('sun-cnn_{}.mat'.format(x), binsize=binsize, mode='test')
        for x in ['test']
    }
    dset_loaders = {
        x: torch.utils.data.DataLoader(dsets[x], batch_size=50, shuffle=False, num_workers=4)
        for x in ['test']
    }

    pred_list = np.zeros((0))
    top1_incorrect = 0

    for mode in ['test']:  # iterate 
        cnn.train(False)  # Set model to Evaluation mode
        cnn.eval()

        for data in dset_loaders[mode]:  # Iterate through all data (each iteration loads a minibatch)
            # Move input data to the GPU
            image = data.to(device).type(torch.FloatTensor)

            pred = cnn(image)  # Forward pass through the network
            _, predicted = torch.max(pred.data, 1)  # from the network output, get the class prediction
            pred_list = np.hstack((pred_list, predicted.cpu().numpy()))

    print("Testing with a binsize of {} degrees - saving predictions to predictions_{}.txt".format(binsize, binsize))

    np.savetxt('predictions_{}.txt'.format(binsize), pred_list.astype(int))
