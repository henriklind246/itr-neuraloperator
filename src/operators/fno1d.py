import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.fft import rfft
from torch.optim import Adam


# --------- FNO MODEL ---------

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()

        """
        1D fourier layer does FFT, linear transform, and Inverse FFT
 
        Its HIGHLY important to note that this layer only represents
        part of the fourier layer
        """

        # these class attributes are required and proced by input
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # complex multiplication (this is a helper method that is going to be used in the forward pass)
    def compl_mul1d(self, input, weights):
        # (batch, in_channels, x (modes)), (in_channels, out_channels, x (modes)) -> (batch, out_channels, x)
        return torch.einsum("bix,iox->box", input, weights)

    # define the forward pass
    def forward(self, x):
        # x.shape = [batch, in_channels, number of grid points]
        batchsize = x.shape[0]
        i_size = x.shape[1]
        N_size = x.shape[2]

        # transform to frequency domain (b, i, N) -> (b, i, N//2+1) where // represents the floor function
        x_ft = rfft(x, dim=-1)
        N_fft = x_ft.size(-1)  # length of the 3rd axis of the tensor -1 is just a shorthand for last axis

        # apply mode-wise channel mixing (b, i, N//2+1),(i, o, N//2+1) -> (b, o, N//2+1)
        out_ft = torch.zeros(batchsize, self.out_channels, N_fft, dtype=torch.cfloat, device=x.device)

        # for safety, if modes1 > N_fft
        m = min(self.modes1, N_fft)
        out_ft[:, :, :m] = self.compl_mul1d(x_ft[:, :, :m], self.weights1[:, :, :m])  # multi on the lower modes1 modes with the weight matrix

        # inverse fourier transform back to physical space: (b, o, N//2+1) -> (b, o, N)
        x = torch.fft.irfft(out_ft, n=N_size, dim=-1)  # ensuring that output has name # of spatial points
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super().__init__()

        """
        This 1D FNO model with have 4 fourier layers including the lift and projection lin. transformations
        """

        self.modes1 = modes
        self.width = width
        self.padding = 1  # pad the domain is input is non-periodic

        self.linear_p = nn.Linear(2,self.width)  # input channel is 2: (u0(x), x) u0(x) is the solution of the intial conidtion at point x

        self.spect0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)  # choosing to have same channel dim. through the fourier layers
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.lin0 = nn.Conv1d(self.width, self.width, 1)
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)
        self.lin3 = nn.Conv1d(self.width, self.width, 1)

        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 1)

        self.activation = nn.Tanh()

    def fourier_layer(self, x, spectral_layer, conv_layer):
        # x.shape = [batch, c_width, N]
        return self.activation(spectral_layer(x) + conv_layer(x))

    def linear_layer(self, x, linear_transformation):
        # x.shape = [b, N, c_width/c_in]
        return linear_transformation(x)

    def forward(self, x):
        """
        x.shape = [batch, N, c_in]
        """
        x_lift = self.linear_layer(x, self.linear_p)
        # x_lift.shape = [batch, N, c_width]
        x_lift = x_lift.permute(0, 2, 1)  # moves tensor dims. so that c_width is the second dimension of the tensor for conv. functions
        # x_lift.shape = [batch, c_width, N]
        x_t0 = self.fourier_layer(x_lift, self.spect0, self.lin0)
        x_t1 = self.fourier_layer(x_t0, self.spect1, self.lin1)
        x_t2 = self.fourier_layer(x_t1, self.spect2, self.lin2)
        x_t3 = self.fourier_layer(x_t2, self.spect3, self.lin3)

        x_t3 = x_t3.permute(0, 2, 1)
        x_project = self.linear_layer(x_t3, self.linear_q)
        x = self.linear_layer(x_project, self.output_layer)

        return x


# ----------- DATA Loader -----------

# data shape = (# samples, # of grid points, # of in_channels)

# define the train/val/test split
n_train = 128
n_val = 32
n_test = 256

# load the data from numpy arrays into torch float tensors
x_data = torch.from_numpy(np.load("x_data.npy")).type(torch.float32)
y_data = torch.from_numpy(np.load("y_data.npy")).type(torch.float32)

if x_data.ndim != 3 or x_data.shape[-1] != 2:
    raise ValueError(f"Input data dimension must be 3 and there must be 2 in_channels, given dim: {x_data.ndim}; number of in_channels: {x_data.shape[-1]}")
if y_data.ndim != 3 or y_data.shape[-1] != 1:
    raise ValueError(f"Output data dimension must be 3 and there must be 1 out_channel, given dim: {y_data.ndim}; number of out_channels: {y_data.shape[-1]}")

input_function_train = x_data[:n_train, :]
output_function_train = y_data[:n_train, :]
input_function_val = x_data[n_train:n_train+n_val, :]
output_function_val = y_data[n_train:n_train+n_val, :]
input_function_test = x_data[n_train+n_val:n_train+n_val+n_test]
output_function_test = y_data[n_train+n_val:n_train+n_val+n_test]

batch_size = 10

training_set = DataLoader(TensorDataset(input_function_train, output_function_train), batch_size=batch_size, shuffle=True)
validation_set = DataLoader(TensorDataset(input_function_val, output_function_val), batch_size=batch_size, shuffle=False)
test_set = DataLoader(TensorDataset(input_function_test, output_function_test), batch_size=batch_size, shuffle=False)

# --------- TRAINING -------------
learning_rate = 0.001
epochs = 250
step_size = 50
gamma = 0.5

modes = 16
width = 64
fno = FNO1d(modes, width) # model

optimizer = Adam(fno.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

l = torch.nn.MSELoss()
for epoch in range(epochs):
    fno.train()
    train_loss = 0.0

    for x_batch, y_batch in training_set:
        optimizer.zero_grad()

        y_pred = fno(x_batch)

        # keep everything (B, N, 1)
        loss = l(y_pred, y_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(training_set)
    scheduler.step()

    print(epoch, train_loss)






