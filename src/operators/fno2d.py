import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- FNO MODEL ---------

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        """
        2D fourier layer does FFT, linear transform, and Inverse FFT for space-time parameterization
        """

        # these class attributes are required and proced by input
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    # complex multiplication (this is a helper method that is going to be used in the forward pass)
    def compl_mul2d(self, input, weights):
        # (batch, in_channels, x, t), (in_channels, out_channels, x, t) -> (batch, out_channels, x, t)
        return torch.einsum("bixt,ioxt->boxt", input, weights)

    # define the forward pass
    def forward(self, x):
        # x.shape = [batch, in_channels, number of grid points]
        batchsize = x.shape[0]
        Nx_size = x.shape[2]
        Nt_size = x.shape[3]

        # transform to frequency domain (b, i, Nx, Nt) -> (b, i, kx, Nt//2+1)
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])
        Nx_fft = x_ft.size(-2)  # full frequency range (pos + neg)
        Nt_fft = x_ft.size(-1) # only nonnegative frequencies

        # apply mode-wise channel mixing (b, i, N//2+1),(i, o, N//2+1) -> (b, o, N//2+1)
        out_ft = torch.zeros(batchsize, self.out_channels, Nx_fft, Nt_fft, dtype=torch.cfloat, device=x.device)

        # for safety, if modes1 > N_fft
        m1 = min(self.modes1, Nx_fft)
        m2 = min(self.modes2, Nt_fft)

        # fill disjoint pieces of out_ft
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        # inverse fourier transform back to physical space: (b, o, Nx//2+1, Nt//2+1) -> (b, o, Nx, Nt)
        x = torch.fft.irfftn(out_ft, s=(Nx_size, Nt_size), dim=[-2, -1])
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()

        """
        Goal: forecasting operator (predict "slab" of temperature given the last k=10 values at each spatial location)
        
        This 2D FNO model with have 4 fourier layers including the lift and projection lin. transformations
        Input: solution of the first 10 timesteps + 2 locations (u(1, x), .... u(10,x), t, x, q(t)), where t in {s+k, s+k+1, ...., s+k+H-1}
        Input Shape: (batchsize, Nx, H, 13)
        Output: the solution of the next 40 timesteps 
        Output Shape: (batchsize, Nx, H, 1) one scalar per (x,t)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding_x = 8  # pad the domain is input is non-periodic

        self.linear_p = nn.Linear(13, self.width)

        self.spect0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.spect1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)  # choosing to have same channel dim. through the fourier layers
        self.spect2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.spect3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.lin0 = nn.Conv2d(self.width, self.width, 1)
        self.lin1 = nn.Conv2d(self.width, self.width, 1)
        self.lin2 = nn.Conv2d(self.width, self.width, 1)
        self.lin3 = nn.Conv2d(self.width, self.width, 1)

        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 1)

        self.activation = nn.Tanh()

    def fourier_layer(self, x, spectral_layer, conv_layer):
        # x.shape = [batch, c_width, N]
        return self.activation(spectral_layer(x) + conv_layer(x))

    def linear_layer(self, x, linear_transformation):
        # x.shape = [b, Nx, Nt, c_width/c_in]
        return linear_transformation(x)

    def forward(self, x):
        """
        x.shape = [batch, Nx, Nt, c_in]
        """
        x_lift = self.linear_layer(x, self.linear_p)
        # x_lift.shape = [batch, Nx, Nt, c_width]
        x_lift = x_lift.permute(0, 3, 1, 2)  # moves tensor dims. so that c_width is the second dimension of the tensor for conv. functions
        # save original Nx before padding
        Nx0 = x_lift.size(-2)
        # x_lift.shape = [batch, c_width, Nx, Nt]

        # pad x only -> (B, c_width, Nx+8, Nt)
        x_lift = F.pad(x_lift, (0,0,0, self.padding_x))

        x_t0 = self.fourier_layer(x_lift, self.spect0, self.lin0)
        x_t1 = self.fourier_layer(x_t0, self.spect1, self.lin1)
        x_t2 = self.fourier_layer(x_t1, self.spect2, self.lin2)
        x_t3 = self.fourier_layer(x_t2, self.spect3, self.lin3)

        # bring back to original Nx
        x_t3 = x_t3[:, :, :Nx0, :]
        x_t3 = x_t3.permute(0, 2, 3, 1)
        x_project = self.linear_layer(x_t3, self.linear_q)
        x = self.linear_layer(x_project, self.output_layer)

        return x





