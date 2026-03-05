import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ----------- DATA Loader -----------

def load_numpy_data(x_path: str, y_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    x_data = torch.from_numpy(np.load(x_path)).type(torch.float32)
    y_data = torch.from_numpy(np.load(y_path)).type(torch.float32)
    return x_data, y_data


def split_tensors(x_data: torch.Tensor, y_data: torch.Tensor, n_train: int, n_val: int, n_test: int) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]:

    # safe guarding tensor sizes
    if x_data.ndim != 3 or x_data.shape[-1] != 2:
        raise ValueError(
            f"Input data dimension must be 3 and there must be 2 in_channels, given dim: {x_data.ndim}; number of in_channels: {x_data.shape[-1]}")
    if y_data.ndim != 3 or y_data.shape[-1] != 1:
        raise ValueError(
            f"Output data dimension must be 3 and there must be 1 out_channel, given dim: {y_data.ndim}; number of out_channels: {y_data.shape[-1]}")

    # splitting the dataset
    input_function_train = x_data[:n_train, :]
    output_function_train = y_data[:n_train, :]
    input_function_val = x_data[n_train:n_train + n_val, :]
    output_function_val = y_data[n_train:n_train + n_val, :]
    input_function_test = x_data[n_train + n_val:n_train + n_val + n_test]
    output_function_test = y_data[n_train + n_val:n_train + n_val + n_test]

    return input_function_train, output_function_train, input_function_val, output_function_val, input_function_test, output_function_test


def create_dataloaders(
        in_f_train: torch.Tensor,
        out_f_train: torch.Tensor,
        in_f_val: torch.Tensor,
        out_f_val: torch.Tensor,
        in_f_test: torch.Tensor,
        out_f_test: torch.Tensor,
        batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:

    # creating dataloaders (X[i], Y[i]) pairs
    training_set = DataLoader(TensorDataset(in_f_train, out_f_train), batch_size=batch_size, shuffle=True)
    validation_set = DataLoader(TensorDataset(in_f_val, out_f_val), batch_size=batch_size, shuffle=False)
    test_set = DataLoader(TensorDataset(in_f_test, out_f_test), batch_size=batch_size, shuffle=False)

    return training_set, validation_set, test_set

if __name__ == '__main__':
    # data shape = (# samples, # of grid points, # of in_channels)
    x_data, y_data = load_numpy_data(x_path="/Users/henriklind/Desktop/no-tps-ihcp/data/x_data.npy", y_path="/Users/henriklind/Desktop/no-tps-ihcp/data/y_data.npy")

    n_samples = 1000
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    n_test = int(n_samples * 0.15)
    batch_size = 10

    in_f_train, out_f_train, in_f_val, out_f_val, in_f_test, out_f_test = split_tensors(x_data=x_data, y_data=y_data, n_train=n_train, n_val=n_val, n_test=n_test)

    training_set, validation_set, test_set = create_dataloaders(in_f_train=in_f_train, out_f_train=out_f_train, in_f_val=in_f_val, out_f_val=out_f_val, in_f_test=in_f_test, out_f_test=out_f_test, batch_size=batch_size)




