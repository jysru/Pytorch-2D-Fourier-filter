import numpy as np
import torch


def numpy_fourier_lowpass_filter(x0, filter_radius):
    # Pad with input shape to ensure Nyquist
    x0_pad = np.pad(x0, pad_width=[int(x0.shape[0]/2), int(x0.shape[1]/2)], mode='constant')
    xf = np.fft.fftshift(np.fft.fft2(x0_pad))
    
    # Define and apply lowpass filter
    w, h = xf.shape
    r = w // 2

    x = np.linspace(0, w-1, w) - w // 2
    y = np.linspace(0, h-1, h) - h // 2
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(np.square(X) + np.square(Y))
    R = R / np.max(R)
    mask = (R > filter_radius)

    xf[mask] = 0

    # Fourier transform back
    xfi = np.fft.ifft2(np.fft.ifftshift(xf))

    # Delete padding
    row_diff = xfi.shape[0] - x0.shape[0]
    col_diff = xfi.shape[1] - x0.shape[1]
    xfi = xfi[row_diff//2:-row_diff//2, col_diff//2:-col_diff//2]

    return (xfi, xf)


def torch_fourier_lowpass_filter(x0, filter_radius):

    # Convert to tensor
    if isinstance(x0, np.ndarray):
        x1 = torch.from_numpy(x0.astype('float32'))
        x1 = torch.polar(x1, torch.zeros_like(x1))

    # Pad with input shape to ensure Nyquist
    pad = (int(x1.size()[0]/2), int(x1.size()[0]/2), int(x1.size()[1]/2), int(x1.size()[1]/2))
    x2 = torch.nn.functional.pad(x1, pad=pad , mode='constant', value=0)

    # Apply Fourier transform
    xf = torch.fft.fftshift(torch.fft.fft2(x2))
    
    # Define and apply lowpass filter
    w, h = xf.shape
    r = w // 2

    x = np.linspace(0, w-1, w) - w // 2
    y = np.linspace(0, h-1, h) - h // 2
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(np.square(X) + np.square(Y))
    R = R / np.max(R)
    mask = (R > filter_radius)
    xf[mask] = 0

    # Fourier transform back
    xfi = torch.fft.ifft2(torch.fft.ifftshift(xf))

    # Delete padding
    row_diff = xfi.shape[0] - x0.shape[0]
    col_diff = xfi.shape[1] - x0.shape[1]
    xfi = xfi[row_diff//2:-row_diff//2, col_diff//2:-col_diff//2]

    return (xfi, xf)