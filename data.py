import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Precomputed mean/std of MNIST pixel values (in [0, 1] range).
# Normalizing to roughly zero mean / unit variance keeps gradients well-scaled
# and matches the input distribution the network's weight init assumes.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_dataloaders(batch_size=64, val_size=5000, data_dir="./data", num_workers=0):
    """Returns train_loader, val_loader, test_loader.

    Train/val come from MNIST's 60k training split (55k/5k by default).
    Test is MNIST's separate 10k split, held out untouched until final evaluation.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])

    full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)} ({len(train_loader.dataset)} samples)")
    print(f"Val batches:   {len(val_loader)} ({len(val_loader.dataset)} samples)")
    print(f"Test batches:  {len(test_loader)} ({len(test_loader.dataset)} samples)")

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, dtype: {images.dtype}")
    print(f"Labels shape: {labels.shape}, sample labels: {labels[:10].tolist()}")
    print(f"Pixel value range after normalize: [{images.min():.3f}, {images.max():.3f}]")
