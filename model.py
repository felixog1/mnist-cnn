import torch.nn as nn


class CNN(nn.Module):
    """2 conv+pool blocks -> flatten -> 2 FC layers -> 10 class logits.

    Configurable so all four variants (baseline/dropout/batchnorm/L2) share
    one architecture and only the flag being studied changes.
    """

    def __init__(self, dropout_p=0.0, use_batchnorm=False):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        # padding=1 keeps spatial size unchanged by the conv itself (28x28 in, 28x28 out);
        # only the maxpool layers reduce spatial size.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)  # halves H and W each time
        self.relu = nn.ReLU()

        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)

        # After conv1+pool: 28x28 -> 14x14, 32 channels
        # After conv2+pool: 14x14 -> 7x7, 64 channels
        # Flattened size: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # p=0.0 makes dropout a no-op, so the baseline can reuse this same class.
        self.dropout_conv = nn.Dropout(dropout_p)
        self.dropout_fc = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.pool(self.relu(x))

        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.pool(self.relu(x))

        x = x.flatten(start_dim=1)  # keep batch dim, flatten (C,H,W) -> single vector
        x = self.dropout_conv(x)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)  # raw logits; CrossEntropyLoss applies softmax internally
        return x
