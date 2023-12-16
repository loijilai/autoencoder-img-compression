import torch
import torch.nn as nn
from model.components.quantizer import Binarizer

class Autoencoder(nn.Module):
    # todo: skip connections
    def __init__(self):
        super().__init__()
        # Normalization:
        # Normalizes each channel across the H and W dimensions (across all pixels in each feature map).
        self.enc = nn.Sequential(
            nn.Conv2d(3,32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32,64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64,128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128,64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64,32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32,3, kernel_size=5 ,stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.binarizer = Binarizer()

    def forward(self, x):
        x = self.enc(x)
        x = self.binarizer(x)
        x = self.dec(x)
        return x


# def test_autoencoder():
#     model = Autoencoder()

#     # Create a dummy input tensor (batch size, channels, height, width)
#     dummy_input = torch.randn(1, 3, 256, 256)

#     # Forward pass through the model
#     try:
#         output = model(dummy_input)
#         print("Forward pass successful.")
#     except Exception as e:
#         print(f"Error during forward pass: {e}")
#         return

#     # Check the output shape
#     assert output.shape == dummy_input.shape, f"Output shape {output.shape} does not match input shape {dummy_input.shape}"
#     print(f"Output shape {output.shape} matches input shape.")

# if __name__ == "__main__":
#     test_autoencoder()
