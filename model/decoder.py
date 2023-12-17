import torch
from torchvision.transforms import functional as TF
import numpy as np
import lzma
from bitstring import BitArray
from model.components.autoencoder import Autoencoder
# for testing
import os

class Decoder():
    def __init__(self, model_path:str):
        self.model = Autoencoder().float()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def decompress(self, in_path:str, out_path:str):
        with lzma.open(in_path, 'rb') as fp:
            pad_w = int.from_bytes(fp.read(1), byteorder='big', signed=False)
            pad_h = int.from_bytes(fp.read(1), byteorder='big', signed=False)
            S2 = int.from_bytes(fp.read(2), byteorder='big', signed=False)
            S3 = int.from_bytes(fp.read(2), byteorder='big', signed=False)
            
            # the last output channel dimension of the encoder is 128
            y = np.empty((1,128,S2,S3)).ravel()

            # Read the latent representation
            j = 0
            byte = fp.read(1)
            eight_bit_array = None
            while byte != b"":
                eight_bit_array = BitArray(byte).bin
                for i in range(len(eight_bit_array)):
                    y[j] = int(eight_bit_array[i])
                    j += 1
                byte =  fp.read(1)
        
        y = y * 2.0 - 1  # (0|1) -> (-1, 1)
        y = torch.from_numpy(y.reshape(1,128,S2,S3)).float()

        output = self.model.dec(y)
        img = TF.to_pil_image(output.squeeze(0)) # remove batch dim

        width, height = img.size
        img = img.crop((pad_w, pad_h, width, height))
        img.save(out_path, "png")
        return y
    
# def create_mock_compressed_file(in_path, S2=32, S3=32):
#     pad_w = pad_h = 0  # No padding in this mock example
#     y = np.random.randint(0, 2, (1, 128, S2, S3))  # Random binary data

#     with lzma.open(in_path, 'wb', preset=9) as fp:
#         fp.write(pad_w.to_bytes(1, byteorder='big'))
#         fp.write(pad_h.to_bytes(1, byteorder='big'))
#         fp.write(S2.to_bytes(2, byteorder='big'))
#         fp.write(S3.to_bytes(2, byteorder='big'))

#         for value in y.flatten():
#             byte = BitArray(bin=str(value)).tobytes()
#             fp.write(byte)

# def test_decoder():
#     # Initialize the Decoder with the mock checkpoint
#     decoder = Decoder("/tmp2/loijilai/itct/vanillaAE/out/debug_checkpoint.pt")

#     # Test the decompress method
#     mock_compressed_path = "/tmp2/loijilai/itct/vanillaAE/model/encoded_output.lzma"
#     output_image_path = "output_image.png"
#     decoder.decompress(mock_compressed_path, output_image_path)

#     # Check if the output image file is created
#     assert os.path.exists(output_image_path), "Output image file was not created"
#     print(f"Output image saved to {output_image_path}")

# if __name__ == "__main__":
#     test_decoder()