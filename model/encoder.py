import torch
import torchvision.transforms.functional as TF
from model.components.autoencoder import Autoencoder
from bitstring import BitArray
from PIL import Image
import lzma
# for testing
import numpy as np

class Encoder():
    def __init__(self, model_path:str):
        # ensure that all parameters are float32
        self.model = Autoencoder().float()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def encode(self, in_path:str):
        img = Image.open(in_path)
        width, height = img.size
        # Ensure that the image is padded to multiple of 32
        pad_w = 32 - (width%32)
        pad_h = 32 - (height%32)
        # pad(left, top, right, bottom)
        padded_img = TF.pad(img, (pad_w, pad_h, 0, 0))
        x = TF.to_tensor(padded_img)
        x = x.unsqueeze(0)
        x = self.model.enc(x)
        x = self.model.binarizer(x)

        # postprocess the binarization result
        y = x.cpu().detach().numpy()
        y[y<0] = 0
        y[y>0] = 1
        return y, pad_w, pad_h
    
    def encode_and_save(self, in_path:str, out_path:str): 
        # y:latent representation[batch, channel, h, w], pad_w, pad_h: padding
        y, pad_w, pad_h = self.encode(in_path)
        comp_pad_w = BitArray(uint=pad_w, length=8) # 8 bits for dw
        comp_pad_h = BitArray(uint=pad_h, length=8) # 8 bits for dh
        comp_S2 = BitArray(uint=y.shape[2], length = 16)
        comp_S3 = BitArray(uint = y.shape[3], length=16)

        y = y.flatten()
        comp_y = BitArray(y)

        # lossless compression, preset=9 for highest compression
        with lzma.open(out_path , 'wb', preset=9) as fp:
            fp.write(comp_pad_w.tobytes())
            fp.write(comp_pad_h.tobytes())
            fp.write(comp_S2.tobytes())
            fp.write(comp_S3.tobytes())
            fp.write(comp_y.tobytes())

# def create_mock_checkpoint(model, path="mock_checkpoint.pth"):
#     torch.save({'model_state': model.state_dict()}, path)

# def test_encoder():
#     # Create a mock checkpoint for testing
#     model = Autoencoder().float()
#     # create_mock_checkpoint(model)

#     # Initialize the Encoder with the mock checkpoint
#     encoder = Encoder("/tmp2/loijilai/itct/vanillaAE/out/debug_checkpoint.pt")

#     # Create a dummy test image (replace with an actual test image path)
#     test_image = np.random.randint(0, 256, (259, 260, 3), dtype=np.uint8)
#     test_image = Image.fromarray(test_image)

#     # Save the test image temporarily
#     test_image_path = "/tmp2/loijilai/itct/lossy-image-compression/dataset_orig/0001.png"
#     # test_image.save(test_image_path)

#     # Test the encoding process
#     encoded_output, pad_w, pad_h = encoder.encode(test_image_path)
#     print(f"Encoded output shape: {encoded_output.shape}")

#     # Test the encode_and_save process
#     output_path = "encoded_output.lzma"
#     encoder.encode_and_save(test_image_path, output_path)
#     print(f"Encoded file saved to {output_path}")

# if __name__ == "__main__":
#     test_encoder()