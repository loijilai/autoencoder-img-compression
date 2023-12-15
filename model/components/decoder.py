import torch
from torchvision.transforms import functional as TF
import numpy as np
import lzma
from bitstring import BitArray
from model.autoencoder import Autoencoder

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
            temp_byte = None
            j = 0

            print('Reading latent representation')
            byte = fp.read(1)
            while byte != b"":
                temp_byte = BitArray(byte).bin
                for i in range(len(temp_byte)):
                    y[j] = int(temp_byte[i])
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