import torch
import torchvision.transforms.functional as TF
from model.autoencoder import Autoencoder
from bitstring import BitArray
from PIL import Image
import lzma

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
