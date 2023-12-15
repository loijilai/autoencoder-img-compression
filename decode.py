import argparse
from model.components.decoder import Decoder
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='./out/main.tar', help='Path for model checkpoint file [default: ./out/main.tar]')
parser.add_argument('--compressed_folder', default='./out/compressed/', help='Directory which holds the compressed files [default: ./out/compressed/]')
parser.add_argument('--out_folder', default='./out/decompressed/', help='Directory which will hold the decompressed images [default: ./out/decompressed/]')
args = parser.parse_args()

inputs = os.listdir(args.compressed_folder)
decoder = Decoder(args.model)

for comp_img in inputs:
    print(f'decompressing {comp_img}...')
    decoder.decompress(os.path.join(args.compressed, comp_img), os.path.join(args.out_folder, f'{comp_img[:-4]}.jpg'))