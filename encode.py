import argparse
from model.components.encoder import Encoder
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='./out/main.tar', help='Path for model checkpoint file [default: ./out/main.tar]')
parser.add_argument('--image_folder', default='./dataset/', help='Directory which holds the images to be compressed [default: ./dataset/]')
parser.add_argument('--out_folder', default='./out/compressed/', help='Directory which will hold the compressed images [default: ./out/compressed/]')
args = parser.parse_args()

inputs = os.listdir(args.image_folder)
encoder = Encoder(args.model)

for image in inputs:
    print(f'compressing {args.image}...')
    encoder.encode_and_save(os.path.join(args.image_folder, image), os.path.join(args.out_folder, f'{image[:-4]}comp.xfr'))