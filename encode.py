import argparse
from model.encoder import Encoder
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    model = "/tmp2/loijilai/itct/vanillaAE/out/debug_checkpoint.pt"
    image_folder = "/tmp2/loijilai/itct/vanillaAE/out/original"
    out_folder = "/tmp2/loijilai/itct/vanillaAE/out/compressed"
    parser.add_argument('--model', default=model, help='Path for model checkpoint file')
    parser.add_argument('--image_folder', default=image_folder, help='Directory which holds the images to be compressed')
    parser.add_argument('--out_folder', default=out_folder, help='Directory which will hold the compressed images')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    inputs = os.listdir(args.image_folder)
    encoder = Encoder(args.model)

    for image in tqdm(inputs):
        print(f'compressing {image}...')
        encoder.encode_and_save(os.path.join(args.image_folder, image), os.path.join(args.out_folder, f'{image[:-4]}comp.xfr'))
    print('Done!')

if __name__ == '__main__':
    main()