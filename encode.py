import argparse
from model.encoder import Encoder
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    model = "./out/final_checkpoint.pth"
    image_folder = "./out/original"
    out_folder = "./out/compressed"
    file_format = 'bmp'
    parser.add_argument('--model', default=model, help='Path for model checkpoint file')
    parser.add_argument('--image_folder', default=image_folder, help='Directory which holds the images to be compressed')
    parser.add_argument('--out_folder', default=out_folder, help='Directory which will hold the compressed images')
    parser.add_argument('--file_format', default=file_format, help='File format of images to be encoded')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    inputs = os.listdir(args.image_folder)
    encoder = Encoder(args.model)

    for image in tqdm(inputs):
        if not image.endswith(args.file_format):
            continue
        print(f'compressing {image}...')
        encoder.encode_and_save(os.path.join(args.image_folder, image), os.path.join(args.out_folder, f'{image[:-4]}comp.xfr'))
    print('Done!')

if __name__ == '__main__':
    main()