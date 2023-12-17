import argparse
from model.decoder import Decoder
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    model = "./out/final_checkpoint.pth"
    compressed_folder = "./out/compressed"
    out_folder = "./out/reconstructed"
    file_format = 'bmp'

    parser.add_argument('--model', default=model, help='Path for model checkpoint file [default: ./out/main.tar]')
    parser.add_argument('--compressed_folder', default=compressed_folder, help='Directory which holds the compressed files [default: ./out/compressed/]')
    parser.add_argument('--out_folder', default=out_folder, help='Directory which will hold the decompressed images [default: ./out/decompressed/]')
    parser.add_argument('--file_format', default=file_format, help='File format of images to decode to [default: bmp]')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    inputs = os.listdir(args.compressed_folder)
    decoder = Decoder(args.model)

    for comp_img in tqdm(inputs):
        print(f'decompressing {comp_img}...')
        decoder.decode_and_save(os.path.join(args.compressed_folder, comp_img), os.path.join(args.out_folder, f'{comp_img[:-4]}.{args.file_format}'))
    print('Done!')

if __name__ == '__main__':
    main()