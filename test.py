import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import json

def psnr(img1_pth, img2_pth):
    image1 = cv2.imread(img1_pth)
    image2 = cv2.imread(img2_pth)

    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    mse = np.mean((image1_gray - image2_gray) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    psnr_value = round(psnr_value, 3)
    print(f'PSNR: {psnr_value} dB')
    return psnr_value

def ssim(img1_pth, img2_pth):
    image1 = cv2.imread(img1_pth)
    image2 = cv2.imread(img2_pth)

    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Ensure the images have the same size
    min_height = min(image1_gray.shape[0], image2_gray.shape[0])
    min_width = min(image1_gray.shape[1], image2_gray.shape[1])

    image1_gray = image1_gray[:min_height, :min_width]
    image2_gray = image2_gray[:min_height, :min_width]

    # Compute SSIM
    ssim_value, _ = compare_ssim(image1_gray, image2_gray, full=True)
    ssim_value = round(ssim_value, 3)
    print(f'SSIM: {ssim_value}')
    return ssim_value

def get_file_path(folder_path):
    files = [f"{folder_path}/{f}" for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return files

def get_pixels(img_pth):
    img = cv2.imread(img_pth)
    height, width, _ = img.shape
    return height * width

def main():
    parser = argparse.ArgumentParser()
    folder_path_original = "./out/original"
    folder_path_compressed = "./out/compressed"
    folder_path_reconstructed = "./out/reconstructed"
    out_filename = "./out/scores/result.json"
    file_format = "bmp"
    parser.add_argument('--folder_path_original', default=folder_path_original, help='Root directory of original images')
    parser.add_argument('--folder_path_compressed', default=folder_path_compressed, help='Root directory of compressed images')
    parser.add_argument('--folder_path_reconstructed', default=folder_path_reconstructed, help='Root directory of reconstructed images')
    parser.add_argument('--out_file_name', default=out_filename, help='Output filename which will hold the scores')
    parser.add_argument('--file_format', default=file_format, help='File format of the images')
    args = parser.parse_args()

    files_path = get_file_path(args.folder_path_original)

    psnr_list = []
    ssim_list = []
    compression_ratio_list = []
    filename_list = []

    for file in files_path:
        filename = file.split("/")[-1].split(".")[0]
        filename_list.append(filename)
        orig = os.path.join(args.folder_path_original, f"{filename}.{args.file_format}")
        comp = os.path.join(args.folder_path_compressed, f"{filename}comp.xfr")
        recon = os.path.join(args.folder_path_reconstructed, f"{filename}comp.{args.file_format}")
        print(f"====== Comparing {filename} ======")
        print(f"Original: {orig}", f"Compressed: {comp}", f"Reconstructed: {recon}", sep="\n")

        # Calculate PSNR and SSIM between orig and recon
        psnr_list.append(psnr(orig, recon))
        ssim_list.append(ssim(orig, recon))

        # Calculate compression ratio between orig and comp
        comp_ratio = round(os.path.getsize(orig) / os.path.getsize(comp), 3)
        compression_ratio_list.append(comp_ratio)
        print(f"Compression ratio: {comp_ratio}")
    
    average_psnr = round(sum(psnr_list) / len(psnr_list), 3)
    average_ssim = round(sum(ssim_list) / len(ssim_list), 3)
    average_compression_ratio = round(sum(compression_ratio_list) / len(compression_ratio_list), 3)
    print(f"====== Result summary ======")
    print(f"Average PSNR: {average_psnr}")
    print(f"Average SSIM: {average_ssim}")
    print(f"Average compression ratio: {average_compression_ratio}")

    # write to json file
    with open(f"{out_filename}", 'w') as f:
        result = {
            "psnr_list": psnr_list,
            "ssim_list": ssim_list,
            "compression_ratio_list": compression_ratio_list,
            "average_psnr": average_psnr,
            "average_ssim": average_ssim,
            "average_compression_ratio": average_compression_ratio,
            "filename_list": filename_list,
        }
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()