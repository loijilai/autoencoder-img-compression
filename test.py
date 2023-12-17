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
    return sorted(files)

def get_pixels(img_pth):
    img = cv2.imread(img_pth)
    height, width, _ = img.shape
    return height * width

def main():
    parser = argparse.ArgumentParser()
    folder_path1 = "/tmp2/loijilai/itct/vanillaAE/out/original"
    folder_path2 = "/tmp2/loijilai/itct/vanillaAE/out/decompressed"
    out_folder = "/tmp2/loijilai/itct/vanillaAE/out/scores"
    parser.add_argument('--folder_path1', default=folder_path1, help='Root directory of original images')
    parser.add_argument('--folder_path2', default=folder_path2, help='Root directory of compressed images')
    parser.add_argument('--out_folder', default=out_folder, help='Directory which will hold the scores')

    files1 = get_file_path(folder_path1)
    files2 = get_file_path(folder_path2)

    psnr_list = []
    ssim_list = []
    compression_ratio_list = []
    original_data_rate_list = []
    compressed_data_rate_list = []

    for img1, img2 in zip(files1, files2):
        img1_name = img1.split("/")[-1]
        img2_name = img2.split("/")[-1]
        print(f"Comparing {img1_name} and {img2_name}")
        psnr_list.append(psnr(img1, img2))
        ssim_list.append(ssim(img1, img2))
        original_data_rate_list.append(round(os.path.getsize(img1) / get_pixels(img1), 3))
        compressed_data_rate_list.append(round(os.path.getsize(img2) / get_pixels(img2), 3))
        compression_ratio_list.append(round(os.path.getsize(img1) / os.path.getsize(img2), 3))
    
    average_psnr = round(sum(psnr_list) / len(psnr_list), 3)
    average_ssim = round(sum(ssim_list) / len(ssim_list), 3)
    average_compression_ratio = round(sum(compression_ratio_list) / len(compression_ratio_list), 3)
    average_original_data_rate = round(sum(original_data_rate_list) / len(original_data_rate_list), 3)
    average_compression_data_rate = round(sum(compressed_data_rate_list) / len(compressed_data_rate_list), 3)
    print(f"Average PSNR: {average_psnr}")
    print(f"Average SSIM: {average_ssim}")
    print(f"Average compression ratio: {average_compression_ratio}")
    print(f"Average original data rate: {average_original_data_rate}")
    print(f"Average compressed data rate: {average_compression_data_rate}")

    # write to json file
    with open(f"{out_folder}/result.json", 'w') as f:
        result = {
            "psnr_list": psnr_list,
            "ssim_list": ssim_list,
            "compression_ratio_list": compression_ratio_list,
            "original_data_rate_list": original_data_rate_list,
            "compressed_data_rate_list": compressed_data_rate_list,
            "average_psnr": average_psnr,
            "average_ssim": average_ssim,
            "average_compression_ratio": average_compression_ratio,
            "average_original_data_rate": average_original_data_rate,
            "average_compression_data_rate": average_compression_data_rate,
        }
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()