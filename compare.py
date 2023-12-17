import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as compare_ssim


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
    print(f'SSIM: {ssim_value}')
    return ssim_value




folder_path1 = "results/test_single_x2/visualization/Single"
folder_path2 = "img_resize_2"


def get_file_path(folder_path):

    files = [f"{folder_path}/{f}" for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    sorted_files = sorted(files)

    # for file_name in sorted_files:
    #     print(file_name)

    return files

files1 = get_file_path(folder_path1)
files2 = get_file_path(folder_path2)

psnr_list = []
ssim_list = []


for img1, img2 in zip(files1, files2):
    print(img1, img2)
    psnr_value = psnr(img1, img2)
    ssim_value = ssim(img1, img2)
    psnr_list.append(psnr_value)
    ssim_list.append(ssim_value)


average_psnr = sum(psnr_list) / len(psnr_list)
print(f"The average psnr is: {average_psnr} dB.")


average_ssim = sum(ssim_list) / len(ssim_list)
print(f"The average ssim is: {average_ssim}.")

    
