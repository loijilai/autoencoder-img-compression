# Autoencoder Image Compression
This is a simple Autoencoder model for image compression, supporting various image format.

## Environment setup
Create a new environment and `pip install -r requirements.txt`  
We structure the project directory as follows:  
```
AutoencoderImageCompression/
├─ model
├─ utils
├─ dataset (put your dataset here)
├─ out 
│  ├─ compressed (compressed imgs in .xfr format)
│  ├─ reconstructed (save reconstructed imgs)
│  ├─ original (put imgs to be compressed here)
│  ├─ scores (save PSNR, SSIM, comp ratio here)
```

## Training
Use [train.py](./train.py) to train your own model.  

```
usage: train.py 

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Root directory of Images
  --checkpoint_path CHECKPOINT_PATH
                        Use to resume training from last checkpoint
  --save_at SAVE_AT     Directory where training state will be saved
  --file_format FILE_FORMAT
                        File format of images to train on
  --num_of_epochs NUM_OF_EPOCHS
                        Epoch to stop training at
  --batch_size BATCH_SIZE
                        Batch size for training
  --validation_split VALIDATION_SPLIT
                        Validation split for training
  --lr_rate LR_RATE     Learning rate for training
```

* Training result (loss) are stored in 'history' in checkpoint

## Testing
Use [test.py](./test.py) to get PSNR, SSIM, and compression ratio on the trained model.  

```
usage: test.py 

options:
  -h, --help            show this help message and exit
  --folder_path_original FOLDER_PATH_ORIGINAL
                        Root directory of original images
  --folder_path_compressed FOLDER_PATH_COMPRESSED
                        Root directory of compressed images
  --folder_path_reconstructed FOLDER_PATH_RECONSTRUCTED
                        Root directory of reconstructed images
  --out_file_name OUT_FILE_NAME
                        Output filename which will hold the scores
  --file_format FILE_FORMAT
                        File format of the images
```

## Usage
After training, put images to be compress into `/out/original` and use [encode.py](./encode.py) to compress an image into `.xfr` file. And use [decode.py](./decode.py) to decompress an image back to the specified format.  

```
usage: encode.py 

options:
  -h, --help            show this help message and exit
  --model MODEL         Path for model checkpoint file
  --image_folder IMAGE_FOLDER
                        Directory which holds the images to be compressed
  --out_folder OUT_FOLDER
                        Directory which will hold the compressed images
  --file_format FILE_FORMAT
                        File format of images to be encoded
```

```
usage: decode.py 

options:
  -h, --help            show this help message and exit
  --model MODEL         Path for model checkpoint file [default: ./out/main.tar]
  --compressed_folder COMPRESSED_FOLDER
                        Directory which holds the compressed files [default: ./out/compressed/]
  --out_folder OUT_FOLDER
                        Directory which will hold the decompressed images [default: ./out/decompressed/]
  --file_format FILE_FORMAT
                        File format of images to decode to [default: bmp]
```

## Some useful tips
* Batch training is available only if all photos have same size. Set batch_size = 1 if photo size varies.
* We provide options to change data format to train on
    * Change the image format to train on: e.g. `python train.py --file_format bmp`
    * Specify the image format to be encoded: e.g. `python encode.py --file_format bmp`
    * Specify the image format to be decode to: e.g. `python decode.py --file_format bmp`
    * In `test.py`, file_format is used to match the filename between original image and the reconstructed image: `python test.py --file_format bmp`

## Further improvement
* The autoencoder architecture consists of only three layers of CNN layers, deeper architecture as well as residual connection may be applied to improve model capability to learn representation
* Our entropy model is not learned, this is not an end-to-end learned optimized image compression. The entropy coding use `lzma` package as a separate module.
* We use binary quantization here, other quantization method may be added

## Reference
This is a simple project built for the purpose of learning autoencoder in image compression task. We rely on other people's work and do not intend to infringe upon anyone's rights.

1. This project is a direct modification on this repository: [lossy-image-compression](https://github.com/abskj/lossy-image-compression).  
2. Our dataset comes from the wiki page of this repository: [cae](https://github.com/alexandru-dinu/cae/wiki).  
3. The model architecture comes from [CompressAI documentation](https://interdigitalinc.github.io/CompressAI/tutorials/tutorial_custom.html) on how to train custom model.