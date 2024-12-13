import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def yuv2rgb(Y, U, V, fw, fh):
    U_resized = cv2.resize(U, (fw, fh), interpolation=cv2.INTER_CUBIC)
    V_resized = cv2.resize(V, (fw, fh), interpolation=cv2.INTER_CUBIC)
    rf = Y + 1.4075 * (V_resized - 128.0)
    gf = Y - 0.3455 * (U_resized - 128.0) - 0.7169 * (V_resized - 128.0)
    bf = Y + 1.7790 * (U_resized - 128.0)
    r = np.clip(rf, 0, 255).astype(np.uint8)
    g = np.clip(gf, 0, 255).astype(np.uint8)
    b = np.clip(bf, 0, 255).astype(np.uint8)
    return r, g, b


def from_folder_yuv_to_folder_png(folder_yuv, folder_png, fw, fh, patchsize=64, patchstep=32):
    os.makedirs(folder_png, exist_ok=True)
    for file in os.listdir(folder_png):
        os.remove(os.path.join(folder_png, file))

    fwuv, fhuv = fw // 2, fh // 2  # Dimensions for U and V components
    dx = np.arange(0, fw - patchsize + 1, patchstep)  # X-axis steps for patches
    dy = np.arange(0, fh - patchsize + 1, patchstep)  # Y-axis steps for patches
    png_frame_num = 0

    for name in os.listdir(folder_yuv):
        if name.endswith('.yuv'):
            filepath = os.path.join(folder_yuv, name)
            with open(filepath, 'rb') as fp:
                size = os.path.getsize(filepath)
                frames = size // (fw * fh + 2 * fwuv * fhuv)  # Correct frame count
                print(f"Processing {filepath}, Total frames: {frames}")

                for frame_idx in range(frames):  # Process all frames
                    Y = np.fromfile(fp, dtype=np.uint8, count=fw * fh).reshape((fh, fw))
                    U = np.fromfile(fp, dtype=np.uint8, count=fwuv * fhuv).reshape((fhuv, fwuv))
                    V = np.fromfile(fp, dtype=np.uint8, count=fwuv * fhuv).reshape((fhuv, fwuv))
                    r, g, b = yuv2rgb(Y, U, V, fw, fh)

                    # Generate patches
                    for i in dx:
                        for j in dy:
                            patch = np.dstack((b[j:j + patchsize, i:i + patchsize],
                                               g[j:j + patchsize, i:i + patchsize],
                                               r[j:j + patchsize, i:i + patchsize]))
                            png_filename = os.path.join(folder_png, f"{png_frame_num}.png")
                            cv2.imwrite(png_filename, patch)
                            png_frame_num += 1

    print(f"Total PNG images generated: {png_frame_num}")
    return png_frame_num



def load_images_from_folder(foldername, patchsize=64):
    files = sorted([os.path.join(foldername, f) for f in os.listdir(foldername) if f.endswith('.png')])
    num_images = len(files)
    
    # Initialize the array for the images
    x = np.zeros((num_images, patchsize, patchsize, 3), dtype=np.float32)
    
    for idx, file in enumerate(files):
        try:
            img = cv2.imread(file)
            if img is None:
                print(f"Error loading image: {file}")
                continue

            # Resize the image to the patchsize, patchsize
            img_resized = cv2.resize(img, (patchsize, patchsize))

            # Convert from BGR to RGB (OpenCV loads in BGR by default)
            x[idx] = img_resized[:, :, ::-1] / 255.0  # Normalize to [0, 1]
        
        except Exception as e:
            print(f"Error processing image {file}: {e}")
    
    return x

def cal_psnr(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def LoadImagesFromFolder (foldername, patchsize):
    dir_list = os.listdir(foldername)
    N = 0
    Nmax = 0
    for name in dir_list:
        fullname = foldername + name
        Nmax = Nmax + 1

    x = np.zeros([Nmax, patchsize, patchsize, 3])
    N = 0
    for name in dir_list:
        fullname = foldername + name
        I1 = cv2.imread(fullname)
        x[N, :, :, 0] = I1[:, :, 2]
        x[N, :, :, 1] = I1[:, :, 1]
        x[N, :, :, 2] = I1[:, :, 0]
        N = N + 1
    return x

def GetRGBFrame (folderyuv,VideoNumber,FrameNumber,fw,fh):
    fwuv = fw // 2
    fhuv = fh // 2
    Y = np.zeros((fh, fw), np.uint8, 'C')
    U = np.zeros((fhuv, fwuv), np.uint8, 'C')
    V = np.zeros((fhuv, fwuv), np.uint8, 'C')

    dir_list = os.listdir(folderyuv)
    v=0
    for name in dir_list:
        fullname = folderyuv + name
        if v!=VideoNumber:
            v = v + 1
            continue
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  
            size = fp.tell()
            fp.close()
            fp = open(fullname, 'rb')
            frames = (2 * size) // (fw * fh * 3)
            for f in range(frames):
                for m in range(fh):
                    for n in range(fw):
                        Y[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        U[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        V[m, n] = ord(fp.read(1))
                if f==FrameNumber:
                    r, g, b = yuv2rgb(Y, U, V, fw, fh)
                    return r,g,b

def GetEngancedRGB (RGBin,enhancer):
    RGBin = np.expand_dims(RGBin, axis=0)
    EnhancedPatches = enhancer.predict(RGBin)
    EnhancedPatches=np.squeeze(EnhancedPatches, axis=0)
    return EnhancedPatches