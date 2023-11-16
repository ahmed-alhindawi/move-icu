import hashlib
import os

import requests
from tqdm import tqdm


def md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def request_if_not_exist(file_name, url, md5sum=None, chunksize=1024):
    if not os.path.isfile(file_name):
        request = requests.get(url, timeout=10, stream=True)
        with open(file_name, 'wb') as fh:
            # Walk through the request response in chunks of 1MiB
            for chunk in tqdm(request.iter_content(chunksize), desc=os.path.basename(file_name),
                              total=int(int(request.headers['Content-length']) / chunksize),
                              unit="KiB"):
                fh.write(chunk)
        if md5sum is not None:
            assert md5sum == md5(
                file_name), "MD5Sums do not match for {}. Please) delete the same file name to re-download".format(
                file_name)


def download_landmark_models():
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models/landmark_1.ckpt"), 
                         "https://liveuclac-my.sharepoint.com/:u:/g/personal/rmhaa84_ucl_ac_uk/EZQ9Mtu_1DdAnWAD6GCuhccB_ZZ8eI8exMXE2HbkZUdBKg?e=YVu04k&download=1", 
                         "2a9abd82b48a98aff95681e7be5170fc")
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models/landmark_2.ckpt"), 
                         "https://liveuclac-my.sharepoint.com/:u:/g/personal/rmhaa84_ucl_ac_uk/EY6Ttc8iPSFCksrzSj0RSwwBs_YekN2_piMCL_-ufTl91g?e=F4avU5&download=1", 
                         "ee192a42cf32f40e2596024d2dc6d880")
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models/landmark_3.ckpt"), 
                         "https://liveuclac-my.sharepoint.com/:u:/g/personal/rmhaa84_ucl_ac_uk/EbBjre_QOCxGmp-swjSmz9kBGrG-pnCTN6k5mhUeQV9F9A?e=KJFiTR&download=1", 
                         "a1d792f8e7d8369a581bcd107a0f70bc")


def download_landmark_jit_models():
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models/landmark_1.ckpt.jit_model"), 
                         "https://liveuclac-my.sharepoint.com/:u:/g/personal/rmhaa84_ucl_ac_uk/ERlkLLMHIXBDo3aDdEIwta8BbWrCqNBLetTBXMF-aMshRw?e=3XduQD&download=1", 
                         "9eac3e709edc5366c5183a43a48369b6")
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models/landmark_2.ckpt.jit_model"), 
                         "https://liveuclac-my.sharepoint.com/:u:/g/personal/rmhaa84_ucl_ac_uk/EVnejtdYFR9BvbrINZbg-uUBKxd4lOHiLKo1shozRapDrA?e=2GpEMp&download=1", 
                         "958e59a7af5d93289a3b401aac15d446")
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models/landmark_3.ckpt.jit_model"), 
                         "https://liveuclac-my.sharepoint.com/:u:/g/personal/rmhaa84_ucl_ac_uk/Ee7YqtynmGtNsv4-iRjtyawBoxHWTjpCCFytA0k5Qy25MQ?e=hW4Vb7&download=1", 
                         "ff70436449c66130f4126e030b943f90")


def download_s3fd_model():
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models/s3fd_facedetector.ckpt"), 
                         "https://liveuclac-my.sharepoint.com/:u:/g/personal/rmhaa84_ucl_ac_uk/EdN3iC9BVqFMrSyvhEoIb1IBRwMdYx3txi1bg7wRyQ9gVA?e=65ywZo&download=1", 
                         "3b5a9888bf0beb93c177db5a18375a6c")
    

download_landmark_models()
download_landmark_jit_models()
download_s3fd_model()
