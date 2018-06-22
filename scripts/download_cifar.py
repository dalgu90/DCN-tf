import os
import sys
import random
import tarfile
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="The name of dataset")
args = parser.parse_args()

if 'cifar-10' == args.dataset:
    # CIFAR-10 download parameters
    dataset_name = 'CIFAR-10'
    dataset_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    dataset_dpath = 'data/cifar-10-binary'
    dataset_fname = 'cifar-10-binary.tar.gz'
    dataset_fsize = 170052171
if 'cifar-100' == args.dataset:
    # CIFAR-100 download parameters
    dataset_name = 'CIFAR-100'
    dataset_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
    dataset_dpath = 'data/cifar-100-binary'
    dataset_fname = 'cifar-100-binary.tar.gz'
    dataset_fsize = 169001437

def download_file(url, path):
    import urllib.request
    file_name = url.split('/')[-1]
    u = urllib.request.urlopen(url)
    f = open(os.path.join(path, file_name), 'wb')
    meta = u.info()
    file_size = int(meta.get_all("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (file_name, file_size))

    download_size = 0
    block_size = 8192
    while True:
        buf = u.read(block_size)
        if not buf:
            break
        download_size += len(buf)
        f.write(buf)
        status = "\r%12d  [%3.2f%%]" % (download_size, download_size * 100. / file_size)
        print(status, end="")
        sys.stdout.flush()
    print("")
    f.close()

if not os.path.exists(dataset_dpath):
    os.makedirs(dataset_dpath)

tar_fpath = os.path.join(dataset_dpath, dataset_fname)
if not os.path.exists(tar_fpath) or os.path.getsize(tar_fpath) != dataset_fsize:
    print('Downloading %s' % dataset_name)
    download_file(dataset_url, dataset_dpath)
    print('Extracting %s' % dataset_name)
    with tarfile.open(tar_fpath) as tar:
        tar.extractall(path=dataset_dpath)
    print('done')
else:
    print('%s  exists\nNothing to be done... Quit!' % dataset_name)
    sys.exit(0)
