from __future__ import print_function
import os
import sys
import gzip
import codecs

# Parameters
dataset_base_url = 'http://yann.lecun.com/exdb/mnist/'
dataset_dpath = 'data/mnist'
dataset_fnames = ['train-images-idx3-ubyte.gz',
                 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz']
dataset_outputs = ['train.bin', 'test.bin']
dataset_fsizes = [9912422, 28881, 1648877, 4542]

def download_file(url, path):
    file_name = url.split('/')[-1]
    try:
        # For Python 3.0 and later
        from urllib.request import urlopen
        u = urlopen(url)
        meta = u.info()
        file_size = int(meta.get_all("Content-Length")[0])
    except ImportError:
        # Fall back to Python 2's urllib2
        from urllib2 import urlopen
        u = urlopen(url)
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (file_name, file_size))

    f = open(os.path.join(path, file_name), 'wb')
    download_size = 0
    block_size = 8192
    while True:
        buf = u.read(block_size)
        if not buf:
            break
        download_size += len(buf)
        f.write(buf)
        status = "\r%12d  [%3.2f%%]" % (download_size, download_size * 100. / file_size)
        print(status, end='')
        sys.stdout.flush()
    print("")
    f.close()

if not os.path.exists(dataset_dpath):
    os.makedirs(dataset_dpath)

download_cnt = 0
for fname, fsize in zip(dataset_fnames, dataset_fsizes):
    gz_fpath = os.path.join(dataset_dpath, fname)
    if not os.path.exists(gz_fpath) or os.path.getsize(gz_fpath) != fsize:
        print('Downloading %s' % fname)
        url = dataset_base_url + fname
        download_file(url, dataset_dpath)
        download_cnt += 1
if download_cnt == 0:
    print('MNIST original dataset exists')

def byte_to_int(bstr):
    return int(codecs.encode(bstr, 'hex'), 16)

generate_cnt = 0
for i, output_fname in enumerate(dataset_outputs):
    output_fpath = os.path.join(dataset_dpath, output_fname)
    if not os.path.exists(output_fpath):
        print('Generating %s' % output_fname)
        image_fname, label_fname = dataset_fnames[2*i:2*(i+1)]
        with gzip.open(os.path.join(dataset_dpath, image_fname)) as infile:
            image = infile.read()
        with gzip.open(os.path.join(dataset_dpath, label_fname)) as infile:
            label = infile.read()
        num_data = byte_to_int(image[4:8])
        print('%d images' % num_data)
        data = bytearray()
        for j in range(num_data):
            data += label[8+j:9+j]
            data += image[16+j*28*28:16+(j+1)*28*28]
        with open(output_fpath, 'wb') as fd:
            fd.write(data)
        generate_cnt += 1
if generate_cnt == 0:
    print('MNIST generated dataset exists')

print('done!')
