"""
Downloads the following:
- Glove vectors
- Stanford Natural Language Inference (SNLI) Corpus

"""

import sys
import os
import zipfile
import gzip

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    os.system('wget {} -O {}'.format(url, filepath))
    return filepath

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def download_wordvecs(dirpath):
    if os.path.exists(dirpath):
        print('Found Glove vectors - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
    unzip(download(url, dirpath))

def download_multinli(dirpath):
    if os.path.exists(dirpath):
        print('Found MultiNLI dataset - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip'
    unzip(download(url, dirpath))


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))
    multinli_dir = os.path.join(base_dir, 'multinli')
    wordvec_dir = os.path.join(base_dir, 'glove')
    download_multinli(multinli_dir)
    # download_wordvecs(wordvec_dir)

