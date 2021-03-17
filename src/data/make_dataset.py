import os
import wget
import shutil
import argparse
import glob

def download_dataset(dataset = "Kvasir_Capsule"):  
    if dataset == "Kvasir_Capsule":
        print("Downloading Kvasir Capsule Dataset from\n 'https://files.osf.io/v1/resources/dv2ag/providers/googledrive/labelled_images/?zip='")
        wget.download('https://files.osf.io/v1/resources/dv2ag/providers/googledrive/labelled_images/?zip=', './data/raw/labelled_images.zip')
    elif dataset == "Hyper_Kvasir":
        None
    elif dataset == "Kvasir":
        None

def make_dataset(dataset):
    download_dataset(dataset)
    if dataset == "Kvasir_Capsule":
      print("Unpacking the Dataset and Preparing")
      shutil.unpack_archive("./data/raw/labelled_images.zip", "./data/raw/kvasir_capsule_dataset")
      os.remove("./data/raw/labelled_images.zip")
      for tar_file in glob.glob("./data/raw/kvasir_capsule_dataset/*.gz"):
        print(tar_file)
        shutil.unpack_archive(tar_file, f"./data/raw/kvasir_capsule_dataset/{tar_file[34:-7]}")
        os.remove(tar_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download and Prepare Datasets')
    parser.add_argument('--dataset', required=True, type=str)
    args = parser.parse_args()

    make_dataset(args.dataset)