import os
import wget
import shutil
import argparse
import glob
from pathlib import Path


def download_dataset(dataset="Kvasir_Capsule"):
    if dataset == "Kvasir_Capsule":
        print("Downloading Kvasir Capsule Dataset from\n 'https://files.osf.io/v1/resources/dv2ag/providers/googledrive/labelled_images/?zip='")
        wget.download('https://files.osf.io/v1/resources/dv2ag/providers/googledrive/labelled_images/?zip=',
                      './data/raw/kvasir-capsule-dataset.zip')
    elif dataset == "Hyper_Kvasir":
        print("Downloading Hyper Kvasir Dataset from\n 'https://datasets.simula.no/hyper-kvasir/hyper-kvasir-labeled-images.zip'")
        wget.download('https://datasets.simula.no/hyper-kvasir/hyper-kvasir-labeled-images.zip',
                      './data/raw/hyper-kvasir-dataset.zip')
    elif dataset == "Kvasir":
        print("Downloading Kvasir Dataset from\n 'https://datasets.simula.no/kvasir/data/kvasir-dataset-v2.zip'")
        wget.download('https://datasets.simula.no/kvasir/data/kvasir-dataset-v2.zip',
                      './data/raw/kvasir-dataset-v2.zip')


def make_dataset(dataset):
    download_dataset(dataset)
    if dataset == "Kvasir_Capsule":
        dir_rename_dict = {
            "Ampulla of vater": "Ampulla_of_vater",
            "Angiectasia": "Angiectasia",
            "Blood - fresh": "Blood_fresh",
            "Blood - hematin": "Blood_hematin",
            "Erosion": "Erosion",
            "Erythema": "Erythema",
            "Foreign body": "Foreign_body",
            "Ileocecal valve": "Ileocecal_valve",
            "Lymphangiectasia": "Lymphangiectasia",
            "Normal clean mucosa": "Normal_clean_mucosa",
            "Polyp": "Polyp",
            "Pylorus": "Pylorus",
            "Reduced mucosal view": "Reduced_mucosal_view",
            "Ulcer": "Ulcer"
        }
        print("Unpacking the Dataset and Preparing")
        shutil.unpack_archive(
            "./data/raw/kvasir-capsule-dataset.zip", "./data/raw/kvasir_capsule_dataset")
        os.remove("./data/raw/kvasir-capsule-dataset.zip")
        for tar_file in glob.glob("./data/raw/kvasir_capsule_dataset/*.gz"):
            print(tar_file)
            shutil.unpack_archive(
                tar_file, "./data/raw/kvasir_capsule_dataset")
            os.remove(tar_file)
        for dir_rename in dir_rename_dict:
            os.rename(f"./data/raw/kvasir_capsule_dataset/{dir_rename}",
                      f"./data/raw/kvasir_capsule_dataset/{dir_rename_dict[dir_rename]}")
    elif dataset == "Kvasir":
        print("Unpacking the Dataset and Preparing")
        shutil.unpack_archive("./data/raw/kvasir-dataset-v2.zip", "./data/raw")
        os.remove("./data/raw/kvasir-dataset-v2.zip")
    elif dataset == "Hyper_Kvasir":
        print("Unpacking the Dataset and Preparing")
        shutil.unpack_archive(
            "./data/raw/hyper-kvasir-dataset.zip", "./data/raw")
        os.remove("./data/raw/hyper-kvasir-dataset.zip")
        p = Path('./data/raw/labeled-images')
        category_dirs = [x for x in p.rglob(
            '*') if x.is_dir() and len(x.parts) >= 6]
        categories = [x.name for x in category_dirs]
        for cat_dir in category_dirs:
            shutil.copytree(cat_dir, "./data/raw/hyper-kvasir/"+cat_dir.name)
        shutil.rmtree('./data/raw/labeled-images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download and Prepare Datasets')
    parser.add_argument('--dataset', required=True, type=str)
    args = parser.parse_args()

    make_dataset(args.dataset)
