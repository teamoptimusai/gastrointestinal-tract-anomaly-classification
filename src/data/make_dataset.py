import os

def download_dataset(dataset = "Kvasir_Capsule"):  
    if dataset == "Kvasir_Capsule":
        os.system(f'cmd /k "!wget -O "labelled_images.zip" https://files.osf.io/v1/resources/dv2ag/providers/googledrive/labelled_images/?zip="')
    elif dataset == "Hyper_Kvasir":
        os.system(f'cmd /k "!wget https://datasets.simula.no/hyper-kvasir/hyper-kvasir-labeled-images.zip"')
    elif dataset == "Kvasir":
        None

def make_dataset(dataset, download_dir = "./data/raw"):
    None
