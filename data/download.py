import os
import tarfile
import gdown

filepath = "mvtec_anomaly_detection.tar.xz"

if not os.path.exists(filepath):
    
    url1 = 'https://drive.google.com/uc?id=124Zz6vsCKaBPPynkPcrJfCkgZ_ftBBTc'
    gdown.download(url1, quiet=False)

    with tarfile.open(filepath) as tar:
        tar.extractall(path="")