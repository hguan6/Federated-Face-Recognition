## Federated-Facial-Recognition
A project to recognize 10k US faces in a federated learning setting

### Installation guide:
Current version of `flwr` does not support Mac with M1 chip. Please consider using an Docker linux image.

We recommend using conda to manage the develpment environment. For Linux machine, installation is simple:
```
conda env create -n fed --file environment.yml  
```
If you already create an environment with python3.10, then run
```
conda env update -n [your-environment-name] --file environment.yml
```

For Mac, we recommend installing `nomkl` before installing other packages.
```
conda env create -n fed python=3.10
conda install -c conda-forge nomkl
conda env update -n fed --file environment.yml
```


**Note:** This guide does not include the installation of the [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) package. 
