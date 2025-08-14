# Inverse design of subwavelength grating (SWG) waveguides in silicon photonic devices

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-306998?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

## CONTENTS 
* [Structure](#STRUCTURE)
* [Project overview](#PROJECT-OVERVIEW)
* [Getting started](#getting-started-on-your-machine)
* [Models](#MODELS)
    * [Feedforward Network](#ii--feedforward-model-ffn)
    * [Conditional Variational AutoEncoder](#iii--CVAE)
* [Contact](#Contact)

## STRUCTURE

### CN2-CVAE_FNN-Inverse-Design/
- **CVAE/**
    - `cvae.py`
    - `encoder.py`
    - `decoder.py`
    - `feedforwardNN.py`
- **saved_models/**
    - `cvae_trained_2.pth`
    - `ffn_big_trained.pth`
    - `xgb_model.json`
-  **utils**
    - `dataload.py`
    - `functions.py`
    - `explore_data.ipynb`
- `cvae_modeling.ipynb`

## PROJECT OVERVIEW 

This project for nanophotonics design optimization involves developing a **FeedForward Neural Network** and a **Conditional Variational Autoencoder** (CVAE) to optimize the design of silicon nanophotonic subwavelength grating waveguides. This tool aims to predict and adjust the waveguide design parameters to achieve a specific refractive index at a given wavelength.

## GETTING STARTED ON YOUR MACHINE

1. **Download the repository** : Clone this Git repository to your local machine
2. **Install python** : Ensure that you have Python and the pip command installed on your system.
3. **Install dependencies** : Run the script *requirements.txt* to install all necessaries dependencies by executing in your terminal the following command : 
```
pip install -r requirements.txt
```
Once you have everything set up, you can run the project by opening `cvae_modeling.ipynb`.

Run the first cells **"Imports et chargement des données"**.

In **"Génération synthétique automatique"**, define the desired values of effective index and frequency, the errors. Then run `hybrid_model_evaluation()`:

<img width="798" height="273" alt="image" src="https://github.com/user-attachments/assets/79059d16-c74b-463a-99eb-a11ee49fec28" />

## MODELS

### I- EDA
Our dataset is made of the frequency spectrums for various combinations of design parameters, results of various FDTD simulations : 
In other words : 
- X_data=[w,DC,pitch,k] -> 4 values corresponding to the four designs parameters 
- y_data=[..,..,..] -> 5000 values of the electrical field for frequency values 

First, we filter our data to keep only the frequency spectrums that shows one peak (|E|>0.01). We then normalize both x_data1_bigger and y_data1_bigger.

### II- Feedforward model (FFN)

This project implements a **Feedforward Neural Network (FNN)** designed to predict the **frequency spectrum** of the electric field for a nanophotonic structure.

The model takes **four design parameters** as input:

* **w** — Width of the waveguide
* **DC** — Duty cycle
* **Pitch** — Distance between adjacent elements
* **k** — Wave vector (computed from `n_desired` and `f_desired`)

Using these inputs, the network outputs **5,000 points** representing the electric field spectrum. From this spectrum, it is possible to extract the **resonance frequency** and the **effective refractive index** of the structure.


### III- Conditional Variational AutoEncoder (CVAE)

...

### IV- Performance

...

# Contact

> Gabriel **LEGROS**
>
> gabriel.legros@ensta-paris.fr

