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

This project includes a Feedforward Neural Network model that predicts the frequency spectrum of the electric field for a nanophotonic structure. The model takes four design parameters as inputs:

- **w** : Width of the waveguide
- **DC** : Duty cycle
- **Pitch** : Distance between adjacent elements
- **k** : Wave vector (deduced from the value of n_desired and f_desired)

<p align="center"><img src="images/1-SWGwaveguide.png" height="200"><p>
<p align="center"><I>Silicon-on-insulator waveguide with a sub-wavelength grating etched longitudinally or transversely</I></p>
The model is based on the idea that, using these four parameters, the network predicts 5000 values of the electric field spectrum, from which the resonance frequency and the effective refractive index of the structure can be derived.

However, the four parameters and the effective index are not directly linked. Using FDTD simulation, we predict the frequency spectrum of the waveguide based on these design parameters, and then obtain the effective index by extracting the resonance frequency and k. 

### I- EDA
Our dataset is made of the frequency spectrums for various combinations of design parameters, results of various FDTD simulations : 
In other words : 
- X_data=[w,DC,pitch,k] -> 4 values corresponding to the four designs parameters 
- y_data=[..,..,..] -> 5000 values of the electrical field for frequency values 

First, we filter our data to keep only the frequency spectrums that shows one peak (|E|>0.01). We then normalize both x_data1_bigger and y_data1_bigger.

### II- Feedforward model (FFN)

Due to the one-to-many nature of the problem, we cannot directly predict the four parameters from one effective index since multiple designs can correspond to a single effective index.

We start by predicting the frequency spectrum corresponding to four design parameters. This is done using a feedforward network as our response prediction network. Its architecture is defined in `Feedforward_network/feedforward_network_model.py`. This fully connected network has six layers, with hyperparameters like learning rate and hidden sizes optimized using Optuna. 

<p align="center"><img src="images/2-FFN.jpg" height="400"><p>
<p align="center"><I>Feedforward Neural Network (FFN) architecture with four design characteristic parameters of the SWG as input and 5000 values of the electric field for multiple frequency values as output</I></p>

The trained model, that is to say the state of the weights and biases after training, is saved at :

### III- Conditional Variational AutoEncoder (CVAE)

...

### IV- Performance

...

# Contact

> Gabriel **LEGROS**
>
> gabriel.legros@ensta-paris.fr

