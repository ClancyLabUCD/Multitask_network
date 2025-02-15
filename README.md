# Multitask-network

We developed a data-driven algorithm intended to address well known shortcomings in the induced pluripotent stem cell-derived cardiomyocyte (iPSC-CMs) platform.  A known concern with iPSC-CMs is that the data collection results in measurements from immature action potentials and it is unclear if these data reliably indicate impact in the adult cardiac environment.  Here, we set out to demonstrate a new deep learning algorithm to classify cells into the drugged and drug free categories and can be used to predict the impact of electrophysiological perturbation across the continuum of aging from the immature iPSC-CM action potential to the adult ventricular myocyte action potential. 

<img src="./results/Fig3/Fig3.png"  width="500" align="center" >

# Dependencies

### Packages

* PyTorch
* Numpy
* Pandas
* sklearn
* Matplotlib
* seaborn

# Directory Structure

* You can find iPSC-CMs and adult-CMs action potential dataset as .txt file format in data folder and the jupyter notebook for data cleaning is shared in jupyter folder to prepare clean and organized data for training the network. 

* Config file in config folder consists all employed hyperparameters for training the network.

* Checkpoints and final model are saved as checkpoint.tar and model_LSTM.pth in results folder.

* You can find the network architecture in network_arch.py file.

* All required parameters for plots are saved in mat.npz file 

* You can change mode to train and train_fram_scatch to True to retrain the network with different hyperparameters using main.py file. 
