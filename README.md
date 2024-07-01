# emc_torch_package

Package to run Bloch equation simulations using PyTorch. GPU acceleration can be enabled.
The implementation is based on Ben-Eliezer et al. 2015 (https://doi.org/10.1002/mrm.25156).

## Install

Recommended and tested is to clone git repository.
From the repository root, run with a name of your choice:

`conda env create -f environment.yml -n <env_name>`

`conda activate <env_name>`

## Usage

The Package contains simulation of dictionary databases as well as the fitting of data to such databases.
#### Simulation
To simulate with configuration found in the "example/simulate" folder run:

`python simulate.py`


To see all options run

`python simulate.py --help`


The simulation needs information about Gradients and RF events.
The basic inputs can be seen in the "example/simulate/emc_params.json" file.
Default configuration will create a database and some plots within the "example/simulate/" folder.


#### Fit
To fit with a configuration found in the "example/fit" folder run:

`python fit.py`


To see all options run

`python fit.py --help`


The fit can take additional input files and needs specification of data and database.
The basic inputs can be seen in the "example/fit/fit_config.json" file.
Default configuration will use the data within the "examples/fit" folder and b1 regularization.


