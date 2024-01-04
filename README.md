# Important Notes
This work draws from https://github.com/neu-spiral/SparCL, using their implementation an personal additions

# Running configurations
Running the code is done through modifying the configurations in ./configurations and baselines/configs (for baselines)
After all sparcl configs are run baselines will be run
It is important to note that if you want to run baselines you msut ensure that you have run a HART + SparCL baseline to set the dataset
If you just want to run baselines just run the icarl baseline folder

# Data Generation
To generate data to run this project please generate the Data using the SensorBasedTransformerTorch library
To generate the CNN data please use the selfhar_testing library /feature_extraction/ part to generate the features