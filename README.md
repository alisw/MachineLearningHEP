# MachineLearningHF


## Prerequisites

Unless you have complicated existing configurations, this should work out of the box for a Mac Sierra. 
With tiny modifications for UBUNTU. 
```
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get -y install python3-pip
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3.6
pip3 install jupyter matplotlib numpy pandas scipy scikit-learn
sudo apt-get install git
pip3 install seaborn
sudo apt-get install python3-tk
pip3 install sklearn-evaluation
```

## Produce the ML ntuples and convert it to dataframes

### Download the input files for MC and Data
Copy the folder MLDsproductions and put it in your HOME directory. The folder is in the public folder in lxplus below:
```
ginnocen@lxplus.cern.ch:/afs/cern.ch/work/g/ginnocen/public/MLDsproductions
```
### Produce the ML ntuples and convert it to dataframes
Simpy run the following code:
```
cd ALICEanalysis/buildsample
source buildMLTree.sh
```

## In case of problems:

For problems ginnocen@cern.ch
