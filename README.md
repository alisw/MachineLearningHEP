# MachineLearningHF


## Prerequisites (fully validated only for MacOs Sierra 10.13.16)

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

Copy the folder MLDsproductions and put it in your HOME directory. The folder is in the public folder in lxplus below:
```
ginnocen@lxplus.cern.ch:/afs/cern.ch/work/g/ginnocen/public/MLDsproductions
```
Simpy run the following code to perform the ML training creating and convertion:
```
cd ALICEanalysis/buildsample
source buildMLTree.sh
```

## In case of problems:

For problems ginnocen@cern.ch
