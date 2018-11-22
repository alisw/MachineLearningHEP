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

### 22/11/2018: 
Above instructions need te be updated for MacOs. One now also need to install:
```
pip3 install keras xgboost
```
If you followed above instructions (with some modifications for apt-get), and are now running into problems with aliBuild, please have a look at the following suggestions that solved it for me:
* Python-modules is build again (and crashes): Probably one of the required Python packages was updated. Following the aliBuild prerequisites for macOS solves it:
```
sudo pip install --upgrade --force-reinstall matplotlib numpy certifi ipython==5.1.0 ipywidgets ipykernel notebook metakernel pyyaml
```
* AliPhysics builds but some of the final tests fail: Probably gcc was updated in the process. Switch back to gcc version used for the earlier aliBuild:
```
brew switch gcc 7.3.0_1
```

### ROOT with Python 3
It is necessary to build ROOT with python 3.6. It is easier to make a new build (independent of aliBuild), using
```
git clone http://github.com/root-project/root.git
cd root
git tag -l
git checkout -b v6-10-08 v6-10-08
mkdir <builddir>
cd <builddir> 
cmake -DPYTHON_EXECUTABLE=/path-to-python/3.6/bin/python3 -Dpython3=ON -DPYTHON_INCLUDE_DIR=/path-to-python/3.6/Headers -DPYTHON_LIBRARY=/path-to-python/3.6/lib/libpython3.6.dylib ../root/
cmake --build .
source /path/to/builddir/dir/bin/thisroot.sh
```

## Prerequisites for Ubuntu (validated for Ubuntu 18.04 at 14/11/18)
### Python 3
This code is based on python3, to install it
```
sudo apt-get update
sudo apt-get install python3.6 python3-tk python3-pip
```
### ROOT with Python 3
It is necessary to build ROOT with python 3.6 while the Ubuntu default is python 2.7, a way to do this using alibuild and update-alternatives is
```
sudo update-alternatives --install /usr/local/bin/python python /usr/bin/python2.7 10
sudo update-alternatives --install /usr/local/bin/python python /usr/bin/python3.6 20
Install alibuild and the python prerequisites of aliBuild with pip3 (instead of pip)
Install the ALICE software normally with aliBuild
```
Then it is possible to switch between python2 and python3 with
```
sudo update-alternatives --config python
```
without affecting ROOT.  
Before running the code the alienv envirovment must be loaded. 

### ML dependencies

```
pip3 install numpy pandas scipy matplotlib seaborn
pip3 install uproot
pip3 install scikit-learn sklearn-evaluation xgboost
pip3 install tensorflow keras
```
To install tensorflow with GPU support please refer to https://www.tensorflow.org/install/gpu

For problems or improvements about Ubuntu prerequisites contact fabio.catalano@cern.ch  

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
