# MachineLearningHF

The current recommended instructions require the installation of a independent ROOT6 framework with Pyhon 3.6. 
Use of the package inside AliPhysics is possible but not recommended for the moment.

## Prerequisites for MacOs Sierra (10.13.16)

### Install python 3.6.6 
Get from the official website https://www.python.org/downloads/mac-osx/ the version Python 3.6.6.
Follow the instructions and install it

### ROOT v6-10-08 with Python 3
REMEMBER to replace "path-to-python" with the path where you have the bin of your python installation
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
### Install all the Machine Learning packages
```
pip3 install numpy pandas scipy matplotlib seaborn
pip3 install uproot
pip3 install scikit-learn sklearn-evaluation xgboost
pip3 install tensorflow keras
pip3 install seaborn
```

## Prerequisites for 18.04

### Install python 3.6.6 
sudo apt-get update
sudo apt-get install python3.6 python3-tk python3-pip

### ROOT with Python 3
REMEMBER to replace "path-to-python" with the path where you have the bin of your python installation
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
### Install all the Machine Learning packages

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
pip3 install keras xgboost
```
To install tensorflow with GPU support please refer to https://www.tensorflow.org/install/gpu

## In case of problems:

For problems ginnocen@cern.ch,fabio.catalano@cern.ch
