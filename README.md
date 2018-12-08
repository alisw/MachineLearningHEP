# MachineLearningHF

The current recommended instructions require the installation of a independent ROOT6 framework with Pyhon 3.6. 
Using the package inside AliPhysics is possible but not described in this README.

Follow:
- step 1a) and 2) if you are using MacOs Sierra 
-  step 1b) and 2) if you are using Ubuntu 18.04 

## 1a) Prerequisites for MacOs Sierra 10.13.16 (Validated 1 december 2018)

### Install git
```
pip3 install git
```

### Install python 3.6.6 
Get from the official website https://www.python.org/downloads/mac-osx/ the version Python 3.6.6.
Follow the instructions and install it. The predefined folder should be 
/Library/Frameworks/Python.framework/Versions/3.6/bin/python3.

### Install ROOT with Python 3
This assumes that you are using the python version you just installed placed in /Library/Frameworks/Python.framework/Versions/3.6/bin/python3.
If you want to use another version change the path accordingly. 
```
git clone http://github.com/root-project/root.git
cd root
git checkout -b v6-10-08 v6-10-08
cd ..
mkdir build
sudo chown -R $(whoami):$(id -g -n $(whoami)) build
cd build
cmake -DPYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.6/bin/python3 ../root/
make -j<num_core_to_use>
source bin/thisroot.sh
```
### Add in your bash_profile (if you use Terminal) 
This assumes you have the build in your home folder. If another choice was made change it accordingly
```
source build/bin/thisroot.sh
```

### Install all the Machine Learning softwares 
```

pip3 install numpy pandas scipy matplotlib seaborn
pip3 install pkgconfig uproot
pip3 install scikit-learn sklearn-evaluation xgboost
pip3 install keras
pip3 install -Iv tensorflow==1.5
brew install graphviz
```

## 1b) Prerequisites for Ubuntu 18.04 (Validated 7 december 2018)

The instruction below requires you to start the installation from your home directory!

### Install cmake and other utilities

```
sudo apt-get update
sudo apt-get install git dpkg-dev cmake g++ gcc binutils libx11-dev libxpm-dev libxft-dev libxext-dev
```
You might need also:
```
sudo add-apt-repository universe
sudo apt-get install build-essential
```

### Install python 3.6.6

```
sudo apt-get install python3.6 python3-tk python3-pip
```

### Install ROOT with Python 3

This assumes that you are using the python version you just installed placed in /usr/bin/python3.
If you want to use another version change the path accordingly.
```
git clone http://github.com/root-project/root.git
cd root
git checkout -b v6-10-08 v6-10-08
cd ..
mkdir build
sudo chown -R $(whoami):$(id -g -n $(whoami)) build
cd build
cmake -DPYTHON_EXECUTABLE=/usr/bin/python3 ../root/
make -j<num_core_to_use>
```

Add in your .bashrc

```
source build/bin/thisroot.sh
```

This assumes you have the build in your home folder. If another choice was made change it accordingly

### Install all the Machine Learning softwares

```
sudo apt-get install graphviz
pip3 install numpy pandas scipy matplotlib seaborn
pip3 install pkgconfig uproot
pip3 install scikit-learn sklearn-evaluation xgboost
pip3 install tensorflow keras
```
You might need also this in case you get errors:
```
pip3 install --upgrade setuptools
```

To install tensorflow with GPU support please refer to https://www.tensorflow.org/install/gpu

## 2) Install the ALICE Machine learning tool

### Configure your github environment 
Please use your own name and mail address :D
```
git config --global user.name "Your name"
git config --global user.email mymail@mail.com
```

### Download the ML package and some test files 
```
git clone https://github.com/ginnocen/MachineLearningHF.git
cd MachineLearningHF/ALICEanalysis/MLproductions
scp <my_cern_user>@lxplus.cern.ch:/afs/cern.ch/work/g/ginnocen/public/exampleInputML/*.root .
```

## In case of problems:

For problems ginnocen@cern.ch, fabio.catalano@cern.ch


