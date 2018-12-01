# MachineLearningHF

The current recommended instructions require the installation of a independent ROOT6 framework with Pyhon 3.6. 
Using the package inside AliPhysics is possible but not described in this README.

Follow:
- step 1a) and 2) if you are using MacOs Sierra 
-  step 1b) and 2) if you are using Ubuntu 18.04 

## 1a) Prerequisites for MacOs Sierra 10.13.16 (Validated 1 december 2018)

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
make -j20
source bin/thisroot.sh
```
### Add in your bash_profile (if you use Terminal) 
This assumes you have the build in your home folder. If another choice was made change it accordingly
```
source build/bin/thisroot.sh
```

### Install all the Machine Learning softwares 
```
pip3 install jupyter matplotlib numpy pandas scipy scikit-learn
pip3 install seaborn
pip3 install sklearn-evaluation
pip3 install keras xgboost
pip3 install --upgrade setuptools
sudo pip3 install uproot
sudo pip3 install -U virtualenv
pip3 install --upgrade tensorflow
pip3 install --upgrade tensorflow-gpu
```

## 1b) Prerequisites for Ubuntu 18.04 (Validated 1 december 2018)

The instruction below requires you to start the installation from your home directory! 

### Install cmake and other utilities 
```
sudo apt-get install git dpkg-dev cmake g++ gcc binutils libx11-dev libxpm-dev libxft-dev libxext-dev
```
### Install python 3.6.6 
```
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install python3.6 python3-tk python3-pip python3-dev
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
make -j20
source bin/thisroot.sh
```
### Add in your bashrc 
This assumes you have the build in your home folder. If another choice was made change it accordingly
```
source build/bin/thisroot.sh
```

### Install all the Machine Learning softwares 
```
sudo apt-get update
sudo apt-get install build-essential
pip3 install jupyter matplotlib numpy pandas scipy scikit-learn
sudo apt-get install git
pip3 install seaborn
pip3 install sklearn-evaluation
pip3 install keras xgboost
pip3 install --upgrade setuptools
sudo pip3 install uproot
sudo pip3 install -U virtualenv
pip3 install --upgrade tensorflow
pip3 install --upgrade tensorflow-gpu
```

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
scp ginnocen@lxplus.cern.ch:/afs/cern.ch/work/g/ginnocen/public/exampleInputML/*.root .
```

## In case of problems:

For problems ginnocen@cern.ch,fabio.catalano@cern.ch























<!-- 

# MachineLearningHF

The current recommended instructions require the installation of a independent ROOT6 framework with Pyhon 3.6. 
Use of the package inside AliPhysics is possible but not recommended for the moment.

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

-->

