# Instructions

### Install python 3.6.6 
Get from the official website https://www.python.org/downloads/mac-osx/ the version Python 3.6.6.
Follow the instructions and install it. The predefined folder should be 
/Library/Frameworks/Python.framework/Versions/3.6/bin/python3.

### Install, create and load virtual environment 

```bash
pip3 install virtualenv
mkdir $HOME/.python-virtual-environments && cd $HOME/.python-virtual-environments
python3 -m venv env
source $HOME/.python-virtual-environments/env/bin/activate
```

### Clone and install the package and all the needed dependences (package is not needed at this stage but it comes with a automatic installation procedure of all needed libraries
```bash
git clone https://github.com/ginnocen/MachineLearningHEP.git
source $HOME/.python-virtual-environments/env/bin/activate
cd MachineLearningHEP
pip3 install -e . 
```
### Install jupyter lab
```bash
pip3 install jupyterlab
```

### Run a jupyter notebook
```bash
jupyter lab
```
This will open a browser. Click on the available example ExampleDataFrame.ipynb in the folder Notebooks to load the example. 
To execute each cell, click on Shift+Return




