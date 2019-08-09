# Getting and processing TTreeCreator output

Instructions to download the output from the LEGO train (can be run as part of the package or stand-alone), and merge the files (only stand-alone). The instructions assume you are an user of the new aliceml server. With some small changes, the instructions are valid for each system though.

## 1) Setup your environment

Start by logging in
```
ssh -X username@lxplus.cern.ch #only when needed
ssh -X username@aliceml
```
> Please have a look at section 4) if you want to use these script on a local system or different server, as some packages are already pre-installed at aliceml which you might need to install yourself first.

### a) Building and loading the virtual environment

While logged in at the server, one should (create/)load your personal virtual environment.
```
ml-create-virtualenv     #only once to create the environment
ml-activate-virtualenv   #start (and enable python) in virtual environment
ml-activate-root         #Enable system-wide ROOT installation
```
and clone+install this git repository (see https://github.com/ginnocen/MachineLearningHEP/wiki)

## 2) Download the train output

Before downloading, one has to enter the JAliEn environment manually. Please make sure your GRID certificates are copied to the server.
```
jalien
#Enter Grid Certificate password
exit
```
> **NB:** If you get the error: "**JBox isn't running, so we won't start JSh.**", your grid certificates probably don't have the right permissions. Correct them in *~/.globus/* using: *"chmod 0440 usercert.pem"* and *"chmod 0400 userkey.pem"*. It seems that JALiEn needs slightly different permissions than usual.
 
 
The scripts are saved in *~/MachineLearningHEP/cplusutilities. There are three ways to run the Download.sh script:
1) Enable the "download alice" flag in default_complete.yaml and run the MLHEP package the usual way. The script will ask you for the required input.
2) Run ./Download.sh **without** arguments. The script will ask you for the required input.
3) Run ./Download.sh **with** arguments: *./Download.sh trainname outputdirectory gridmergingstage* 
 
The first argument is the trainname, which has the format: *trainnumber_date-time*. With this info, the script automatically loads the correct dataset name and AliPhysics tag. The second argument is the output directory. Please do *not* use your local folder. The last argument is the GRID merging stage, which should be in the format Stage_#. If this argument is empty, JAliEn will download the unmerged files from GRID. If (some of the) arguments are empty, the script will ask for your input.
 

### a) Hardcoded values

A few variables are hardcoded in *Download.sh*:
1) The number of files to download from GRID. By default all files will be downloaded: **nfiles="/*/"**. For test runs, one can add some zeros ("/000*/", assuming 1000 < jobs < 9999) to download less files.
2) The file to be downloaded is by default: **AnalysisResults.root**.
3) There are hardcoded paths for the different datasets from where to get the LEGO train output. Unfortunately, it is not possible to automatically get these from the train config, as some of the child are splitted into multiple paths when the output is too big. For debugging purposes, the script will print the hardcoded paths with the ones it can get from the train config.

### b) The screen program

If one will download and process the full statistics, the *screen* program can be very usefull. This program allows you to keep programs running on the remote computer even if you disconnect from it. To use it, one should do:
```
ssh -X username@lxplus.cern.ch
screen    #A empty terminal will pop up
#Do everything till the Download.sh script is running
#Important to do this from lxplus, JAliEn together with screen on aliceml will not work
```
When the script is running, you can detach from it by pressing **Ctrl-a d**. You will find yourself back in the previous terminal. Before quiting, there is some information you should remember:
```
screen -list     #Should print something like: "There is a screen on: 32693.pts-30.lxplus008    (Detached)"
hostname         #Should print something like: "lxplus008.cern.ch"
```
Save this information somewhere, disconnect from lxplus, and start doing something else.

When the script is ready, or you want to check the progress, just do:
```
ssh username@lxplus008.cern.ch          #Change to your situation
screen -list
screen -rD 32693.pts-30.lxplus008       #Change to your situation
```
Is the download finished? Exit the *screen* program with **Ctrl-d**. Is the script still running, detach again with **Ctrl-a d**.


## 3) Post download merging
[Instructions to be improved] Run the post_download.sh script. One has to be in the AliPhysics environment before starting.
```
./post_download.sh --input /path/where/data/is/stored/upto/trainID --target-size 500000 --jobs 50
```


## 4) Installation on a local system/different server

The JAliEn tool is needed for downloading from the GRID. An installation of alibuild is needed (follow https://alice-doc.github.io/alice-analysis-tutorial/building/). Afterwards, one can build JAliEn (the installation should take a few minutes only).
```
mkdir -p ~/alice
cd ~/alice/
git clone https://github.com/alisw/alidist
aliBuild build JAliEn --defaults jalien -z jalien
```
> JAliEn is already installed at lxplus. To enter the environment, do '/cvmfs/alice.cern.ch/bin/alienv enter JAliEn'

You may need to edit the hard-coded jalien path (`/opt/jalien/src/jalien/jalien`) in `downloader.sh` to 
something appropriate for your system, e.g. simply `jalien`. 

ROOT is needed for the merging of the files. If this is not yet installed, please follow the instructions below. **Please note that these instructions don't build against a specific python version, which you might need for ML studies.**
```
git clone http://github.com/root-project/root.git
cd root
git checkout -b v6-14-04 v6-14-04    #Or any other version listed by 'git tag -l'
mkdir -p ../rootbuild
cd ../rootbuild
cmake ../root
```
Please change **N** into the number of cores to be used.
```
cmake --build . -- -jN
```
When ready, source ROOT in your ~/.bashrc
```
source $HOME/alice/rootbuild/bin/thisroot.sh    #change directory if needed
```
and source your .bashrc
```
source ~/.bashrc
```

## In case of problems:

For problems luuk.vermunt@cern.ch
