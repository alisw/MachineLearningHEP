#############################################################################
##  © Copyright CERN 2023. All rights not expressly granted are reserved.  ##
##                 Author: Gian.Michele.Innocenti@cern.ch                  ##
## This program is free software: you can redistribute it and/or modify it ##
##  under the terms of the GNU General Public License as published by the  ##
## Free Software Foundation, either version 3 of the License, or (at your  ##
## option) any later version. This program is distributed in the hope that ##
##  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  ##
##     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    ##
##           See the GNU General Public License for more details.          ##
##    You should have received a copy of the GNU General Public License    ##
##   along with this program. if not, see <https://www.gnu.org/licenses/>. ##
#############################################################################

import os
import shutil

from .logger import get_logger

logger = get_logger()

def list_folders(main_dir, filenameinput, maxfiles, select=None): # pylint: disable=too-many-branches
    """
    List all files in a subdirectory structure
    """
    if not os.path.isdir(main_dir):
        logger.error("the input directory = <%s> does not exist", main_dir)
    list_subdir0 = os.listdir(main_dir)
    listfolders = []
    for subdir0 in list_subdir0: # pylint: disable=too-many-nested-blocks
        subdir0full = os.path.join(main_dir, subdir0)
        if os.path.isdir(subdir0full):
            list_subdir1 = os.listdir(subdir0full)
            for subdir1 in list_subdir1:
                subdir1full = os.path.join(subdir0full, subdir1)
                if os.path.isdir(subdir1full):
                    if os.listdir(subdir1full)[0] == filenameinput:
                        list_files_ = os.listdir(subdir1full)
                        for myfile in list_files_:
                            filefull = os.path.join(subdir1full, myfile)
                            if os.path.isfile(filefull) and \
                               myfile == filenameinput:
                                listfolders.append(os.path.join(subdir0, subdir1))
                    else:
                        list_subdir2 = os.listdir(subdir1full)
                        for subdir2 in list_subdir2:
                            subdir2full = os.path.join(subdir1full, subdir2)
                            if os.path.isdir(subdir2full):
                                list_files_ = os.listdir(subdir2full)
                                for myfile in list_files_:
                                    filefull = os.path.join(subdir2full, myfile)
                                    if os.path.isfile(filefull) and \
                                       myfile == filenameinput:
                                        listfolders.append(os.path.join(subdir0, subdir1, subdir2))
    if select:
        # Select only folders with a matching sub-string in their paths
        list_folders_tmp = []
        for sel_sub_string in select:
            list_folders_tmp.extend([folder for folder in listfolders if sel_sub_string in folder])
        listfolders = list_folders_tmp

    if maxfiles != -1:
        listfolders = listfolders[:maxfiles]
    return  listfolders

def create_folder_struc(maindir, listpath):
    """
    Reproduce the folder structure as input
    """
    for path in listpath:
        path = path.split("/")
        folder = maindir
        for _, element in enumerate(path):
            folder = os.path.join(folder, element)
            if not os.path.exists(folder):
                os.makedirs(folder)

def checkdirlist(dirlist):
    """
    Checks if list of folder already exist, to not overwrite by accident
    """
    exfolders = 0
    for mydir in dirlist:
        if os.path.exists(mydir):
            print("rm -rf ", mydir)
            exfolders = exfolders - 1
    return exfolders

def checkdir(mydir):
    """
    Checks if folder already exist, to not overwrite by accident
    """
    exfolders = 0
    if os.path.exists(mydir):
        print("rm -rf ", mydir)
        exfolders = -1
    return exfolders

def checkmakedir(mydir):
    """
    Makes directory using 'mkdir'
    """
    if os.path.exists(mydir):
        logger.warning("Using existing folder %s", mydir)
        return
    logger.debug("creating folder %s", mydir)
    os.makedirs(mydir)

def checkmakedirlist(dirlist):
    """
    Makes directories from list using 'mkdir'
    """
    for mydir in dirlist:
        checkmakedir(mydir)

def delete_dir(path: str):
    """
    Delete directory if it exists. Return True if success, False otherwise.
    """
    if not os.path.isdir(path):
        logger.warning("Directory %s does not exist", path)
        return True
    logger.warning("Deleting directory %s", path)
    try:
        shutil.rmtree(path)
    except OSError:
        logger.error("Error: Failed to delete directory %s", path)
        return False
    return True

def delete_dirlist(dirlist: str):
    """
    Delete directories from list. Return True if success, False otherwise.
    """
    for path in dirlist:
        if not delete_dir(path):
            return False
    return True

def appendfiletolist(mylist, namefile):
    """
    Append filename to list
    """
    return [os.path.join(path, namefile) for path in mylist]

def appendmainfoldertolist(prefolder, mylist):
    """
    Append base foldername to paths in list
    """
    return [os.path.join(prefolder, path) for path in mylist]

def createlist(prefolder, mylistfolder, namefile):
    """
    Appends base foldername + filename in list
    """
    listfiles = appendfiletolist(mylistfolder, namefile)
    listfiles = appendmainfoldertolist(prefolder, listfiles)
    return listfiles
