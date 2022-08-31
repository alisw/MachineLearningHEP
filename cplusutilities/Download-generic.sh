#!/bin/bash
#IMPORTANT: Before running one should have entered jAliEn environment:
#    jalien
#    <Enter Grid certificate password>
#    exit
#
#Arguments to this bash:
#   $1 is trainname (e.g. 297_20181120-2315)
#   $2 is path to place to save output (e.g. "" or ../ALICEanalysis/MLproductions/)
#   $3 is GRID merging Stage_X (e.g. "" for no merging, or Stage_1)
#
#To set in script (find with "#toset"):
#   OUTPUTFILE (name of file to download)




#----THINGS TO SET----#
nfiles=-1 # to be implemented
outputfile="AnalysisResults.root" #toset
stage=""


# Global variables
ENV_SH="TreeCreator_env.sh"
CHILDREN=""
GENERIC_PATHS=""
INTERACTION_TYPES=""
DATASET_WITH_CHILDREN=""

PWD=$(pwd)

##################
# SOME FUNCTIONS #
##################

make_dir_inside()
{
    # Create a directory inside another one

    local top_dir=$1
    local inside_dir=$2
    local create_dir="$top_dir/$inside_dir"
    [[ "$inside_dir" == "" ]] && create_dir="$top_dir"
    if ! mkdir -p -m 777 "$create_dir"; then
      printf "\n\e[1;31mError: Could not create output directory %s. Is $top_dir writable? Returning... \e[0m\n\n" "$create_dir"
      exit
    else
      printf "Created directory: \e[1m%s\e[0m \n" "$create_dir"
    fi
}

download_env()
{
    # Download the environmnet settings for this train

    echo "Attempt to download $ENV_SH"
    local children=(_child_1 _child_2 _child_3 _child_4 _child_5 _child_6 _child_7 _child_8 _child_9 _child_10 "")
    for c in "${children[@]}"
    do
        if [ ! -e "$ENV_SH" ]
        then
          cmd=$(printf "cp -T 32 /alice/cern.ch/user/a/alitrain/PWGHF/HF_TreeCreator/%s%s/env.sh file://%s/%s\n" "$trainname" "$c" "$PWD" $ENV_SH)
          jalien << EOF
          $cmd
          exit
EOF
          [[ "$c" != "" ]] && DATASET_WITH_CHILDREN="1"
        else
            break
        fi
    done
    if [ ! -e "$ENV_SH" ]
    then
        printf "   \e[1;31mERROR: Downloading %s failed \e[0m\n\n" "$ENV_SH"
    fi
}

make_generic_path()
{
    local out_dir_read="$1"

    local sim_or_data=$(echo $out_dir_read | cut -d '/' -f 3)
    # This is the year
    local year=$(echo $out_dir_read | cut -d '/' -f 4)
    # That is the anchoring
    local anchor=$(echo $out_dir_read | cut -d '/' -f 5)

    local run_number=$(echo $out_dir_read | cut -d '/' -f 6)

    local generic_path="/alice/$sim_or_data/$year/$anchor/*"

    # Check now if we need the pass and potential AOD parts for the path
    local pass_aod=${out_dir_read##*$run_number/}
    if [[ "$(echo $pass_aod | grep $run_number)" == "" ]]
    then
        generic_path="$generic_path/$pass_aod"
    fi
    echo "$generic_path/PWGHF/HF_TreeCreator"
}


get_paths()
{
    # Obtain the generic save paths on the grid from the environment

    if [[ "$DATASET_WITH_CHILDREN" != "1" ]]
    then
        echo "Data set without children"
        GENERIC_PATHS="$(make_generic_path "$ALIEN_JDL_OUTPUTDIR")"
        GENERIC_PATHS=("$GENERIC_PATHS/$trainname/AOD/*/$outputfile" )
        CHILDREN=("child_0" )
        INTERACTION_TYPES=("$ALIEN_JDL_LPMINTERACTIONTYPE" )
    else
        echo "Data set with children"

        # Find children, use exported variable TEST_DIR_child_i
        local children_raw=$(env | grep "TEST_DIR_" | cut -d "=" -f 1)

        for cr in $children_raw
        do
            child="${cr##*TEST_DIR_}"

            # Check if that was enabled
            enabled=$(eval "echo $"RUNNO_${child})
            if [[ "$enabled" == "-1" ]]
            then

                continue
            fi
            CHILDREN+=" $child"

            out_dir_prefix=$(eval "echo $"ALIEN_JDL_${child}"_OUTPUTDIR")
            echo "out_dir_prefix $out_dir_prefix"
            local out_dir="$(make_generic_path $out_dir_prefix)"
            out_dir="$out_dir/${trainname}_${child}/AOD/*/$outputfile"
            GENERIC_PATHS+=" $out_dir"
            INTERACTION_TYPES+=" $(eval "echo $"ALIEN_JDL_${child}"_LPMINTERACTIONTYPE")"
            newline=" $(eval "echo $"ALIEN_JDL_${child}"_LPMINTERACTIONTYPE")"
            echo "Extending INTERACTION_TYPES with $newline"
        done
        # Make arrays
        CHILDREN=($CHILDREN)
        GENERIC_PATHS=($GENERIC_PATHS)
        INTERACTION_TYPES=($INTERACTION_TYPES)
    fi
}

run()
{


    #####################################
    #            PREPARATION            #
    #####################################

    printf "\n\n\n\e[1m----RUNNING THE DOWNLOADER----\e[0m\n\n"

    #Confirm with user if hardcoded values are what he/she wants
    printf "\e[1mYou set the following setters in the script. Please check them carefully before continuing.\e[0m\n"
    printf "   Number of files to download from grid: \e[1m%s\e[0m\n" "$nfiles"
    printf "   Outputfile to be downloaded from grid: \e[1m%s\e[0m\n" "$outputfile"
    printf "   I will download \e[1mnon-merged files\e[0m from GRID\n"
    printf "   Did you authenticate to \e[1mJAliEn\e[0m before running this script?\n"

    printf "\n\e[1m   Are you okay with these settings [y/n]: \e[0m"
    read answer
    if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
      printf "   Thanks for confirming. Continuing...\n\n"
    else
      printf "   \e[1;31mERROR: Please correct... \e[0m\n\n"
      exit
    fi

    #----INITIALIZING----#
    #When arguments are not given, user should provide them while macro is started
    #Checking argument 1, trainname
    if [ -z "$1" ]; then
      printf "Please enter train name: "
      read trainname
      printf "  Will download \e[1m%s\e[0m output from train: \e[1m%s\e[0m \n\n" "$outputfile" "$trainname"
    else
      trainname=$1
      printf "Will download \e[1m%s\e[0m output from train: \e[1m%s\e[0m \n\n" "$outputfile" "$trainname"
    fi

    #Checking argument 2, output directoy
    if [ -z "$2" ]; then
      printf "Please enter output directory: "
      read placetosave
      printf "  Output will be saved in: \e[1m%s\e[0m \n\n" "$placetosave"
    else
      placetosave=$2
      printf "\nOutput will be saved in: \e[1m%s\e[0m \n\n" "$placetosave"
    fi

    #Checking argument 3, GRID merging stage
    if [ -z "$3" ] && [ -z "$stage" ] ; then
      printf "No GRID merging stage was entered. I will download non-merged files\n"
    elif [ -n "$stage" ] ; then
      printf "I will download files from GRID merging: \e[1m%s\e[0m    (if not in format Stage_#, download will fail)\n\n" "$stage"
    else
      stage=$3
      printf "I will download files from GRID merging: \e[1m%s\e[0m    (if not in format Stage_#, download will fail)\n\n" "$stage"
    fi


    #####################################
    #             MAIN PART             #
    #####################################

    # Immediately abort if there is not enough disk space left (meaning less than 1TB)
    local free_space="$(df "$placetosave" | sed -n '2 p' | awk '{print $4}')"
    echo "$free_space"
    if (( free_space < 1000000000 ))
    then
        printf "\e[1;34mWARNING\e[0m: Less than 1 TB available on target disk"
        printf "\n\e[1m   Are you sure that what will be downloaded will fit? [y/n]: \e[0m"
        read answer
        if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
          printf "   Thanks for confirming. Continuing...\n\n"
        else
          printf "   \e[1;31mERROR: Check disk space ... \e[0m\n\n"
          exit
        fi
    fi

    # Try to download train env.sh
    rm $ENV_SH 2>/dev/null
    download_env

    [[ "$?" != "0" ]] && exit 1

    # Get everything into our environment
    source $ENV_SH 2>/dev/null
    [[ "$?" != "0" ]] && exit 1

    # Obtain grid paths
    get_paths

    TAG=${ALIROOT_VERSION:21:20}


    #Creating output directory, and checking if user has writing permission. + dir with used TAG, so it is easier to see which production should be used/can be deleted

    #make_dir_inside $placetosave $TAG
    OUT_DIR_TOP="$placetosave/$TAG"

    placetosave=$OUT_DIR_TOP

    for ch_index in ${!CHILDREN[*]}
    do
        echo "------ INTERACTION_TYPES ${INTERACTION_TYPES[$ch_index]}"
        echo "------ GENERIC_PATHS ${GENERIC_PATHS[$ch_index]}"
        # Obtain anchoring
        sim_or_data=$( echo "${GENERIC_PATHS[$ch_index]}" | cut -d "/" -f 3)
        dataset_short=${INTERACTION_TYPES[$ch_index]}_${sim_or_data}
        # Now it's $TOP_DIR/ALIROOTTAG/pp_{sim,data}
        placetosavearr[$ch_index]=$OUT_DIR_TOP/$dataset_short

        save_paths[$ch_index]="$OUT_DIR_TOP/$dataset_short/$trainname/unmerged/${CHILDREN[$ch_index]}"
    done

    echo "###########################################"
    echo "The downloads' source and target paths:"
    for sp in ${!save_paths[*]}
    do
        echo "${GENERIC_PATHS[$sp]}"
        echo "to"
        echo "${save_paths[$sp]}"
        echo "-----"
    done
    echo "###########################################"

    printf "\n\e[1m   Does the above look right to you? [y/n]: \e[0m"
    read answer
    if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
      printf "   Thanks for confirming. Continuing...\n\n"
    else
      printf "   \e[1;31mERROR: Check train %s ... \e[0m\n\n" "$trainname"
      exit
    fi


    # Check whether save directories already exist
    existing_save_paths=""
    for p in "${save_paths[@]}"
    do
        [[ -d $p ]] && existing_save_paths+=" $p"
    done

    if [[ "$existing_save_paths" != "" ]]
    then
        echo "At least some of the target dorectories already exist:"
        echo "If you are sure, it was also created from this script, you can just continue. JAliEn will just add what is not yet there"
        echo "In case you are not sure how that directory came there, please check first and run again"
        for p in $existing_save_paths
        do
            echo "$p"
        done
        printf "\n\e[1m   You want to continue?... [y/n]: \e[0m"
        read answer
        if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
          printf "   Thanks for confirming. Continuing...\n\n"
        else
          printf "   \e[1;31mERROR: Check target directories ... \e[0m\n\n"
          exit
        fi
    fi




    #Log files ('D' is for download) + trainID, date, and timestamp
    datestamp="$(date +"%d-%m-%Y")"
    timestamp="$(date +"%H-%M-%S")"
    if [ -z "$4" ]; then
      stdoutputfile=$(printf "D_stdout_%s_%s-%s.txt" "$trainname" "$datestamp" "$timestamp")
      stderrorfile=$(printf "D_stderr_%s_%s-%s.txt" "$trainname" "$datestamp" "$timestamp")
    else
      stdoutputfile=$(printf "D_stdout_%s_%s_%s-%s.txt" "$trainname" "$stage" "$datestamp" "$timestamp")
      stderrorfile=$(printf "D_stderr_%s_%s_%s-%s.txt" "$trainname" "$stage" "$datestamp" "$timestamp")
    fi

    #----RUNNING THE DOWNLOADER----#
    printf "\n\n\e[1m----RUNNING THE DOWNLOADER----\e[0m\n\n"
    printf "  Output of downloaders stored in:            \e[1m%s\e[0m\n  Warnings/Errors of downloader stored in:    \e[1m%s\e[0m\n" $i "$stdoutputfile" "$stderrorfile"
    rundownloader="sh ../cplusutilities/downloader-generic.sh"

    printf "\n\n\n\nOutput downloading starts here\n\n" > "$stdoutputfile"
    printf "\n\n\n\nErrors downloading starts here\n\n" > "$stderrorfile"

    for ch_index in ${!CHILDREN[*]}
    do
      make_dir_inside "${save_paths[$ch_index]}"
      cp $ENV_SH "${save_paths[$ch_index]}"

      sh ../cplusutilities/run_downloader $rundownloader "${GENERIC_PATHS[$ch_index]}" "${save_paths[$ch_index]}" $outputfile >> "$stdoutputfile" 2>> "$stderrorfile"
    done

    exit


    #for input_index in ${!inputpaths[*]}
    #do
    #  placetosave=${placetosavearr[$input_index]}
    #  #give all permissions to all directories downloaded from the GRID
    #  chmod -R 777 $placetosave/$trainname/unmerged/
    #  if [ $splitchildsdifferentpaths -eq 0 ]; then
    #    break
    #  fi
    #done

    #Check logs for the comman 'jalien command not found' error. If this is the case, no files were downloaded.
    if grep -q "jalien\|command not found" "$stderrorfile"
    then
      printf "\e[1;31m  Warning: The 'jalien' command was not found, so no new files were downloaded. Did you already connect to JAliEn? Check log if this was not intended!\e[0m\n\n"
    fi
    #Check logs for the comman 'JBox agent could not be started' error. If this is the case, no files were downloaded.
    if grep -q "JBox agent\|could not be started" "$stderrorfile"
    then
    printf "\e[1;31m  Warning: The 'JBox agent' could not be started, so no new files were downloaded. Did you already connect to JAliEn? Check log if this was not intended!\e[0m\n\n"
    fi

    rm $ENV_SH

    for input_index in ${!inputpaths[*]}
    do
      placetosave=${placetosavearr[$input_index]}
      #Saving log files in output directory
      cp "$stdoutputfile" "$placetosave/$trainname/"
      cp "$stderrorfile" "$placetosave/$trainname/"
      printf "\e[1mMoved log files to %s/%s/\e[0m\n" "$placetosave" "$trainname"
      printf "\e[1m----DOWNLOADER FINISHED----\e[0m\n\n"
      if [ "$splitchildsdifferentpaths" -eq 0 ]; then
        break
      fi
    done
    rm "$stdoutputfile"
    rm "$stderrorfile"

    printf "\n\e[1m<<<Ready downloading? Please kill JAliEn daemons>>>\e[0m\n"
    printf "  killall java\n"
    printf "\e[1m<<<And remove alien logs if you like>>>\e[0m\n"
    printf "  rm alien-config* alien-fine* alien-info* alien-severe* alien-warning*\n"
    printf "  rm ../cplusutilities/alien-config* ../cplusutilities/alien-fine* ../cplusutilities/alien-info* ../cplusutilities/alien-severe* ../cplusutilities/alien-warning*\n"
}


# Don't source it
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]
then
    echo "Don't source but run in sub-shell"
else
    run $@
fi
