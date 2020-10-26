#!/bin/bash

function packing()
{
    local output_dir="$1"
    local file_pack="$2"
    local logfile="$3"
    echo "===> Write pack to $output_dir"
    mkdir -p $output_dir
    hadd $output_dir/AnalysisResults.root $file_pack > $logfile 2>&1
}

function n_job_delay()
{
    local n_max_jobs="$1"
    local sleep_time="$2"
    while true
    do
        n_packing=$(jobs | grep "packing" | grep "Running" | wc -l )
        if (( $n_packing >= $n_max_jobs ))
        then
            sleep $sleep_time
        else
            break
        fi
    done
}

# Some colors
STREAM_START_RED="\033[0;31m"
STREAM_START_GREEN="\033[0;32m"
STREAM_START_YELLOW="\033[0;33m"
STREAM_START_BOLD="\033[1m"
STREAM_END_FORMAT="\033[0m"

# Top directory where pointing down to the train number
# In that directory the further structure then should be
# .../unmerged/child_N/nnnn/AnalysisResults.root
INPUT_PATH="$1"

# We start from here
CURR_DIR=$(pwd)

# To do some logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=$HOME/train_post_download/$TIMESTAMP

# Normal upper bound for merged files
TARGET_PACK_SIZE="1000000"

# Accepted size if one input file is already bigger
MAX_ACCEPTED_INPUT_SIZE="5000000"
# Number of those
MAX_ACCEPTED_BIG_INPUT="45"

# Number of packing jobs
N_PACKING_JOBS=20

MERGED_DIR="merged"

MAX_SEARCH_DEPTH="2"

FORCE=false

function print_usage()
{
    echo -e "Usage: post_download \n" \
                                   "[[ [-i | --input <input_directory> top directory where \"unmerged\" directory can be found] \n" \
                                   "   [-o | --output <output_directory> output directory where \"merged\" directory will be placed]\n" \
                                   "   [-s | --target-size <size> target size in kB]\n" \
                                   "   [-m | --max-input-size <max_input_size> in case input file might be bigger that target size]\n" \
                                   "   [-n | --n-max-size <n_max_size> number of big files accepted]\n" \
                                   "   [-j | --jobs <n_jobs> number of ROOT \"hadd\" jobs]]\n" \
                                   "   [-f | --force do not ask for confirmation]]\n" \
                                   " | [-h | --help print this help message and exit]]"
}

function check_settings()
{
    [[ "$INPUT_PATH" == "" ]] && { echo -e "${STREAM_START_RED}ERROR:${STREAM_END_FORMAT} Input file required"; exit 1; }
    [[ ! -d $INPUT_PATH ]] && { echo -e "${STREAM_START_RED}ERROR:${STREAM_END_FORMAT} $INPUT_PATH is no directory"; exit 1; }
    echo -e "${STREAM_START_BOLD}Settings chosen${STREAM_END_FORMAT}\n" \
            "input path $INPUT_PATH \n" \
            "target size: $TARGET_PACK_SIZE kB \n" \
            "maximum accepted input size: $MAX_ACCEPTED_INPUT_SIZE kB\n" \
            "number of big files accepted: $MAX_ACCEPTED_BIG_INPUT \n" \
            "number of jobs: $N_PACKING_JOBS \n"
    echo
    echo -e "Merged files will be written to ${STREAM_START_BOLD}$OUTPUT_PATH/merged${STREAM_END_FORMAT}."
    echo -e "Log files will be written to ${STREAM_START_BOLD}${LOG_DIR}${STREAM_END_FORMAT}."
    echo

    if [ "${FORCE}" = true ]; then
        sleep 5;
    else
        echo -e "${STREAM_START_BOLD}Do you agree with these settings? [y/n]${STREAM_END_FORMAT}"
        read answer
        if [[ "$answer" != "Y" && "$answer" != "y" ]]
        then
            echo "Abort, you were not satisfied apparently. Set your desired values and start again. If you need help, use \"--help\" flag."
            exit 0
        fi
    fi
}

# Check whether ROOT is loaded.
if [ -z "$ROOTSYS" ]
then
    echo "Error: ROOT has not been loaded."
    exit 1
fi

# Command line arguments
while [ "$1" != "" ]
do
    case $1 in
        -i | --input )              shift
                                    INPUT_PATH="$1"
                                    ;;
        -o | --output )             shift
                                    OUTPUT_PATH="$1"
                                    ;;
        -s | --target-size )        shift
                                    TARGET_PACK_SIZE="$1"
                                    ;;
        -m | --max-input-size )     shift
                                    MAX_ACCEPTED_INPUT_SIZE="$1"
                                    ;;
        -n | --n-max-size )         shift
                                    MAX_ACCEPTED_BIG_INPUT="$1"
                                    ;;
        -j | --jobs )               shift
                                    N_PACKING_JOBS="$1"
                                    ;;
        -d | --max-search-depth )   shift
                                    MAX_SEARCH_DEPTH="$1"
                                    ;;
        -f | --force )              FORCE=true
                                    ;;
        -h | --help )               shift
                                    print_usage
                                    exit 0
                                    ;;
        * )                         shift
                                    echo -e "${STREAM_START_RED}ERROR:${STREAM_END_FORMAT} Unknown argument $1"
                                    print_usage
                                    exit 0
                                    ;;
    esac
    shift
done

# Make it an absolute path...
INPUT_PATH=$(realpath $INPUT_PATH)
OUTPUT_PATH=${OUTPUT_PATH:-$INPUT_PATH}

check_settings

echo "#####"
echo "#####"
echo "MERGING GRID DATA up to target file size of $TARGET_PACK_SIZE kB"
echo "#####"
echo "#####"
echo

# ... and go there
cd $INPUT_PATH

# check for unmerged directory
[[ ! -d "./unmerged" ]] && { echo -e "${STREAM_START_RED}ERROR${STREAM_END_FORMAT}: Cannot find unmgered directory"; exit 1; }

# keep the old data savely and produce the merged data as well. This assumes
# there is nothing but the ROOT data from the grid
unmerged_size="$(du -s | awk '{print $1}' )"
free_space="$(df . | grep "/dev" | awk '{print $4}')"

if (( $free_space < $unmerged_size ))
then
    echo -e "${STREAM_START_RED}ERROR${STREAM_END_FORMAT}: Not enough disk space left"
fi


# Fail if "merged" directory exists already
[[ -d "${OUTPUT_PATH}/$MERGED_DIR" ]] && { echo -e "${STREAM_START_RED}ERROR${STREAM_END_FORMAT}: Seems that the merge directory already exists"; exit 1; }

# To do some logging
mkdir -p $LOG_DIR
rm -rf $LOG_DIR/* > /dev/null 2>&1
echo "===> Find log files in $LOG_DIR"
echo

# If we are here, things sould be fine
#mkdir "merged"

# Merge per child so find out children we have
children=$(find unmerged -maxdepth 1 -type d -name "child_*" | sort -u)

echo "===> Found children"
echo "$children"
echo

# Make the merged dir
mkdir $MERGED_DIR
for c in $children
do
    c_stripped=${c##unmerged/}
    echo "===> Process $c_stripped"
    # For each child_i there will be a pack_i
    MERGED_CHILD_DIR=$MERGED_DIR/$c_stripped
    mkdir -p $MERGED_CHILD_DIR
    root_files_children=""
    # Search down to certain depth if requested, standard is 2
    if [[ "$MAX_SEARCH_DEPTH" != "-1" ]]
    then
        root_files_children=$(find $c -maxdepth $MAX_SEARCH_DEPTH -type f -name "AnalysisResults.root")
    else
        root_files_children=$(find $c -type f -name "AnalysisResults.root")
    fi

    if [[ "$root_files_children" == "" ]]
    then
        echo -e "${STREAM_START_YELLOW}WARNING${STREAM_END_FORMAT}: No ROOT files found in $c"
        continue
    fi
    n_packs="0"
    file_pack=""
    current_size="0"
    n_big_files="0"
    for rfc in $root_files_children
    do
        next_size=$(du -s $rfc | awk '{print $1}')
        if (( $next_size > $MAX_ACCEPTED_INPUT_SIZE ))
        then
            echo -e "${STREAM_START_RED}ERROR${STREAM_END_FORMAT}: File $rfc is bigger than $MAX_ACCEPTED_INPUT_SIZE kB. Not accepted..."
            exit 1
        fi

        if [[ "$file_pack" == "" ]]
        then
            file_pack+="$rfc "
            current_size=$(( $current_size + $next_size ))
            if (( ( $current_size < $MAX_ACCEPTED_INPUT_SIZE ) && ( $current_size > $TARGET_PACK_SIZE ) ))
            then
                n_big_files=$(( $n_big_files + 1 ))
                if (( $n_big_files > $MAX_ACCEPTED_BIG_INPUT ))
                then
                    echo -e "${STREAM_START_RED}ERROR${STREAM_END_FORMAT}: More than $MAX_ACCEPTED_BIG_INPUT big files are not accepted"
                    exit 1
                fi
            fi

            continue
        fi
        if (( ($current_size < $TARGET_PACK_SIZE)  && ( $TARGET_PACK_SIZE > ( $next_size + $current_size ) )))
        then
            file_pack+="$rfc "
            current_size=$(( $current_size + $next_size ))
        else
            output_dir="$MERGED_CHILD_DIR/pack_${n_packs}"
            n_job_delay $N_PACKING_JOBS 10
            log_file=$LOG_DIR/${c_stripped}_${n_packs}.log
            packing $OUTPUT_PATH/$output_dir "$file_pack" $log_file &
            # Need to add that since it would be skipped otherwise
            file_pack="$rfc "
            current_size="$next_size"
            n_packs="$(( $n_packs + 1 ))"
        fi
    done

    # Handle the last pack
    output_dir="$MERGED_CHILD_DIR/pack_${n_packs}"
    n_job_delay $N_PACKING_JOBS 10
    log_file=$LOG_DIR/${c_stripped}_${n_packs}.log
    packing $OUTPUT_PATH/$output_dir "$file_pack" $log_file &

done

echo "Wait for all jobs to finish"
n_job_delay 1 10

echo "DONE"
echo "You may want to delete log files in $LOG_DIR"
