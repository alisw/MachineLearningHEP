#!/bin/bash

if [[ -z "$3" ]]
then
    echo "Usage: $0 <default database> <variation database> <analysis>"
    exit 0
fi

DB_DEFAULT="$1"
DB_VARIATION="$2"
ANALYSIS="$3"
DO_PROC="$4"


#CMD_VAR="nice python do_variations.py ${DB_DEFAULT} ${DB_VARIATION} -p ${DO_PROC}"
CMD_VAR="nice python do_variations.py ${DB_DEFAULT} ${DB_VARIATION} "

echo -e "\n $CMD_VAR"

${CMD_VAR}


# Exit if error.
if [ ! $? -eq 0 ]; then echo "Error"; exit 1; fi

RUN=0

echo -e "\nDo you wish to run these variations?"
while true; do
    read -p "Answer: " yn
    case $yn in
        [y] ) echo "Proceeding"; RUN=1; break;;
        [n] ) echo "Aborting"; break;;
        * ) echo "Please answer y or n.";;
    esac
done

if ((RUN))
then
    echo -e "\nRunning variations"
    ${CMD_VAR} -a ${ANALYSIS}
else
    echo -e "\nCleaning databases"
    ${CMD_VAR} -c
fi

exit 0

