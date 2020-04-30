#!/bin/bash -e

TEST=$1
#set -o pipefail
cd "$(dirname "$0")"/..

#############################################################################
##  © Copyright CERN 2018. All rights not expressly granted are reserved.  ##
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


# this function strip comments, empty lines and double spaces from input stream
function clean-lines()
{
    sed 's,^#.*,,' | sed '/^$/ d' | sed 's,  *, ,g'
}

ERR=0

# Configure ignore
PYLINT_IGNORE=""
PYLINT_IGNORE_FILE="ci/pylint-ignore"
FLAKE8_IGNORE=""
FLAKE8_IGNORE_FILE="ci/flake8-ignore"
[[ -e $PYLINT_IGNORE_FILE ]] && PYLINT_IGNORE=$(cat $PYLINT_IGNORE_FILE | clean-lines)
[[ -e $FLAKE8_IGNORE_FILE ]] && FLAKE8_IGNORE=$(cat $FLAKE8_IGNORE_FILE | clean-lines)

function install-package()
{
    pip3 install --upgrade --force-reinstall --no-deps -e .
}

function ignore-file()
{
    local f=$1
    shift
    local ignore="$@"
    local found=""
    for i in $ignore
    do
        found="$(echo $f | grep $i || :)"
        [[ "$found" != "" ]] && break
    done
    echo $found
}

function swallow() {
    local ERR=0
    local TMPF=$(mktemp /tmp/swallow.XXXX)
    local MSG=$1
    shift
    printf "[    ] $MSG" >&2
    "$@" &> $TMPF || ERR=$?
    if [[ $ERR != 0 ]]; then
        printf "\r[\033[31mFAIL\033[m] $MSG (log follows)\n" >&2
    else
        printf "\r[ \033[32mOK\033[m ] $MSG\n" >&2
    fi
    cat $TMPF
    printf "\n" >&2
    rm -f $TMPF
    return $ERR
}


function test-pylint()
{
    local test_files=$@
    echo "run test: pylint"
    type pylint
    if [[ "$test_files" != "" ]]
    then
        for tf in $test_files; do
            [[ -e "$tf" ]] || continue
            [[ "$(echo $tf | grep '.py$' || :)" != "" ]] || continue
            [[ "$(ignore-file $tf $PYLINT_IGNORE)" != "" ]] && { echo "File $tf set to be ignored"; continue; }
            echo "File $tf "
            swallow "linting $tf" pylint $tf || ERR=1
        done
    fi
}


function test-flake8()
{
    local test_files=$@
    echo "run test: flake8"
    type flake8
    if [[ "$test_files" != "" ]]
    then
        for tf in $test_files; do
            [[ -e "$tf" ]] || continue
            [[ "$(echo $tf | grep '.py$' || :)" != "" ]] || continue
            [[ "$(ignore-file $tf $FLAKE8_IGNORE)" != "" ]] && { echo "File $tf set to be ignored"; continue; }
            echo "File $tf "
            swallow "flaking $tf" flake8  $tf --count --select=E9,F63,F7,F82 --show-source --statistics || ERR=1
            swallow "flaking (treat as warnings) $tf" flake8  $tf --exit-zero --max-complexity=10 --max-line-length=127 --statistics
        done
    fi
}


function test-pytest()
{
    install-package
    echo "run test: pytest"
    type pytest
    swallow "pytest ci/pytest" pytest ci/pytest || ERR=1
}


function test-case()
{
    case "$1" in
        pylint)
            shift
            test-pylint $@
            ;;
        flake8)
            shift
            test-flake8 $@
            ;;
        pytest)
            shift
            test-pytest
            ;;
        *)
            echo "Unknown test case $1"
            ;;
    esac
}


function test-all()
{
    echo "Run all tests"
    test-pylint $@
    test-flake8 $@
    test-pytest
}


function print-help()
{
    echo
    echo "run_tests.sh usage to run CI tests"
    echo ""
    echo "run_tests.sh [  --tests pylint|pytest (all tests if not given)"
    echo "              | --files <files> (all tracked Python files if not given) ]  # defaults to all"
    echo ""
    echo "Possible test cases are:"
    echo "  pylint                        # run style tests for copyright and pylint"
    echo ""
    echo "--help|-h                       # Show this message and exit"
}


[[ $# == 0 ]] && { echo "ERROR: Arguments required" ; print-help ; exit 1; }

FILES=""
TESTS=""
CURRENT_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in

        --tests)
            CURRENT_ARG="tests"
            ;;
        --files)
            CURRENT_ARG="files"
            ;;
        --help|-h)
            print-help
            exit 1
            ;;

        *)
            case "$CURRENT_ARG" in
                files)
                    FILES+=" $1"
                    ;;
                tests)
                    TESTS+=" $1"
                    ;;
                *)
                    echo "Unknown option $1"
                    print-help
                    exit 2
                    ;;
            esac
            ;;
    esac
    shift
done

# No files given, take all there is tracked by git
if [[ "$FILES" == "" ]]
then
    # Do it like this because otherwise so that all .py files which are not tracked
    # are not tested
    FILES="$(git ls-tree -r HEAD --name-only | grep '.py$')"
fi

# If there are still no files, nothing to do
[[ "$FILES" == "" ]] && { echo "Nothing to do..."; exit 0; }

if [[ "$TESTS" == "" ]]
then
    test-all $FILES
else
    for t in $TESTS
    do
        echo "Do test for $t"
        test-case $t $FILES
    done
fi

exit $ERR
