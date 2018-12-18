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


#!/bin/bash -e
cd "$(dirname "$0")"/..

function swallow() {
  local ERR=0
  local TMPF=$(mktemp /tmp/swallow.XXXX)
  printf "$1: " >&2
  shift
  "$@" &> $TMPF || ERR=$?
  if [[ $ERR != 0 ]]; then
    printf "FAILED (log follows)\n" >&2
    cat $TMPF
    printf "\n" >&2
  else
    printf "OK\n" >&2
  fi
  rm -f $TMPF
  return $ERR
}

# Pylint
ERRPY=
while read PY; do
  swallow "Linting $PY" pylint "$PY" || ERRPY="$ERRPY $PY"
done < <(find . -name '*.py' -a -not -name setup.py)
[[ ! $ERRPY ]] || { printf "\n\npylint errors in:$ERRPY\n" >&2; exit 1; }
