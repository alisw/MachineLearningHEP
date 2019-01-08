#!/bin/bash -e

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

cd "$(dirname "$0")"/..

function swallow() {
  local ERR=0
  local TMPF=$(mktemp /tmp/swallow.XXXX)
  local MSG=$1
  shift
  printf "[    ] $MSG" >&2
  "$@" &> $TMPF || ERR=$?
  if [[ $ERR != 0 ]]; then
    printf "\r[\033[31mFAIL\033[m] $MSG (log follows)\n" >&2
    cat $TMPF
    printf "\n" >&2
  else
    printf "\r[ \033[32mOK\033[m ] $MSG\n" >&2
  fi
  rm -f $TMPF
  return $ERR
}

function check_copyright() {
  local COPYRIGHT="$(cat <<'EOF'
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
EOF
)"
  local COPYRIGHT_LINES=$(echo "$COPYRIGHT" | wc -l)
  [[ "$(head -n$COPYRIGHT_LINES "$1")" == "$COPYRIGHT" ]] || { printf "$1: missing or malformed copyright notice\n"; return 1; }
  return 0
}

if [[ $TRAVIS_PULL_REQUEST != "false" && $TRAVIS_COMMIT_RANGE ]]; then
  # Only check changed Python files (snappier)
  CHANGED_FILES=($(git diff --name-only $TRAVIS_COMMIT_RANGE | grep -E '\.py$' || true))
else
  # Check all Python files
  CHANGED_FILES=($(find . -name '*.py' -a -not -name setup.py))
fi

ERRCHECK=
for PY in "${CHANGED_FILES[@]}"; do
  ERR=
  swallow "$PY: linting" pylint "$PY" || ERR=1
  swallow "$PY: checking copyright notice" check_copyright "$PY" || ERR=1
  [[ ! $ERR ]] || ERRCHECK="$ERRCHECK $PY"
done
[[ ! $ERRCHECK ]] || { printf "\n\nErrors found in:$ERRCHECK\n" >&2; exit 1; }
