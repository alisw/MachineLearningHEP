#!/bin/bash -e

TEST=$1
set -o pipefail
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


FILES_CHANGED="$1"

ERR=0

# Find only python files

echo "$FILES_CHANGED"

[[ "$FILES_CHANGED" == "" ]] && exit 0

for PY in $FILES_CHANGED; do
    [[ -e "$PY" ]] || continue
    check_copyright "$PY" || ERR=1
done
exit $ERR
