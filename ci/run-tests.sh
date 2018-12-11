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
