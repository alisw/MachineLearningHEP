#!/bin/bash
MCORDATA=$1
ISML=$2

#AliPhysics is needed for the mass fit
if [ -z "${ALICE_PHYSICS}" ]
then
  #load yesterday's tag
  eval `/cvmfs/alice.cern.ch/bin/alienv printenv AliPhysics/vAN-$(date -v-1d +%Y%m%d)_ROOT6-1`
fi

root -b -l <<EOF
.L mass_fitter.C+
mass_fitter("$1","$2");
.q
EOF
