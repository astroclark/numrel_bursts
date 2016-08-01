#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016 James Clark <james.clark@ligo.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# nrburst_bwpostsub.sh
#
# USAGE: nrburst_bwpostsub.sh <injdir> <sub-file>
#
# Write a trivial condor submission file to speed up BW match calculations

executable="${HOME}/Projects/numrel_bursts/nrburst_utils/nrburst_bwpost.sh"
log_dir="condor_logs"

workdir_globpattern=${1}
subfile=${2}.sub
dagfile=${2}.dag
logdir=${2}_logs

test ! -d ${logdir} && mkdir -p ${logdir}
test ! ${dagfile} && rm ${dagfile}
test ! ${subfile} && rm ${subfile}

echo '
######################
# bayeswave archives #
######################

executable = '${executable}'
universe   = vanilla 
getenv=True

arguments  = $(macroworkdir)
output     = '${logdir}'/$(workdir).out
error      = '${logdir}'/$(workdir).err
log        = '${logdir}'/$(workdir).log
queue 1

' > ${subfile}

for workdir in ${workdir_globpattern}*
do

    echo ${workdir}

    echo '
    JOB '${workdir}'
    RETRY '${workdir}' 0
    VARS '${workdir}' macroworkdir="'${workdir}'"

    ' >> ${dagfile}



done

#python -c "import uuid; print uuid.uuid4()"
