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
# bw_match_sub.sh
#
# Write a trivial condor submission file to speed up BW match calculations

tag="GW150914"
ini="theevent.ini"

executable="${HOME}/Projects/numrel_bursts/nrburst_utils/nrburst_netmatch.py"
log_dir="condor_logs"
hdf5files="${1}"

subfiletag=`echo ${hdf5files} | sed "s/.txt//g"`
subfile="bw_matches_${subfiletag}.sub"


test ! -d ${log_dir} && mkdir -p ${log_dir}

echo '
####################
# nrburst_match.py #
####################

executable = '${executable}'
universe   = vanilla 
getenv=True
' > ${subfile}

for hdf5 in `cat ${hdf5files}`
do

    for minsample in `seq 0 100 900`
    do 
        maxsample=$((${minsample}+99))

        echo '

        arguments  = '${ini}' --user-tag='${tag}' --hdf5file '${hdf5}' --min-sample '${minsample}' --max-sample '${maxsample}'
        output     = condor_logs/'${hdf5}'.out
        error      = condor_logs/'${hdf5}'.err
        log        = condor_logs/'${hdf5}'.log
        queue 1
        ' >> ${subfile}

    done

done

