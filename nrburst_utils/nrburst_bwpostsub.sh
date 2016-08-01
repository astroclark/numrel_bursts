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

executable="${HOME}/Projects/numrel_bursts/nrburst_utils/nrburst_bwpost.sh"
log_dir="condor_logs"

usertag=${1}
subfile=${usertag}.sub

test ! -d ${log_dir} && mkdir -p ${log_dir}

echo '
######################
# bayeswave archives #
######################

executable = '${executable}'
universe   = vanilla 
getenv=True
' > ${subfile}

for injrun in injruns
do
    echo '

    arguments  = '${injrun}'
    output     = condor_logs/'${injrun}'.out
    error      = condor_logs/'${injrun}'.err
    log        = condor_logs/'${injrun}'.log
    queue 1
    ' >> ${subfile}

done

