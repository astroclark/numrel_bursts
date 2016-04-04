#!/bin/bash

catalogfile="GaTechCatalog.xml.gz"
gpsstart=1126621184 # first CO2 frame for O1
gpsend=`python -c "print ${gpsstart} + 1000"`
outfile="GaTechIMBBH.xml"
#outfile_with_fref="GaTechIMBBH_with_fref.xml"

lalapps_inspinj \
    --seed 1234 --f-lower 30 --gps-start-time ${gpsstart} \
    --gps-end-time ${gpsend} --waveform NR_hdf5threePointFivePN \
    --amp-order 0 \
    --ninja2-mass --nr-file ${catalogfile} \
    --time-step 1 --time-interval 0 --l-distr random \
    --dchirp-distr uniform --i-distr uniform \
    --min-distance 50000 --max-distance 50000 \
    --m-distr nrwaves --disable-spin \
    --min-mtotal 100. --max-mtotal 100.\
    --taper-injection start --output ${outfile}

# add the fref
#pycbc_add_fref_to_sim_inspiral_table \
#    --catalog-file ${catalogfile} \
#    --input-file ${outfile} \
#    --output-file ${outfile_with_fref}
