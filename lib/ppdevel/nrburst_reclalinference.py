#!/usr/bin/env python
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
"""
nrburst_reclalinference.py

Post-process posterior samples from lalinference to produce the time-domain
reconstructed waveform

"""

import sys, os
from optparse import OptionParser

import numpy as np
from matplotlib import pyplot as pl

import pycbc.types
import pycbc.waveform 
import pycbc.waveform.utils as wfutils
from pycbc.detector import Detector

from pylal import bayespputils as bppu

def make_waveform(pos=None,ifos=['H1','L1']):#,'V1']):
    
    # time and freq data handling variables
    srate=4096.0
    seglen=60.
    length=srate*seglen # lenght of 60 secs, hardcoded. May call a LALSimRoutine to get an idea
    deltaT=1/srate
    deltaF = 1.0 / (length * deltaT);

    # build window for FFT
    pad=0.4
    timeToFreqFFTPlan = CreateForwardREAL8FFTPlan(int(length), 1 );
    window=CreateTukeyREAL8Window(int(length),2.0*pad*srate/length);
    WinNorm = sqrt(window.sumofsquares/window.data.length);
    # time and freq domain strain:
    segStart=100000000
    strainT=CreateREAL8TimeSeries("strainT",segStart,0.0,1.0/srate,DimensionlessUnit,int(length));
    strainF= CreateCOMPLEX16FrequencySeries("strainF",segStart,	0.0,	deltaF,	DimensionlessUnit,int(length/2. +1));

    f_min=25 # hardcoded default (may be changed below)
    f_ref=100 # hardcoded default (may be changed below)
    f_max=srate/2.0
    plot_fmax=f_max

    inj_strains=dict((i,{"T":{'x':None,'strain':None},"F":{'x':None,'strain':None}}) for i in ifos)
    rec_strains=dict((i,{"T":{'x':None,'strain':None},"F":{'x':None,'strain':None}}) for i in ifos)

    inj_domain=None
    rec_domain=None
    font_size=26

    # Select the maxP sample
    _,which=pos._posMap()

    if 'time' in pos.names:
        REAL8time=pos['time'].samples[which][0]
    elif 'time_maxl' in pos.names:
        REAL8time=pos['time_maxl'].samples[which][0]
    elif 'time_min' in pos.names and 'time_max' in pos.names:
        REAL8time=pos['time_min'].samples[which][0]+0.5*(pos['time_max'].samples[which][0]-pos['time_min'].samples[which][0])
    else:
        print "ERROR: could not find any time parameter in the posterior file. Failing...\n"
        return None

    # first check we have approx in posterior samples, otherwise skip
    skip=0
    try:
        approximant=int(pos['LAL_APPROXIMANT'].samples[which][0])
        amplitudeO=int(pos['LAL_AMPORDER'].samples[which][0])
        phaseO=int(pos['LAL_PNORDER'].samples[which][0])
    except:
        skip=1
    if skip==0:
        GPStime=LIGOTimeGPS(REAL8time)

    q=pos['q'].samples[which][0]
    mc=pos['mc'].samples[which][0]
    M1,M2=bppu.q2ms(mc,q)
    if 'dist' in pos.names:
        D=pos['dist'].samples[which][0]
    elif 'distance' in pos.names:
        D=pos['distance'].samples[which][0]
    elif 'logdistance' in pos.names:
        D=exp(pos['distance'].samples[which][0])

    m1=M1*LAL_MSUN_SI
    m2=M2*LAL_MSUN_SI

    if 'phi_orb' in pos.names:
        phiRef=pos['phi_orb'].samples[which][0]
    elif 'phase_maxl' in pos.names:
        phiRef=pos['phase_maxl'].samples[which][0]
        print 'INFO: phi_orb not estimated, using maximum likelihood value'
    else:
        print 'WARNING: phi_orb not found in posterior files. Defaulting to 0.0 which is probably *not* what you want\n'
        phiRef=0.0
    
    try:
        for name in ['flow','f_lower']:
            if name in pos.names:
                f_min=pos[name].samples[which][0]
    except:
        pass
    try:
        for name in ['fref','f_ref','f_Ref','fRef']:
            if name in pos.names:
                fname=name

    Fref = np.unique(pos[fname].samples)
    if len(Fref) > 1:
        print "ERROR: Expected f_ref to be constant for all samples.  Can't tell which value was injected! Defaulting to 100 Hz\n"
        print Fref
    else:
        f_ref = Fref[0]
    except ValueError:
      print "WARNING: Could not read fref from posterior file! Defaulting to 100 Hz\n"

    try:
        a = pos['a1'].samples[which][0]
        the = pos['theta_spin1'].samples[which][0]
        phi = pos['phi_spin1'].samples[which][0]
        s1x = (a * sin(the) * cos(phi));
        s1y = (a * sin(the) * sin(phi));
        s1z = (a * cos(the));
        a = pos['a2'].samples[which][0]
        the = pos['theta_spin2'].samples[which][0]
        phi = pos['phi_spin2'].samples[which][0]
        s2x = (a * sin(the) * cos(phi));
        s2y = (a * sin(the) * sin(phi));
        s2z = (a * cos(the));
        iota=pos['inclination'].samples[which][0]
    except:
          try:
              iota, s1x, s1y, s1z, s2x, s2y, s2z=lalsim.SimInspiralTransformPrecessingNewInitialConditions(pos['theta_jn'].samples[which][0], pos['phi_JL'].samples[which][0], pos['tilt1'].samples[which][0], pos['tilt2'].samples[which][0], pos['phi12'].samples[which][0], pos['a1'].samples[which][0], pos['a2'].samples[which][0], m1, m2, f_ref)
          except:
              if 'a1z' in pos.names:
                  s1z=pos['a1z'].samples[which][0]
              elif 'a1' in pos.names:
                  s1z=pos['a1'].samples[which][0]
              else:
                  s1z=0
            if 'a2z' in pos.names:
                s2z=pos['a2z'].samples[which][0]
            elif 'a2' in pos.names:
                s2z=pos['a2'].samples[which][0]
            else:
                s2z=0
            s1x=s1y=s2x=s2y=0.0
            if 'inclination' in pos.names:
                iota=pos['inclination'].samples[which][0]
            else:
                iota=pos['theta_jn'].samples[which][0]

      r=D*LAL_PC_SI*1.0e6

      lambda1=0
      lambda2=0
      waveFlags=None
      nonGRparams=None
      approximant=int(pos['LAL_APPROXIMANT'].samples[which][0])
      amplitudeO=int(pos['LAL_AMPORDER'].samples[which][0])
      phaseO=int(pos['LAL_PNORDER'].samples[which][0])

      if SimInspiralImplementedFDApproximants(approximant):
        rec_domain='F'
        [plus,cross]=SimInspiralChooseFDWaveform(phiRef, deltaF,  m1, m2, s1x, s1y, s1z,s2x,s2y,s2z,f_min, f_max,   f_ref,r,   iota, lambda1,   lambda2,waveFlags, nonGRparams, amplitudeO, phaseO, approximant)
      elif SimInspiralImplementedTDApproximants(approximant):
        rec_domain='T'
        [plus,cross]=SimInspiralChooseTDWaveform(phiRef, deltaT,  m1, m2, s1x, s1y, s1z,s2x,s2y,s2z,f_min, f_ref,  r,   iota, lambda1,   lambda2,waveFlags, nonGRparams, amplitudeO, phaseO, approximant)
      else:
        print "The approximant %s doesn't seem to be recognized by lalsimulation!\n Skipping WF plots\n"%approximant
        return None

      ra=pos['ra'].samples[which][0]
      dec=pos['dec'].samples[which][0]
      psi=pos['psi'].samples[which][0]
      fs={}
      for ifo in ifos:
        (fp,fc,fa,qv)=ant.response(REAL8time,ra,dec,iota,psi,'radians',ifo)
        if rec_domain=='T':
          # strain is a temporary container for this IFO strain.
          # Take antenna pattern into accout and window the data
          for k in np.arange(strainT.data.length):
            if k<plus.data.length:
              strainT.data.data[k]=((fp*plus.data.data[k]+fc*cross.data.data[k]))
            else:
              strainT.data.data[k]=0.0
            strainT.data.data[k]*=window.data.data[k]
          # now copy in the dictionary only the part of strain which is not null (that is achieved using plus.data.length as length)
          rec_strains[ifo]["T"]['strain']=np.array([strainT.data.data[k] for k in arange(plus.data.length)])
          rec_strains[ifo]["T"]['x']=np.array([REAL8time - deltaT*(plus.data.length-1-k) for k in np.arange(plus.data.length)])

          # Take the FFT
          for j in arange(strainF.data.length):
            strainF.data.data[j]=0.0
          REAL8TimeFreqFFT(strainF,strainT,timeToFreqFFTPlan);
          for j in arange(strainF.data.length):
            strainF.data.data[j]/=WinNorm
          # copy in the dictionary
          rec_strains[ifo]["F"]['strain']=np.array([strainF.data.data[k] for k in arange(int(strainF.data.length))])
          rec_strains[ifo]["F"]['x']=np.array([strainF.f0+ k*strainF.deltaF for k in arange(int(strainF.data.length))])
        elif rec_domain=='F':
          for k in np.arange(strainF.data.length):
            if k<plus.data.length:
              strainF.data.data[k]=((fp*plus.data.data[k]+fc*cross.data.data[k]))
            else:
              strainF.data.data[k]=0.0
          # copy in the dictionary
          rec_strains[ifo]["F"]['strain']=np.array([strainF.data.data[k] for k in arange(int(strainF.data.length))])
          rec_strains[ifo]["F"]['x']=np.array([strainF.f0+ k*strainF.deltaF for k in arange(int(strainF.data.length))])

  myfig=plt.figure(1,figsize=(23,15))

  rows=len(ifos)
  cols=2

  #this variables decide which domain will be plotted on the left column of the plot.
  # only plot Time domain if both injections and recovery are TD
  global_domain="F"
  if rec_domain is not None and inj_domain is not None:
    if rec_domain=="T" and inj_domain=="T":
      global_domain="T"
  elif rec_domain is not None:
    if rec_domain=="T":
      global_domain="T"
  elif inj_domain is not None:
    if inj_domain=="T":
      global_domain="T"

  A,axes=plt.subplots(nrows=rows,ncols=cols,sharex=False,sharey=False)
  plt.setp(A,figwidth=23,figheight=15)
  for (r,i) in zip(np.arange(rows),ifos):
    for c in np.arange(cols):
      ax=axes[r]
      if type(ax)==np.ndarray:
        ax=ax[c]
      else:
        ax=axes[c]
      if rec_strains[i]["T"]['strain'] is not None or rec_strains[i]["F"]['strain'] is not None:
        if c==0:
          if global_domain=="T":
            ax.plot(rec_strains[i]["T"]['x'],rec_strains[i]["T"]['strain'],colors_rec[i],alpha=0.5,label='%s maP'%i)
          else:
            data=rec_strains[i]["F"]['strain']
            f=rec_strains[i]["F"]['x']
            mask=np.logical_and(f>=f_min,f<=plot_fmax)
            ys=data
            ax.semilogx(f[mask],ys[mask].real,'.-',color=colors_rec[i],alpha=0.5,label='%s maP'%i)
        else:
            data=rec_strains[i]["F"]['strain']
            f=rec_strains[i]["F"]['x']
            mask=np.logical_and(f>=f_min,f<=plot_fmax)
            ys=data
            ax.loglog(f[mask],abs(ys[mask]),'--',color=colors_rec[i],alpha=0.5,linewidth=4)
            ax.set_xlim([min(f[mask]),max(f[mask])])
            ax.grid(True,which='both')
      if inj_strains[i]["T"]['strain'] is not None or inj_strains[i]["F"]['strain'] is not None:
        if c==0:
          if global_domain=="T":
            ax.plot(inj_strains[i]["T"]['x'],inj_strains[i]["T"]['strain'],colors_inj[i],alpha=0.5,label='%s inj'%i)
          else:
            data=inj_strains[i]["F"]['strain']
            f=inj_strains[i]["F"]['x']
            mask=np.logical_and(f>=f_min,f<=plot_fmax)
            ys=data
            ax.plot(f[mask],ys[mask].real,'.-',color=colors_inj[i],alpha=0.5,label='%s inj'%i)
        else:
            data=inj_strains[i]["F"]['strain']
            f=inj_strains[i]["F"]['x']
            mask=np.logical_and(f>=f_min,f<=plot_fmax)
            ys=data
            ax.loglog(f[mask],abs(ys[mask]),'--',color=colors_inj[i],alpha=0.5,linewidth=4)
            ax.set_xlim([min(f[mask]),max(f[mask])])
            ax.grid(True,which='both')

      if r==0:
        if c==0:
          if global_domain=="T":
            ax.set_title(r"$h(t)$",fontsize=font_size)
          else:
            ax.set_title(r"$\Re[h(f)]$",fontsize=font_size)
        else:
          ax.set_title(r"$|h(f)|$",fontsize=font_size)
      elif r==rows-1:
        if c==0:
          if global_domain=="T":
            ax.set_xlabel("time [s]",fontsize=font_size)
          else:
            ax.set_xlabel("frequency [Hz]",fontsize=font_size)
        else:
          ax.set_xlabel("frequency [Hz]",fontsize=font_size)

      ax.legend(loc='best')
      ax.grid(True)

      #ax.tight_layout()
  A.savefig(os.path.join(path,'WF_DetFrame.png'),bbox_inches='tight')
  return inj_strains,rec_strains

def parser():

    # --- Command line input
    parser = OptionParser()
    parser.add_option("-d", "--delta-t", type=float, default=1./512)
    parser.add_option("-L", "--datalen", type=float, default=4.0)
    parser.add_option("-a", "--approximant", type=str, default=None)
    parser.add_option("-p", "--psds", type=str, default=None)

    (opts,args) = parser.parse_args()


    if len(args)==0:
        print >> sys.stderr, "Must supply a posterior samples file as a commandline argument"
        sys.exit()

    if not os.path.isfile(args[0]):
        print >> sys.stderr, "posterior samples file requested: %s does not exist"%args[0]
        sys.exit()

    return opts, args


def main(opts,args):

    #
    # Check waveform is valid and choose td/fd generator
    #
    td_approximants = pycbc.waveform.td_approximants()
    fd_approximants = pycbc.waveform.fd_approximants()
    if opts.approximant is in td_approximants:
        get_waveform = pycbc.waveform.get_td_waveform
    elif opts.approximant is in fd_approximants:
        get_waveform = pycbc.waveform.get_fd_waveform
    else:
        print >> sys.stderr, "waveform %s not recognised"%(
                opts.approximant)


    #
    # Load and parse posterior samples file
    #
    peparser = bppu.PEOutputParser('common')
    resultsObj = peparser.parse(open(args[0], 'r'))
    posterior = bppu.Posterior(resultsObj)

    #
    # Reconstructions
    #

    # Loop through posterior samples, build waveform(s)
    for s in xrange(len(posterior)):
        # Component masses
        mass1, mass2 = bppu.q2ms(mc,q)

        





    return h_signal, l_signal, hplus, hcross


if __name__ == "__main__":

    opts, args = parser()

    # return Hanford observation, Livingston observation & polarisations
    h_signal, l_signal, hplus, hcross = main(opts,args)



