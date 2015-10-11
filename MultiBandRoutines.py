import pdb, sys, os
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import scipy.optimize
from bayes.pyhm_dev import pyhm

def white_mle( MultiBand ):

    ntrials = MultiBand.mle_ntrials
    mbundle = MultiBand.mbundle
    par_ranges = MultiBand.par_ranges # initial starting ranges to sample randomly from

    # Initialise the walker values in a tight Gaussian ball centered 
    # at the maximum likelihood location:
    mp = pyhm.MAP( mbundle )

    print '\nRunning MLE trials:'
    mle_trial_results = []
    trial_logps = np.zeros( ntrials )
    for i in range( ntrials ):
        print '\n ... trial {0} of {1}'.format( i+1, ntrials )
            print '\nInitial values:'
            for key in mp.model.free.keys():
                mp.model.free[key].value = par_ranges[key].random()
                print key, mp.model.free[key].value
            mp.fit( xtol=1e-4, ftol=1e-4, maxfun=10000, maxiter=10000 )
            print 'Fit results:'
            for key in mp.model.free.keys():
                print key, mp.model.free[key].value
            print 'logp = {0}'.format( mp.logp() )
            if np.isfinite( mp.logp() ):
                # If the fit has nonzero finite probability, proceed:
                mle_ok = True
            else:
                # If we've ended up in zero-probability region, re-initialise
                # the model and re-attempt the fit:
                print 
                for key in cpar_ranges.keys():
                    startpos_ok_mle = False
                    while startpos_ok_mle==False:
                        mp.model.free[key].value = cpar_ranges[key].random()
                        if np.isfinite( mp.model.free[key].logp() )==True:
                            startpos_ok_mle = True
                            print key, mp.model.free[key].value
        mle_trial_results_i = {}
        for key in mp.model.free.keys():
            mle_trial_results_i[key] = mp.model.free[key].value
        mle_trial_results += [ mle_trial_results_i ]
        trial_logps[i] = mp.logp()
    ix = np.argmax( trial_logps )
    mle_vals = {}
    print '\n{0}\nBest MLE solution:'.format( 50*'#' )
    for key in mp.model.free.keys():
        mle_vals[key] = mle_trial_results[ix][key]
        print key, mle_vals[key]
    print 'logp', trial_logps[ix]
    print '{0}\n'.format( 50*'#' )

    MultiBand.mle_vals = mle_vals

    return None

def white_lc( MultiBand ):

    cuton_micron = MultiBand.cuton_micron
    cutoff_micron = MultiBand.cutoff_micron
    auxvars = MultiBand.auxvars
    spectra = MultiBand.spectra
    wavsol_micron = MultiBand.wavsol_micron
    
    # Integrate over the dispersion axis...
    print '\nBinning the lightcurve...'
    nframes, ndisp = np.shape( spectra )
    ninterp = ndisp*1000
    wavsol_micron_f = np.r_[ wavsol_micron.min():wavsol_micron.max():1j*ninterp ]
    dwav = np.median( np.diff( wavsol_micron_f ) ) # in microns
    dwav_pix = ndisp*dwav/( wavsol_micron.max()-wavsol_micron.min() ) # in pixels
    white_lc_flux = np.zeros( nframes )
    white_lc_uncs = np.zeros( nframes )
    for j in range( nframes ):
        print '... frame {0} of {1}'.format( j+1, nframes )
        wav_micron_j = wavsol_micron[j,:]
        wav_micron_f = np.r_[ wav_micron_j.min():wav_micron_j.max():1j*ninterp ]
        spectrum_j = spectra[j,:]
        spectrum_f = np.interp( wav_micron_f, wav_micron_j, spectrum_j )
        ixs_full = ( wav_micron_f>=cuton_micron )*( wav_micron_f<cutoff_micron )
        ixs_full_native = ( wav_micron_j>=cuton_micron )*( wav_micron_j<cutoff_micron )
        wav_micron_full = wav_micron_f[ixs_full]
        flux_full_f = np.sum( spectrum_f[ixs_full] )
        flux_full_native = np.sum( spectrum_j[ixs_full_native] )
        if cuton_micron>wav_micron_f.min():
            low_frac = ( wav_micron_full.min() - cuton_micron )/float( dwav )
            ixs_low = ( wav_micron_j<cuton_micron )
            flux_lowedge_f = low_frac*spectrum_f[ixs_low][-1]
        else:
            flux_lowedge_f = 0
        if cutoff_micron<wav_micron_f.max():
            upp_frac = ( cutoff_micron - wav_micron_full.max() )/float( dwav )
            ixs_upp = ( wav_micron_j>cutoff_micron )
            flux_uppedge_f = upp_frac*spectrum_f[ixs_upp][0]
        else:
            flux_uppedge_f = 0
        white_lc_flux[j] = dwav_pix*( flux_full_f + flux_lowedge_f + flux_uppedge_f )
        white_lc_uncs[j] = np.sqrt( white_lc_flux[j] )

    if cuton_micron<wavsol_micron.min():
        cuton_micron = wavsol_micron.min()
    if cutoff_micron>wavsol_micron.max():
        cutoff_micron = wavsol_micron.max()
    MultiBand.wavsol_micron = wavsol_micron
    MultiBand.auxvars = auxvars
    MultiBand.white_lc_flux = white_lc_flux
    MultiBand.white_lc_uncs = white_lc_uncs
    MultiBand.cuton_micron = cuton_micron
    MultiBand.cutoff_micron = cutoff_micron

    return None

