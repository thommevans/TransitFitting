import pdb, sys, os
import numpy as np
import cPickle
import matplotlib.pyplot as plt


def white_lc( hst ):

    ifile = open( hst.raw_input_data_file )
    z = cPickle.load( ifile )
    ifile.close()

    auxvars = z['auxvars']
    spectra = z['spectra']
    wavsol_micron = z['wavsol_micron']
    
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
    hst.white_lc_flux = white_lc_flux
    hst.white_lc_uncs = white_lc_uncs
    hst.cuton_micron = cuton_micron
    hst.cutoff_micron = cutoff_micron

    return None

