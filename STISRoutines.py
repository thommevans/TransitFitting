import pdb, sys, os
import idlsave
import fitsio
import numpy as np
import matplotlib.pyplot as plt

"""
This module takes STIS data in a format that is very specific to 
our reduction pipeline, then produces generic input files that
can be used by the rest of the routines for making lightcurves etc.
"""


def prep_data( hst ):
    
    z = idlsave.read( self.dpath )

    x1d_files = glob.glob( os.path.join( ddir_full, '*x1d*fits' ) )
    hdu = fitsio.FITS( x1d_files[0] )
    wavsol_micron = (1e-4)*hdu[1].read()['WAVELENGTH'].flatten()

    # Put spectra and auxiliary variables from the data
    # reduction into allspectrace and spectra arrays:
    allspectrace = z['allspectrace']
    spectra = z['allspec_no_weight']*z['gain']
    allspectrace = allspectrace.T
    spectra = spectra.T

    # Remove last column of G750M datasets:
    if ( config=='G750M' ):
        wavsol_micron = wavsol_micron[:-1]
        spectra = spectra[:,:-1]

    # Put auxiliary variables into x and y:
    x = allspectrace[:,-2]
    y = allspectrace[:,-1]
    
    # Read in the headers to get the timestamps:
    h = z['allheader']
    nfields, nimages = np.shape( h )
    tstarts = np.zeros( nimages )
    tends = np.zeros( nimages )
    for i in range( nimages ):
        for j in range( nfields ):
            entry = h[j,i]
            ix1 = entry.find( '=' )
            ix2 = entry.find( '/' )
            key = entry[:8].replace( ' ', '' )
            if key=='EXPSTART':
                tstarts[i] = float( entry[ix1+1:ix2] )
            elif key=='EXPEND':
                tends[i] = float( entry[ix1+1:ix2] )
    mjd = 0.5*( tstarts + tends )
    jd = mjd + 2400000.5

    # Ensure the data is in time-order:
    ixs = np.argsort( jd )
    jd = jd[ixs]
    x = x[ixs]
    y = y[ixs]
    spectra = spectra[ixs,:]

    # UPDATE: ACTUALLY, MAYBE I SHOULD DISCARD THE FIRST EXPOSURE
    # OF EACH ORBIT AND THE FIRST ORBIT BEFORE SMOOTHING AND GETTING
    # THE DSPECS ETC... CHECK HOW IT IS DONE IN WFC3 (THIS AFTERNOON)

    # Discard the first exposures of each orbit:
    # THIS SHOULD BE DONE AT THE FITTING STAGE!!
    #ndat = len( jd )
    #delt = np.diff( jd )*24*60
    #ixs = np.arange( ndat )[delt>10]+1
    #ixs = np.concatenate( [ [0], ixs, [ndat-1] ] )
    #norbs = len( ixs )-1
    #orbixs = []
    #keepixs = []
    #for i in range( norbs ):
    #    orbixs_i = np.arange( ixs[i], ixs[i+1] )
    #    orbixs += [ orbixs_i[1:]-i-1 ]
    #    keepixs += [ orbixs_i[1:] ]
    #keepixs = np.concatenate( keepixs )
    #jd = jd[keepixs]
    #x = x[keepixs]
    #y = y[keepixs]
    #ecube = ecube[keepixs,:]

    ### Nikolay hasn't discarded the first orbit, so must do this:
    # THIS SHOULD BE DONE AT THE FITTING STEP, NOT HERE:
    #if reduction=='nik':
    #    orbixs = orbixs[1:]
    #    keepixs = np.concatenate( orbixs )
    #    jd = jd[keepixs]
    #    x = x[keepixs]
    #    y = y[keepixs]
    #    spectra = spectra[keepixs,:]
    #    ndat = len( keepixs )

    # Get the hstphase:
    delt = jd - jd.min()
    hstphase = np.mod( delt, HST_ORB_PERIOD_DAYS )/float( HST_ORB_PERIOD_DAYS )
    ixs = ( hstphase>0.9 )
    hstphase[ixs] -= 1
    
    # Smooth the spectra:
    nframes, ndisp = np.shape( spectra )
    spectra_smooth = np.zeros( [ nframes, ndisp ] )
    sig = float( smoothing_fwhm )/2.354
    for i in range( nframes ):
        cut = spectra[i,:]
        scut = scipy.ndimage.filters.gaussian_filter1d( cut, sig, mode='nearest' )
        spectra_smooth[i,:] = scut

    # Load the best-fit white lightcurve in order to determine the
    # out-of-transit spectra for constructing a reference spectrum:
    model = 10 # currently this is hard-wired
    # NOTE: there's no 'smoothing' for the white stis analysis currently, unlike WFC3; 
    # however, not that the smoothed lightcurves are not actually fit for WFC3 (?)
    ifilename = 'white.{0}.{1}.{2:.2f}-{3:.2f}micron.model{4:.0f}.mle_refined.pkl'\
                .format( config, visit_str, white_wav_range_micron[0], white_wav_range_micron[1], model )
    ipath = os.path.join( stis_results_dir_full, ifilename )
    ifile = open( ipath )
    white_mle = cPickle.load( ifile )
    ifile.close()
    white_psignal = white_mle['psignal']
    pdb.set_trace()

    ixs_in = white_psignal<1
    ndat, ndisp = np.shape( spectra )
    ixs1 = np.concatenate( [ np.arange( ixs_in[0]-10 ), np.arange( ixs_in[-1]+10, ndat ) ] )
    ref_spectrum = np.median( spectra[ixs,:], axis=0 )
    #ref_spectrum = np.median( ecube[:10,:], axis=0 ) #old

    # Discard the first settling orbit:
    ixs2 = get_first_orbit_ixs( jd[ixs1] )
    ref_spectrum_smooth = np.median( spectra_smooth[ixs1,:][ixs2,:], axis=0 )


    # Note that max_wavshift, dwav and wavshifts are all in units
    # of pixels along the dispersion axis, not wavelength:
    print '\n\nComputing wavelength shifts:'
    dspec, wavshift, vstretch = wfc3_routines.calc_spectra_variations( spectra, \
                                                                       ref_spectrum, \
                                                                       max_wavshift=5, \
                                                                       dwav=0.01, \
                                                                       smoothing_fwhm=None )
    # NOTE: Possible should smooth spectra when calculating shift+stretch?

    dispersion = ( wavsol_micron.max() - wavsol_micron.min() )/float( len( wavsol_micron ) )
    wavshift_micron = wavshift*dispersion
    wavsol_micron_corr = np.zeros( np.shape( spectra ) )
    for i in range( ndat ):
        wavsol_micron_corr[i,:] = wavsol_micron - wavshift_micron[i]

    # Save to output:
    odir = os.path.join( stis_spec_dir, extension )

    header = 'jd, hstphase, x, y, wavshift_micron'
    auxvars = np.column_stack( [ jd, hstphase, x, y, wavshift_micron ] )
    ofile = dfile.replace( 'STEP3.sav', 'auxvars.txt' )
    opath = os.path.join( odir, ofile )
    np.savetxt( opath, auxvars, header=header )
    print '\nSaved {0}'.format( opath )

    header = 'frame number along vertical, dispersion column along horizontal'
    ofile = dfile.replace( 'STEP3.sav', 'spectra.txt' )
    opath = os.path.join( odir, ofile )
    np.savetxt( opath, spectra, header=header )
    print 'Saved {0}'.format( opath )
    
    header = 'frame number along vertical, dispersion column along horizontal'
    ofile = dfile.replace( 'STEP3.sav', 'wavsol.txt' )
    opath = os.path.join( odir, ofile )
    np.savetxt( opath, wavsol_micron_corr, header=header )
    print 'Saved {0}'.format( opath )


    x1d_files = glob.glob( os.path.join( ddir_full, '*x1d*fits' ) )
    hdu = fitsio.FITS( x1d_files[0] )
    wavsol_micron = (1e-4)*hdu[1].read()['WAVELENGTH'].flatten()

    # Put spectra and auxiliary variables from the data
    # reduction into allspectrace and spectra arrays:
    allspectrace = z['allspectrace']
    spectra = z['allspec_no_weight']*z['gain']
    allspectrace = allspectrace.T
    spectra = spectra.T

    # Remove last column of G750M datasets:
    if ( config=='G750M' ):
        wavsol_micron = wavsol_micron[:-1]
        spectra = spectra[:,:-1]

    # Put auxiliary variables into x and y:
    x = allspectrace[:,-2]
    y = allspectrace[:,-1]
    
    # Read in the headers to get the timestamps:
    h = z['allheader']
    nfields, nimages = np.shape( h )
    tstarts = np.zeros( nimages )
    tends = np.zeros( nimages )
    for i in range( nimages ):
        for j in range( nfields ):
            entry = h[j,i]
            ix1 = entry.find( '=' )
            ix2 = entry.find( '/' )
            key = entry[:8].replace( ' ', '' )
            if key=='EXPSTART':
                tstarts[i] = float( entry[ix1+1:ix2] )
            elif key=='EXPEND':
                tends[i] = float( entry[ix1+1:ix2] )
    mjd = 0.5*( tstarts + tends )
    jd = mjd + 2400000.5

    ### For some reason Nikolay's data isn't in time order:
    reduction = 'nik' # because I use Nikolay's reduction always now
    if reduction=='nik':
        ixs = np.argsort( jd )
        jd = jd[ixs]
        x = x[ixs]
        y = y[ixs]
        spectra = spectra[ixs,:]

    # UPDATE: ACTUALLY, MAYBE I SHOULD DISCARD THE FIRST EXPOSURE
    # OF EACH ORBIT AND THE FIRST ORBIT BEFORE SMOOTHING AND GETTING
    # THE DSPECS ETC... CHECK HOW IT IS DONE IN WFC3 (THIS AFTERNOON)

    # Discard the first exposures of each orbit:
    # THIS SHOULD BE DONE AT THE FITTING STAGE!!
    #ndat = len( jd )
    #delt = np.diff( jd )*24*60
    #ixs = np.arange( ndat )[delt>10]+1
    #ixs = np.concatenate( [ [0], ixs, [ndat-1] ] )
    #norbs = len( ixs )-1
    #orbixs = []
    #keepixs = []
    #for i in range( norbs ):
    #    orbixs_i = np.arange( ixs[i], ixs[i+1] )
    #    orbixs += [ orbixs_i[1:]-i-1 ]
    #    keepixs += [ orbixs_i[1:] ]
    #keepixs = np.concatenate( keepixs )
    #jd = jd[keepixs]
    #x = x[keepixs]
    #y = y[keepixs]
    #ecube = ecube[keepixs,:]

    ### Nikolay hasn't discarded the first orbit, so must do this:
    # THIS SHOULD BE DONE AT THE FITTING STEP, NOT HERE:
    #if reduction=='nik':
    #    orbixs = orbixs[1:]
    #    keepixs = np.concatenate( orbixs )
    #    jd = jd[keepixs]
    #    x = x[keepixs]
    #    y = y[keepixs]
    #    spectra = spectra[keepixs,:]
    #    ndat = len( keepixs )

    # Get the hstphase:
    delt = jd - jd.min()
    hstphase = np.mod( delt, HST_ORB_PERIOD_DAYS )/float( HST_ORB_PERIOD_DAYS )
    ixs = ( hstphase>0.9 )
    hstphase[ixs] -= 1
    
    # Smooth the spectra:
    nframes, ndisp = np.shape( spectra )
    spectra_smooth = np.zeros( [ nframes, ndisp ] )
    sig = float( smoothing_fwhm )/2.354
    for i in range( nframes ):
        cut = spectra[i,:]
        scut = scipy.ndimage.filters.gaussian_filter1d( cut, sig, mode='nearest' )
        spectra_smooth[i,:] = scut

    # Load the best-fit white lightcurve in order to determine the
    # out-of-transit spectra for constructing a reference spectrum:
    model = 10 # currently this is hard-wired
    # NOTE: there's no 'smoothing' for the white stis analysis currently, unlike WFC3; 
    # however, not that the smoothed lightcurves are not actually fit for WFC3 (?)
    ifilename = 'white.{0}.{1}.{2:.2f}-{3:.2f}micron.model{4:.0f}.mle_refined.pkl'\
                .format( config, visit_str, white_wav_range_micron[0], white_wav_range_micron[1], model )
    ipath = os.path.join( stis_results_dir_full, ifilename )
    ifile = open( ipath )
    white_mle = cPickle.load( ifile )
    ifile.close()
    white_psignal = white_mle['psignal']
    pdb.set_trace()

    ixs_in = white_psignal<1
    ndat, ndisp = np.shape( spectra )
    ixs1 = np.concatenate( [ np.arange( ixs_in[0]-10 ), np.arange( ixs_in[-1]+10, ndat ) ] )
    ref_spectrum = np.median( spectra[ixs,:], axis=0 )
    #ref_spectrum = np.median( ecube[:10,:], axis=0 ) #old

    # Discard the first settling orbit:
    ixs2 = get_first_orbit_ixs( jd[ixs1] )
    ref_spectrum_smooth = np.median( spectra_smooth[ixs1,:][ixs2,:], axis=0 )


    # Note that max_wavshift, dwav and wavshifts are all in units
    # of pixels along the dispersion axis, not wavelength:
    print '\n\nComputing wavelength shifts:'
    dspec, wavshift, vstretch = wfc3_routines.calc_spectra_variations( spectra, \
                                                                       ref_spectrum, \
                                                                       max_wavshift=5, \
                                                                       dwav=0.01, \
                                                                       smoothing_fwhm=None )
    # NOTE: Possible should smooth spectra when calculating shift+stretch?

    dispersion = ( wavsol_micron.max() - wavsol_micron.min() )/float( len( wavsol_micron ) )
    wavshift_micron = wavshift*dispersion
    wavsol_micron_corr = np.zeros( np.shape( spectra ) )
    for i in range( ndat ):
        wavsol_micron_corr[i,:] = wavsol_micron - wavshift_micron[i]

    # Save to output:
    odir = os.path.join( stis_spec_dir, extension )

    header = 'jd, hstphase, x, y, wavshift_micron'
    auxvars = np.column_stack( [ jd, hstphase, x, y, wavshift_micron ] )
    ofile = dfile.replace( 'STEP3.sav', 'auxvars.txt' )
    opath = os.path.join( odir, ofile )
    np.savetxt( opath, auxvars, header=header )
    print '\nSaved {0}'.format( opath )

    header = 'frame number along vertical, dispersion column along horizontal'
    ofile = dfile.replace( 'STEP3.sav', 'spectra.txt' )
    opath = os.path.join( odir, ofile )
    np.savetxt( opath, spectra, header=header )
    print 'Saved {0}'.format( opath )
    
    header = 'frame number along vertical, dispersion column along horizontal'
    ofile = dfile.replace( 'STEP3.sav', 'wavsol.txt' )
    opath = os.path.join( odir, ofile )
    np.savetxt( opath, wavsol_micron_corr, header=header )
    print 'Saved {0}'.format( opath )

    return None
    
    
