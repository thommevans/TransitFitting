import pdb, sys, os, time
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
        mle_ok = False
        while mle_ok==False:
            startpos_ok_mle = False
            while startpos_ok_mle==False:
                for key in mp.model.free.keys():
                    mp.model.free[key].value = par_ranges[key].random()
                if np.isfinite( mp.logp() )==True:
                    startpos_ok_mle = True
            mp.fit( xtol=1e-4, ftol=1e-4, maxfun=10000, maxiter=10000 )
            print 'Fit results:'
            for key in mp.model.free.keys():
                print key, mp.model.free[key].value
            print 'logp = {0}'.format( mp.logp() )
            if np.isfinite( mp.logp() ):
                # If the fit has nonzero finite probability, proceed:
                mle_ok = True
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


def spec_lcs( MultiBand ):
    """
    Need to have dspec and white_psignal provided before passing in here.
    Assumes that dspec has already been trimmed along the dispersion axis,
    so it starts at ix=0. Need to provide npix_perbin.

    Need to calculate the ld elsewhere outside after this routine has 
    been run, using the wavedges for the channels that get recorded here.
    """
    
    wavsol_micron = MultiBand.wavsol_micron
    dspec = MultiBand.dspec
    white_psignal = MultiBand.white_psignal
    npix_perbin = MultiBand.npix_perbin

    # Construct the spectroscopic lightcurves using data that is 
    # contained within the range defined by disp_bound_ixs:
    nframes, nlam = np.shape( dspec )
    wav_edges = []
    wav_centers = []
    spec_lc_flux = []
    spec_lc_uncs = []
    ld_nonlin = []
    ld_quad = []
    terminate = False
    counter = 0
    while terminate==False:
        # Get the wavelength range for the 
        # current spectroscopic channel:
        a = counter*npix_perbin
        #b = min( [ a+npix_perbin, nlam-2 ] )
        b = min( [ a+npix_perbin, nlam-1 ] )
        if b<=a:
            terminate = True
            continue
        wav_edges += [ [ wavsol[a], wavsol[b] ] ]
        wav_centers += [ 0.5*( wavsol[a] + wavsol[b] ) ]
        # Bin the differential fluxes over the current channel:
        dspec_binned = np.mean( dspec[:,a:b+1], axis=1 )
        # Since the differential fluxes correspond to the raw spectroscopic
        # fluxes corrected for wavelength-common-mode systematics minus the 
        # white transit, we simply add back in the white transit signal to
        # obtain the systematics-corrected spectroscopic lightcurve:
        spec_lc_flux += [ dspec_binned + white_psignal ]
        # Computed the binned uncertainties for the wavelength channel:
        enoise_binned = np.mean( enoise[:,a:b+1], axis=1 )/np.sqrt( float( b-a ) )
        spec_lc_uncs += [ enoise_binned ]
        # Proceed to next channel if required:
        counter += 1
        if b+1>=nlam:
            terminate = True
    # Convert variables into arrays:
    MultiBand.wav_centers = np.array( [ wav_centers ] ).T
    MultiBand.wav_edges = np.column_stack( wav_edges ).T
    MultiBand.spec_lc_flux = np.column_stack( spec_lc_flux )
    MultiBand.spec_lc_uncs = np.column_stack( spec_lc_uncs )
    nframes, nchannels = np.shape( spec_lc_flux )
    MultiBand.nframes = nframes
    MultiBand.nchannels = nchannels


    return None



def white_mcmc( MultiBand ):

    mbundle = MultiBand.mbundle
    par_ranges = MultiBand.par_ranges
    nchains = MultiBand.nchains
    nsteps = MultiBand.nsteps
    nwalkers = MultiBand.nwalkers

    # Initialise the emcee sampler:
    mcmc = pyhm.MCMC( MultiBand.mbundle )
    mcmc.assign_step_method( pyhm.BuiltinStepMethods.AffineInvariant )

    # Sample for each chain, i.e. group of walkers:
    walker_chains = []
    acor_funcs = []
    acor_integs = []
    print '\nRunning the MCMC sampling:'
    for i in range( nchains ):
        print '\n... chain {0} of {1}'.format( i+1, nchains )
        init_walkers = {}
        for key in mcmc.model.free.keys():
            init_walkers[key] = np.zeros( nwalkers )
        for i in range( nwalkers ):
            for key in mcmc.model.free.keys():
                startpos_ok = False
                counter = 0
                while startpos_ok==False:
                    startpos = par_ranges[key].random()
                    mcmc.model.free[key].value = startpos
                    if np.isfinite( mcmc.model.free[key].logp() )==True:
                        startpos_ok = True
                    else:
                        counter += 1
                    if counter>100:
                        print '\n\nTrouble initialising walkers!\n\n'
                        for key in mcmc.model.free.keys():
                            print key, mcmc.model.free[key].value, mcmc.model.free[key].logp()
                        pdb.set_trace()
                init_walkers[key][i] = startpos

        # Run the sampling:
        t1 = time.time()
        mcmc.sample( nsteps=nsteps, init_walkers=init_walkers, verbose=False )
        t2 = time.time()
        print 'Done. Time taken = {0:.2f} minutes'.format( ( t2-t1 )/60. )
        acor_func, acor_integ = pyhm.walker_chain_autocorr( mcmc.walker_chain, nburn=None, maxlag=50 )

        print '\nIntegrated correlation times (total/corr):'
        for key in acor_integ.keys():
            print '{0} --> {1:.2f} ({2:.2f})'.format( key, acor_integ[key], float( nsteps )/acor_integ[key] )
    
        walker_chains += [ mcmc.walker_chain ]
        acor_funcs += [ acor_func ]
        acor_integs += [ acor_integ ]

    # Refine the best-fit solution and make a plot:
    mp = pyhm.MAP( mbundle )
    ix = np.argmax( mcmc.walker_chain['logp'] )
    ix = np.unravel_index( ix, mcmc.walker_chain['logp'].shape )
    for key in mp.model.free.keys():
        mp.model.free[key].value = mcmc.walker_chain[key][ix]
    print '\nRefining the best-fit solution to produce plot...'
    mp.fit( xtol=1e-4, ftol=1e-4, maxfun=10000, maxiter=10000 )
    print 'Done.'
    mle_refined = {}
    for key in mp.model.free.keys():
        mle_refined[key] = mp.model.free[key].value

    MultiBand.walker_chains = walker_chains
    MultiBand.acor_funcs = acor_funcs
    MultiBand.acor_integs = acor_integs
    MultiBand.mle_refined = mle_refined

    MultiBand.freepars = mcmc.model.free.keys()

    return None

def white_lc( MultiBand ):

    cuton_micron = MultiBand.cuton_micron # need this??
    cutoff_micron = MultiBand.cutoff_micron # need this??
    auxvars = MultiBand.auxvars
    spectra = MultiBand.spectra # SHOULD BE TRIMMED PRIOR TO PASSING IN!
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
    white_lc_flux_norm = white_lc_flux[0]
    MultiBand.white_lc_flux = white_lc_flux/white_lc_flux_norm
    MultiBand.white_lc_uncs = white_lc_uncs/white_lc_flux_norm
    MultiBand.cuton_micron = cuton_micron
    MultiBand.cutoff_micron = cutoff_micron

    return None

