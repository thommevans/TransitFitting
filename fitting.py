import numpy as np
import matplotlib.pyplot as plt
from bayes.gps_dev.gps import gp_class, gp_routines, kernels
from bayes.pyhm_dev import pyhm
import cPickle
import prep
import glob
from planetc import transit
import pdb, time, shutil
import numexpr
from utils import triangle

"""

Note:
  Currently output is generated to a hard-

    model 1 - Linear trend in time only.
    model 2 - Sqexp in hstphase, time.
    model 3 - Matern in hstphase, time.
    model 4 - Sqexp in hstphase + linear trend in time.
    model 5 - Matern in hstphase + linear trend in time.
    model 6 - Sqexp in hstphase, wavshift, time.
    model 7 - Matern in hstphase, wavshift, time.
    model 8 - Sqexp in hstphase, wavshift + linear trend in time.
    model 9 - Matern in hstphase, wavshift + linear trend in time.
    model 10 - Sqexp in hstphase, time + linear trend in time.
    model 11 - Matern in hstphase, time + linear trend in time.

TODO:

  Update all of the get_modelXorX etc so that it works for the 
  spectroscopic lightcurves. Right now, I've tried getting it
  to work for models 1 and 10. But will need to adapt the others
  as well.

"""


def mle_white_stis( config='G430L', visit=1, cuton_micron=0.29, cutoff_micron=0.57, model=1, \
                    ld_type='nonlin', ntrials=5, discard_first_orbit=True, orbpars='free', \
                    save_output=True, lc_ipath='', results_opath='' ):
    """
    model 1 - Linear trend in time only.
    model 2 - Sqexp in hstphase, time.
    model 3 - Matern in hstphase, time.
    model 4 - Sqexp in hstphase + linear trend in time.
    model 5 - Sqexp in hstphase + linear trend in time.
    model 6 - Sqexp in hstphase, wavshift, time.
    model 7 - Matern in hstphase, wavshift, time.
    model 8 - Sqexp in hstphase, wavshift + linear trend in time.
    model 9 - Matern in hstphase, wavshift + linear trend in time.
    """

    lc_ifile = open( lc_ipath )
    lc = cPickle.load( lc_ifile )
    if discard_first_orbit==True:
        ixs = prep.get_first_orbit_ixs( lc['jd'] )
        lc['jd'] = lc['jd'][ixs]
        lc['hstphase'] = lc['hstphase'][ixs]
        lc['x'] = lc['x'][ixs]
        lc['y'] = lc['y'][ixs]
        lc['wavshift_micron'] = lc['wavshift_micron'][ixs]
        lc['white_lc_flux'] = lc['white_lc_flux'][ixs]
        lc['white_lc_uncs'] = lc['white_lc_uncs'][ixs]

    z = mle_white( lc, model=model, ld_type=ld_type, ntrials=ntrials, orbpars=orbpars )

    if save_output==True:
        ofile = open( results_opath, 'w' )
        cPickle.dump( z, ofile )
        ofile.close()
        print '\nSaved:\n{0}'.format( opath )

    return None


def mle_white_wfc3( config='G141', apradius=75, smoothing_fwhm=4, cuton_micron=0.77, cutoff_micron=1.95, \
                    model=1, ld_type='nonlin', ntrials=5, discard_first_orbit=True, orbpars='free', \
                    save_output=True, lc_ipath='', results_opath='' ):
    """
    model 1 - Linear trend in time only.
    model 2 - Sqexp in hstphase, time.
    model 3 - Matern in hstphase, time.
    model 4 - Sqexp in hstphase + linear trend in time.
    model 5 - Sqexp in hstphase + linear trend in time.
    model 6 - Sqexp in hstphase, wavshift, time.
    model 7 - Matern in hstphase, wavshift, time.
    model 8 - Sqexp in hstphase, wavshift + linear trend in time.
    model 9 - Matern in hstphase, wavshift + linear trend in time.
    """

    lc_ifile = open( lc_ipath )
    lc = cPickle.load( lc_ifile )
    if discard_first_orbit==True:
        ixs = prep.get_first_orbit_ixs( lc['jd'] )
        lc['jd'] = lc['jd'][ixs]
        lc['hstphase'] = lc['hstphase'][ixs]
        lc['x'] = lc['x'][ixs]
        lc['y'] = lc['y'][ixs]
        lc['wavshift_micron'] = lc['wavshift_micron'][ixs]
        lc['white_lc_flux'] = lc['white_lc_flux'][ixs]
        lc['white_lc_uncs'] = lc['white_lc_uncs'][ixs]

    z = mle_white( lc, model=model, ld_type=ld_type, ntrials=ntrials, orbpars=orbpars )

    if save_output==True:
        ofile = open( results_opath, 'w' )
        cPickle.dump( z, ofile )
        ofile.close()
        print '\nSaved:\n{0}'.format( opath )

    return None


def mle_spec_wfc3( config='G141', apradius=75, smoothing_fwhm=4, npix_perbin=4, ld_type='nonlin', \
                   channel_ixs='all', model=1, ntrials=5, discard_first_orbit=True, save_output=True, \
                   lc_ipath='', results_opath='' ):
    """
    model 1 - Linear trend in time only.
    model 2 - Sqexp in hstphase, time.
    model 3 - Matern in hstphase, time.
    model 4 - Sqexp in hstphase + linear trend in time.
    model 5 - Sqexp in hstphase + linear trend in time.
    model 6 - Sqexp in hstphase, wavshift, time.
    model 7 - Matern in hstphase, wavshift, time.
    model 8 - Sqexp in hstphase, wavshift + linear trend in time.
    model 9 - Matern in hstphase, wavshift + linear trend in time.
    """

    ifile = open( lc_ipath )
    lc = cPickle.load( ifile )
    ifile.close()
    if save_output==True:
        if os.path.isdir( wfc3_results_dir )==False:
            os.makedirs( wfc3_results_dir )

    if discard_first_orbit==True:
        ixs = prep.get_first_orbit_ixs( lc['jd'] )
        lc['jd'] = lc['jd'][ixs]
        lc['hstphase'] = lc['hstphase'][ixs]
        lc['x'] = lc['x'][ixs]
        lc['y'] = lc['y'][ixs]
        lc['wavshift_micron'] = lc['wavshift_micron'][ixs]
        lc['spec_lc_flux'] = lc['spec_lc_flux'][ixs,:]
        lc['spec_lc_uncs'] = lc['spec_lc_uncs'][ixs,:]
    nframes, nchannels = np.shape( lc['spec_lc_flux'] )
    if channel_ixs=='all':
        channel_ixs = np.arange( nchannels )
    nmle = len( channel_ixs )
    print '\n\nRunning spectroscopic lightcurve fits:'
    for j in range( nmle ):
        print '\n ... channel {0} of {1}'.format( j+1, nmle )
        mle_results = mle_spec( lc, channel_ixs[j], model=model, ld_type=ld_type, ntrials=ntrials, \
                                channel_ixs=channel_ixs, discard_first_orbit=discard_first_orbit )
        if save_output==True:
            print '\n'
            ofile = open( results_opath, 'w' )
            cPickle.dump( mle_results, ofile )
            ofile.close()
            print 'Saved: {0}'.format( results_opath )

    return None




def mcmc_white_wfc3( config='G141', apradius=75, smoothing_fwhm=4, cuton_micron=0.77, cutoff_micron=1.95, \
                     model=1, ld_type='nonlin', nchains=2, nsteps=400, nwalkers=200, discard_first_orbit=True, \
                     orbpars='free', save_output=True, lc_ipath='', mle_ipath='', results_opath='' ):
    """
    model 1 - Linear trend in time only.
    model 2 - Sqexp in hstphase, time.
    model 3 - Matern in hstphase, time.
    model 4 - Sqexp in hstphase + linear trend in time.
    model 5 - Sqexp in hstphase + linear trend in time.
    model 6 - Sqexp in hstphase, wavshift, time.
    model 7 - Matern in hstphase, wavshift, time.
    model 8 - Sqexp in hstphase, wavshift + linear trend in time.
    model 9 - Matern in hstphase, wavshift + linear trend in time.
    """

    mle_ipath = os.path.join( wfc3_results_dir_full, mle_filename )
    ifile = open( mle_ipath )
    mle = cPickle.load( ifile )
    ifile.close()

    lc_ifile = open( lc_ipath )
    lc = cPickle.load( lc_ifile )
    if discard_first_orbit==True:
        ixs = prep.get_first_orbit_ixs( lc['jd'] )
        lc['jd'] = lc['jd'][ixs]
        lc['hstphase'] = lc['hstphase'][ixs]
        lc['x'] = lc['x'][ixs]
        lc['y'] = lc['y'][ixs]
        lc['wavshift_micron'] = lc['wavshift_micron'][ixs]
        lc['white_lc_flux'] = lc['white_lc_flux'][ixs]
        lc['white_lc_uncs'] = lc['white_lc_uncs'][ixs]

    if os.path.isdir( wfc3_results_dir_full )==False:
        os.makedirs( wfc3_results_dir_full )
    z = mcmc_white( lc, mle, model=model, ld_type=ld_type, orbpars=orbpars, \
                    nchains=nchains, nsteps=nsteps, nwalkers=nwalkers )
    
    # ALL OF THIS SHOULD BE DONE SEPARATELY; JUST PASS THE Z AS OUTPUT
    #if save_output==True:
    #
    #    ofigname_walkers = 'white.{0}.apradius{1:.2f}.smooth{2:.2f}pix.wav{3:.2f}-{4:.2f}micron'\
    #                       .format( config, apradius, smoothing_fwhm, cuton_micron, cutoff_micron )
    #    ofigname_walkers += '.model{0:.0f}.walkers.pdf'.format( model )
    #    ofigpath_walkers = os.path.join( wfc3_results_dir_full, ofigname_walkers )
    #    z['fig_walkers'].savefig( ofigpath_walkers )#

    #    ofigname_bestfit = 'white.{0}.apradius{1:.2f}.smooth{2:.2f}pix.wav{3:.2f}-{4:.2f}micron'\
    #                       .format( config, apradius, smoothing_fwhm, cuton_micron, cutoff_micron )
    #    ofigname_bestfit += '.model{0:.0f}.bestfit.pdf'.format( model )
    #    ofigpath_bestfit = os.path.join( wfc3_results_dir_full, ofigname_bestfit )
    #    z['fig_bestfit'].savefig( ofigpath_bestfit )

    #    ofilename_walkers = 'white.{0}.apradius{1:.2f}.smooth{2:.2f}pix.wav{3:.2f}-{4:.2f}micron'\
    #                        .format( config, apradius, smoothing_fwhm, cuton_micron, cutoff_micron )
    #    ofilename_walkers += '.model{0:.0f}.walkers.pkl'.format( model )
    #    opath_walkers = os.path.join( wfc3_results_dir_full, ofilename_walkers )
    #    ofile = open( opath_walkers, 'w' )
    #    y = { 'walker_chains':z['walker_chains'], 'acor_integs':z['acor_integs'], 'acor_funcs':z['acor_funcs'] }
    #    cPickle.dump( y, ofile )
    #    ofile.close()

    #    mle_ofilename = mle_filename.replace( '.mle_prelim.', '.mle_refined.' )
    #    ofile = open( mle_ofilename, 'w' )
    #    cPickle.dump( z['mle_refined'], ofile )
    #    ofile.close()

    #    print '\nSaved:\n{0}\n{1}\n{2}\n{3}'\
    #          .format( ofigpath_walkers, ofigpath_bestfit, opath_walkers, mle_ofilename )

    return z


def mcmc_spec_wfc3( config='G141', apradius=75, smoothing_fwhm=4, npix_perbin=4, channel_ixs='all', \
                    model=1, ld_type='nonlin', nchains=2, nsteps=400, nwalkers=200, discard_first_orbit=True, \
                    lc_ipath='' ):
    """
    model 1 - Linear trend in time only.
    model 2 - Sqexp in hstphase, time.
    model 3 - Matern in hstphase, time.
    model 4 - Sqexp in hstphase + linear trend in time.
    model 5 - Sqexp in hstphase + linear trend in time.
    model 6 - Sqexp in hstphase, wavshift, time.
    model 7 - Matern in hstphase, wavshift, time.
    model 8 - Sqexp in hstphase, wavshift + linear trend in time.
    model 9 - Matern in hstphase, wavshift + linear trend in time.
    """

    # This used to loop over all the channels within this 

    ifile = open( lc_ipath )
    lc = cPickle.load( ifile )
    ifile.close()
    if discard_first_orbit==True:
        ixs = prep.get_first_orbit_ixs( lc['jd'] )
        lc['jd'] = lc['jd'][ixs]
        lc['hstphase'] = lc['hstphase'][ixs]
        lc['x'] = lc['x'][ixs]
        lc['y'] = lc['y'][ixs]
        lc['wavshift_micron'] = lc['wavshift_micron'][ixs]
        lc['spec_lc_flux'] = lc['spec_lc_flux'][ixs]
        lc['spec_lc_uncs'] = lc['spec_lc_uncs'][ixs]

    nframes, nchannels = np.shape( lc['spec_lc_flux'] )
    if channel_ixs=='all':
        channel_ixs = np.arange( nchannels )
    nmcmc = len( channel_ixs )
    for i in range( nmcmc ):
        mle_filename = 'spec.{0}.apradius{1:.2f}.smooth{2:.2f}pix.{3:.0f}pixpbin.ch{4:.0f}.'\
                       .format( config, apradius, smoothing_fwhm, npix_perbin, channel_ixs[i]+1 )
        mle_filename += 'model{0:.0f}.mle_prelim.pkl'.format( model )
        ipath = os.path.join( wfc3_results_dir, mle_filename )
        ifile = open( ipath )
        mle = cPickle.load( ifile )
        ifile.close()
        z = mcmc_spec( lc, channel_ixs[i], mle, model=model, ld_type=ld_type, \
                        nchains=nchains, nsteps=nsteps, nwalkers=nwalkers, \
                        discard_first_orbit=discard_first_orbit )

        if save_output==True:
            ofigname_walkers = 'spec.{0}.apradius{1:.2f}.{2}.{3:.0f}pixpbin.ch{4:.0f}'\
                               .format( config, apradius, smoothing_str, npix_perbin, channel_ixs[i]+1 )
            ofigname_walkers += '.model{0:.0f}.walkers.pdf'.format( model )
            ofigpath_walkers = os.path.join( wfc3_results_dir, ofigname_walkers )
            z['fig_walkers'].savefig( ofigpath_walkers )

            ofigname_bestfit = 'spec.{0}.apradius{1:.2f}.{2}.{3:.0f}pixpbin.ch{4:.0f}'\
                               .format( config, apradius, smoothing_str, npix_perbin, channel_ixs[i]+1 )
            ofigname_bestfit += '.model{0:.0f}.bestfit.pdf'.format( model )
            ofigpath_bestfit = os.path.join( wfc3_results_dir, ofigname_bestfit )
            z['fig_bestfit'].savefig( ofigpath_bestfit )

            ofilename_walkers = 'spec.{0}.apradius{1:.2f}.{2}.{3:.0f}pixpbin.ch{4:.0f}'\
                               .format( config, apradius, smoothing_str, npix_perbin, channel_ixs[i]+1 )
            ofilename_walkers += '.model{0:.0f}.walkers.pkl'.format( model )
            opath_walkers = os.path.join( wfc3_results_dir, ofilename_walkers )
            ofile = open( opath_walkers, 'w' )
            walkers = { 'walker_chains':z['walker_chains'], 'acor_integs':z['acor_integs'], \
                        'acor_funcs':z['acor_funcs'] }
            cPickle.dump( walkers, ofile )
            ofile.close()

            mle_ofilename = mle_filename.replace( '.mle_prelim.', '.mle_refined.' )
            ofile = open( mle_ofilename, 'w' )
            cPickle.dump( z['mle_refined'], ofile )
            ofile.close()

            print '\nSaved:\n{0}\n{1}\n{2}\n{3}'.format( ofigpath_walkers, ofigpath_bestfit, opath_walkers, mle_ofilename )

    return z


def mcmc_white_stis( config='G430L', visit=1, cuton_micron=0.29, cutoff_micron=0.57, model=1, \
                     ld_type='nonlin', nchains=2, nsteps=400, nwalkers=200, discard_first_orbit=True, \
                     orbpars='free', save_output=True, lc_dir='', results_dir='' ):
    """
    model 1 - Linear trend in time only.
    model 2 - Sqexp in hstphase, time.
    model 3 - Matern in hstphase, time.
    model 4 - Sqexp in hstphase + linear trend in time.
    model 5 - Sqexp in hstphase + linear trend in time.
    model 6 - Sqexp in hstphase, wavshift, time.
    model 7 - Matern in hstphase, wavshift, time.
    model 8 - Sqexp in hstphase, wavshift + linear trend in time.
    model 9 - Matern in hstphase, wavshift + linear trend in time.
    """

    stis_lc_dir = os.path.join( univvars['STIS_ADIR'], 'lightcurves' )
    stis_results_dir = os.path.join( univvars['STIS_ADIR'], 'results/white' )
    if ( config=='G430L' ):
        extension = 'Proposal_9447_G430L+G750L/visit{0}_G430L'.format( visit )
    elif config=='G750L':
        extension = 'Proposal_9447_G430L+G750L/visit{0}_G750L'.format( visit )
    elif config=='G750M':
        extension = 'Proposal_8789_G750M/visit{0}_G750M'.format( visit )
    else:
        pdb.set_trace()
    stis_lc_dir_full = os.path.join( stis_lc_dir, extension )

    if orbpars=='free':
        extension = os.path.join( extension, 'orbpars_free' )
    elif orbpars=='prior':
        extension = os.path.join( extension, 'orbpars_prior' )
    else:
        pdb.set_trace()
    stis_results_dir_full = os.path.join( stis_results_dir, extension )

    mle_filename = 'white.{0}.visit{1}.{2:.2f}-{3:.2f}micron.model{4:.0f}.mle.pkl'\
                   .format( config, visit, cuton_micron, cutoff_micron, model )
    ipath = os.path.join( stis_results_dir_full, mle_filename )
    ifile = open( ipath )
    mle = cPickle.load( ifile )
    ifile.close()

    lcfile = 'white_{0}_visit{1}_{2:.2f}-{3:.2f}micron.pkl'\
             .format( config, visit, cuton_micron, cutoff_micron )
    lc_path = os.path.join( stis_lc_dir_full, lcfile )
    lc_ifile = open( lc_path )
    lc = cPickle.load( lc_ifile )
    if discard_first_orbit==True:
        ixs = prep.get_first_orbit_ixs( lc['jd'] )
        lc['jd'] = lc['jd'][ixs]
        lc['hstphase'] = lc['hstphase'][ixs]
        lc['x'] = lc['x'][ixs]
        lc['y'] = lc['y'][ixs]
        lc['wavshift_micron'] = lc['wavshift_micron'][ixs]
        lc['white_lc_flux'] = lc['white_lc_flux'][ixs]
        lc['white_lc_uncs'] = lc['white_lc_uncs'][ixs]

    if os.path.isdir( stis_results_dir_full )==False:
        os.makedirs( stis_results_dir_full )
    z = mcmc_white( lc, mle, model=model, ld_type=ld_type, orbpars=orbpars, \
                    nchains=nchains, nsteps=nsteps, nwalkers=nwalkers )
    
    if save_output==True:
        ofigname_walkers = 'white.{0}.visit{1}.{2:.2f}-{3:.2f}micron.model{4:.0f}.walkers.pdf'\
                           .format( config, visit, cuton_micron, cutoff_micron, model )
        ofigpath_walkers = os.path.join( stis_results_dir_full, ofigname_walkers )
        z['fig_walkers'].savefig( ofigpath_walkers )
        ofigname_bestfit = 'white.{0}.visit{1}.{2:.2f}-{3:.2f}micron.model{4:.0f}.bestfit.pdf'\
                           .format( config, visit, cuton_micron, cutoff_micron, model )
        ofigpath_bestfit = os.path.join( stis_results_dir_full, ofigname_bestfit )
        z['fig_bestfit'].savefig( ofigpath_bestfit )

        ofilename_walkers = 'white.{0}.visit{1:.0f}.{2:.2f}-{3:.2f}micron'\
                            .format( config, visit, cuton_micron, cutoff_micron )
        ofilename_walkers += '.model{0:.0f}.walkers.pkl'.format( model )
        opath_walkers = os.path.join( stis_results_dir_full, ofilename_walkers )
        ofile = open( opath_walkers, 'w' )
        y = { 'walker_chains':z['walker_chains'], 'acor_integs':z['acor_integs'], 'acor_funcs':z['acor_funcs'] }
        cPickle.dump( y, ofile )
        ofile.close()

        mle_ofilename = mle_filename.replace( '.mle_prelim.', '.mle_refined.' )
        ofile = open( mle_ofilename, 'w' )
        cPickle.dump( z['mle_refined'], ofile )
        ofile.close()

        print '\nSaved:\n{0}\n{1}\n{2}'.format( ofigpath_walkers, ofigpath_bestfit, opath_walkers )

    return None


def plot_walker_hist_stis( lc_type='white', config='G430L', visit=1, model=1, orbpars='free', \
                           cuton_micron=0.29, cutoff_micron=0.57, ncorr_burn=3, results_dir='' ):
    """
    Creates figure using triangle.corner() to display the chain results.
    """

    stis_results_dir = os.path.join( univvars['STIS_ADIR'], 'results' )
    if ( lc_type=='white' ):
        if ( config=='G430L' ):
            extension = 'Proposal_9447_G430L+G750L/visit{0}_G430L'.format( visit )
        elif config=='G750L':
            extension = 'Proposal_9447_G430L+G750L/visit{0}_G750L'.format( visit )
        elif config=='G750M':
            extension = 'Proposal_8789_G750M/visit{0}_G750M'.format( visit )
        else:
            pdb.set_trace()
    if orbpars=='free':
        extension = os.path.join( extension, 'white/orbpars_free' )
    elif orbpars=='prior':
        extension = os.path.join( extension, 'white/orbpars_prior' )
    else:
        pdb.set_trace()
    stis_results_dir_full = os.path.join( stis_results_dir, extension )
    ifilename = 'white.{0}.visit{1:.0f}.{2:.2f}-{3:.2f}micron'\
                .format( config, visit, cuton_micron, cutoff_micron )
    ifilename += '.model{0:.0f}.walkers.pkl'.format( model )
    ipath = os.path.join( stis_results_dir_full, ifilename )
    ifile = open( ipath )
    z = cPickle.load( ifile )
    ifile.close()
    title_str = 'white (cuton={0:.2f}, cutoff={1:.2f} ) - config={2}, model={3:.0f}'\
               .format( cuton_micron, cutoff_micron, config, model )
    fig, sample_properties, grs = plot_walker_hist( z['walker_chains'], z['acor_integs'], title_str=title_str, \
                                                    ncorr_burn=ncorr_burn, which_type=lc_type )

    #title_str = 'white (cuton={0:.2f}, cutoff={1:.2f} ) - config={2}, model={3:.0f}'\
    #           .format( cuton_micron, cutoff_micron, config, model )
    #fig, sample_properties = plot_walker_hist( z['walker_chains'], z['acor_integs'], title_str=title_str )
    ofigname = 'white.{0}.visit{1}.{2:.2f}-{3:.2f}micron'\
               .format( config, visit, cuton_micron, cutoff_micron )
    ofigname += '.model{0:.0f}.histogram.pdf'.format( model )
    ofigpath = os.path.join( stis_results_dir_full, ofigname )
    fig.savefig( ofigpath )
    print '\nSaved: {0}'.format( ofigpath )

    med = sample_properties['median']
    l34 = sample_properties['l34']
    u34 = sample_properties['u34']
    keys = med.keys()
    out_str = '# {0}, {1}, visit={2:.0f}, model={3:.0f}'\
              .format( config, lc_type, visit, model )
    out_str += '\n# wavelength range = {0:.2f}-{1:.2f} micron'.format( cuton_micron, cutoff_micron )
    out_str += '\n#\n# parameter, median, l34, u34'
    for key in keys:
        print key, med[key], np.abs( l34[key] ), u34[key], grs[key] 
        out_str += '\n{0} {1:.6f} {2:.6f} {3:.6f} {4:.4f}'\
                   .format( key, med[key], np.abs( l34[key] ), u34[key], grs[key] )
    opath = ofigpath.replace( '.histogram.pdf', '.mcmc.txt' )
    ofile = open( opath, 'w' )
    ofile.write( out_str )
    ofile.close()
    print 'Saved: {0}'.format( opath )

    return None


def plot_walker_hist_wfc3( lc_type='white', config='G141', model=1, apradius=75, smoothing_fwhm=4, \
                           orbpars='free', cuton_micron=0.77, cutoff_micron=1.95, ncorr_burn=3, \
                           npix_perbin=4, walker_filepaths=[''] ):
    """
    Creates figure using triangle.corner() to display the chain results.
    """

    n = len( walker_filepaths )
    title_strs = []
    for j in range( n ):
        title_strs += [ os.path.basename( walker_filepaths[j] ) ]

    for i in range( n ):
        print '\n\n\n'
        print i+1, n
        ifile = open( ipaths[i] )
        z = cPickle.load( ifile )
        ifile.close()
        fig, sample_properties, grs = plot_walker_hist( z['walker_chains'], z['acor_integs'], \
                                                        title_str=title_strs[i], \
                                                        ncorr_burn=ncorr_burn, which_type=lc_type )
        med = sample_properties['median']
        l34 = sample_properties['l34']
        u34 = sample_properties['u34']
        keys = med.keys()
        out_str = '# {0}, {1}, model={2:.0f}, apradius={3:.2f}, smoothing_fwhm={4:.2f}'\
                  .format( config, lc_type, model, apradius, smoothing_fwhm )
        out_str += '\n#\n# parameter, median, l34, u34'
        for key in keys:
            out_str += '\n{0} {1:.6f} {2:.6f} {3:.6f} {4:.3f}'\
                       .format( key, med[key], np.abs( l34[key] ), u34[key], grs[key] )

    return fig, out_str



####################################################################################
# Backend routines:

def plot_walker_hist( walker_chains, acor_integs, title_str=None, ncorr_burn=3, which_type='' ):
    """
    Backend to plot_walker_hist_stis() and plot_walker_hist_wfc3().
    Generates and returns the figure for the triangle.corner() plot.
    Also computes the sample statistics and Gelman-Rubin values.
    """

    nchains = len( walker_chains )
    acor = np.zeros( nchains )
    nsteps = np.zeros( nchains )
    nwalkers = np.zeros( nchains )
    for i in range( nchains ):
        walker_chain = walker_chains[i]
        nsteps[i], nwalkers[i] = np.shape( walker_chain['logp'] )
        keys = walker_chain.keys()
        keys.remove( 'logp' )
        npar = len( keys )
        acor_vals = np.zeros( npar )
        for j in range( npar ):
            acor_vals[j] = acor_integs[i][keys[j]]
        acor[i] = np.max( np.abs( acor_vals ) )
    y = nsteps/acor
    if y.min()<ncorr_burn:
        print '\nChains only run for {0:.2f}x correlation times'.format( y.min() )
        pdb.set_trace()
    else:
        nburn = int( np.round( ncorr_burn*acor.max() ) )
        chain_dicts = []
        chain_arrs = []
        for i in range( nchains ):
            chain_i = pyhm.collapse_walker_chain( walker_chains[i], nburn=nburn )
            if which_type=='white':
                chain_i['incl'] = np.rad2deg( np.arccos( chain_i['b']/chain_i['aRs'] ) )
            elif which_type=='spec':
                pass
            else:
                pdb.set_trace()
            chain_dicts += [ chain_i ]
        grs = pyhm.gelman_rubin( chain_dicts, nburn=0, thin=1 )
        chain = pyhm.combine_chains( chain_dicts, nburn=nburn, thin=1 )        
        logp_arr = chain['logp']
        chain.pop( 'logp' )
        keys = chain.keys()
        npar = len( keys )
        nsamples = len( logp_arr )
        chain_arr = np.zeros( [ nsamples, npar ] )
        for j in range( npar ):
            chain_arr[:,j] = chain[keys[j]]

    plt.ioff()
    fig = triangle.corner( chain_arr, labels=keys )

    title_fs = 20
    text_fs = 18
    title_x = 0.2
    if title_str!=None:
        fig.text( title_x, 0.98, title_str, fontsize=title_fs, horizontalalignment='left', verticalalignment='top' )

    text_str = 'nchains={0:.0f}, nwalkers={1:.0f}, nsteps={2:.0f}, acor={3:.0f}, nburn={4:.0f}, nsamples={5:.0f}'\
               .format( nchains, nwalkers[0], nsteps[0], acor.max(), nburn, nsamples )
    fig.text( title_x, 0.95, text_str, fontsize=title_fs, horizontalalignment='left', verticalalignment='top' )
    y = pyhm.chain_properties( chain, nburn=0, thin=None, print_to_screen=True )
    text_str = 'Sample properties:\n\n'
    for key in keys:
        text_str += '{0} --> {1:.6f} (-{2:.6f}, +{3:.6f}) GR={4:.3f}\n'\
                    .format( key, y['median'][key], np.abs( y['l34'][key] ), y['u34'][key], grs[key] )
    fig.text( 0.47, 0.9, text_str, fontsize=text_fs, horizontalalignment='left', verticalalignment='top' )
    plt.ion()

    return fig, y, grs


def mle_white( lc, model=1, ld_type='nonlin', ntrials=5, orbpars='free' ):
    """
    Note that first orbit should be discarded from lc prior to passing it in here.
    """

    print '\nUsing {0} limb darkening with coefficients:'.format( ld_type )
    if ld_type=='quad':
        print 'gam1 = {0:.3f}'.format( lc['ld_quad'][0] )
        print 'gam2 = {0:.3f}'.format( lc['ld_quad'][1] )
    elif ld_type=='nonlin':
        print 'c1 = {0:.3f}'.format( lc['ld_nonlin'][0] )
        print 'c2 = {0:.3f}'.format( lc['ld_nonlin'][1] )
        print 'c3 = {0:.3f}'.format( lc['ld_nonlin'][2] )
        print 'c4 = {0:.3f}'.format( lc['ld_nonlin'][3] )
    else:
        pdb.set_trace()

    # Get the model bundle and GP object:
    mbundle, gp, cpar_ranges = get_white_mbundle( lc, model, ld_type, orbpars=orbpars )

    # Initialise the walker values in a tight Gaussian ball centered 
    # at the maximum likelihood location:
    mp = pyhm.MAP( mbundle )

    # Do a simple linear trend plus transit fit to start:
    if model!=1:
        mbundle_simple, gp_simple, cpar_ranges_simple = get_white_mbundle( lc, 1, ld_type, orbpars=orbpars )
        mp_simple = pyhm.MAP( mbundle_simple )
        mp_simple.fit( xtol=1e-4, ftol=1e-4, maxfun=10000, maxiter=10000 )
    print '\nResults of simple preliminary MLE fit:'
    for key in mp_simple.model.free.keys():
        print key, mp_simple.model.free[key].value

    print '\nRunning MLE trials:'
    mle_trial_results = []
    trial_logps = np.zeros( ntrials )
    for i in range( ntrials ):
        print '\n ... trial {0} of {1}'.format( i+1, ntrials )
        mle_ok = False
        while mle_ok==False:
            # Install the best fit parameters from the simple model
            # as starting guesses for the actual fit below:
            if model!=1: # assuming we're not doing the simple linear-time-trend fit
                for key in mp_simple.model.free.keys():
                    try:
                        mp.model.free[key].value = mp_simple.model.free[key].value
                    except:
                        pass
                # However, lets ensure the planet parameters we start
                # with are roughly what we expect:
                mp.RpRs.value = 0.1205+0.0010*np.random.random()
                if orbpars=='free':
                    mp.aRs.value = 8.75+0.1*np.random.random()
                    mp.b.value = 0.499+0.002*np.random.random()
                elif orbpars=='prior':
                    mp.aRs.value = mp.aRs.random()
                    mp.b.value = mp.b.random()
                mp.beta.value = 0.5*np.random.random()
                mp.delT.value = mp_simple.delT.value
                mp.foot.value = 1
                mp.tgrad.value = 0
                # Install reasonable starting values for the cpars:
                for key in cpar_ranges.keys():
                    mp.model.free[key].value = cpar_ranges[key].random()
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

    return mle_vals


def mle_spec( lc, channel_ix, model=1, ld_type='nonlin', ntrials=5, channel_ixs='all' ):
    """
    Note that first orbit should be discarded from lc prior to passing it in here.
    """

    print '\nUsing {0} limb darkening with coefficients:'.format( ld_type )
    if ld_type=='quad':
        print 'gam1 = {0:.3f}'.format( lc['ld_quad'][channel_ix,0] )
        print 'gam2 = {0:.3f}'.format( lc['ld_quad'][channel_ix,1] )
    elif ld_type=='nonlin':
        print 'c1 = {0:.3f}'.format( lc['ld_nonlin'][channel_ix,0] )
        print 'c2 = {0:.3f}'.format( lc['ld_nonlin'][channel_ix,1] )
        print 'c3 = {0:.3f}'.format( lc['ld_nonlin'][channel_ix,2] )
        print 'c4 = {0:.3f}'.format( lc['ld_nonlin'][channel_ix,3] )
    else:
        pdb.set_trace()
    # Get the model bundle and GP object:
    mbundle, gp, cpar_ranges = get_spec_mbundle( lc, model, ld_type, channel_ix )

    # Initialise the walker values in a tight Gaussian ball centered 
    # at the maximum likelihood location:
    mp = pyhm.MAP( mbundle )

    #plt.figure()
    #plt.errorbar( lc['jd'], lc['spec_lc_flux'][:,channel_ix], yerr=lc['spec_lc_uncs'][:,channel_ix], fmt='ok' )
    #pdb.set_trace()

    # Do a simple linear trend plus transit fit to start:
    if model!=1:
        mbundle_simple, gp_simple, cpar_ranges_simple = get_spec_mbundle( lc, 1, ld_type, channel_ix )
        mp_simple = pyhm.MAP( mbundle_simple )
        mp_simple.fit( xtol=1e-4, ftol=1e-4, maxfun=10000, maxiter=10000 )

    print '\nResults of simple preliminary MLE fit:'
    for key in mp_simple.model.free.keys():
        print key, mp_simple.model.free[key].value

    print '\nRunning MLE trials:'
    mle_trial_results = []
    trial_logps = np.zeros( ntrials )
    for i in range( ntrials ):
        print ' ... trial {0} of {1}'.format( i+1, ntrials )
        mle_ok = False
        while mle_ok==False:
            # Install the best fit parameters from the simple model
            # as starting guesses for the actual fit below:
            if model!=1: # assuming we're not doing the simple linear-time-trend fit
                for key in mp_simple.model.free.keys():
                    try:
                        mp.model.free[key].value = mp_simple.model.free[key].value
                    except:
                        pass
                # However, lets ensure the planet parameters we start
                # with are roughly what we expect:
                # NOTE: This needs to be generalised!!
                mp.RpRs.value = 0.1205+0.0010*np.random.random()
                mp.beta.value = 0.5*np.random.random()
                mp.foot.value = 1
                mp.tgrad.value = 0
                # Install reasonable starting values for the cpars:
                for key in cpar_ranges.keys():
                    mp.model.free[key].value = cpar_ranges[key].random()
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
    print '\nBest MLE solution:'
    for key in mp.model.free.keys():
        mle_vals[key] = mle_trial_results[ix][key]
        print key, mle_vals[key]
    print 'logp', trial_logps[ix]
    return mle_vals


def mcmc_white( lc, init_vals, model=1, ld_type='nonlin', nchains=2, nsteps=400, nwalkers=200, orbpars='free' ):
    """
    Note that first orbit should be discarded from lc prior to passing it in here.
    """

    # Get the model bundle and GP object:
    mbundle, gp, cpar_ranges = get_white_mbundle( lc, model, ld_type, orbpars=orbpars )

    # Initialise the emcee sampler:
    mcmc = pyhm.MCMC( mbundle )
    mcmc.assign_step_method( pyhm.BuiltinStepMethods.AffineInvariant )
    
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
                    startpos = init_vals[key]*( 1 + 0.0001*np.random.randn() )
                    mcmc.model.free[key].value = startpos
                    if np.isfinite( mcmc.model.free[key].logp() )==True:
                        startpos_ok = True
                        #print 'Walker {0} starting position for {1} verified.'.format( i+1, key )
                    elif ( mcmc.model.free[key].value<0 )*( mcmc.model.free[key].value>-1e-6 ):
                        mcmc.mode.free[key].value = np.abs( mcmc.model.free[key].value )
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

    # Make a plot of the walker chains, to get a feel for convergence:
    green_l = np.array( [ 135., 220., 140. ] )/256.
    green_d = np.array( [ 15., 120., 25. ] )/256.
    purple_l = np.array( [ 185., 160., 200. ] )/256.
    purple_d = np.array( [ 95., 30., 135. ] )/256.
    c_chains = [ green_l, purple_l ]
    c_avgs = [ green_d, purple_d ]
    n_to_plot = min( [ nchains, 2 ] )
    plt.ioff()
    figw = 14
    figh = 14
    fig_walkers = plt.figure( figsize=[ figw, figh ] )
    
    keys = mcmc.model.free.keys()
    npar = len( keys )
    hbuff = 0.05
    vbuff = 0.05
    epsi = 0.2
    axh = ( 1-1.5*vbuff-npar*vbuff*epsi )/float( npar+1 )
    axw = 1-2*hbuff
    xlow = 1.7*hbuff
    axs = []
    # Axis for the log likelihood chains:
    ylow0 = 1-0.5*vbuff-axh
    ax0 = fig_walkers.add_axes( [ xlow, ylow0, axw, axh ] )
    ax0.set_ylabel( 'logp' )
    axs += [ ax0 ]
    # Axes for the free parameter chains:
    for i in range( 1, npar+1 ):
        ylow = 1-0.5*vbuff-axh*( i+1 )-i*epsi*vbuff
        axi = fig_walkers.add_axes( [ xlow, ylow, axw, axh ], sharex=ax0 )
        axi.set_ylabel( keys[i-1] )
        axs += [ axi ]
    for k in range( n_to_plot ):
        for j in range( nwalkers ):
            ax0.plot( walker_chains[k]['logp'][:,j], '-', color=c_chains[k], lw=1, zorder=0 )
        ax0.plot( np.mean( walker_chains[k]['logp'], axis=1 ), '-', color=c_avgs[k], lw=2, zorder=1 )
        acor_vals = np.zeros( npar )
        for i in range( 1, npar+1 ):
            for j in range( nwalkers ):
                axs[i].plot( walker_chains[k][keys[i-1]][:,j], '-', color=c_chains[k], lw=1, zorder=0 )
            axs[i].plot( np.mean( walker_chains[k][keys[i-1]], axis=1 ), '-', color=c_avgs[k], lw=2, zorder=1 )
            #axi.axvline( nburn, ls='-', lw=2, color='b' )
            #axi.axvline( nburn+np.abs( acor_integ[keys[i-1]] ), ls='--', lw=2, color='b' )
            acor_vals[i-1] = np.abs( acor_integs[k][keys[i-1]] )
            axs[i].axvline( acor_vals[i-1], ls='--', lw=2, color=c_avgs[k], zorder=2 )
            text_str = 'acor={0:.1f}'.format( acor_vals[i-1] )
            text_fs = 14
            ylims = axs[i].get_ylim()
            ytext = ylims[1]-0.1*( ylims[1]-ylims[0] )
            axs[i].text( acor_vals[i-1]+0.01*nsteps, ytext, text_str, \
                         horizontalalignment='left', verticalalignment='top', \
                         fontsize=text_fs, color=c_avgs[k], zorder=4 )
        for axi in axs:
            axi.axvline( acor_vals.max(), ls='-', lw=2, color=c_avgs[k], zorder=3 )
        
    for i in range( npar ):
        plt.setp( axs[i].xaxis.get_ticklabels(), visible=False )
    axs[-1].set_xlabel( 'nsteps' )
    ax0.set_xlim( [ 0, nsteps-1 ] )

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
    mle_refined['incl'] = np.rad2deg( np.arccos( mle_refined['b']/mle_refined['aRs'] ) )
    mle_refined['Tmid'] = mbundle['Tmid_lit'] + mle_refined['delT']
    systematics, psignal, sig = eval_model_bestfit( mbundle, gp, mle_refined, model, which_type='white' )
    jd = mbundle['jd']
    flux_vals = mbundle['flux_vals']
    figw = 12
    figh = 8
    fig_bestfit = plt.figure( figsize=[ figw, figh ] )
    hbuff = 0.05
    vbuff = 0.05
    axw = 1-2.5*hbuff
    axh1 = 0.8*( 1-3*vbuff )
    axh2 = 0.2*( 1-3*vbuff )
    xlow = 2.0*hbuff
    ylow1 = 1-1.2*vbuff-axh1
    ylow2 = ylow1-0.5*vbuff-axh2
    ax1 = fig_bestfit.add_axes( [ xlow, ylow1, axw, axh1 ] )
    plt.setp( ax1.xaxis.get_ticklabels(), visible=False )
    ax2 = fig_bestfit.add_axes( [ xlow, ylow2, axw, axh2 ], sharex=ax1 )
    ax1.plot( jd, flux_vals, 'ok' )
    resids_ppm = (1e6)*( flux_vals-systematics*psignal )
    errbars_ppm = (1e6)*mbundle['sigw_approx']*mle_refined['beta']
    ax2.errorbar( jd, resids_ppm, yerr=errbars_ppm, fmt='ok' )
    ax1.plot( jd, systematics, '-r' )
    ax1.plot( jd, systematics*psignal, '-b' )
    ax1.set_xlim( [ jd.min()-10./60./24., jd.max()+10./60./24. ] )

    text_fs = 18
    text_str = 'RpRs = {0:.5f}\naRs = {1:.4f}\nb = {2:.4f}\ni = {3:.4f}deg'\
               .format( mle_refined['RpRs'], mle_refined['aRs'], mle_refined['b'], mle_refined['incl'] )
    text_str += '\ndelT = {0:.2f}sec\nbeta = {1:.4f}'\
                .format( mle_refined['delT']*24.*60.*60., mle_refined['beta'] )
    ax1.text( 0.05, 0.5, text_str, horizontalalignment='left', verticalalignment='top', \
              fontsize=text_fs, transform=ax1.transAxes )
    ax2.set_xlabel( 'JD' )
    ax1.set_ylabel( 'Rel flux' )
    ax2.set_ylabel( 'Resids (ppm )' )
    ax2.axhline( 0. )

    
    z = { 'walker_chains':walker_chains, 'acor_funcs':acor_funcs, 'acor_integs':acor_integs, \
          'fig_walkers':fig_walkers, 'fig_bestfit':fig_bestfit, 'mle_refined':mle_refined }
    plt.ion()

    return z


def mcmc_spec( lc, channel_ix, init_vals, model=1, ld_type='nonlin', nchains=2, nsteps=400, nwalkers=200 ):
    """
    Note that first orbit should be discarded from lc prior to passing it in here.
    """

    # Get the model bundle and GP object:
    mbundle, gp, cpar_ranges = get_spec_mbundle( lc, model, ld_type, channel_ix )
    z = mcmc_backend( mbundle, gp, init_vals, nchains, nsteps, nwalkers, model=model, which_type='spec' )

    return z

def mcmc_backend( mbundle, gp, init_vals, nchains, nsteps, nwalkers, model=1, which_type='' ):
    """
    This should be called by both mcmc_white and mcmc_spec.
    """

    # Initialise the emcee sampler:
    mcmc = pyhm.MCMC( mbundle )
    mcmc.assign_step_method( pyhm.BuiltinStepMethods.AffineInvariant )
    
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
                    startpos = init_vals[key]*( 1 + 0.0001*np.random.randn() )
                    mcmc.model.free[key].value = startpos
                    if np.isfinite( mcmc.model.free[key].logp() )==True:
                        startpos_ok = True
                        #print 'Walker {0} starting position for {1} verified.'.format( i+1, key )
                    elif ( mcmc.model.free[key].value<0 )*( mcmc.model.free[key].value>-1e-6 ):
                        mcmc.mode.free[key].value = np.abs( mcmc.model.free[key].value )
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
        

    # Make a plot of the walker chains, to get a feel for convergence:
    green_l = np.array( [ 135., 220., 140. ] )/256.
    green_d = np.array( [ 15., 120., 25. ] )/256.
    purple_l = np.array( [ 185., 160., 200. ] )/256.
    purple_d = np.array( [ 95., 30., 135. ] )/256.
    c_chains = [ green_l, purple_l ]
    c_avgs = [ green_d, purple_d ]
    n_to_plot = min( [ nchains, 2 ] )
    plt.ioff()
    figw = 14
    figh = 14
    fig_walkers = plt.figure( figsize=[ figw, figh ] )
    
    keys = mcmc.model.free.keys()
    npar = len( keys )
    hbuff = 0.05
    vbuff = 0.05
    epsi = 0.2
    axh = ( 1-1.5*vbuff-npar*vbuff*epsi )/float( npar+1 )
    axw = 1-2*hbuff
    xlow = 1.7*hbuff
    axs = []
    # Axis for the log likelihood chains:
    ylow0 = 1-0.5*vbuff-axh
    ax0 = fig_walkers.add_axes( [ xlow, ylow0, axw, axh ] )
    ax0.set_ylabel( 'logp' )
    axs += [ ax0 ]
    # Axes for the free parameter chains:
    for i in range( 1, npar+1 ):
        ylow = 1-0.5*vbuff-axh*( i+1 )-i*epsi*vbuff
        axi = fig_walkers.add_axes( [ xlow, ylow, axw, axh ], sharex=ax0 )
        axi.set_ylabel( keys[i-1] )
        axs += [ axi ]
    for k in range( n_to_plot ):
        for j in range( nwalkers ):
            ax0.plot( walker_chains[k]['logp'][:,j], '-', color=c_chains[k], lw=1, zorder=0 )
        ax0.plot( np.mean( walker_chains[k]['logp'], axis=1 ), '-', color=c_avgs[k], lw=2, zorder=1 )
        acor_vals = np.zeros( npar )
        for i in range( 1, npar+1 ):
            for j in range( nwalkers ):
                axs[i].plot( walker_chains[k][keys[i-1]][:,j], '-', color=c_chains[k], lw=1, zorder=0 )
            axs[i].plot( np.mean( walker_chains[k][keys[i-1]], axis=1 ), '-', color=c_avgs[k], lw=2, zorder=1 )
            #axi.axvline( nburn, ls='-', lw=2, color='b' )
            #axi.axvline( nburn+np.abs( acor_integ[keys[i-1]] ), ls='--', lw=2, color='b' )
            acor_vals[i-1] = np.abs( acor_integs[k][keys[i-1]] )
            axs[i].axvline( acor_vals[i-1], ls='--', lw=2, color=c_avgs[k], zorder=2 )
            text_str = 'acor={0:.1f}'.format( acor_vals[i-1] )
            text_fs = 14
            ylims = axs[i].get_ylim()
            ytext = ylims[1]-0.1*( ylims[1]-ylims[0] )
            axs[i].text( acor_vals[i-1]+0.01*nsteps, ytext, text_str, \
                         horizontalalignment='left', verticalalignment='top', \
                         fontsize=text_fs, color=c_avgs[k], zorder=4 )
        for axi in axs:
            axi.axvline( acor_vals.max(), ls='-', lw=2, color=c_avgs[k], zorder=3 )
        
    for i in range( npar ):
        plt.setp( axs[i].xaxis.get_ticklabels(), visible=False )
    axs[-1].set_xlabel( 'nsteps' )
    ax0.set_xlim( [ 0, nsteps-1 ] )

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
    if which_type=='white':
        mle_refined['incl'] = np.rad2deg( np.arccos( mle_refined['b']/mle_refined['aRs'] ) )
    systematics, psignal, sig = eval_model_bestfit( mbundle, gp, mle_refined, model, which_type=which_type )
    jd = mbundle['jd']
    flux_vals = mbundle['flux_vals']
    figw = 12
    figh = 8
    fig_bestfit = plt.figure( figsize=[ figw, figh ] )
    hbuff = 0.05
    vbuff = 0.05
    axw = 1-2.5*hbuff
    axh1 = 0.8*( 1-3*vbuff )
    axh2 = 0.2*( 1-3*vbuff )
    xlow = 2.0*hbuff
    ylow1 = 1-1.2*vbuff-axh1
    ylow2 = ylow1-0.5*vbuff-axh2
    ax1 = fig_bestfit.add_axes( [ xlow, ylow1, axw, axh1 ] )
    plt.setp( ax1.xaxis.get_ticklabels(), visible=False )
    ax2 = fig_bestfit.add_axes( [ xlow, ylow2, axw, axh2 ], sharex=ax1 )
    ax1.plot( jd, flux_vals, 'ok' )
    resids_ppm = (1e6)*( flux_vals-systematics*psignal )
    errbars_ppm = (1e6)*mbundle['sigw_approx']*mle_refined['beta']
    ax2.errorbar( jd, resids_ppm, yerr=errbars_ppm, fmt='ok' )
    ax1.plot( jd, systematics, '-r' )
    ax1.plot( jd, systematics*psignal, '-b' )
    ax1.set_xlim( [ jd.min()-10./60./24., jd.max()+10./60./24. ] )

    text_fs = 18
    if which_type=='white':
        text_str = 'RpRs = {0:.5f}\naRs = {1:.4f}\nb = {2:.4f}\ni = {3:.4f}deg'\
                   .format( parsfit['RpRs'], parsfit['aRs'], parsfit['b'], parsfit['incl'] )
        text_str += '\ndelT = {0:.2f}sec\nbeta = {1:.4f}'\
                    .format( parsfit['delT']*24.*60.*60., parsfit['beta'] )
    elif which_type=='spec':
        text_str = 'RpRs = {0:.5f}'.format( mle_refined['RpRs'] )
    else:
        pdb.set_trace()
    ax1.text( 0.05, 0.5, text_str, horizontalalignment='left', verticalalignment='top', \
              fontsize=text_fs, transform=ax1.transAxes )
    ax2.set_xlabel( 'JD' )
    ax1.set_ylabel( 'Rel flux' )
    ax2.set_ylabel( 'Resids (ppm )' )
    ax2.axhline( 0. )
    
    mle_refined = { 'pars':mle_refined, 'psignal':psignal, 'systematics':systematics }

    z = { 'walker_chains':walker_chains, 'acor_funcs':acor_funcs, 'acor_integs':acor_integs, \
          'fig_walkers':fig_walkers, 'fig_bestfit':fig_bestfit, 'mle_refined':mle_refined }
    plt.ion()

    return z


def get_white_mbundle( lc, model, ld_type, orbpars='free' ):
    """
    Note that first orbit should be discarded from lc prior to passing it in here.
    """

    jd = lc['jd']
    hstphase = lc['hstphase']
    x = lc['jd']
    y = lc['jd']
    wavshift_micron = lc['wavshift_micron']
    flux_vals = lc['white_lc_flux']
    flux_uncs = lc['white_lc_uncs']
    ld_quad = lc['ld_quad']
    ld_nonlin = lc['ld_nonlin']

    syspars = prep.get_syspars()
    syspars['ld'] = ld_type
    syspars['gam1'] = lc['ld_quad'][0]
    syspars['gam2'] = lc['ld_quad'][1]
    syspars['c1'] = lc['ld_nonlin'][0]
    syspars['c2'] = lc['ld_nonlin'][1]
    syspars['c3'] = lc['ld_nonlin'][2]
    syspars['c4'] = lc['ld_nonlin'][3]

    RpRs_lit = syspars['RpRs']
    aRs_lit = syspars['aRs']
    b_lit = syspars['b']
    Tmid_lit = syspars['T0']
    while Tmid_lit<lc['jd'].min():
        Tmid_lit += syspars['P']
    while Tmid_lit>lc['jd'].max():
        Tmid_lit -= syspars['P']
    sigw_approx = np.median( flux_uncs )

    flux_norm = np.median( flux_vals[:11] )
    flux_vals /= flux_norm
    flux_uncs /= flux_norm
    sigw_approx /= flux_norm

    # Free parameters common to all white lightcurve models:
    RpRs = pyhm.Uniform( 'RpRs', lower=0.05, upper=0.25, value=0.12 )
    beta = pyhm.Uniform( 'beta', lower=0, upper=100.0, value=0.0 )
    foot = pyhm.Uniform( 'foot', lower=0.5, upper=1.5, value=1.0 )
    delT = pyhm.Uniform( 'delT', lower=-1./24., upper=1./24., value=0.0 )
    if orbpars=='free':
        aRs = pyhm.Uniform( 'aRs', lower=4., upper=14., value=8.85 )
        b = pyhm.Uniform( 'b', lower=0.3, upper=0.7, value=0.5 )
    elif orbpars=='prior':
        aRs_mu, aRs_sig, b_mu, b_sig = calculate_wmean_orbpars( make_plot=False )
        aRs = pyhm.Gaussian( 'aRs', mu=aRs_mu, sigma=aRs_sig, value=aRs_mu )
        b = pyhm.Gaussian( 'b', mu=b_mu, sigma=b_sig, value=b_mu )
    else:
        pdb.set_trace()

    # Standardise the input variables:
    tv = ( lc['jd']-np.mean( lc['jd'] ) )/np.std( lc['jd'] )
    xv = ( lc['x']-np.mean( lc['x'] ) )/np.std( lc['x'] )
    yv = ( lc['y']-np.mean( lc['y'] ) )/np.std( lc['y'] )
    phiv = ( lc['hstphase']-np.mean( lc['hstphase'] ) )/np.std( lc['hstphase'] )
    wv = ( lc['wavshift_micron']-np.mean( lc['wavshift_micron'] ) )/np.std( lc['wavshift_micron'] )

    if model==1:
        z = { 'jd':jd, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit, \
              'syspars':syspars, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
              'tv':tv  }
        mbundle, gp, cpar_ranges = get_model1( z, which_type='white' )
    elif ( model==2 )+( model==3 ):
        z = { 'jd':jd, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit, \
              'syspars':syspars, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
              'phiv':phiv, 'tv':tv  }
        mbundle, gp, cpar_ranges = white_model2or3( z, model, which_type='white' )
    elif ( model==4 )+( model==5 ):
        z = { 'jd':jd, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit, \
              'syspars':syspars, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
              'phiv':phiv, 'tv':tv  }
        mbundle, gp, cpar_ranges = white_model4or5( z, model, which_type='white' )
    elif ( model==6 )+( model==7 ):
        z = { 'jd':jd, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit, \
              'syspars':syspars, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
              'phiv':phiv, 'tv':tv, 'wv':wv  }
        mbundle, gp, cpar_ranges = white_model6or7( z, model, which_type='white' )
    elif ( model==8 )+( model==9 ):
        z = { 'jd':jd, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit, \
              'syspars':syspars, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
              'phiv':phiv, 'tv':tv, 'wv':wv  }
        mbundle, gp, cpar_ranges = white_model8or9( z, model, which_type='white' )
    elif ( model==10 )+( model==11 ):
        z = { 'jd':jd, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit, \
              'syspars':syspars, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
              'phiv':phiv, 'tv':tv, 'wv':wv  }
        mbundle, gp, cpar_ranges = get_model10or11( z, model, which_type='white' )

    else:
        pdb.set_trace()

    return mbundle, gp, cpar_ranges


def get_spec_mbundle( lc, model, ld_type, ix ):
    """
    Note that first orbit should be discarded from lc prior to passing it in here.
    """

    jd = lc['jd']
    hstphase = lc['hstphase']
    x = lc['jd']
    y = lc['jd']
    wavshift_micron = lc['wavshift_micron']
    flux_vals = lc['spec_lc_flux'][:,ix]
    flux_uncs = lc['spec_lc_uncs'][:,ix]

    syspars = prep.get_syspars()
    syspars['ld'] = ld_type
    syspars['gam1'] = lc['ld_quad'][ix,:][0]
    syspars['gam2'] = lc['ld_quad'][ix,:][1]
    syspars['c1'] = lc['ld_nonlin'][ix,:][0]
    syspars['c2'] = lc['ld_nonlin'][ix,:][1]
    syspars['c3'] = lc['ld_nonlin'][ix,:][2]
    syspars['c4'] = lc['ld_nonlin'][ix,:][3]
    white_mle = lc['white_mle']['mle_pars']
    syspars['aRs'] = white_mle['aRs']
    syspars['b'] = white_mle['b']
    syspars['incl'] = white_mle['incl']
    syspars['T0'] = white_mle['Tmid']

    RpRs_white = white_mle['RpRs']
    sigw_approx = 0.5*np.median( flux_uncs )

    flux_norm = np.median( flux_vals[:11] )
    flux_vals /= flux_norm
    flux_uncs /= flux_norm
    sigw_approx /= flux_norm

    # Free parameters common to all models:
    RpRs = pyhm.Uniform( 'RpRs', lower=0.05, upper=0.25, value=0.12 )
    delT = pyhm.Uniform( 'delT', lower=-1./24., upper=1./24., value=0.0 )
    beta = pyhm.Uniform( 'beta', lower=0.0, upper=100.0, value=0.0 )
    foot = pyhm.Uniform( 'foot', lower=0.5, upper=1.5, value=1.0 )

    tv = ( lc['jd']-np.mean( lc['jd'] ) )/np.std( lc['jd'] )
    xv = ( lc['x']-np.mean( lc['x'] ) )/np.std( lc['x'] )
    yv = ( lc['y']-np.mean( lc['y'] ) )/np.std( lc['y'] )
    phiv = ( lc['hstphase']-np.mean( lc['hstphase'] ) )/np.std( lc['hstphase'] )
    wv = ( lc['wavshift_micron']-np.mean( lc['wavshift_micron'] ) )/np.std( lc['wavshift_micron'] )

    if model==1:
        z = { 'jd':jd, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, \
              'syspars':syspars, 'RpRs':RpRs, 'beta':beta, 'foot':foot, \
              'tv':tv  }
        mbundle, gp, cpar_ranges = get_model1( z, which_type='spec' )
    elif ( model==2 )+( model==3 ):
        z = { 'jd':jd, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit, \
              'syspars':syspars, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
              'phiv':phiv, 'tv':tv  }
        mbundle, gp, cpar_ranges = spec_model2or3( z, model, which_type='spec' )
    elif ( model==4 )+( model==5 ):
        z = { 'jd':jd, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit, \
              'syspars':syspars, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
              'phiv':phiv, 'tv':tv  }
        mbundle, gp, cpar_ranges = spec_model4or5( z, model, which_type='spec' )
    elif ( model==6 )+( model==7 ):
        z = { 'jd':jd, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit, \
              'syspars':syspars, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
              'phiv':phiv, 'tv':tv, 'wv':wv  }
        mbundle, gp, cpar_ranges = spec_model6or7( z, model, which_type='spec' )
    elif ( model==8 )+( model==9 ):
        z = { 'jd':jd, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit, \
              'syspars':syspars, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
              'phiv':phiv, 'tv':tv, 'wv':wv  }
        mbundle, gp, cpar_ranges = spec_model8or9( z, model, which_type='spec' )
    elif ( model==10 )+( model==11 ):
        z = { 'jd':jd, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, \
              'syspars':syspars, 'RpRs':RpRs, 'beta':beta, 'foot':foot, \
              'phiv':phiv, 'tv':tv, 'wv':wv  }
        mbundle, gp, cpar_ranges = get_model10or11( z, model, which_type='spec' )
    else:
        pdb.set_trace()

    return mbundle, gp, cpar_ranges


def eval_model_bestfit( mbundle, gp, parsfit, model, which_type='' ):

    sigw_approx = mbundle['sigw_approx']
    syspars = mbundle['syspars']
    jd = mbundle['jd']
    flux_vals = mbundle['flux_vals']
    ndat = len( flux_vals )
    syspars['RpRs'] = parsfit['RpRs']
    if which_type=='white':
        syspars['aRs'] = parsfit['aRs']
        syspars['b'] = parsfit['b']
        syspars['incl'] = np.rad2deg( np.arccos( parsfit['b']/parsfit['aRs'] ) )
        parsfit['Tmid'] = mbundle['Tmid_lit'] + parsfit['delT']
    elif which_type=='spec':
        pass
    else:
        pdb.set_trace()
    psignal = transit.ma02_aRs( jd, **syspars )
    if model==1:
        #syspars['RpRs'] = parsfit['RpRs']
        #syspars['aRs'] = parsfit['aRs']
        #syspars['b'] = parsfit['b']
        #syspars['incl'] = np.rad2deg( np.arccos( parsfit['b']/parsfit['aRs'] ) )
        #syspars['T0'] = Tmid_lit + parsfit['delT']
        #psignal = transit.ma02_aRs( jd, **syspars )
        #systematics = parsfit['foot'] + parsfit['tgrad']*mbundle['tv']
        #sig = None
        psignal, baseline = eval_model1( jd, mbundle['tv'], flux_vals, syspars, parsfit, \
                                         which_type=which_type )
    elif ( model==2 )+( model==3 ):
        psignal, baseline, gp = eval_model2or3( jd, flux_vals, syspars, parsfit, gp, sigw_approx, \
                                                which_type=which_type )
        mu, sig = gp.predictive( xnew=gp.xtrain, enew=gp.etrain )
        systematics = baseline + mu.flatten()
    elif ( model==4 )+( model==5 ):
        psignal, baseline, gp = eval_model4or5( jd, flux_vals, syspars, parsfit, gp, sigw_approx, \
                                                mbundle['tv'], which_type=which_type )
        mu, sig = gp.predictive( xnew=gp.xtrain, enew=gp.etrain )
        systematics = baseline + mu.flatten()
    elif ( model==6 )+( model==7 ):
        psignal, baseline, gp = eval_model6or7( jd, flux_vals, syspars, parsfit, gp, sigw_approx, \
                                                which_type=which_type )
        mu, sig = gp.predictive( xnew=gp.xtrain, enew=gp.etrain )
        systematics = baseline + mu.flatten()
    elif ( model==8 )+( model==9 ):
        psignal, baseline, gp = eval_model8or9( jd, flux_vals, syspars, parsfit, gp, sigw_approx, \
                                                mbundle['tv'], which_type=which_type )
        mu, sig = gp.predictive( xnew=gp.xtrain, enew=gp.etrain )
        systematics = baseline + mu.flatten()
    elif ( model==10 )+( model==11 ):
        psignal, baseline, gp = eval_model10or11( jd, flux_vals, syspars, parsfit, gp, sigw_approx, \
                                                  mbundle['tv'], which_type=which_type )
        mu, sig = gp.predictive( xnew=gp.xtrain, enew=gp.etrain )
        systematics = baseline + mu.flatten()
    
    return systematics, psignal, sig


def get_model1( z, which_type='white' ):
    
    jd = z['jd']
    flux_vals = z['flux_vals']
    sigw_approx = z['sigw_approx']
    Tmid_lit = z['Tmid_lit']
    syspars = z['syspars']
    RpRs = z['RpRs']
    if which_type=='white':

        aRs = z['aRs']
        b = z['b']
        delT = z['delT']
    beta = z['beta']
    foot = z['foot']
    tv = z['tv']
    ndat = len( flux_vals )
    sigw_approx_arr = sigw_approx*np.ones( ndat )

    tgrad = pyhm.Uniform( 'tgrad', lower=-1, upper=1, value=0.0 )

    if which_type=='white':
        parents = { 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'Tmid_lit':Tmid_lit, 'delT':delT, 'beta':beta, \
                    'foot':foot, 'tgrad':tgrad }
    elif which_type=='spec':
        parents = { 'RpRs':RpRs, 'beta':beta, 'foot':foot, 'tgrad':tgrad }
    else:
        pdb.set_trace()

    @pyhm.stochastic( observed=True )
    def loglikelihood( value=flux_vals, parents=parents ):
        def logp( value, parents=parents ):
            parents['Tmid'] = parents['Tmid_lit'] + parents['delT']
            psignal, baseline = eval_model1( jd, tv, flux_vals, syspars, parents, which_type=which_type )
            psignal = transit.ma02_aRs( jd, **syspars )
            baseline = parents['foot'] + parents['tgrad']*tv
            resids_arr = value - psignal*baseline
            uncs_arr = sigw_approx_arr*parents['beta']
            logp_val = logp_mvnormal_whitenoise( resids_arr, uncs_arr, ndat )
            #print 'aaaa', syspars['RpRs'], parents['beta']
            if 0: # delete this stuff
                plt.figure()
                plt.errorbar(jd,flux_vals,fmt='ok',yerr=gp.etrain)
                plt.plot(jd,psignal*baseline,'-r')
                plt.plot(jd,psignal*baseline,'.c')
                pdb.set_trace()
            
            return logp_val

    if which_type=='white':
        mbundle = { 'syspars':syspars, 'loglikelihood':loglikelihood, 'RpRs':RpRs, 'aRs':aRs, 'b':b, \
                    'delT':delT, 'beta':beta, 'foot':foot, 'tgrad':tgrad, 'flux_vals':flux_vals, \
                    'jd':jd, 'tv':tv, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit }
    elif which_type=='spec':
        mbundle = { 'syspars':syspars, 'loglikelihood':loglikelihood, 'RpRs':RpRs, \
                    'beta':beta, 'foot':foot, 'tgrad':tgrad, 'flux_vals':flux_vals, \
                    'jd':jd, 'tv':tv, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit }
    else:
        pdb.set_trace()

    
    gp = None
    cpar_ranges = None

    return mbundle, gp, cpar_ranges

    
def white_model2or3( z, model ):
    
    jd = z['jd']
    flux_vals = z['flux_vals']
    sigw_approx = z['sigw_approx']
    Tmid_lit = z['Tmid_lit']
    syspars = z['syspars']
    RpRs = z['RpRs']
    aRs = z['aRs']
    b = z['b']
    delT = z['delT']
    beta = z['beta']
    foot = z['foot']
    phiv = z['phiv']
    tv = z['tv']
    ndat = len( flux_vals )

    gp = gp_class.gp( which_type='full' )
    gp.xtrain = np.column_stack( [ phiv, tv ] )
    if model==2:
        gp.cfunc = kernels.sqexp_invL_ard
    elif model==3:
        gp.cfunc = kernels.matern32_invL_ard
    Amp = pyhm.Gamma( 'Amp', alpha=1, beta=1e-1, value=5e-3 )
    iLhstphase = pyhm.Gamma( 'iLhstphase', alpha=1, beta=1e-4, value=1. )
    iLt = pyhm.Gamma( 'iLt', alpha=1, beta=1e2, value=0.1 )

    parents = { 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
                'Amp':Amp, 'iLhstphase':iLhstphase, 'iLt':iLt }
    Amp_range = pyhm.Uniform( 'Amp', lower=5e-4, upper=8e-3 )
    iLhstphase_range = pyhm.Uniform( 'iLhstphase', lower=0.7, upper=1.4 )
    iLt_range = pyhm.Uniform( 'iLt', lower=0.7, upper=1.4 )
    cpar_ranges = { 'Amp':Amp_range, 'iLhstphase':iLhstphase_range, 'iLt':iLt }

    @pyhm.stochastic( observed=True )
    def loglikelihood( value=flux_vals, parents=parents ):
        def logp( value, parents=parents ):
            psignal, baseline, gp_updated = eval_model2or3( jd, flux_vals, syspars, parents, gp, \
                                                            Tmid_lit, sigw_approx )
            logp_val = gp_updated.logp_builtin()
            return logp_val

    mbundle = { 'syspars':syspars, 'loglikelihood':loglikelihood, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, \
                'beta':beta, 'foot':foot, 'Amp':Amp, 'iLhstphase':iLhstphase, 'iLt':iLt, 'jd':jd, 'tv':tv, \
                'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'xtrain':gp.xtrain, 'Tmid_lit':Tmid_lit }
    
    return mbundle, gp, cpar_ranges
    
    

def white_model4or5( z, model ):
    
    jd = z['jd']
    flux_vals = z['flux_vals']
    sigw_approx = z['sigw_approx']
    Tmid_lit = z['Tmid_lit']
    syspars = z['syspars']
    RpRs = z['RpRs']
    aRs = z['aRs']
    b = z['b']
    delT = z['delT']
    beta = z['beta']
    foot = z['foot']
    phiv = z['phiv']
    tv = z['tv']
    ndat = len( flux_vals )

    gp = gp_class.gp( which_type='full' )
    gp.xtrain = np.column_stack( [ phiv ] )
    if model==4:
        gp.cfunc = kernels.sqexp_invL
    elif model==5:
        gp.cfunc = kernels.matern32_invL
    else:
        pdb.set_trace()
    Amp = pyhm.Gamma( 'Amp', alpha=1, beta=1e-1, value=1e-3 )
    iLhstphase = pyhm.Gamma( 'iLhstphase', alpha=1, beta=1e-1, value=1. )
    tgrad = pyhm.Uniform( 'tgrad', lower=-1, upper=1, value=0.0 )

    Amp_range = pyhm.Uniform( 'Amp', lower=5e-4, upper=8e-3 )
    iLhstphase_range = pyhm.Uniform( 'iLhstphase', lower=0.7, upper=1.4 )
    cpar_ranges = { 'Amp':Amp_range, 'iLhstphase':iLhstphase_range }

    parents = { 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
                'Amp':Amp, 'iLhstphase':iLhstphase, 'tgrad':tgrad }
    
    @pyhm.stochastic( observed=True )
    def loglikelihood( value=flux_vals, parents=parents ):
        def logp( value, parents=parents ):
            psignal, baseline, gp_updated = eval_model4or5( jd, flux_vals, syspars, parents, gp, \
                                                            Tmid_lit, sigw_approx, tv )
            logp_val = gp_updated.logp_builtin()
            return logp_val

    mbundle = { 'syspars':syspars, 'loglikelihood':loglikelihood, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, \
                'beta':beta, 'foot':foot, 'Amp':Amp, 'iLhstphase':iLhstphase, 'tgrad':tgrad, 'jd':jd, 'tv':tv, \
                'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'xtrain':gp.xtrain, 'Tmid_lit':Tmid_lit }
    
    return mbundle, gp, cpar_ranges
    
    
def white_model6or7( z, model ):
    
    jd = z['jd']
    flux_vals = z['flux_vals']
    sigw_approx = z['sigw_approx']
    Tmid_lit = z['Tmid_lit']
    syspars = z['syspars']
    RpRs = z['RpRs']
    aRs = z['aRs']
    b = z['b']
    delT = z['delT']
    beta = z['beta']
    foot = z['foot']
    phiv = z['phiv']
    tv = z['tv']
    wv = z['wv']
    ndat = len( flux_vals )

    gp = gp_class.gp( which_type='full' )
    gp.xtrain = np.column_stack( [ phiv, tv, wv ] )
    if model==6:
        gp.cfunc = kernels.sqexp_invL_ard
    elif model==7:
        gp.cfunc = kernels.matern32_invL_ard
    Amp = pyhm.Gamma( 'Amp', alpha=1, beta=1e-1, value=1e-3 )
    iLhstphase = pyhm.Gamma( 'iLhstphase', alpha=1, beta=1e-4, value=1. )
    iLt = pyhm.Gamma( 'iLt', alpha=1, beta=1e-4, value=1. )
    iLwavshift = pyhm.Gamma( 'iLwavshift', alpha=1, beta=1e-4, value=1. )

    parents = { 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
                'Amp':Amp, 'iLhstphase':iLhstphase, 'iLt':iLt, 'iLwavshift':iLwavshift }
    
    Amp_range = pyhm.Uniform( 'Amp', lower=5e-4, upper=8e-3 )
    iLhstphase_range = pyhm.Uniform( 'iLhstphase', lower=0.7, upper=1.4 )
    iLt_range = pyhm.Uniform( 'iLt', lower=0.7, upper=1.4 )
    iLwavshift_range = pyhm.Uniform( 'iLwavshift', lower=0.7, upper=1.4 )
    cpar_ranges = { 'Amp':Amp_range, 'iLhstphase':iLhstphase_range, 'iLt':iLt, 'iLwavshift':iLwavshift }

    @pyhm.stochastic( observed=True )
    def loglikelihood( value=flux_vals, parents=parents ):
        def logp( value, parents=parents ):
            psignal, baseline, gp_updated = eval_model6or7( jd, flux_vals, syspars, parents, gp, \
                                                            Tmid_lit, sigw_approx )
            logp_val = gp_updated.logp_builtin()
            return logp_val

    mbundle = { 'syspars':syspars, 'loglikelihood':loglikelihood, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, \
                'beta':beta, 'foot':foot, 'Amp':Amp, 'iLhstphase':iLhstphase, 'iLt':iLt, 'iLwavshift':iLwavshift, \
                'jd':jd, 'tv':tv, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'xtrain':gp.xtrain, \
                'Tmid_lit':Tmid_lit }
    
    return mbundle, gp, cpar_ranges
    

def white_model8or9( z, model ):
    
    jd = z['jd']
    flux_vals = z['flux_vals']
    sigw_approx = z['sigw_approx']
    Tmid_lit = z['Tmid_lit']
    syspars = z['syspars']
    RpRs = z['RpRs']
    aRs = z['aRs']
    b = z['b']
    delT = z['delT']
    beta = z['beta']
    foot = z['foot']
    phiv = z['phiv']
    tv = z['tv']
    wv = z['wv']
    ndat = len( flux_vals )

    gp = gp_class.gp( which_type='full' )
    gp.xtrain = np.column_stack( [ phiv, wv ] )
    if model==8:
        gp.cfunc = kernels.sqexp_invL_ard
    elif model==9:
        gp.cfunc = kernels.matern32_invL_ard
    else:
        pdb.set_trace()
    Amp = pyhm.Gamma( 'Amp', alpha=1, beta=1e-1, value=1e-3 )
    iLhstphase = pyhm.Gamma( 'iLhstphase', alpha=1, beta=1e-4, value=1. )
    iLwavshift = pyhm.Gamma( 'iLwavshift', alpha=1, beta=1e-4, value=1. )
    tgrad = pyhm.Uniform( 'tgrad', lower=-1, upper=1, value=0.0 )

    parents = { 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
                'Amp':Amp, 'iLhstphase':iLhstphase, 'iLwavshift':iLwavshift, 'tgrad':tgrad }
    
    Amp_range = pyhm.Uniform( 'Amp', lower=5e-4, upper=8e-3 )
    iLhstphase_range = pyhm.Uniform( 'iLhstphase', lower=0.7, upper=1.4 )
    iLwavshift_range = pyhm.Uniform( 'iLwavshift', lower=0.7, upper=1.4 )
    cpar_ranges = { 'Amp':Amp_range, 'iLhstphase':iLhstphase_range, 'iLwavshift':iLwavshift }

    @pyhm.stochastic( observed=True )
    def loglikelihood( value=flux_vals, parents=parents ):
        def logp( value, parents=parents ):
            psignal, baseline, gp_updated = eval_model8or9( jd, flux_vals, syspars, parents, gp, \
                                                            Tmid_lit, sigw_approx, tv )
            logp_val = gp_updated.logp_builtin()
            return logp_val

    mbundle = { 'syspars':syspars, 'loglikelihood':loglikelihood, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, \
                'beta':beta, 'foot':foot, 'Amp':Amp, 'iLhstphase':iLhstphase, 'iLwavshift':iLwavshift, \
                'tgrad':tgrad, 'jd':jd, 'tv':tv, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, \
                'xtrain':gp.xtrain, 'Tmid_lit':Tmid_lit }
    
    return mbundle, gp, cpar_ranges
    
    
def get_model10or11( z, model, which_type=None ):
    
    jd = z['jd']
    flux_vals = z['flux_vals']
    #sigw_approx = 0.6*(2e-4)
    sigw_approx = z['sigw_approx']
    Tmid_lit = z['Tmid_lit']

    syspars = z['syspars']
    RpRs = z['RpRs']
    if which_type=='white':
        aRs = z['aRs']
        b = z['b']
        delT = z['delT']

    beta = z['beta']
    foot = z['foot']
    phiv = z['phiv']
    tv = z['tv']
    ndat = len( flux_vals )

    gp = gp_class.gp( which_type='full' )
    gp.xtrain = np.column_stack( [ phiv, tv ] )
    if model==10:
        gp.cfunc = kernels.sqexp_invL_ard
    elif model==11:
        gp.cfunc = kernels.matern32_invL_ard
    # These priors are important for convergence and also ensuring that
    # iLt doesn't get too big, which is a reasonable restriction:
    #Amp = pyhm.Gamma( 'Amp', alpha=1, beta=1e2, value=5e-3 )
    #iLhstphase = pyhm.Gamma( 'iLhstphase', alpha=1, beta=1e1, value=1. )
    #iLt = pyhm.Gamma( 'iLt', alpha=1, beta=1e1, value=0.1 )
    Amp = pyhm.Gamma( 'Amp', alpha=1, beta=1e-1 )

    iLhstphase = pyhm.Gamma( 'iLhstphase', alpha=1, beta=1e-4 ) # THIS WORKED WELL

    iLt = pyhm.Gamma( 'iLt', alpha=1, beta=1e2 ) # THIS WORKED WELL
    #iLt = pyhm.Gamma( 'iLt', alpha=1, beta=1e1 ) # THIS WAS NOT IDEAL
    #iLt = pyhm.Gamma( 'iLt', alpha=1, beta=1e0 ) # THIS DID **NOT** WORK 
    #iLt = pyhm.Gamma( 'iLt', alpha=1, beta=1e-4 ) # THIS DID **NOT** WORK

    tgrad = pyhm.Uniform( 'tgrad', lower=-1000, upper=1000, value=0.0 )
    if which_type=='white':
        parents = { 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, 'beta':beta, 'foot':foot, \
                    'Amp':Amp, 'iLhstphase':iLhstphase, 'iLt':iLt, 'tgrad':tgrad, 'Tmid_lit':Tmid_lit }
    elif which_type=='spec':
        parents = { 'RpRs':RpRs, 'beta':beta, 'foot':foot, \
                    'Amp':Amp, 'iLhstphase':iLhstphase, 'iLt':iLt, 'tgrad':tgrad }

    Amp_range = pyhm.Uniform( 'Amp', lower=0.00005, upper=0.0001 )
    iLhstphase_range = pyhm.Uniform( 'iLhstphase', lower=1, upper=2 )
    iLt_range = pyhm.Uniform( 'iLt', lower=0.01, upper=0.5 )
    cpar_ranges = { 'Amp':Amp_range, 'iLhstphase':iLhstphase_range, 'iLt':iLt }

    @pyhm.stochastic( observed=True )
    def loglikelihood( value=flux_vals, parents=parents ):
        def logp( value, parents=parents ):
            parents['Tmid'] = parents['Tmid_lit'] + parents['delT']
            psignal, baseline, gp_updated = eval_model10or11( jd, flux_vals, syspars, parents, gp, \
                                                              sigw_approx, tv, which_type=which_type )
            logp_val = gp_updated.logp_builtin()
            return logp_val

    if which_type=='white':
        mbundle = { 'syspars':syspars, 'loglikelihood':loglikelihood, 'RpRs':RpRs, 'aRs':aRs, 'b':b, \
                    'delT':delT, 'beta':beta, 'foot':foot, 'tgrad':tgrad, 'Amp':Amp, 'iLhstphase':iLhstphase, \
                    'iLt':iLt, 'jd':jd, 'tv':tv, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, \
                    'xtrain':gp.xtrain, 'Tmid_lit':Tmid_lit }
    elif which_type=='spec':
        mbundle = { 'syspars':syspars, 'loglikelihood':loglikelihood, 'RpRs':RpRs, 'beta':beta, 'foot':foot, \
                    'tgrad':tgrad, 'Amp':Amp, 'iLhstphase':iLhstphase, 'iLt':iLt, 'xtrain':gp.xtrain, \
                    'jd':jd, 'tv':tv, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'Tmid_lit':Tmid_lit }
    else:
        pdb.set_trace()
    
    return mbundle, gp, cpar_ranges
    

def spec_model10or11( z, model ):
    
    jd = z['jd']
    flux_vals = z['flux_vals']
    #sigw_approx = 0.6*(2e-4)
    sigw_approx = z['sigw_approx']
    Tmid_lit = z['Tmid_lit']
    syspars = z['syspars']
    RpRs = z['RpRs']
    beta = z['beta']
    foot = z['foot']
    phiv = z['phiv']
    tv = z['tv']
    ndat = len( flux_vals )

    gp = gp_class.gp( which_type='full' )
    gp.xtrain = np.column_stack( [ phiv, tv ] )
    if model==10:
        gp.cfunc = kernels.sqexp_invL_ard
    elif model==11:
        gp.cfunc = kernels.matern32_invL_ard
    # These priors are important for convergence and also ensuring that
    # iLt doesn't get too big, which is a reasonable restriction:
    #Amp = pyhm.Gamma( 'Amp', alpha=1, beta=1e2, value=5e-3 )
    #iLhstphase = pyhm.Gamma( 'iLhstphase', alpha=1, beta=1e1, value=1. )
    #iLt = pyhm.Gamma( 'iLt', alpha=1, beta=1e1, value=0.1 )
    Amp = pyhm.Gamma( 'Amp', alpha=1, beta=1e-1 )
    iLhstphase = pyhm.Gamma( 'iLhstphase', alpha=1, beta=1e-4 )
    iLt = pyhm.Gamma( 'iLt', alpha=1, beta=1e-4 )
    tgrad = pyhm.Uniform( 'tgrad', lower=-1000, upper=1000, value=0.0 )

    parents = { 'RpRs':RpRs, 'beta':beta, 'foot':foot, \
                'Amp':Amp, 'iLhstphase':iLhstphase, 'iLt':iLt, 'tgrad':tgrad }
    Amp_range = pyhm.Uniform( 'Amp', lower=0.00005, upper=0.0001 )
    iLhstphase_range = pyhm.Uniform( 'iLhstphase', lower=1, upper=2 )
    iLt_range = pyhm.Uniform( 'iLt', lower=0.01, upper=0.5 )
    cpar_ranges = { 'Amp':Amp_range, 'iLhstphase':iLhstphase_range, 'iLt':iLt }

    #plt.figure()
    #plt.plot( jd, flux_vals, '.k' )
    #pdb.set_trace()

    @pyhm.stochastic( observed=True )
    def loglikelihood( value=flux_vals, parents=parents ):
        def logp( value, parents=parents ):
            psignal, baseline, gp_updated = eval_spec_model10or11( jd, flux_vals, syspars, parents, gp, \
                                                                   Tmid_lit, sigw_approx, tv )
            logp_val = gp_updated.logp_builtin()
            return logp_val

    mbundle = { 'syspars':syspars, 'loglikelihood':loglikelihood, 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT, \
                'beta':beta, 'foot':foot, 'tgrad':tgrad, 'Amp':Amp, 'iLhstphase':iLhstphase, 'iLt':iLt, \
                'jd':jd, 'tv':tv, 'flux_vals':flux_vals, 'sigw_approx':sigw_approx, 'xtrain':gp.xtrain, \
                'Tmid_lit':Tmid_lit }
    
    return mbundle, gp, cpar_ranges
    


def logp_mvnormal_whitenoise( r, u, n  ):
    """
    Log likelihood of a multivariate normal distribution
    with diagonal covariance matrix.
    """
    term1 = -np.sum( numexpr.evaluate( 'log( u )' ) )
    term2 = -0.5*np.sum( numexpr.evaluate( '( r/u )**2.' ) )
    return term1 + term2 - 0.5*n*np.log( 2*np.pi )


def eval_model1( jd, tv, flux_vals, syspars, parsfit, which_type='white' ):
    syspars['RpRs'] = parsfit['RpRs']
    if which_type=='white':
        syspars['aRs'] = parsfit['aRs']
        syspars['b'] = parsfit['b']
        syspars['incl'] = np.rad2deg( np.arccos( parsfit['b']/parsfit['aRs'] ) )
        syspars['T0'] = parsfit['Tmid_lit'] + parsfit['delT']
    psignal = transit.ma02_aRs( jd, **syspars )
    baseline = parsfit['foot'] + parsfit['tgrad']*tv

    return psignal, baseline



def eval_model2or3( jd, flux_vals, syspars, parsfit, gp, Tmid_lit, sigw_approx ):
    syspars['RpRs'] = parsfit['RpRs']
    syspars['aRs'] = parsfit['aRs']
    syspars['b'] = parsfit['b']
    syspars['incl'] = np.rad2deg( np.arccos( parsfit['b']/parsfit['aRs'] ) )
    syspars['T0'] = Tmid_lit + parsfit['delT']
    psignal = transit.ma02_aRs( jd, **syspars )
    baseline = parsfit['foot']
    resids_arr = flux_vals - psignal*baseline
    gp.dtrain = np.reshape( resids_arr, [ resids_arr.size, 1 ] )
    gp.etrain = sigw_approx*parsfit['beta']
    iLscales = np.array( [ parsfit['iLhstphase'], parsfit['iLt'] ] )
    gp.cpars = { 'amp':parsfit['Amp'], 'iscale':iLscales }
    return psignal, baseline, gp

def eval_model4or5( jd, flux_vals, syspars, parsfit, gp, Tmid_lit, sigw_approx, tv ):
    syspars['RpRs'] = parsfit['RpRs']
    syspars['aRs'] = parsfit['aRs']
    syspars['b'] = parsfit['b']
    syspars['incl'] = np.rad2deg( np.arccos( parsfit['b']/parsfit['aRs'] ) )
    syspars['T0'] = Tmid_lit + parsfit['delT']
    psignal = transit.ma02_aRs( jd, **syspars )
    baseline = parsfit['foot'] + parsfit['tgrad']*tv
    resids_arr = flux_vals - psignal*baseline
    gp.dtrain = np.reshape( resids_arr, [ resids_arr.size, 1 ] )
    gp.etrain = sigw_approx*parsfit['beta']
    gp.cpars = { 'amp':parsfit['Amp'], 'iscale':parsfit['iLhstphase'] }
    return psignal, baseline, gp

def eval_model6or7( jd, flux_vals, syspars, parsfit, gp, Tmid_lit, sigw_approx ):
    syspars['RpRs'] = parsfit['RpRs']
    syspars['aRs'] = parsfit['aRs']
    syspars['b'] = parsfit['b']
    syspars['incl'] = np.rad2deg( np.arccos( parsfit['b']/parsfit['aRs'] ) )
    syspars['T0'] = Tmid_lit + parsfit['delT']
    psignal = transit.ma02_aRs( jd, **syspars )
    baseline = parsfit['foot']
    resids_arr = flux_vals - psignal*baseline
    gp.dtrain = np.reshape( resids_arr, [ resids_arr.size, 1 ] )
    gp.etrain = sigw_approx*parsfit['beta']
    iLscales = np.array( [ parsfit['iLhstphase'], parsfit['iLt'], parsfit['iLwavshift'] ] )
    gp.cpars = { 'amp':parsfit['Amp'], 'iscale':iLscales }
    return psignal, baseline, gp

def eval_model8or9( jd, flux_vals, syspars, parsfit, gp, Tmid_lit, sigw_approx, tv ):
    syspars['RpRs'] = parsfit['RpRs']
    syspars['aRs'] = parsfit['aRs']
    syspars['b'] = parsfit['b']
    syspars['incl'] = np.rad2deg( np.arccos( parsfit['b']/parsfit['aRs'] ) )
    syspars['T0'] = Tmid_lit + parsfit['delT']
    psignal = transit.ma02_aRs( jd, **syspars )
    baseline = parsfit['foot'] + parsfit['tgrad']*tv
    resids_arr = flux_vals - psignal*baseline
    gp.dtrain = np.reshape( resids_arr, [ resids_arr.size, 1 ] )
    gp.etrain = sigw_approx*parsfit['beta']
    iLscales = np.array( [ parsfit['iLhstphase'], parsfit['iLwavshift'] ] )
    gp.cpars = { 'amp':parsfit['Amp'], 'iscale':iLscales }
    return psignal, baseline, gp

def eval_model10or11( jd, flux_vals, syspars, parsfit, gp, sigw_approx, tv, which_type='white' ):
    syspars['RpRs'] = parsfit['RpRs']
    if which_type=='white':
        syspars['aRs'] = parsfit['aRs']
        syspars['b'] = parsfit['b']
        syspars['incl'] = np.rad2deg( np.arccos( parsfit['b']/parsfit['aRs'] ) )
        syspars['T0'] = parsfit['Tmid']
    elif which_type=='spec':
        pass
    else:
        pdb.set_trace() # something wrong
    psignal = transit.ma02_aRs( jd, **syspars )
    baseline = parsfit['foot'] + parsfit['tgrad']*tv
    resids_arr = flux_vals - psignal*baseline
    gp.dtrain = np.reshape( resids_arr, [ resids_arr.size, 1 ] )
    gp.etrain = sigw_approx*parsfit['beta'] 
    iLscales = np.array( [ parsfit['iLhstphase'], parsfit['iLt'] ] )
    gp.cpars = { 'amp':parsfit['Amp'], 'iscale':iLscales }
    return psignal, baseline, gp


