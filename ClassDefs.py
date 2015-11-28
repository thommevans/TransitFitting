import pdb, sys, os
import MultiBandRoutines


class MultiBand():
    """
    Intended for spectroscopic transit datasets...
    """

    def __init__( self ):
        # default attributes?
        self.lc_type = ''
        self.spectra = None
        self.dspec = None
        self.enoise = None
        self.npix_perbin = None
        self.auxvars = {}
        self.shiftstretch = False
        self.wav_centers = None
        self.wav_edges = None
        self.spec_lc_flux = None
        self.spec_lc_uncs = None
        self.ld_quad = None
        self.ld_nonlin = None

        return None

    def prep_white_lc( self ):
        # requires hst.input_data_file to exist already, 
        # which contains variables auxvars, spectra, wavsol_micron at least.
        MultiBandRoutines.white_lc( self )
        return None

    def prep_spectra( self ):
        # uses the white mle to identify out-of-tr
        return None

    def prep_spec_lcs( self ):
        # should this just handle one channel at a time?
        MultiBandRoutines.spec_lcs( self )
        return None

    def mle( self ):
        MultiBandRoutines.mle( self )
        return None

    def mcmc( self ):
        MultiBandRoutines.mcmc( self )
        return None

    def mcmc_old( self ):
        # todo - should probably handle both white and spec with a single backend routine?
        if self.lc_type=='white':
            MultiBandRoutines.white_mcmc( self )
        else:
            pdb.set_trace()

        return None


