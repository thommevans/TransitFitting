import pdb, sys, os
import MultiBandRoutines


class MultiBand():
    """
    """

    def __init__( self ):
        # default attributes?
        return None

    def prep_white_lc( self ):
        # requires hst.input_data_file to exist already, 
        # which contains variables auxvars, spectra, wavsol_micron at least.
        MultiBandRoutines.white_lc( self ) # adapt from the lc_white_stis() and lc_white_wfc3() routines...
        return None

    def prep_spectra( self ):
        # uses the white mle to identify out-of-tr
        return None

    def prep_spec_lc( self ):
        # should this just handle one channel at a time?
        return None

    def mle( self ):
        # todo - should handle white and spec
        if self.lc_type=='white':
            MultiBandRoutines.white_mle( self )
        else:
            pdb.set_trace()
        return None

    def mcmc( self ):
        # todo - should handle white and spec
        return None


# then define the same kind of class for WFC3...
