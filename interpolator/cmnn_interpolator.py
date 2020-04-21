import os
import numpy as np
import argparse
import pickle

def return_interp_stats(user_maglims,user_redshift,verbose=False):
    ### Input values: 
    ###   user_maglims   = array of 5-sigma magnitude limits, ugrizy
    ###   user_redshift  = redshift
    ###   verbose        = boolean, if True additional outputs are printed
    ### Return values:
    ###   zbin           = the bin that this user-supplied redshift falls into
    ###   stats          = a 5-element array of interpolated statistics
    ###   verify_pass    = boolean, True if inputs are acceptable, else False

    ### Print inputs to screen
    if verbose:
        print('')
        print('Inputs')
        print('User-supplied magnitude limits in u g r i z y: ',user_maglims)
        print('User-supplied redshift: ',user_redshift)

    ### Define the acceptable ranges for magnitude limits and redshift
    filters = ['u','g','r','i','z','y']
    magmins = [23.90, 25.00, 24.70, 24.00, 23.30, 22.10]
    magmaxs = [26.78, 27.88, 27.84, 27.15, 26.37, 25.17]
    zbinmin = 0.30
    zbinmax = 3.00

    ### Verify that the input values are within an acceptable range
    verify_pass = True
    if verbose:
        print(' ')
    for m in range(6):
        if (user_maglims[m] < magmins[m]) | (user_maglims[m] > magmaxs[m]):
            verify_pass = False
            if verbose:
                print('BAD INPUT: ',filters[m],' magnitude limit ',user_maglims[m],' out of range (',magmins[m],' to ',magmaxs[m],')')
    if (user_redshift < zbinmin) | (user_redshift > zbinmax):
        verify_pass = False
        if verbose:
            print('BAD INPUT: Redshift ',user_redshift,' out of range (',zbinmin,' to ',zbinmax,')')
    if verbose:
        if verify_pass == True:
            print('Input values are within an acceptable range.')

    ### If user inputs are good proceed with the interpolation, else return Nan
    if verify_pass:
        ### Figure out which redshift bin (zbins) the user-supplied redshift falls into
        zfind = np.genfromtxt( 'pickled_models/zbins.dat', dtype='str', usecols={0} )
        zbins = np.loadtxt( 'pickled_models/zbins.dat', dtype='float', usecols={1} )
        zx = np.argmin( np.abs(user_redshift-zbins) )
        zbin = zbins[zx]
        ### For each of the statistics, load the pickle file for this zbin and interpolate
        stats = np.zeros( 5, dtype='float' )
        for s,statname in enumerate(['stdd','stdderr','bias','biaserr','fout']):
            fnm = 'pickled_models/'+statname+'_'+zfind[zx]+'.pkl'
            fin = open(fnm,'r+b')
            reg = pickle.load(fin)
            stats[s] = reg.predict(user_maglims.reshape(1,-1))
            fin.close()
            del fnm,fin,reg
    else:
        zbin = float('NaN')
        stats = np.zeros( 5, dtype='float' )
        for s in range(5):
            stats[s] = float('NaN')

    if verbose:
        print(' ')
        print('Outputs')
        print('Redshift bin center:',zbin)
        print('Predicted statistical measures (value, error):')
        print('standard deviation   %10.6f %10.6f ' % (stats[0],stats[1]))
        print('bias                 %10.6f %10.6f ' % (stats[2],stats[3]))
        print('fraction of outliers %10.6f ' % stats[4])

    return zbin,stats


if __name__ == '__main__':

    ### Example command line input
    ### python cmnn_interpolator.py -user_maglims 26.1 27.4 27.5 26.8 26.1 24.9 -user_redshift 0.35

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    ### Parse the input for the user-supplied magnitude limits and redshift
    parser = argparse.ArgumentParser(description='Allow for passing of arguments.')
    parser.add_argument('-user_maglims', nargs='+', action='store', dest='user_maglims', type=float,\
        help='user-provided magnitude limits u g r i z y')
    parser.add_argument('-user_redshift', action='store', dest='user_redshift', type=float,\
        help='user_provided redshift')
    args = parser.parse_args()
    user_maglims = np.asarray(args.user_maglims, dtype='float')
    user_redshift = float(args.user_redshift)

    ### Run module with verbose=True to write results to screen
    zbin,stats = return_interp_stats(user_maglims,user_redshift,verbose=True)


