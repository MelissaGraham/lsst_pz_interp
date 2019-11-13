import os
import numpy as np
import argparse
import pickle

def interpolate_statistics(user_maglims,user_redshift,verbose=False):
    ### This task will return:
    ###   zbins[zx] = the redshift bin that this user-supplied redshift falls into
    ###   stats     = a 5-element array of interpolated values

    ### Figure out which redshift bin (zbins) the user-supplied redshift falls into
    zbins = np.arange( 27, dtype='float' )*0.10 + 0.35
    zx = np.argmin( np.abs(user_redshift-zbins) )
    ### Write out the zbin information
    if verbose:
        print('user_redshift = ',user_redshift)
        print('zx = ',zx)
        print('zbins[zx] = ',zbins[zx])

    ### Initialize the 5-element array of interpolated statistics that this task will return
    stats = np.zeros( 5, dtype='float' )

    ### For each of the statistics, load the pickle file for this zbin and interpolate
    for s,statname in enumerate(['stddevs','stddevserr','biases','biaseserr','fouts']):
        fnm = 'pickles/'+statname+'_'+str(zx)+'.pkl'
        fin = open(fnm,'r+b')
        reg = pickle.load(fin)
        stats[s] = reg.predict(user_maglims.reshape(1,-1))
        fin.close()
        del fnm,fin,reg

    return zbins[zx],stats


if __name__ == '__main__':
    ### Example command line input
    ### python return_pz_stats.py -user_maglims 26.1 27.4 27.5 26.8 26.1 24.9 -user_redshift 0.35

    ### Parse the input for the user-supplied magnitude limits and redshift
    parser = argparse.ArgumentParser(description='Allow for passing of arguments.')
    parser.add_argument('-user_maglims', nargs='+', action='store', dest='user_maglims', type=float, help='user-provided magnitude limits u g r i z y')
    parser.add_argument('-user_redshift', action='store', dest='user_redshift', type=float, help='user_provided redshift')
    args = parser.parse_args()
    user_maglims = np.asarray(args.user_maglims, dtype='float')
    user_redshift = float(args.user_redshift)
    print('User-supplied magnitude limits in u g r i z y: ',user_maglims)
    print('User-supplied redshift: ',user_redshift)

    ### Impose quality assessment on input values
    ### If user input is outside acceptable ranges, return a warning
    filters = ['u','g','r','i','z','y']
    magmins = [23.15, 24.25, 23.95, 23.25, 22.55, 21.35]
    magmaxs = [26.78, 27.88, 27.84, 27.15, 26.37, 25.17]
    zbinmin = 0.30
    zbinmax = 3.00
    qa = np.zeros( 7, dtype='int' )
    for m in range(6):
        if (user_maglims[m] < magmins[m]) | (user_maglims[m] > magmaxs[m]):
            qa[m] = 1
            if qa[m] == 1:
                print('Warning: ',filters[m],' magnitude limit out of range (',magmins[m],' to ',magmaxs[m],')')
    if (user_redshift < zbinmin) | (user_redshift > zbinmax):
        qa[6] = 1
        if qa[6] == 1:
            print('Warning: redshift out of range (',zbinmin,' to ',zbinmax,')')
    sumqa = np.sum(qa)
    if sumqa > 0:
        print('Fix inputs and try again.')

    ### If user input is not outside acceptable ranges, proceed
    if sumqa == 0:
        zbin,stats = interpolate_statistics(user_maglims,user_redshift)
        print('Predicted statistical measures (value, error):')
        print('standard deviation   %10.6f %10.6f ' % (stats[0],stats[1]))
        print('bias                 %10.6f %10.6f ' % (stats[2],stats[3]))
        print('fraction of outliers %10.6f ' % stats[4])


