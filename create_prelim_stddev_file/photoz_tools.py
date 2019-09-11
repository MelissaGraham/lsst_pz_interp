import os
import numpy as np
import lsst.sims.photUtils.Bandpass as Bandpass
import lsst.sims.photUtils.Sed as Sed

def importBandpass(filename, name):
    '''Given the name of a bandpass read it from file into Bandpass class
    assumes filters are differentiated by #NAME filter_name filter comments'''
    filter = Bandpass()
    data = []
    with open(filename, 'r') as input:
        for line in input:
            line = line.strip().split()
            if (line[0].startswith("#NAME") and (line[1] == name)):
                for line in input:
                    if line.startswith("#NAME"):
                        break
                    else:
                        data.append(map(float, line.split()))
    data = np.asarray(data).T
    # convert to nanometers and set limits for wavelength
    filter.setWavelenLimits(data[0].min()/10., data[0].max()/10., 1.)
    try:
        filter.setBandpass(data[0]/10., data[1])
    except IndexError:
        print "Unable to find data values for ", name
        exit()
    #resample to
    filter.resampleBandpass(wavelen_min=300.0, wavelen_max=1150.0, wavelen_step=0.1)
    return filter


def read_filter_atmos_files(in_fa_list):
    throughput_dir = '/Users/melissagraham/Science/LSST/photoz_experiments/filters/airmass/prepare_transmissions'
    throughputs = {}
    for fa in in_fa_list:
        throughputs[fa] = Bandpass()
        throughputs[fa].readThroughput(os.path.join(throughput_dir, fa+'.dat'))
    return throughputs


def read_sed(inseddir,infnm):
    sed = Sed()
    sed.readSED_flambda(os.path.join(inseddir, infnm))
    return sed


def calculate_magnitudes(infilterlist, insed, inthru):
    out_mags = {}
    for f in infilterlist:
        out_mags[f] = insed.calcMag(inthru[f])
    return out_mags


def stats(in_zSpec,in_zPhot,zbinlow=0.0,zbinhigh=3.0,bias_inliers=False):
    # Definitions of outputs. 
    #  meanz     = mean zPhot of galaxies in bin
    #  fout      = fraction of outliers (SRD calls them catastrophic)
    #  bias      = mean dz of galaxies in IQR
    #  stddev    = standard deviation in dz
    #  IQR       = interquartile range of dz
    #  IQRs      = IQR / 1.349, equivalent standard deviation

    # Bootstrap resample with replacement
    # Nmc = number of times to repeat measurement
    Nmc  = 1000

    # Need the standard deviation over the full range in order to identify outliers
    index      = np.where( (in_zPhot > 0.3) & (in_zPhot <= 3.0) )[0]
    allzSpec   = in_zSpec[index]
    allzPhot   = in_zPhot[index]
    alldz      = (allzSpec-allzPhot)
    q75, q25   = np.percentile( alldz, [75 ,25])
    sigma      = (q75-q25)/1.349
    threesigma = 3.00*sigma

    # Identify galaxies in the requested bin
    index = np.where( (in_zPhot > zbinlow) & (in_zPhot <= zbinhigh) )[0]
    zSpec = in_zSpec[index]
    zPhot = in_zPhot[index]
    del index

    # calculate the mean zPhot in the bin
    meanz = np.mean(zPhot)
    # define dz for use in all stats
    dz = (zSpec-zPhot)

    # fout
    tx = np.where( (np.fabs(dz) > 0.06) & (np.fabs(dz) > threesigma) )[0]
    fout = float(len(tx)) / float(len(dz))
    del tx

    # stddev, IQR and IQRs
    stddev = np.std(dz)
    q75, q25 = np.percentile( dz, [75 ,25])
    IQR  = (q75 - q25)
    IQRs = (q75 - q25) / 1.349
    # err_stddev, err_IQR and err_IQRs
    temp1 = np.zeros( Nmc, dtype='float')
    temp2 = np.zeros( Nmc, dtype='float')
    temp3 = np.zeros( Nmc, dtype='float')
    for i in range(Nmc):
        tx = np.random.choice(len(dz), size=len(dz), replace=True, p=None)
        temp1[i] = np.std( dz[tx])  # values of tx will be a subset of index2, not indices of index2
        tq75, tq25 = np.percentile( dz[tx] , [75 ,25])
        temp2[i] = (tq75 - tq25)
        temp3[i] = (tq75 - tq25) / 1.349
        del tx,tq75,tq25
    err_stddev = np.mean( np.fabs( stddev-temp1 ) )
    err_IQR  = np.mean( np.fabs( IQR-temp2) )
    err_IQRs = np.mean( np.fabs( IQRs-temp3) )
    del temp1,temp2,temp3

    if bias_inliers == False:
        # bias in the IQR of this bin
        tx = np.where( (dz > q25) & (dz < q75) )[0]
        if len(tx) > 0:
            tdz = dz[tx]
            del tx
            bias = np.mean(tdz)
            # err_bias
            temp = np.zeros( Nmc, dtype='float')
            for i in range(Nmc):
                tx = np.random.choice(len(tdz), size=len(tdz), replace=True, p=None)
                temp[i] = np.mean( tdz[tx])  # values of tx will be a subset of index2, not indices of index2
                del tx
            err_bias = np.mean( np.fabs( bias-temp ) )
            del temp
        else:
            bias = None
            err_bias = None

    if bias_inliers:
        # bias in the inliers
        tx = np.where( (np.fabs(dz) <= 0.06) | (np.fabs(dz) <= threesigma) )[0]
        if len(tx) > 0:
            tdz = dz[tx]
            del tx
            bias = np.mean(tdz)
            # err_bias
            temp = np.zeros( Nmc, dtype='float')
            for i in range(Nmc):
                tx = np.random.choice(len(tdz), size=len(tdz), replace=True, p=None)
                temp[i] = np.mean( tdz[tx])  # values of tx will be a subset of index2, not indices of index2
                del tx
            err_bias = np.mean( np.fabs( bias-temp ) )
            del temp
        else:
            bias = None
            err_bias = None

    return meanz, fout, bias,err_bias, stddev,err_stddev, IQR,err_IQR, IQRs,err_IQRs



def stats_d1pz(in_zSpec,in_zPhot,zbinlow=0.0,zbinhigh=3.0,bias_inliers=False):
    # Definitions of outputs. 
    #  meanz     = mean zPhot of galaxies in bin
    #  fout      = fraction of outliers (SRD calls them catastrophic)
    #  bias      = mean dz/(1+z) of galaxies in IQR
    #  stddev    = standard deviation in dz/(1+z)
    #  IQR       = interquartile range of dz/(1+z)
    #  IQRs      = IQR / 1.349, equivalent standard deviation

    # Bootstrap resample with replacement
    # Nmc = number of times to repeat measurement
    Nmc  = 1000

    # Need the standard deviation over the full range in order to identify outliers
    index      = np.where( (in_zPhot > 0.3) & (in_zPhot <= 3.0) )[0]
    allzSpec   = in_zSpec[index]
    allzPhot   = in_zPhot[index]
    alldz      = (allzSpec-allzPhot)/(1.0 + allzPhot)
    q75, q25   = np.percentile( alldz, [75 ,25])
    sigma      = (q75-q25)/1.349
    threesigma = 3.00*sigma

    # Identify galaxies in the requested bin
    index = np.where( (in_zPhot > zbinlow) & (in_zPhot <= zbinhigh) )[0]
    zSpec = in_zSpec[index]
    zPhot = in_zPhot[index]
    del index

    # calculate the mean zPhot in the bin
    meanz = np.mean(zPhot)
    # define dz for use in all stats
    dz = (zSpec-zPhot)/(1.0+zPhot)

    # fout
    tx = np.where( (np.fabs(dz) > 0.06) & (np.fabs(dz) > threesigma) )[0]
    fout = float(len(tx)) / float(len(dz))
    del tx

    # stddev, IQR and IQRs
    stddev = np.std(dz)
    q75, q25 = np.percentile( dz, [75 ,25])
    IQR  = (q75 - q25)
    IQRs = (q75 - q25) / 1.349
    # err_stddev, err_IQR and err_IQRs
    temp1 = np.zeros( Nmc, dtype='float')
    temp2 = np.zeros( Nmc, dtype='float')
    temp3 = np.zeros( Nmc, dtype='float')
    for i in range(Nmc):
        tx = np.random.choice(len(dz), size=len(dz), replace=True, p=None)
        temp1[i] = np.std( dz[tx])  # values of tx will be a subset of index2, not indices of index2
        tq75, tq25 = np.percentile( dz[tx] , [75 ,25])
        temp2[i] = (tq75 - tq25)
        temp3[i] = (tq75 - tq25) / 1.349
        del tx,tq75,tq25
    err_stddev = np.mean( np.fabs( stddev-temp1 ) )
    err_IQR  = np.mean( np.fabs( IQR-temp2) )
    err_IQRs = np.mean( np.fabs( IQRs-temp3) )
    del temp1,temp2,temp3

    if bias_inliers == False:
        # bias in the IQR of this bin
        tx = np.where( (dz > q25) & (dz < q75) )[0]
        if len(tx) > 0:
            tdz = dz[tx]
            del tx
            bias = np.mean(tdz)
            # err_bias
            temp = np.zeros( Nmc, dtype='float')
            for i in range(Nmc):
                tx = np.random.choice(len(tdz), size=len(tdz), replace=True, p=None)
                temp[i] = np.mean( tdz[tx])  # values of tx will be a subset of index2, not indices of index2
                del tx
            err_bias = np.mean( np.fabs( bias-temp ) )
            del temp
        else:
            bias = None
            err_bias = None

    if bias_inliers:
        # bias in the inliers
        tx = np.where( (np.fabs(dz) <= 0.06) | (np.fabs(dz) <= threesigma) )[0]
        if len(tx) > 0:
            tdz = dz[tx]
            del tx
            bias = np.mean(tdz)
            # err_bias
            temp = np.zeros( Nmc, dtype='float')
            for i in range(Nmc):
                tx = np.random.choice(len(tdz), size=len(tdz), replace=True, p=None)
                temp[i] = np.mean( tdz[tx])  # values of tx will be a subset of index2, not indices of index2
                del tx
            err_bias = np.mean( np.fabs( bias-temp ) )
            del temp
        else:
            bias = None
            err_bias = None

    return meanz, fout, bias,err_bias, stddev,err_stddev, IQR,err_IQR, IQRs,err_IQRs



