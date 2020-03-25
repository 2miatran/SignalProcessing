import h5py
import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.io
from scipy import sparse as sp
from scipy import signal
#from termcolor import colored

"--------------------------------'EXTRACTING DATA FROM HDF5 FORMAT'--------------------------------"
def single_extractOSAdata(file, path_to_outfile, patient, signal_list):
    '''
    Extract data from h5py file.
    Extract the specific channel name instead of using fixed indexes as in theory it can happen that some channels
    will be missing.
    Return: npz file containing oldecg (ECGs from left/right arm), ecg(ECG from leg), oxy(So2), and other info
    ----------------------------------------------
    Parameters:
    file: name of the .mat file
    patient: name of the patient
    '''

    with h5py.File(file, 'r') as f:

        channels = f['ch']

        s = {}
        for i in range(channels['data'].shape[0]):
            label = f[channels['label'][i, 0]][()].tobytes()[::2].decode()

            if label in signal_list:
                s[label] = f[channels['data'][i, 0]][()]  # .value


        s['ECG'] = s[signal_list[0]] - s[signal_list[1]]


        np.savez_compressed(path_to_outfile + patient,
                            flags=np.stack([s[label] for label in signal_list[5:]])[:, :, 0],  # stacking all other signals
                            ecg=s['ECG'],  # from leg



"--------------------------------'DATA VISUALIZATION TOOL '--------------------------------"
def draw_hist(path_to_folder, fs = 200,
              cutoff = None, order = 5, 
              transform_normalize = False, transform_scale = False):
    
    '''
    Draw histogram distribution of the ECGs data. 
    Applied normalization or lowpass filter befor drawing if needed
    WARNING: This takes long time!!! Could be easily modified to draw a few instead of all.
    ''' 
    import os
    import seaborn as sns
    from sklearn.preprocessing import normalize, scale
  
    f = plt.figure(1, figsize=(16, 3))

    for filename in os.listdir(path_to_folder):
        #print(filename)
        file = path_to_folder+filename
        m = np.load(file)
        ecg = m.f.ecg[:]

        if transform_normalize: ecg = normalize(ecg, axis = 0)
        if transform_scale: ecg = scale(ecg, axis=0)
        
        if cutoff: ecg = butter_lowpass_filter(ecg, cutoff, fs, order=5, plot=False) #plot = True to compare lowpass filter
        
        sns.distplot(ecg, hist = True, kde = True, kde_kws = {'linewidth': 1})



"--------------------------------'DATA PROCESSING TOOLS'--------------------------------"
from scipy.signal import butter, lfilter, freqz,filtfilt

def butter_lowpass(cutoff, fs, order=5):

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5, plot = False):
    '''

    :param data: input data
    :param cutoff: float, highpass value. recommended of 50
    :param fs: sampling frequency
    :param order: 5
    :param plot: if True, data before and after are ploted
    :return: same dimension signals
    '''
    y = np.transpose(data)
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, y)
    y = np.transpose(y)
    plt.figure(1, figsize=(16, 5))
    plt.grid=True
    if plot:
        plt.plot(data, label='Before filter')
        plt.plot(y, label = 'After filter')
        plt.legend(loc='upper left')
        plt.title('Applying lowpass filter')
    
    return y


from scipy.signal import butter, lfilter, freqz, filtfilt, sosfilt, sosfreqz, sosfiltfilt

def butter_bandpass(lowcut, highcut, fs, order=5, mode='ba'):
    '''

    :param lowcut: lowpass
    :param highcut: highpass
    :param fs: sampling frequency of data/signals
    :param order: 5
    :param mode: norm or ba
    :return: same dimension signals
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if mode == 'ba':
        return butter(order, [low, high], btype='band', analog=False)
    if mode == 'sos':
        return butter(order, [low, high], analog=False, btype='band', output='sos')


def ButterBandpassFilter(data, lowcut, highcut, fs, order=5, mode='ba', axis=0, plot=False, pdf=None, name='sample'):

    '''
    apply bandpass filtering along the specified dimension. Currently only 2D input
    :param data: signals, size 2D
    :param lowcut: lowpass
    :param highcut: highpass frequency
    :param fs: sampling frequency
    :param order:
    :param mode: norm / ba
    :param axis: the axis along the dimension to be filtered.
    :param plot: if plot, plot data before / after
    :param pdf: #silly things to extract all the graph, not needed
    :param name: #go with pdf option to name graph
    :return:2D output
    '''
    assert len(data.shape) > axis, 'Dimension is greater than input shape'
    assert data.shape[axis] > 1, 'Filter dimension should be greater than 1. '

    if len(data.shape) == 3:
        assert axis == 2, 'Filter only apply to signal dimension. Specify [bz, ch, signal] for input and axis=2 instead'
        bz, ch, s = data.shape
        axis = 2

    elif len(data.shape) == 2:
        s = data.shape[-1]

    elif len(data.shape) == 1:
        s = 1
        # assert s>1, 'Signal dimension should be greater than 1. Specify shape as (sig,0) instead! '

    if axis == 0:  # axis==1 mean no tranpose, reshape needed
        y = np.transpose(data.reshape(-1, s))  # [bs, sig]
    else:
        y = data

    #print(y.shape)

    if mode == 'ba':
        b, a = butter_bandpass(lowcut, highcut, fs, order, mode)  # apply along the last dimension
        y = lfilter(b, a, y)
    if mode == 'sos':
        sos = butter_bandpass(lowcut, highcut, fs, order, mode)
        y = sosfiltfilt(sos, y)  # keep phase

    if axis == 0:
        if len(data.shape) == 3:
            y = np.transpose(y).reshape(-1, ch, s)
        elif len(data.shape) == 2:
            y = np.transpose(y)
        elif len(data.shape) == 1:
            y = y.flatten()

    plt.figure(1, figsize=(30, 5))
    plt.grid = True
    if plot:
        plt.plot(data.flatten(), label='Before filter')
        plt.plot(y.flatten(), label='After filter')
        plt.legend(loc='upper left')
        plt.title('Applying lowpass filter')
        # plt.show()
    if pdf:
        pdf.savefig(fig)

    plt.show()

    return y

def Transformer(data, mode='normal', axis=0):
    '''

    :param data: 2D input signals
    :param mode: 'normal', 'standard', 'minmax','robust'
    :param axis: axis along which the transformation is performed.
    For ex: for input: (bs, signals voltage), axis should be 0
            for input: (bs, features), axis should be 1
    :return: return array of same dimension
    '''

    from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, RobustScaler
    assert (len(data.shape) ==2), 'transformer only support 2D. Use np.apply_along_axis to apply to high dimension input'
    assert mode in ['normal', 'standard', 'minmax','robust'], "Invalid mode!"
    s = data.shape[1]

    if axis == 0:  # axis==1 mean no tranpose, reshape needed
        r_data = np.transpose(data.reshape(-1, s))
    else:
        r_data = data

    if mode == 'normal':
        transform = Normalizer()

    if mode == 'standard':
        transform = StandardScaler()

    if mode == 'minmax':
        transform = MinMaxScaler()

    if mode == 'robust':
        transform = RobustScaler()
    #print(transform.mean)
    t_data = transform.fit_transform(r_data)

    if axis == 0:
            t_data = np.transpose(t_data)
    return t_data

def data_segmenting(data, split_dur = 1000, fs = 200, overlap = None, axis = 0):
    '''

    :param data: 2D input
    :param split_dur: time to split in second
    :param fs: sampling frequency
    :param overlap: overlapped time in second
    :param axis: the axis to segment. For ex:
    input of (bs, signal_length)=> axis = 1
    input of (signal_length, bs) => axis = 0
    :return:
    '''
    
    total_length = data.shape[axis]
    split_length = split_dur * fs
    if overlap: overlap_length = overlap * fs
    else: overlap_length= 0
    
    assert overlap<split_dur, "Overlap is equal or greater than segment length"
    assert total_length>=split_length, "Segment length is equal or greater than total length"    
    
    num_seq = int((total_length-split_length)/(split_length-overlap_length))

    step = [(split_length-overlap_length)*i for i in range(num_seq+1)]

    if axis ==0:
        out = np.array([data[i:i+split_length] for i in step])

    if axis == 1:
        out = np.array([data[:,i:i+split_length] for i in step])
        out = out.reshape(-1, split_length)
    
    return out
    
class DataGenerator():
    
    def __init__(self, path_to_infile, path_to_outfile = 'ECG/Dataset/', suffix = '',
                  event = 'any_event', test_ratio = 0.4, valid_ratio = 0.1,
                  split_dur = 60, overlap = 30, 
                 transform_normalize = None, transform_scale = None,
                 fs = [500, 50], fs_out = [200,1], cutoff = 60):
                 
        self.path_to_infile = path_to_infile
        self.path_to_outfile = path_to_outfile
        
        'For splitting to train/test/valid'
        self.event = event
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        
        'For segmenting data'
        self.split_dur = split_dur
        self.overlap = overlap
        self.suffix = suffix
        
        'For signal processing'
        self.transform_normalize = transform_normalize
        self.transform_scale = transform_scale
        self.cutoff = cutoff
        self.ecg_fs, self.lb_fs = fs
        self.ecg_fs_out, self.lb_fs_out = fs_out 
        
        '-----"Main" part to generate data-------'
        train, valid, test = self.get_split_id()
        self.train = self.__data_generation(train, 'train')
        self.valid = self.__data_generation(valid, 'valid')
        self.test = self.__data_generation(test, 'test')
        
   
    def get_split_id(self):
        
        from random import shuffle, seed
        import os 
        seed(3)
        
        'Extract and shuffle index'
        record_list = [record for record in os.listdir(self.path_to_infile)]
        shuffle(record_list)

        dataSize = len(record_list)              ####################   CHANGE THIS LINE :) !!!!  ######################
        
        nTest = int(self.test_ratio * dataSize)
        nValid = int(self.valid_ratio * dataSize)
        
        testlist= record_list[:nTest]
        validlist = record_list[nTest:nTest+nValid]
        trainlist = record_list[nTest+nValid:]
        
        return trainlist, validlist, testlist
    
    def signal_processing(self, ecg, label):
        
        'Apply lowpass filter'
        if self.cutoff: ecg = butter_lowpass_filter(ecg, self.cutoff, self.ecg_fs, order=5) #plot = True to compare lowpass filter
        
        'Down sample ECG data'
        ecg, _ = resampling(ecg, self.ecg_fs, self.ecg_fs_out, axis = 0, mode = 'downfft')
        
        'Normalie / standardize'
        if self.transform_normalize: ecg = normalize(ecg, axis = 0)
        if self.transform_scale: ecg = scale(ecg, axis=0)
        
        'Resample label data'
        if self.lb_fs_out>= self.lb_fs:    
            label, _ = resampling(label, self.lb_fs, self.lb_fs_out, axis = 0, mode = 'up')
        else:
            label,_ = resampling(label, self.lb_fs, self.lb_fs_out, axis = 0, mode = 'down')

        return ecg, label
    
    
    def __data_generation(self, id_list, dataname = 'train'):
        
        #print('DATA PROCESSING AND SEGMENTING ...!')
        LABEL_DICT = {'psg1':0, 'hyp':1, 'csa':2, 'msa':3, 'osa':4, 'any_event':5}
        i = 0
        #for fileset, dataname in zip([validlist, testlist, trainlist], ['valid', 'test', 'train']):

        print(colored(f'------------Processing data into {dataname} set -------------', 'green'))
        ecglist, labelist = [], []

        for file in id_list:

            'loading ECGs and label data'
            m = np.load(self.path_to_infile + file)
            ecg, label = m.f.ecg[:,0], m.f.flags[LABEL_DICT[self.event]]
            
            'signal processing part'
            ecg, label = self.signal_processing(ecg, label)

            'segmentation part'
            ecg = data_segmenting(ecg, split_dur = self.split_dur, fs = self.ecg_fs_out, overlap = self.overlap, axis = 0)
            label = data_segmenting(label, split_dur= self.split_dur, fs=self.lb_fs_out, overlap= self.overlap, axis = 0)
            #print(ecg.shape, label.shape)

            ecglist.append(ecg)
            labelist.append(label)
            
            i = i+1
            if i%20==0: print(f'finished {i} samples')
        ecgdata = np.vstack(ecglist)
        labeldata = np.vstack(labelist)
        
        print(f'{dataname} set contain ecg {ecgdata.shape} & label {labeldata.shape}')
        
        return ecgdata, labeldata


def Resampling(sig, in_fs=50, out_fs=1, plot=False, axis=0, mode='down'):
    '''
    Resampling signals with 3 modes: down (slicing), up (repeat), downfft ()
    if input is an array of size (n,) => axis= 0 (1 otherwise) (app: ecg vs all other channels)
    Plotting to compare before vs after sampling

    Accept any 2d input, dimension to resample could be 0 or 1
    For 3d: dimension should be specified as [bz, #channel, signal], dimension to resample could be 0, 1, 2
    ------------------
    Parameters:

    ------------------
    Example:
    file = 'rawdata/F30_01.npz'
    m = np.load(file)
    flags = m.f.flags[3:5,int(990000*50/500):int(1050000*5/50)]
    flags = np.expand_dims(flags, axis=1)
    print(flags.shape) #(2, 1, 6000)
    resample_flags= resampling(flags, in_fs=50, out_fs=1, axis = 2, mode = 'down', plot= True)


    ecg = m.f.ecg[990000:993000]
    ecg = np.expand_dims(ecg, axis=1)
    ecg = ecg.reshape(1, 1, 3000) #ecg.shape = (1,1,3000)
    resample_ecg= resampling(ecg, in_fs=500, out_fs=200, mode = 'downfft', plot= True)

    '''

    assert mode in ['down', 'up', 'downfft'], f'Mode is not in list. Choosing from ["down", "up", "downfft"]!'
    assert len(sig.shape)<=3, 'Input dimension greater than 3 are not supported'
    if in_fs == out_fs:
        return sig

    if len(sig.shape) == 3:

        if axis != 2:
            print('WARNING: Resampling currently supports dimension 2 for 3D array! Specify input as [bz, #channel, signal_to_resampled]')
        axis = 2
        bs, ch, _ = sig.shape
        assert sig.shape[2] > 1, 'Resampled dimension 2 of input array should be greater than 1!'

    if axis >= 1:

        sig = np.transpose(sig)  # from (2, 1, 6000) to (6000, 2)
        n = sig.shape[-1]  # for plotting

    else:
        n = 1

    time = np.arange(sig.shape[0]).astype('float64')



    ratio = (in_fs / out_fs)

    new_length = int(sig.shape[0] / ratio)  # get new time domain
    resampled_time = np.linspace(0, max(time), new_length)

    if mode == 'down':
        assert (in_fs >= out_fs), 'In_fs should be greater than out_fs for downsampling!'
        if in_fs>1:
            assert ((in_fs % out_fs) == 0), 'Downsampling by splicing currently support when (in_fs%out_fs)==0, \
choose downfft mode instead!'
        resampled_signal = sig[::int(ratio)]

    if mode == 'up':
        assert (in_fs <= out_fs), 'In_fs should be smaller than out_fs for upsampling!'
        resampled_signal = sig.repeat(int(1 / ratio), axis=0)

    if mode == 'downfft':
        assert (in_fs >= out_fs), 'In_fs should be greater than out_fs for downsampling!'
        # print(sig.shape)
        resampled_signal, resampled_time = signal.resample(sig, axis=0, num=new_length, t=time)  # resampling

    if plot:

        fig = plt.figure(figsize=(20, n * 4))
        fig.suptitle(f'Resampling Segment of {sig.shape[0] / in_fs} seconds, from {in_fs} Hz to {out_fs} Hz',
                     color='blue', fontsize=18)
        if axis == 2:
            sig = sig.reshape(-1, sig.shape[2])
            print(sig.shape)
            resampled_signal = resampled_signal.reshape(-1, resampled_signal.shape[2])
            print(resampled_signal.shape)

        for i in range(n):
            ax = fig.add_subplot(n, 1, i + 1)

            if axis >= 1:
                ax.plot(time, sig[:, i], 'black', resampled_time, resampled_signal[:, i], 'red')
            else:
                ax.plot(time, sig, 'black', resampled_time, resampled_signal, 'red')

    if axis == 2:
        resampled_signal = (np.transpose(resampled_signal)).reshape(bs, ch, new_length)
        # print(resampled_signal.shape)
    elif axis == 1:
        resampled_signal = np.transpose(resampled_signal)

    return resampled_signal

def LoadData(sampleName, dataPath='training2017/', plot = True):
    """
    Loads data from MatLab file for given example
    :return: features for given example
    
    """
    import scipy.io as sio
    import matplotlib.pyplot as plt
    sample = sio.loadmat(dataPath + sampleName + '.mat')
    signals = sample['val'][0]
    print(f'ECGs from file {sampleName}, length of {len(signals)}')
    return signals

def FindPeakPropteries(x, dist=75, plot=False):
    from scipy.signal import find_peaks
    # print(x.shape)
    peaks, properties = find_peaks(x, distance=dist, prominence=1)
    height = (properties['prominences'])  # get the prominence propteries only

    if plot:
        plt.figure(figsize=(20, 5))
        plt.plot(x)
        plt.plot(peaks, x[peaks], "x")
        plt.show()
    return (len(x[peaks]), height.mean())
    
    
def artifactrejection(data, label=None, seg_axis=0, plot=False, mode='whole', zerocross = True,
                      mpercentthreshold=1 / (200 * 20), maxthreshold=1000, minthreshold=-700,
                      applyfs=200, pdf=None, name='sample'):
    '''
    mode: seg for segment-based mean and variance, whole for whole patient mean and variance, pop for population (enter your mean, std)
    Input: 2D
    ----------
    Ex:
    file = 'FilesForDL/F30_01.npz'
    m = np.load(file)
    ecg = m.f.ecg[:20000].reshape(2,10000)
    artifactrejection(ecg, seg_axis=1)
    '''
    import scipy

    if maxthreshold & minthreshold:
        from data_processing import transformer


        sum_outlier1 = ((data > maxthreshold) | (data < minthreshold)).sum(seg_axis)  # (seg_axis)
        percent_outlier1 = (sum_outlier1 / (data.shape[seg_axis]))
        criterion1 = percent_outlier1 < mpercentthreshold  # total percent of outliers smaller than threshold
        # print(np.where(data<minthreshold), percent_outlier1, mpercentthreshold, data.shape[seg_axis])
        # print('cri1', criterion1)
        'keep indice'
        indice = np.where(criterion1 == True)[0]
        # print('keep indice',indice)

    tdata = data[criterion1]

    mean, std = np.mean(tdata), np.std(tdata)
    zdata = (tdata - mean) / std
    var = np.std(zdata, axis=seg_axis)
    # print('var', var)
    'get var std'
    # varvar, varmean  = np.std(var), np.mean(var)
    # zvar = (var - varmean)/varvar
    # print('get var std', zvar, varmean, varvar)
    # criterion3 = (var>0.90)&(var<1.1)
    criterion3 = (var > 0.5) & (var < 1.1)

    zerocrossing = np.sum((np.diff(np.sign(zdata)) != 0), axis=seg_axis)
    if (zdata.shape[0] > 0) and (zerocross == True):
        peaks = np.apply_along_axis(peakpropteries, arr=zdata, axis=1)
        peakcount, peakheight = peaks.mean(axis=0)
        # weight = 10/peakheight
        n = data.shape[seg_axis]
        peakcount = peakcount/n
        weight = 1

        zerothreshold = 10 * peakcount

        criterion4 = zerocrossing < zerothreshold * weight

        indice = indice[criterion4]



    if plot:
        'set up'
        plt.figure(2, figsize=(30, 10))

        totallength = np.prod((data.shape))
        x = np.linspace(0, totallength, num=(totallength / data.shape[seg_axis]) + 1)

        pdata = np.zeros((data.shape[0], data.shape[1]))
        pdata[indice] = data[indice]

        'plot real data'
        orig_time = np.arange(0, totallength)
        new_time = resampling(orig_time, in_fs=200, out_fs=applyfs)

        plt.plot(orig_time, data.flatten(), label='Ecg', color="#FEC8D8", alpha=0.5)
        plt.plot(new_time, pdata.flatten(), label='Jump Difference', color="#957DAD", alpha=0.6)

        'plot reference (min, max, zscore) lines'
        plt.hlines(minthreshold, xmin=x[0], xmax=x[-1], colors='grey', linestyles='dotted', label='Min-Max')
        plt.hlines(maxthreshold, xmin=x[0], xmax=x[-1], colors='grey', linestyles='dotted')

        'Finishing set up'
        plt.ylabel('Voltage')
        plt.xlabel('Time')
        plt.legend(loc='upper left')
        plt.title(f'Artifact rejection for {name}')
        plt.show()

    #if label:
    return data[indice], label[indice]
    #else:
    #    return data[indice]


import numpy as np

def DataLookUp(path_to_file, start=0, stop=4000, loadall=False,
               in_fs=[300, 50], out_fs=[300, 200],  # apply resampling
               transform = None,  # apply transform
               cutoff=None, order=5,  # Apply lowpass filter
               plot=False, summary=False):
    import os
    import scipy.io as sio
    from sklearn.preprocessing import normalize, scale
    import numpy as np
    import matplotlib.pyplot as plt

    data = sio.loadmat(path_to_file) 
    ecg = data['val'][0]

    ecg_fs_in = in_fs[0]
    ecg_fs_out = out_fs[0]

    'Summary data when needed'
    if summary:
        print(f'Total length of signal {len(ecg)} which is {len(ecg) / (ecg_fs_in * 60)} minutes')

    if loadall:
        start, stop = 0, len(ecg)
    #print(start, stop)

    'Parse and process ECGs'
   
    ecg = ecg[start:stop]
    
    # resampling
    ecg = Resampling(ecg, ecg_fs_in, ecg_fs_out, mode='downfft', axis=0, plot=False)

    if transform:
        ecg = Transformer(ecg[:, np.newaxis], mode=transform, axis=1)

    if cutoff:
        ecg = ButterBandpassFilter(ecg, cutoff[0], cutoff[1], ecg_fs_out, order=5,
                                     mode='sos')  # plot = True to compare lowpass filter


    if plot:

        import datetime
        import matplotlib.ticker as ticker

        fig, ax = plt.subplots(figsize= (20,5))
        ax.plot(ecg)
        ax.set_title(path_to_file, fontsize=20, color = 'blue')
        ax.set_xlabel(' Time (Hour:Minute:Second)', color = 'green', fontsize = 20)

        
        xticks = np.linspace(int(start), int(stop), 5)
        ticklabels = [f'{datetime.timedelta(seconds=x)}' for x in xticks/ecg_fs_in]
        plt.xticks(xticks*ecg_fs_out/ecg_fs_in, ticklabels)
        
        #ax.set_xlim([0, max(xticks)])
        
        plt.tight_layout()