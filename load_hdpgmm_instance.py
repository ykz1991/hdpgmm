import numpy as np
import pickle
from model_loglikelihood import dict2mix, mixture_logpdf

segLen = 120
run0 = 0
run1 = 1
run2 = 2
directory0 = './results/2_model/no_CV_average_hyper_param/time_freq_%ds_feats_%dth_run/' % (segLen/4, (run0+4))
directory1 = './results/2_model/no_CV_average_hyper_param/time_freq_%ds_feats_%dth_run/' % (segLen/4, (run1+4))
directory2 = './results/2_model/no_CV_average_hyper_param/time_freq_%ds_feats_%dth_run/' % (segLen/4, (run2+4))

iter0 = 80
iter1 = 90
iter2 = 100
hdpgmm_un_0 = pickle.load(open(directory0 + 'hdpgmm_un_%d-th_iter' % iter0))
hdpgmm_un_1 = pickle.load(open(directory1 + 'hdpgmm_un_%d-th_iter' % iter0))
hdpgmm_un_2 = pickle.load(open(directory2 + 'hdpgmm_un_%d-th_iter' % iter0))

print 'loading finished.'
