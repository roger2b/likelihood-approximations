# Copyright (C) 2016 Steven Gratton
# adapted RdB
# test-hacking to go from EB to TE...
#
from numpy import *
import itertools
import sys
import multiprocessing
import time
import healpy
from numpy.linalg import inv, slogdet
import matplotlib
#matplotlib.use('Agg')
from matplotlib.pyplot import *
from numba import jit
import scipy.integrate
import scipy.interpolate

# for pixel-likelihood, compute CAMB C_ell at each iteration 
# sys.path.insert(0,'/home/rmvd2/.local/lib/python3.6/site-packages')
# import camb

############################################################
# parameters for computation
############################################################

do_linear       = True
use_offset_k2   = False
doEEonly        = True
only_compute_ps = False
show_plot       = False
use_roger_sims  = False

#likelihoods to be computed
dofidlike       = True
dogdlike        = True
donewlike       = True

#healpix conditions
nside = 16
npix  = 12*nside*nside

#power spectrum computations
lmin      = 2
lmax      = 10
lmin=int(sys.argv[1])
lmax=int(sys.argv[2])
lmaxalias = 47

#multiprocessing for romberg integration
no_processes = 4
int_method = 'romberg'
#int_method = 'quad'

#likelihood parameters
lminlike  = 2
lmaxlike  = 10

#choose tau max: 10-200
imin=20
imax=100
#choose step size for tau computations
istep=1
istep=int(sys.argv[3])
#Reijo and pixel window smoothing 
smooth_reijo        = False
smooth_reijo_pixwin = True

#data_set='Planck_2018'
data_set='SROLL2.0'

# read in which frequency and split combination to perform
# eg. 100fullx143full
freq1=int(sys.argv[4])
split1=str(sys.argv[5])
freq2=int(sys.argv[6])
split2=str(sys.argv[7])

##########################################################
if do_linear:
    linear_approx = '_lin'
else:
    linear_approx = ''

if doEEonly:
    pol_approx = '_EE'
else:
    pol_approx = '_TEEE'


usegeneralfitsdata_roger=True
dosync_roger        = True
doclean_cross_roger = True
compsepnoiseboost   = True
testdestroymonodi   = True
usesims_roger       = False
reducemeanfac       = False
addallhighlpower    = True
if use_roger_sims:
    # no dust & sync cleaning
    usegeneralfitsdata_roger=False
    dosync_roger        = False
    doclean_cross_roger = False
    compsepnoiseboost   = True
    testdestroymonodi   = True
    usesims_roger       = True
    reducemeanfac       = True
    addallhighlpower    = True
    data_set            = 'SIMS'

############################################################
# Files for computations
############################################################

basedir        = '/home/rmvd2/cmb/glass/'
ncm_dir        = '/rds/user/rmvd2/hpc-work/cmb/cov_matrix/'
tau_dir        = '/rds/user/rmvd2/hpc-work/cmb/cov_matrix/tau_scan/'
reijo_dir      = '/rds/user/rmvd2/hpc-work/cmb/reijonoise/'
planck2018_dir = '/rds/user/rmvd2/hpc-work/cmb/Planck_2018/frequency_maps/'
sroll20_dir    = '/rds/user/rmvd2/hpc-work/cmb/SRoll2.0/frequency_maps/'

tmaskname      = basedir + 'tmaskfrom0p70.dat'
qumaskname     = basedir + 'mask_pol_nside16.dat'

reijo_mat_100_TT = basedir + 'reijo_100GHz_TT_ring.dat'
reijo_mat_143_TT = basedir + 'reijo_143GHz_TT_ring.dat'
reijo_mat_100_PP = basedir + 'reijo_100GHz_PP_ring.dat'
reijo_mat_143_PP = basedir + 'reijo_143GHz_PP_ring.dat'

#number of modes 
eigfilename=basedir+'sphharmTE_lmax47_md.dat'
# fiducial tau file
tau_fid_file = tau_dir + 'tau_scan_0_060_lensedCls.dat'


#insert smooth template here
'''
smooth_template_100ds1_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits
smooth_template_100ds2_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits
smooth_template_100full_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits
smooth_template_100hm1_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits
smooth_template_100hm2_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits
smooth_template_143ds1_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits
smooth_template_143ds2_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits
smooth_template_143full_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits
smooth_template_143hm1_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits
smooth_template_143hm2_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits
'''


if data_set=='Planck_2018':
    #Planck 2018 data
    inname30_1  = planck2018_dir+'30hm1_LFI_FFP10_16_pixwin.fits'
    inname30_2  = planck2018_dir+'30hm2_LFI_FFP10_16_pixwin.fits'
    inname100   = planck2018_dir+'100full_FFP10_16R_pixwin_new.fits'
    inname143   = planck2018_dir+'143full_FFP10_16R_pixwin_new.fits'
    inname353_1 = planck2018_dir+'353hm1_FFP10_16R_pixwin_new.fits'
    inname353_2 = planck2018_dir+'353hm2_FFP10_16R_pixwin_new.fits'
    smooth_mean_100 = reijo_dir+'smooth_template_100full_FFP10sim_16R_pixwin_300_sims_lmax4.fits'
    smooth_mean_143 = reijo_dir+'smooth_template_143full_FFP10sim_16R_pixwin_300_sims_lmax4.fits'
# use Stevens ols NCM
    #ncm_11_file = basedir + 'joint_noise_100x100_f32_EB_lmax4_projected.dat'
    #ncm_22_file = basedir + 'joint_noise_143x143_f32_EB_lmax4_projected.dat'
    # my NCM
    ncm_11_file = ncm_dir + 'noise_FFP10_100fullx100full_EB_lmax4_pixwin_300sims_smoothmean_AC.dat'
    ncm_22_file = ncm_dir + 'noise_FFP10_143fullx143full_EB_lmax4_pixwin_300sims_smoothmean_AC.dat'
elif data_set=='SROLL2.0':
    #SROLL2.0 2018 data
    inname30_1  = planck2018_dir+'30hm1_LFI_FFP10_16_pixwin.fits'
    inname30_2  = planck2018_dir+'30hm2_LFI_FFP10_16_pixwin.fits'
    inname353_1 = sroll20_dir+'353hm1_SR20_16R_pixwin_new.fits'
    inname353_2 = sroll20_dir+'353hm2_SR20_16R_pixwin_new.fits'
    #inname100   = sroll20_dir+'100full_SR20_16R_pixwin_new.fits'
    #inname143   = sroll20_dir+'143full_SR20_16R_pixwin_new.fits'
    #smooth_mean_100 = reijo_dir+'smooth_template_100full_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits'
    #smooth_mean_143 = reijo_dir+'smooth_template_143full_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits'
    #ncm_11_file = ncm_dir + 'noise_SROLL20_100fullx100full_EB_lmax4_pixwin_400sims_smoothmean_AC.dat'
    #ncm_22_file = ncm_dir + 'noise_SROLL20_143fullx143full_EB_lmax4_pixwin_400sims_smoothmean_AC.dat'
    inname100   = sroll20_dir+'{}{}_SR20_16R_pixwin_new.fits'.format(freq1,split1)
    inname143   = sroll20_dir+'{}{}_SR20_16R_pixwin_new.fits'.format(freq2,split2)
    smooth_mean_100 = reijo_dir+'smooth_template_{}{}_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits'.format(freq1,split1)
    smooth_mean_143 = reijo_dir+'smooth_template_{}{}_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits'.format(freq2,split2)
    ncm_11_file = ncm_dir + 'noise_SROLL20_{}{}x{}{}_EB_lmax4_pixwin_400sims_smoothmean_AC.dat'.format(freq1,split1,freq1,split1)
    ncm_22_file = ncm_dir + 'noise_SROLL20_{}{}x{}{}_EB_lmax4_pixwin_400sims_smoothmean_AC.dat'.format(freq2,split2,freq2,split2)
elif data_set=='SIMS':
    sims_dir    = '/rds/user/rmvd2/hpc-work/cmb/sims/new_realisations/sims/'
    #sims
    inname30_1  = planck2018_dir+'30hm1_LFI_FFP10_16_pixwin.fits'
    inname30_2  = planck2018_dir+'30hm2_LFI_FFP10_16_pixwin.fits'
    #inname100   = sims_dir+'100fullx143full_qml_0.060_sim_v4_SROLL20_new_10_sim100.fits'
    #inname143   = sims_dir+'100fullx143full_qml_0.060_sim_v4_SROLL20_new_10_sim143.fits'
    inname100   = sims_dir+'sim_noise_100_5.fits'
    inname143   = sims_dir+'sim_noise_143_5.fits'
    inname353_1 = sroll20_dir+'353hm1_SR20_16R_pixwin_new.fits'
    inname353_2 = sroll20_dir+'353hm2_SR20_16R_pixwin_new.fits'
    smooth_mean_100 = reijo_dir+'smooth_template_100full_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits'
    smooth_mean_143 = reijo_dir+'smooth_template_143full_SR20sim-FFP10sky-FFP10cmb_16R_pixwin_400_sims_lmax4.fits'
    ncm_11_file = ncm_dir + 'noise_SROLL20_100fullx100full_EB_lmax4_pixwin_400sims_smoothmean_AC.dat'
    ncm_22_file = ncm_dir + 'noise_SROLL20_143fullx143full_EB_lmax4_pixwin_400sims_smoothmean_AC.dat'


#import resource
#print (resource.getrlimit(resource.RLIMIT_STACK))
#resource.setrlimit(resource.RLIMIT_STACK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))

############################################################
# Write everything to logfile and screen
############################################################
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('log/logfile_{}_{}{}x{}{}_glass_ell_{}-{}_tau_0.{:03d}-0.{:03d}{}{}.log'.format(data_set,freq1,split1,freq2,split2,lminlike,lmaxlike,imin,imax,linear_approx,pol_approx), 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    

#sys.stdout = Logger()

############################################################
# PRINT SETTING OF CODE
############################################################
print('==================================', flush=True)
print('          GLASS SETTINGS          ', flush=True)
print('==================================', flush=True)
print('linear approximation == {}'.format(do_linear), flush=True)
print('EE polarisation only == {}'.format(doEEonly), flush=True)
print('fiducial likelihood  == {}'.format(dofidlike), flush=True)
print('Gaussian det like    == {}'.format(dogdlike), flush=True)
print('GLASS likelihood     == {}'.format(donewlike), flush=True)
print('integration method   == {}'.format(int_method), flush=True)
print(' ', flush=True)
print('compute for: {}'.format(data_set), flush=True)
print(' ', flush=True)
print('multipole range: {} - {}'.format(lmin, lmax), flush=True)
print(' ', flush=True)
print('likelihood: ', flush=True)
print('multipole range: {} - {}'.format(lminlike, lmaxlike), flush=True)
print(' ', flush=True)
print('tau values', flush=True)
print('tau range      : 0.{:03d} - 0.{:03d}'.format(imin, imax), flush=True)
print('tau step size  : 0.{:03d}'.format(istep), flush=True)
print('==================================', flush=True)

############################################################
# Define functions
############################################################

def inpvec(fname):
    return fromfile(fname,dtype=float32).astype(float)

def inpcovmat(fname):
    dat=fromfile(fname,dtype=float32).astype(float)
    n=int(sqrt(dat.shape))
    return reshape(dat,(n,n))

def inpdoubcovmat(fname):
    dat=fromfile(fname,dtype=float).astype(float)
    n=int(sqrt(dat.shape))
    return reshape(dat,(n,n))

def corr(mat):
    mat2=mat.copy()
    oosd=(1./sqrt(diag(mat))).copy()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat2[i,j]=mat[i,j]*oosd[i]*oosd[j]
    return mat2

def unmask(x):
    y=zeros(len(qumask))
    y[qumask==1] = x
    return y

def corr3(mat):
    mat2=mat.copy()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            for k in range(mat.shape[2]):
                mat2[i,j,k]=mat[i,j,k]/(mat[i,i,i]*mat[j,j,j]*mat[k,k,k])**(1./3.)
    return mat2

def corrcovnorm(mat,matcov):
    mat2=mat.copy()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            for k in range(mat.shape[2]):
                mat2[i,j,k]=mat[i,j,k]/sqrt(matcov[i,i]*matcov[j,j]*matcov[k,k])
    return mat2

tmask=inpvec(tmaskname)
polmask=inpvec(qumaskname)
qumask=concatenate((polmask,polmask))
fullmask=concatenate((tmask,polmask,polmask))

tl=int(sum(tmask))
pl=int(sum(polmask))

al100i=0.0
al100p=0.0
al143i=0.0
al143p=0.0
be100i=0.0
be100p=0.0
be143i=0.0
be143p=0.0

dosyncforsims=False
if dosyncforsims:
    be100i=0.027
    be100p=0.027
    be143i=0.0
    be143p=0.0

dosync=False
if dosync:
    be100i=0.01
    be100p=0.01
    be143i=0.0
    be143p=0.0

#roger values from intermediate planck paper
#roger values from intermediate planck paper
if dosync_roger:
    if freq1==100 and freq2==143:
        be100i=0.02
        be100p=0.02
        be143i=0.0076
        be143p=0.0076
    elif freq1==100 and freq2==100:
        be100i=0.02
        be100p=0.02
        be143i=0.02
        be143p=0.02
    elif freq1==143 and freq2==143:
        be100i=0.0076
        be100p=0.0076
        be143i=0.0076
        be143p=0.0076

#roger values (GPE)
if doclean_cross_roger:
    if freq1==100 and freq2==143:
        al100i=0.019
        al100p=0.024
        al143i=0.041
        al143p=0.038
    elif freq1==100 and freq2==100:
        al100i=0.019
        al100p=0.024
        al143i=0.019
        al143p=0.024
    elif freq1==143 and freq2==143:
        al100i=0.041
        al100p=0.038
        al143i=0.041
        al143p=0.038

doclean_cross_stg=False
#steven values 
if doclean_cross_stg:
    al100i=0.019
    al100p=0.019
    al143i=0.038
    al143p=0.038


do100x100coeffs=False
if do100x100coeffs:
    al100i=0.019
    al100p=0.019
    al143i=0.019
    al143p=0.019
    be100i=0.01
    be100p=0.01
    be143i=0.01
    be143p=0.01

do143x143coeffs=False
if do143x143coeffs:
    al100i=0.038
    al100p=0.038
    al143i=0.038
    al143p=0.038
    be100i=0.0
    be100p=0.0
    be143i=0.0
    be143p=0.0

# will default to uK^2...
#n_11=1.e12*inpcovmat('joint_noise_100x100_f32.dat')
#n_22=1.e12*inpcovmat('joint_noise_143x143_f32.dat')
#n_12=1.e12*inpcovmat('joint_noise_100x143_f32.dat')

#n_11=1.*inpcovmat('joint_noise_100x100_f32_EB_lmax4_projected.dat')
#n_22=1.*inpcovmat('joint_noise_143x143_f32_EB_lmax4_projected.dat')
#n_12=1.e12*inpcovmat('joint_noise_100x143_f32.dat')

#should be 0.062336 ?...
#noisefac=0.062336
noisefac=65363.807
n_11=noisefac*inpcovmat(basedir+'gpemask_rbeam_QU_RC4_v2_tp_100-all_r06.dat')
n_22=noisefac*inpcovmat(basedir+'gpemask_rbeam_QU_RC4_v2_tp_143-all_r06.dat')
n_12=zeros_like(n_11)
#n_12[tl:,tl:]=noisefac*1.e12*inpcovmat('joint_noise_100x143_f32_lmax4_projected.dat')
    
# to match ffp8
boostPPnoise=False
if boostPPnoise:
    n_11[tl:,tl:]*=1.3
    n_22[tl:,tl:]*=2.5

replaceTTblock=False
if replaceTTblock:
    n_11[:tl,:tl]=1.e12*inpcovmat(reijo_mat_100_TT)[tmask>0][:,tmask>0]
    n_22[:tl,:tl]=1.e12*inpcovmat(reijo_mat_143_TT)[tmask>0][:,tmask>0]

'''
noise_DX11d_100ds1x100ds1_EB_lmax4_pixwin_400sims_smoothmean_AC.dat
noise_DX11d_100ds2x100ds2_EB_lmax4_pixwin_400sims_smoothmean_AC.dat
noise_DX11d_100fullx100full_EB_lmax4_pixwin_400sims_smoothmean_AC.dat
noise_DX11d_100hm1x100hm1_EB_lmax4_pixwin_400sims_smoothmean_AC.dat
noise_DX11d_100hm2x100hm2_EB_lmax4_pixwin_400sims_smoothmean_AC.dat
noise_DX11d_100x100_EB_lmax4_pixwin_400sims_smoothmean_AC.dat
noise_DX11d_143ds1x143ds1_EB_lmax4_pixwin_400sims_smoothmean_AC.dat
noise_DX11d_143ds2x143ds2_EB_lmax4_pixwin_400sims_smoothmean_AC.dat
noise_DX11d_143fullx143full_EB_lmax4_pixwin_400sims_smoothmean_AC.dat
noise_DX11d_143hm1x143hm1_EB_lmax4_pixwin_400sims_smoothmean_AC.dat
noise_DX11d_143hm2x143hm2_EB_lmax4_pixwin_400sims_smoothmean_AC.dat
noise_DX11d_143x143_EB_lmax4_pixwin_400sims_smoothmean_AC.dat
'''

replacePPblock=True
if replacePPblock:
#    n_11[tl:,tl:]=inpcovmat('RC4/RC4_noise_100x100_ri_al_1p04_f32_EB_lmax4_projected.dat')
#    n_22[tl:,tl:]=inpcovmat('RC4/RC4_noise_143x143_ri_al_1p2_f32_EB_lmax4_projected.dat')
#    n_11[tl:,tl:]=inpcovmat('RC4/reijo_100x100_f32_EB_lmax4_projected.dat')
#    STG20: former matrices
#    n_11[tl:,tl:]=inpcovmat(basedir+'reijo_100x100_f32_EB_lmax4_projected.dat')
#    n_22[tl:,tl:]=inpcovmat(basedir+'reijo_143x143_f32_EB_lmax4_projected.dat')
    n_11[tl:,tl:]=inpcovmat(ncm_11_file)
    n_22[tl:,tl:]=inpcovmat(ncm_22_file)
#    n_11[tl:,tl:]=inpcovmat('RC4/luca/luca_reijo_100x100_f32_EB_lmax4_projected.dat')
#    n_22[tl:,tl:]=inpcovmat('RC4/luca/luca_reijo_143x143_f32_EB_lmax4_projected.dat')
#    n_11[tl:,tl:]=inpcovmat('RC4/g2_100GHz_fullx100GHz_full_f32_EB_l23f_8p.dat')
#    n_22[tl:,tl:]=inpcovmat('RC4/g2_143GHz_fullx143GHz_full_f32_EB_l23f_8p.dat')
#    n_11[tl:,tl:]=inpcovmat('RC4/my_100GHz_full_projected.dat')
#    n_22[tl:,tl:]=inpcovmat('RC4/my_143GHz_full_projected.dat')
#    n_11[tl:,tl:]=inpcovmat('RC4/ecl_reijo_100x100_custom_projected_EB_v4.dat')
#    n_22[tl:,tl:]=inpcovmat('RC4/ecl_reijo_143x143_custom_projected_EB_v4.dat')
#factors to match Tab 6 of low-l paper...
#    n_11*=1.8
#
#    n_22*=1.27
    pass

combinePPblock=False
if combinePPblock:
    n_11[tl:,tl:]*=1.
    n_22[tl:,tl:]*=1.
    n_11[tl:,tl:]+=inpcovmat(ncm_11_file)
    n_22[tl:,tl:]+=inpcovmat(ncm_22_file)
    n_11*=1.3
    n_22*=2.5

replacewithReijoMat=False
if replacewithReijoMat:
    n_11*=0.
    n_22*=0
    n_11[:tl,:tl]=1.e12*inpcovmat(reijo_mat_100_TT)[tmask>0][:,tmask>0]
    n_22[:tl,:tl]=1.e12*inpcovmat(reijo_mat_143_TT)[tmask>0][:,tmask>0]
    n_11[tl:,tl:]=1.e12*inpcovmat(reijo_mat_100_PP)[qumask>0][:,qumask>0]
    n_22[tl:,tl:]=1.e12*inpcovmat(reijo_mat_143_PP)[qumask>0][:,qumask>0]
#    n_11[tl:,tl:]*=2.3
#    n_22[tl:,tl:]*=2.3

combinewithReijoMat=False
if combinewithReijoMat:
    #n_11[:tl,:tl]=1.e12*inpcovmat('reijo_100GHz_TT_ring.dat')[tmask>0][:,tmask>0]
    #n_22[:tl,:tl]=1.e12*inpcovmat('reijo_143GHz_TT_ring.dat')[tmask>0][:,tmask>0]
    n_11[tl:,tl:]*=1.
    n_22[tl:,tl:]*=1.
#    n_11[tl:,tl:]+=0.*1.e12*inpcovmat('reijo_100GHz_PP_ring.dat')[qumask>0][:,qumask>0]
    n_11[tl:,tl:]+=1.*1.e12*inpcovmat(reijo_mat_100_PP)[qumask>0][:,qumask>0]
    n_22[tl:,tl:]+=1.*1.e12*inpcovmat(reijo_mat_143_PP)[qumask>0][:,qumask>0]

if compsepnoiseboost:
    n_11[:tl,:tl]*=1./(1-al100i-be100i)**2
    n_11[:tl,tl:]*=1./(1-al100i-be100i)/(1-al100p-be100p)
    n_11[tl:,:tl]*=1./(1-al100i-be100i)/(1-al100p-be100p)
    n_11[tl:,tl:]*=1./(1-al100p-be100p)**2
    n_22[:tl,:tl]*=1./(1-al143i-be143i)**2
    n_22[:tl,tl:]*=1./(1-al143i-be143i)/(1-al143p-be143p)
    n_22[tl:,:tl]*=1./(1-al143i-be143i)/(1-al143p-be143p)
    n_22[tl:,tl:]*=1./(1-al143p-be143p)**2


# move this copy up or down depending on what needs to be regulated...
n_11_cov=n_11.copy()
n_22_cov=n_22.copy()
n_12_cov=n_12.copy()


restoresimplenoise=False
if restoresimplenoise:
    n_11*=0.
    n_22*=0
    n_11[:tl,:tl]=1.e12*inpcovmat(reijo_mat_100_TT)[tmask>0][:,tmask>0]
    n_22[:tl,:tl]=1.e12*inpcovmat(reijo_mat_143_TT)[tmask>0][:,tmask>0]
    n_11[tl:,tl:]=1.e12*inpcovmat(reijo_mat_100_PP)[qumask>0][:,qumask>0]
    n_22[tl:,tl:]=1.e12*inpcovmat(reijo_mat_143_PP)[qumask>0][:,qumask>0]

destroyTPnoise=False
if destroyTPnoise:
    n_11[:tl,tl:]=0.
    n_11[tl:,:tl]=0.
    n_22[:tl,tl:]=0.
    n_22[tl:,:tl]=0.
    n_12[:tl,tl:]=0.
    n_12[tl:,:tl]=0.



# I've used 3.e-4 for both 11 and 22 to regulate noise
# George used about 1% of the diagonal covariance, so
# this is about 25 for TT and .001 for 11 PP and .0006 for 22 PP...
    
regulatenoise=False
if regulatenoise:
    n_11+=diag(3.e-4*ones(n_11.shape[0]))
    n_22+=diag(3.e-4*ones(n_22.shape[0]))
    
regulatenoise=False
if regulatenoise:
    n_11+=diag(.001*ones(n_11.shape[0]))
    n_22+=diag(.001*ones(n_22.shape[0]))

regulatenoiseT=True
if regulatenoiseT:
    n_11[:tl,:tl]+=diag(25.*ones(tl))
    n_22[:tl,:tl]+=diag(25.*ones(tl))

#n_11=1.*inpcovmat('full_joint_noise_100ds2x100ds2_for_100ds2+143ds2_f32_EB_lmax4_projected.dat')
#n_22=1.*inpcovmat('full_joint_noise_143ds2x143ds2_for_100ds2+143ds2_f32_EB_lmax4_projected.dat')
#n_12=1.*inpcovmat('full_joint_noise_100ds2x143ds2_for_100ds2+143ds2_f32_EB_lmax4_projected.dat')

#n_11=1.*inpcovmat('joint_noise_100x100_f32_EB_lmax4_projected.dat')
#n_22=1.*inpcovmat('joint_noise_143x143_f32_EB_lmax4_projected.dat')
#n_12=0.*inpcovmat('full_joint_noise_100ds1x143ds2_for_100ds1+143ds2_f32_EB_lmax4_projected.dat')

#n_11=1.e12*inpcovmat('joint_noise_100x100_f32.dat')
#n_22=1.e12*inpcovmat('joint_noise_143x143_f32.dat')
#n_12=0.*inpcovmat('full_joint_noise_100ds1x143ds2_for_100ds1+143ds2_f32_EB_lmax4_projected.dat')

diagonalizenoise=False
if diagonalizenoise:
    nfac=.1
    n11mean=mean(diag(n_11))
    n22mean=mean(diag(n_22))
    nsize=n_11.shape[0]
    n_11=nfac*n11mean*identity(nsize)
    n_22=nfac*n22mean*identity(nsize)
    n_12*=0.0

#n_11=1.e12*inpdoubcovmat('noise_100x100.dat')
#n_22=1.e12*inpdoubcovmat('noise_143x143.dat')
#n_12=1.e12*inpcovmat('joint_noise_100x143_f32.dat')

reducecrossnoise=False
if reducecrossnoise:
    n_12*=0.0

setqunoisetozero=False
if setqunoisetozero:
    sz=n_11.shape[0]/2
    n_11[:sz,sz:]=0.0
    n_11[sz:,:sz]=0.0
    n_22[:sz,sz:]=0.0
    n_22[sz:,:sz]=0.0
    n_12[:sz,sz:]=0.0
    n_12[sz:,:sz]=0.0


    #trying removing the 0.062336...
mean_11=1.0*inpvec(basedir+'gpemask_fixedw_IQU_100GHz_freq_mean.dat')
mean_22=1.0*inpvec(basedir+'gpemask_fixedw_IQU_143GHz_freq_mean.dat')
#mean_11[tl:]=inpvec('RC4/DEC16v1_0_82_nmd_QU_100GHz_full_mean.dat')
#mean_22[tl:]=inpvec('RC4/DEC16v1_0_82_nmd_QU_143GHz_full_mean.dat')

#mean_11[tl:]=inpvec(basedir+'DEC16v1_0_82_ri_QU_100GHz_full_mean.dat')
#mean_22[tl:]=inpvec(basedir+'DEC16v1_0_82_ri_QU_143GHz_full_mean.dat')

tmp100 = healpy.read_map(smooth_mean_100, field=(1,2))
tmp143 = healpy.read_map(smooth_mean_143, field=(1,2))

mean_11[tl:]= concatenate((tmp100[0,:], tmp100[1,:]))[qumask==1]
mean_22[tl:]= concatenate((tmp143[0,:], tmp143[1,:]))[qumask==1]


removefidCMBfrommeans=False
if removefidCMBfrommeans:
    import healpy
    cmb100i,cmb100q,cmb100u=healpy.read_map('RC4/cmb_fid_100_lr_no_md.fits',field=(0,1,2))
    cmb143i,cmb143q,cmb143u=healpy.read_map('RC4/cmb_fid_143_lr_no_md.fits',field=(0,1,2))
    mean_11[:tl]-=cmb100i[tmask>0]
    #mean_11[:tl]=0.
    mean_11[tl:tl+pl]-=cmb100q[polmask>0]
    mean_11[tl+pl:tl+2*pl]-=cmb100u[polmask>0]
    mean_22[:tl]-=cmb143i[tmask>0]
    #mean_22[:tl]=0.
    mean_22[tl:tl+pl]-=cmb143q[polmask>0]
    mean_22[tl+pl:tl+2*pl]-=cmb143u[polmask>0]

removefidskyfrommeans=False
if removefidskyfrommeans:
    import healpy
    inname='RC4/inp100_lr_nmd.fits'
    cmb100i,cmb100q,cmb100u=healpy.read_map(inname,field=(0,1,2))
    inname='RC4/inp143_lr_nmd.fits'
    cmb143i,cmb143q,cmb143u=healpy.read_map(inname,field=(0,1,2))
    inname='RC4/inp353_lr_nmd.fits'
    cmb353i,cmb353q,cmb353u=healpy.read_map(inname,field=(0,1,2))
    t_1=(cmb100i-al100i*cmb353i)/(1.-al100i)
    t_2=(cmb143i-al143i*cmb353i)/(1.-al143i)
    q_1=(cmb100q-al100p*cmb353q)/(1.-al100p)
    q_2=(cmb143q-al143p*cmb353q)/(1.-al143p)
    u_1=(cmb100u-al100p*cmb353u)/(1.-al100p)
    u_2=(cmb143u-al143p*cmb353u)/(1.-al143p)
    testzeroTsubtraction=False
    if testzeroTsubtraction:
        t_1*=0.
        t_2*=0.
    mean_11[:tl]-=t_1[tmask>0]
    mean_11[tl:tl+pl]-=q_1[polmask>0]
    mean_11[tl+pl:tl+2*pl]-=u_1[polmask>0]
    mean_22[:tl]-=t_2[tmask>0]
    mean_22[tl:tl+pl]-=q_2[polmask>0]
    mean_22[tl+pl:tl+2*pl]-=u_2[polmask>0]

adjustmeans=False
if adjustmeans:
    meanlen=mean_11.shape[0]
    mean_11[:meanlen/2]*=.43
    mean_11[meanlen/2:]*=.51
    mean_22[:meanlen/2]*=0.04
    mean_22[meanlen/2]*=0.36

projectmeans=False
if projectmeans:
    n_11+=9.*outer(mean_11,mean_11)
    n_22+=9.*outer(mean_22,mean_22)
    n_12+=9.*outer(mean_11,mean_22)

useprepareddata=False
if useprepareddata:
    use_cosine_pixwin_data=False
    if use_cosine_pixwin_data:
        datnamestart='../RD12ll/'
        datnameend='-353_full_cosine_pixwin.dat'
    else:
        datnamestart='RC4/'
        datnameend='-353_full.dat'
    t_1=inpvec(datnamestart+'I_100'+datnameend)
    q_1=inpvec(datnamestart+'Q_100'+datnameend)
    u_1=inpvec(datnamestart+'U_100'+datnameend)
    t_2=inpvec(datnamestart+'I_143'+datnameend)
    q_2=inpvec(datnamestart+'Q_143'+datnameend)
    u_2=inpvec(datnamestart+'U_143'+datnameend)

usefitsdata=False
if usefitsdata:
    import healpy
    inname='RC4/30GHz_dx12_lr_nmd.fits'
    cmb30i,cmb30q,cmb30u=healpy.read_map(inname,field=(0,1,2))
    inname='RC4/100GHz_RC4_lr_nmd.fits'
    cmb100i,cmb100q,cmb100u=healpy.read_map(inname,field=(0,1,2))
    inname='RC4/143GHz_RC4_lr_nmd.fits'
    cmb143i,cmb143q,cmb143u=healpy.read_map(inname,field=(0,1,2))
    inname='RC4/353GHz_RC4_lr_nmd.fits'
    cmb353i1,cmb353q1,cmb353u1=healpy.read_map(inname,field=(0,1,2))
    inname='RC4/353GHz_RC4_lr_nmd.fits'
    cmb353i2,cmb353q2,cmb353u2=healpy.read_map(inname,field=(0,1,2))
    t_1=(cmb100i-al100i*cmb353i1-be100i*cmb30i)/(1.-al100i-be100i)
    t_2=(cmb143i-al143i*cmb353i2-be143i*cmb30i)/(1.-al143i-be143i)
    q_1=(cmb100q-al100p*cmb353q1-be100p*cmb30q)/(1.-al100p-be100p)
    q_2=(cmb143q-al143p*cmb353q2-be143p*cmb30q)/(1.-al143p-be143p)
    u_1=(cmb100u-al100p*cmb353u1-be100p*cmb30u)/(1.-al100p-be100p)
    u_2=(cmb143u-al143p*cmb353u2-be143p*cmb30u)/(1.-al143p-be143p)

'''
#SROLL2.0
100ds1_SR20_16R_pixwin_new.fits
100ds2_SR20_16R_pixwin_new.fits
100full_SR20_16R_pixwin_new.fits
143ds1_SR20_16R_pixwin_new.fits
143ds2_SR20_16R_pixwin_new.fits
143full_SR20_16R_pixwin_new.fits
353ds1_SR20_16R_pixwin_new.fits
353ds2_SR20_16R_pixwin_new.fits
353hm1_SR20_16R_pixwin_new.fits
353hm2_SR20_16R_pixwin_new.fits
#PLANCK2018
100full_FFP10_16R_pixwin_new.fits
143full_FFP10_16R_pixwin_new.fits
30hm1_LFI_FFP10_16_pixwin.fits
30hm2_LFI_FFP10_16_pixwin.fits
353hm1_FFP10_16R_pixwin_new.fits
353hm2_FFP10_16R_pixwin_new.fits
'''


if usegeneralfitsdata_roger:
    print('Reading in data: {}'.format(data_set),flush=True)
    print(inname100,flush=True)
    print(inname143,flush=True)
    print('Foreground cleaning using:',flush=True)
    print(inname30_1,flush=True)
    print(inname30_2,flush=True)
    print(inname353_1,flush=True)
    print(inname353_2,flush=True)
    print('',flush=True)
    cmb30i1,cmb30q1,cmb30u1=healpy.read_map(inname30_1,field=(0,1,2))
    cmb30i2,cmb30q2,cmb30u2=healpy.read_map(inname30_2,field=(0,1,2))
    cmb100i,cmb100q,cmb100u=healpy.read_map(inname100,field=(0,1,2))
    cmb143i,cmb143q,cmb143u=healpy.read_map(inname143,field=(0,1,2))
    cmb353i1,cmb353q1,cmb353u1=healpy.read_map(inname353_1,field=(0,1,2))
    cmb353i2,cmb353q2,cmb353u2=healpy.read_map(inname353_2,field=(0,1,2))
    t_1=(cmb100i-al100i*cmb353i1-be100i*cmb30i1)/(1.-al100i-be100i)
    t_2=(cmb143i-al143i*cmb353i2-be143i*cmb30i2)/(1.-al143i-be143i)
    q_1=(cmb100q-al100p*cmb353q1-be100p*cmb30q1)/(1.-al100p-be100p)
    q_2=(cmb143q-al143p*cmb353q2-be143p*cmb30q2)/(1.-al143p-be143p)
    u_1=(cmb100u-al100p*cmb353u1-be100p*cmb30u1)/(1.-al100p-be100p)
    u_2=(cmb143u-al143p*cmb353u2-be143p*cmb30u2)/(1.-al143p-be143p)

if usesims_roger:
    print('Reading in sims: {}'.format(data_set),flush=True)
    print(inname100,flush=True)
    print(inname143,flush=True)
    print('Foreground cleaning using:',flush=True)
    print(inname30_1,flush=True)
    print(inname30_2,flush=True)
    print(inname353_1,flush=True)
    print(inname353_2,flush=True)
    print('',flush=True)
    cmb30i1,cmb30q1,cmb30u1=healpy.read_map(inname30_1,field=(0,1,2))
    cmb30i2,cmb30q2,cmb30u2=healpy.read_map(inname30_2,field=(0,1,2))
    cmb100i,cmb100q,cmb100u=healpy.read_map(inname100,field=(0,1,2))
    cmb143i,cmb143q,cmb143u=healpy.read_map(inname143,field=(0,1,2))
    cmb353i1,cmb353q1,cmb353u1=healpy.read_map(inname353_1,field=(0,1,2))
    cmb353i2,cmb353q2,cmb353u2=healpy.read_map(inname353_2,field=(0,1,2))
    t_1=(cmb100i-al100i*cmb353i1-be100i*cmb30i1)/(1.-al100i-be100i)
    t_2=(cmb143i-al143i*cmb353i2-be143i*cmb30i2)/(1.-al143i-be143i)
    q_1=(cmb100q-al100p*cmb353q1-be100p*cmb30q1)/(1.-al100p-be100p)
    q_2=(cmb143q-al143p*cmb353q2-be143p*cmb30q2)/(1.-al143p-be143p)
    u_1=(cmb100u-al100p*cmb353u1-be100p*cmb30u1)/(1.-al100p-be100p)
    u_2=(cmb143u-al143p*cmb353u2-be143p*cmb30u2)/(1.-al143p-be143p)


usegeneralfitsdata=False
if usegeneralfitsdata:
    inname30=basedir+'30GHz_dx12_lr_nmd.fits'
    inname100=basedir+'100GHz_ful.all_ful.RD12_RC4.P_lr.fits'
    inname143=basedir+'143GHz_ful.all_ful.RD12_RC4.P_lr.fits'
    inname353_1=basedir+'353GHz_ful.all_ful.RD12_RC4.P_lr.fits'
    inname353_2=basedir+'353GHz_ful.all_ful.RD12_RC4.P_lr.fits'
    print('Doing:',flush=True)
    print(inname100,flush=True)
    print(inname143,flush=True)
    print('',flush=True)
    cmb30i,cmb30q,cmb30u=healpy.read_map(inname30,field=(0,1,2))
    cmb100i,cmb100q,cmb100u=healpy.read_map(inname100,field=(0,1,2))
    cmb143i,cmb143q,cmb143u=healpy.read_map(inname143,field=(0,1,2))
    cmb353i1,cmb353q1,cmb353u1=healpy.read_map(inname353_1,field=(0,1,2))
    cmb353i2,cmb353q2,cmb353u2=healpy.read_map(inname353_2,field=(0,1,2))
    t_1=(cmb100i-al100i*cmb353i1-be100i*cmb30i)/(1.-al100i-be100i)
    t_2=(cmb143i-al143i*cmb353i2-be143i*cmb30i)/(1.-al143i-be143i)
    q_1=(cmb100q-al100p*cmb353q1-be100p*cmb30q)/(1.-al100p-be100p)
    q_2=(cmb143q-al143p*cmb353q2-be143p*cmb30q)/(1.-al143p-be143p)
    u_1=(cmb100u-al100p*cmb353u1-be100p*cmb30u)/(1.-al100p-be100p)
    u_2=(cmb143u-al143p*cmb353u2-be143p*cmb30u)/(1.-al143p-be143p)



usesimulation_stg=False
if usesimulation_stg:
    import healpy
    simnum=98
    simdir='/scratch/stg20/planck/DEC16v1/'
    instart=simdir+'DEC16v1_CMBfid_100ghz_'
    inname=instart+'{0:0>3}_full_IQU_lr_nmd.fits'.format(simnum)
    cmb100i,cmb100q,cmb100u=healpy.read_map(inname,field=(0,1,2))
    instart=simdir+'DEC16v1_CMBfid_143ghz_'
    inname=instart+'{0:0>3}_full_IQU_lr_nmd.fits'.format(simnum)
    cmb143i,cmb143q,cmb143u=healpy.read_map(inname,field=(0,1,2))
    instart=simdir+'DEC16v1_CMBfid_353ghz_'
    inname=instart+'{0:0>3}_full_IQU_lr_nmd.fits'.format(simnum)
    cmb353i,cmb353q,cmb353u=healpy.read_map(inname,field=(0,1,2))
    inname='RC4/sync30_lr_nmd.fits'
    cmb30i,cmb30q,cmb30u=healpy.read_map(inname,field=(0,1,2))
    t_1=(cmb100i-al100i*cmb353i-be100i*cmb30i)/(1.-al100i)
    t_2=(cmb143i-al143i*cmb353i-be143i*cmb30i)/(1.-al143i)
    q_1=(cmb100q-al100p*cmb353q-be100p*cmb30q)/(1.-al100p)
    q_2=(cmb143q-al143p*cmb353q-be143p*cmb30q)/(1.-al143p)
    u_1=(cmb100u-al100p*cmb353u-be100p*cmb30u)/(1.-al100p)
    u_2=(cmb143u-al143p*cmb353u-be143p*cmb30u)/(1.-al143p)


replaceTwithfid=False
if replaceTwithfid:
    import healpy
    cmb100i,cmb100q,cmb100u=healpy.read_map('RC4/cmb_fid_100_lr_no_md.fits',field=(0,1,2))
    cmb143i,cmb143q,cmb143u=healpy.read_map('RC4/cmb_fid_143_lr_no_md.fits',field=(0,1,2))
    t_1=cmb100i
    t_2=cmb143i

zerotmean=False
if zerotmean:
    t_1-=mean(t_1)
    t_2-=mean(t_2)

zerotsubtraction=False
if zerotsubtraction:
    mean_11[:tl]=0.
    mean_22[:tl]=0.

d_1=concatenate((t_1[tmask==1],q_1[polmask==1],u_1[polmask==1]))
d_2=concatenate((t_2[tmask==1],q_2[polmask==1],u_2[polmask==1]))

meanfac=1.0
if reducemeanfac:
    meanfac=0.

individualmeanfac=False
if individualmeanfac:
    meanfac1=1.0
    meanfac2=0.0
else:
    meanfac1=meanfac
    meanfac2=meanfac

d_1-=meanfac1*mean_11
d_2-=meanfac2*mean_22

fiddledata=False
if fiddledata:
    d_1=mean_11
    d_2=mean_22

#clroot='trimmed/'
#clroot='/scratch/stg20/tau_scan/trimmed/'

fidcldatfile = tau_fid_file
#b_l=inpvec(basedir+'beam_from_symbol.dat')
#b_l=inpvec('cosine_pixwin_beam.dat')

def reijo(ell, nside):
    if (ell<=nside): wl = 1.
    if (nside < ell <= 3.*nside): wl = 0.5*(1+ cos(np.pi * (ell-nside)/(2*nside)))
    if (ell > 3.*nside): wl = 0.
    return wl

vreijo = vectorize(reijo)
if smooth_reijo:
    b_l    = vreijo(np.arange(0, 3*nside), nside)
elif smooth_reijo_pixwin:
    b_l    = vreijo(np.arange(0, 3*nside), nside)*healpy.pixwin(nside)

print(fidcldatfile,flush=True)

#eigfilename='/scratch/stg20/planck/storeddats/sphharmE.dat'

#eigfilename='sphharmEB_lmax40.dat'
#eigfilename='sphharmTE_lmax40.dat'
#eigfilename='sphharmTE_lmax47.dat'
#eigfilename=basedir+'sphharmTE_lmax47_md.dat'

y_full=reshape(inpvec(eigfilename),(3*npix,-1),order='F').copy()
# sum should be npix/4pi...
#for i in range(0,100):
#    print(i, sum(y_full[:3072,i]**2),  sum(y_full[3072:,i]**2),  sum(y_full[:,i]**2))

# Also note one can separate the e and b parts by:
# y_full2=reshape(inpvec(eigfilename),(2*npix,2,-1),order='F')
# Then y_full2[:,0,l**2+2m+1] is the e vec, 
#      y_full2[:,1,l**2+2m+1] is the b vec. 


if testdestroymonodi:
    print('destroy mono and dipole for temperature')
    dmask=concatenate((tmask,zeros(npix),zeros(npix)))    
    yd=y_full[dmask==1,0:8:2].copy()
    d1d=d_1[:tl].copy()
    d2d=d_2[:tl].copy()
    ydtyd=dot(transpose(yd),yd)
    ydtydi=inv(ydtyd)
    d_1[:tl]-=dot(yd,dot(ydtydi,dot(transpose(yd),d1d)))
    d_2[:tl]-=dot(yd,dot(ydtydi,dot(transpose(yd),d2d)))
    

#lmin=2
#lmax=29
#lmaxalias=47

def getcls(filename, wantedcol):
    inpdat=loadtxt(filename)
    ells=inpdat[lmin-2:lmax-1,0]
    dls=inpdat[lmin-2:lmax-1,wantedcol]
    return 2.0*math.pi*dls/ells/(ells+1.)*b_l[lmin:lmax+1]**2

#CHECK: Should this have the beam in or not?qqqq
def getclsmulti(filename, wantedcols):
    inpdat=loadtxt(filename)
    ells=inpdat[lmin-2:lmax-1,0]
    ncols=len(wantedcols)
    cls=zeros((lmax+1-lmin)*ncols)
    wcol=0
    for col in wantedcols:
        cls[wcol::ncols]=2.0*math.pi*inpdat[lmin-2:lmax-1,col]/ells/(ells+1.)*b_l[lmin:lmax+1]**2
        wcol+=1
    return cls

# y is for unmasked sky, starting at the quadrupole...
# 2 in front of lmax since e and b
y=y_full[fullmask==1,2*lmin**2:2*(lmax+1)**2].copy()
yt=transpose(y).copy()

print('compute Y matrix only for polarisation E and B')
yqu=y_full[npix:,2*lmin**2:2*(lmax+1)**2].copy()
yqumask=yqu[qumask==1,:].copy()
yqumaskt=transpose(yqumask).copy()

print('my NCM only for QQ, QU, UQ, UU ')
ncm100 = n_11[tl:, tl:]
ncm143 = n_22[tl:, tl:]

yal=y_full[fullmask==1,2*(lmax+1)**2:2*(lmaxalias+1)**2].copy()
yalt=transpose(yal).copy()

def makecllong_EEEBBB(filename):
    inpdat=loadtxt(filename)
    inpells=inpdat[lmin-2:lmax-1,0] # ells
    inpclee=inpdat[lmin-2:lmax-1,2] # i.e. EE
    inpclbb=inpdat[lmin-2:lmax-1,3] # i.e. BB
    inpcleb=inpdat[lmin-2:lmax-1,4]  # i.e. EB
    inpclbe=inpcleb
    outlen=2*((lmax+1)**2-lmin**2)
    cldiag=zeros((outlen,outlen))
    for l in range(lmin,lmax+1):
        for m in range(0,2*l+1):
            outpos=2*(l**2+m-lmin**2)
            cldiag[outpos,outpos]=2.0*math.pi*inpclee[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos,outpos+1]=2.0*math.pi*inpcleb[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos+1,outpos]=2.0*math.pi*inpclbe[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos+1,outpos+1]=2.0*math.pi*inpclbb[l-lmin]/l/(l+1)*b_l[l]**2
    return cldiag

def makecllong(filename):
    # could be called makecllong_TTTEEE
    inpdat=loadtxt(filename)
    inpells=inpdat[lmin-2:lmax-1,0] # ells
    inpclee=inpdat[lmin-2:lmax-1,1] # i.e. TT
    inpclbb=inpdat[lmin-2:lmax-1,2] # i.e. EE
    inpcleb=inpdat[lmin-2:lmax-1,4]  # i.e. TE
    inpclbe=inpcleb
    outlen=2*((lmax+1)**2-lmin**2)
    cldiag=zeros((outlen,outlen))
    for l in range(lmin,lmax+1):
        for m in range(0,2*l+1):
            outpos=2*(l**2+m-lmin**2)
            cldiag[outpos,outpos]=2.0*math.pi*inpclee[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos,outpos+1]=2.0*math.pi*inpcleb[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos+1,outpos]=2.0*math.pi*inpclbe[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos+1,outpos+1]=2.0*math.pi*inpclbb[l-lmin]/l/(l+1)*b_l[l]**2
    return cldiag

def makecllong_altlmax(filename,altlmax):
    inpdat=loadtxt(filename)
    inpells=inpdat[lmin-2:altlmax-1,0]
    inpclee=inpdat[lmin-2:altlmax-1,1] # i.e. TT
    inpclbb=inpdat[lmin-2:altlmax-1,2] # i.e. EE
    inpcleb=inpdat[lmin-2:altlmax-1,4]  # i.e. TE
    inpclbe=inpcleb
    outlen=2*((altlmax+1)**2-lmin**2)
    cldiag=zeros((outlen,outlen))
    for l in range(lmin,altlmax+1):
        for m in range(0,2*l+1):
            outpos=2*(l**2+m-lmin**2)
            cldiag[outpos,outpos]=2.0*math.pi*inpclee[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos,outpos+1]=2.0*math.pi*inpcleb[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos+1,outpos]=2.0*math.pi*inpclbe[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos+1,outpos+1]=2.0*math.pi*inpclbb[l-lmin]/l/(l+1)*b_l[l]**2
    return cldiag

regcleeval=0.
regclbbval=0.
regcl=True
if regcl:
    regcleeval=.1
    regclbbval=5.e-5

def makecllong_regulate(filename):
    inpdat=loadtxt(filename)
    inpells=inpdat[lmin-2:lmax-1,0]
    inpclee=inpdat[lmin-2:lmax-1,1] # i.e. TT
    inpclbb=inpdat[lmin-2:lmax-1,2] # i.e. EE
    inpcleb=inpdat[lmin-2:lmax-1,4]  # i.e. TE
    inpclbe=inpcleb
    outlen=2*((lmax+1)**2-lmin**2)
    cldiag=zeros((outlen,outlen))
    for l in range(lmin,lmax+1):
        for m in range(0,2*l+1):
            outpos=2*(l**2+m-lmin**2)
            cldiag[outpos,outpos]=2.0*math.pi*inpclee[l-lmin]/l/(l+1)*b_l[l]**2+regcleeval
            cldiag[outpos,outpos+1]=2.0*math.pi*inpcleb[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos+1,outpos]=2.0*math.pi*inpclbe[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos+1,outpos+1]=2.0*math.pi*inpclbb[l-lmin]/l/(l+1)*b_l[l]**2+regclbbval
    return cldiag
    
def makecllong_reshaped(filename):
    inpdat=loadtxt(filename)
    inpells=inpdat[lmin-2:lmax-1,0]
    inpclee=inpdat[lmin-2:lmax-1,1] # i.e. TT
    inpclbb=inpdat[lmin-2:lmax-1,2] # i.e. EE
    inpcleb=zeros_like(inpclee) 
    inpclbe=zeros_like(inpclee)
    outlen=2*((lmax+1)**2-lmin**2)
    cldiag=zeros((outlen,outlen))
    for l in range(lmin,lmax+1):
        for m in range(0,2*l+1):
            outpos=2*(l**2+m-lmin**2)
            cldiag[outpos,outpos]=2.0*math.pi*inpclee[l-lmin]/l/(l+1)*b_l[l]**2+regcleeval
            cldiag[outpos,outpos+1]=2.0*math.pi*inpcleb[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos+1,outpos]=2.0*math.pi*inpclbe[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos+1,outpos+1]=2.0*math.pi*inpclbb[l-lmin]/l/(l+1)*b_l[l]**2+regclbbval
    return cldiag

def makecllong_reshaped_skip(filename,startl):
    inpdat=loadtxt(filename)
    inpells=inpdat[lmin-2:lmax-1,0]
    inpclee=inpdat[lmin-2:lmax-1,1] # i.e. TT
    inpclbb=inpdat[lmin-2:lmax-1,2] # i.e. EE
    inpcleb=zeros_like(inpclee) 
    inpclbe=zeros_like(inpclee)
    outlen=2*((lmax+1)**2-lmin**2)
    cldiag=zeros((outlen,outlen))
    for l in range(startl,lmax+1):
        for m in range(0,2*l+1):
            outpos=2*(l**2+m-lmin**2)
            cldiag[outpos,outpos]=2.0*math.pi*inpclee[l-lmin]/l/(l+1)*b_l[l]**2+regcleeval
            cldiag[outpos,outpos+1]=2.0*math.pi*inpcleb[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos+1,outpos]=2.0*math.pi*inpclbe[l-lmin]/l/(l+1)*b_l[l]**2
            cldiag[outpos+1,outpos+1]=2.0*math.pi*inpclbb[l-lmin]/l/(l+1)*b_l[l]**2+regclbbval
    return cldiag


def makecllong_from_cls(cls):
    nells=lmax-lmin+1
    inpclee=cls[::3] # i.e. TT
    inpclbb=cls[2::3] # i.e. EE
    inpcleb=cls[1::3]  # i.e. TE
    inpclbe=inpcleb
    outlen=2*((lmax+1)**2-lmin**2)
    cldiag=zeros((outlen,outlen))
    for l in range(lmin,lmax+1):
        for m in range(0,2*l+1):
            outpos=2*(l**2+m-lmin**2)
            cldiag[outpos,outpos]=inpclee[l-lmin]*b_l[l]**2
            cldiag[outpos,outpos+1]=inpcleb[l-lmin]*b_l[l]**2
            cldiag[outpos+1,outpos]=inpclbe[l-lmin]*b_l[l]**2
            cldiag[outpos+1,outpos+1]=inpclbb[l-lmin]*b_l[l]**2
    return cldiag

#assert(False)


if addallhighlpower:
    print('adding highl power...',flush=True)
    n_alias=dot(yal,dot(makecllong_altlmax(fidcldatfile,lmaxalias)[2*(lmax+1)**2-2*lmin**2:,2*(lmax+1)**2-2*lmin**2:],yalt))
    n_11_cov+=n_alias
    n_22_cov+=n_alias
    n_12_cov+=n_alias    
    n_11+=n_alias
    n_22+=n_alias
    n_12+=n_alias
    print('...done adding highl power',flush=True)

addhighlpowernomixing=False
if addhighlpowernomixing:
    n_alias=dot(yal,dot(makecllong_altlmax(fidcldatfile,lmaxalias)[2*(lmax+1)**2-2*lmin**2:,2*(lmax+1)**2-2*lmin**2:],yalt))
    n_11_cov+=n_alias
    n_22_cov+=n_alias
    n_12_cov+=n_alias
#move as desired...
    n_alias[:tl,tl:]=0.
    n_alias[tl:,:tl]=0.    
    n_11+=n_alias
    n_22+=n_alias
    n_12+=n_alias

ninvy_11=dot(inv(n_11),y)
ninvy_22=dot(inv(n_22),y)
ytninvy_11=dot(yt,ninvy_11)
ytninvy_22=dot(yt,ninvy_22)


# for the regnoise to be negligible compared to signal, need C_l * factor <<1,
# so factor << 1/C_l
# tightest bound comes from largest C_l
# for pol, 1/.036=27
# for T, 1/1000=.0001

regulateytninvy=False
if regulateytninvy:
    ytninvy_11+=5.*identity(ytninvy_11.shape[0])
    ytninvy_22+=5.*identity(ytninvy_22.shape[0])


    # emprically a factor of 50 in front of both leads to reasonable spectra...
regulateytninvyT=False
if regulateytninvyT:
    ytninvy_11[::2,::2]+=.001*identity(ytninvy_11[1::2,1::2].shape[0])
    ytninvy_22[::2,::2]+=.001*identity(ytninvy_22[1::2,1::2].shape[0])

regulateytninvyP=False
if regulateytninvyP:
    ytninvy_11[1::2,1::2]+=10.*identity(ytninvy_11[1::2,1::2].shape[0])
    ytninvy_22[1::2,1::2]+=10.*identity(ytninvy_22[1::2,1::2].shape[0])

ytninvyinv_11=inv(ytninvy_11)
ytninvyinv_22=inv(ytninvy_22)

ytninvnninvy_11=dot(transpose(ninvy_11),dot(n_11_cov,ninvy_11))
ytninvnninvy_22=dot(transpose(ninvy_22),dot(n_22_cov,ninvy_22))
ytninvnninvy_12=dot(transpose(ninvy_11),dot(n_12_cov,ninvy_22))

nterm_11=dot(transpose(ytninvyinv_11),dot(ytninvnninvy_11,ytninvyinv_11))
nterm_22=dot(transpose(ytninvyinv_22),dot(ytninvnninvy_22,ytninvyinv_22))
nterm_12=dot(transpose(ytninvyinv_11),dot(ytninvnninvy_12,ytninvyinv_22))




# consider reshaping here to stop t data influencing pseudo-E etc...
# (seem to get better TE spectra if you don't bother funnily enough...)

#d_11=makecllong_reshaped(fidcldatfile)+ytninvyinv_11
#d_22=makecllong_reshaped(fidcldatfile)+ytninvyinv_22





#f_11=inv(d_11)
#f_22=inv(d_22)
#yn_11=dot(transpose(f_11),dot(nterm_11,f_11))
#yn_22=dot(transpose(f_22),dot(nterm_22,f_22))
#yn_12=dot(transpose(f_11),dot(nterm_12,f_22))
#yn_21=transpose(yn_12)
#y_1=dot(f_11,dot(ytninvyinv_11,dot(transpose(ninvy_11),d_1)))
#y_2=dot(f_22,dot(ytninvyinv_22,dot(transpose(ninvy_22),d_2)))


clforweighting=makecllong_reshaped(fidcldatfile)
#clforweighting=makecllong_regulate(fidcldatfile)

clwi=inv(clforweighting)

zz_11=dot(clwi, inv(clwi+ytninvy_11) )
zz_22=dot(clwi, inv(clwi+ytninvy_22) )

f_11=dot(zz_11, ytninvy_11)
f_22=dot(zz_22, ytninvy_22)

yn_11=dot(zz_11, dot (ytninvnninvy_11, transpose(zz_11) ) )
yn_22=dot(zz_22, dot (ytninvnninvy_22, transpose(zz_22) ) )
yn_12=dot(zz_11, dot (ytninvnninvy_12, transpose(zz_22) ) )
yn_21=transpose(yn_12)

y_1=dot(zz_11, dot(transpose(ninvy_11),d_1) )
y_2=dot(zz_22, dot(transpose(ninvy_22),d_2) )

ym_1=dot(zz_11, dot(transpose(ninvy_11), mean_11) )
ym_2=dot(zz_22, dot(transpose(ninvy_22), mean_22) )


def mul_rows(mat,d):
    mat2=mat.copy()
    for i in range(len(d)):
        mat2[i,:]*=d[i]
    return mat2

def mul_cols(mat,d):
    mat2=mat.copy()
    for i in range(len(d)):
        mat2[:,i]*=d[i]
    return mat2

def make_m(mat1, mat2, cld,mat3):
    return (dot(transpose(mat1),mul_rows(mat2,cld))+mat3)

# is the transpose needed?
def make_m_slow(mat1, clmat, mat2, mat3):
    return(dot(transpose(mat1),dot(clmat,mat2))+mat3)

def vecpart(p,v):
    pstart=p**2-lmin**2
    return v[2*pstart: 2*(pstart+2*p+1)].copy()

def sum_vv(v1,v2):
    means=zeros(4*(lmax-lmin+1))
    for l in range(lmin,lmax+1):
        s_ee=0.0
        s_eb=0.0
        s_be=0.0
        s_bb=0.0
        v1part=vecpart(l,v1)
        v2part=vecpart(l,v2)        
        for m in range(0,2*l+1):
            s_ee+=v1part[2*m]*v2part[2*m]
            s_eb+=v1part[2*m]*v2part[2*m+1]
            s_be+=v1part[2*m+1]*v2part[2*m]
            s_bb+=v1part[2*m+1]*v2part[2*m+1]
        means[4*(l-lmin)]=s_ee
        means[4*(l-lmin)+1]=s_eb
        means[4*(l-lmin)+2]=s_be
        means[4*(l-lmin)+3]=s_bb
    return means

def testmakedlfromalm(alms,lmin,lmax):
    dlvec=zeros(lmax-lmin+1)
    pos=0
    for l in range(lmin,lmax+1):
        dl=0.
        for m in range(0,2*l+1):
            dl+=alms[pos]**2
            pos+=1
        dl/=(2.*l+1)
        dlvec[l-lmin]=dl*l*(l+1)/2./pi
    return dlvec

def testmakedlfromxalm(alm1,alm2,lmin,lmax):
    dlvec=zeros(lmax-lmin+1)
    pos=0
    for l in range(lmin,lmax+1):
        dl=0.
        for m in range(0,2*l+1):
            dl+=alm1[pos]*alm2[pos]
            pos+=1
        dl/=(2.*l+1)
        dlvec[l-lmin]=dl*l*(l+1)/2./pi
    return dlvec

yp_11=sum_vv(y_1,y_1)
yp_22=sum_vv(y_2,y_2)
yp_12=sum_vv(y_1,y_2)

@jit(nopython=True)
def matpart(p,q,mat):
    pstart=p**2-lmin**2
    qstart=q**2-lmin**2
    return(mat[2*pstart:2*(pstart+2*p+1),2*qstart:2*(qstart+2*q+1)])

def sum_mat(mat):
    means=zeros(4*(lmax-lmin+1))
    for l in range(lmin,lmax+1):
        s_ee=0.0
        s_eb=0.0
        s_be=0.0
        s_bb=0.0
        mpart=matpart(l,l,mat)
        #        print mat.shape, mpart.shape
        for m in range(0,2*l+1):
            s_ee+=mpart[2*m,2*m]
            s_eb+=mpart[2*m,2*m+1]
            s_be+=mpart[2*m+1,2*m]
            s_bb+=mpart[2*m+1,2*m+1]
        means[4*(l-lmin)]=s_ee
        means[4*(l-lmin)+1]=s_eb
        means[4*(l-lmin)+2]=s_be
        means[4*(l-lmin)+3]=s_bb
    return means

@jit(nopython=True)
def pairtrace(m1, m2):
   return sum(m1*(m2.transpose()))

def indcheck():
    for a1 in range(0,2):
        for a2 in range(0,2):
            for b1 in range(0,2):
                for b2 in range(0,2):
                    a=2*a1+a2
                    b=2*b1+b2
                    c=2*a2+b1
                    d=2*b2+a1
                    e=2*a2+b2
                    f=2*b1+a1
                    #print a, b, ": ", a1,a2,",", b1,b2,": ", c, d, ", ", e, f
                    print(a, b, ": ", a1, a2, ",", b1, b2,": ", a2, b1, ",", b2, a1,  " + " , a2, b2, ", ", b1, a1,flush=True)

'''
def indcheck3():
    for a1 in range(0,2):
        for a2 in range(0,2):
            for b1 in range(0,2):
                for b2 in range(0,2):
                    for c1 in range(0,2):
                        for c2 in range(0,2):
                            a=2*a1+a2
                            b=2*b1+b2
                            c=2*c1+c2
                            print a, b, c,": ",\
                            a1, a2, ",", b1, b2,",", c1, c2, ": ",\
                            a2, b1, ",", b2, c1, ",", c2, a1, " + "

'''
                    
def k3contrib(p,q,r, m23,m24,m25,m26, m31,m35,m36, m41,m45,m46, m51,m53,m54, m61,m63,m64):
    z=zeros((4,4,4))
    for m1 in range(0,2*p+1):
        for m2 in range(0,2*q+1):
            for m3 in range(0,2*r+1):
                for a1 in range(0,2):
                    for a2 in range(0,2):
                        for b1 in range(0,2):
                            for b2 in range(0,2):
                                for c1 in range(0,2):
                                    for c2 in range(0,2):
                                        a=2*a1+a2
                                        b=2*b1+b2
                                        c=2*c1+c2
                                        z[a,b,c]+=(
                                            + m23[2*m1+a2, 2*m2+b1] * m45[2*m2+b2, 2*m3+c1] * m61[2*m3+c2, 2*m1+a1]
                                            + m23[2*m1+a2, 2*m2+b1] * m46[2*m2+b2, 2*m3+c2] * m51[2*m3+c1, 2*m1+a1]
                                            + m24[2*m1+a2, 2*m2+b2] * m35[2*m2+b1, 2*m3+c1] * m61[2*m3+c2, 2*m1+a1]
                                            + m24[2*m1+a2, 2*m2+b2] * m36[2*m2+b1, 2*m3+c2] * m51[2*m3+c1, 2*m1+a1]
                                            + m25[2*m1+a2, 2*m3+c1] * m63[2*m3+c2, 2*m2+b1] * m41[2*m2+b2, 2*m1+a1]
                                            + m25[2*m1+a2, 2*m3+c1] * m64[2*m3+c2, 2*m2+b2] * m31[2*m2+b1, 2*m1+a1]
                                            + m26[2*m1+a2, 2*m3+c2] * m53[2*m3+c1, 2*m2+b1] * m41[2*m2+b2, 2*m1+a1]
                                            + m26[2*m1+a2, 2*m3+c2] * m54[2*m3+c1, 2*m2+b2] * m31[2*m2+b1, 2*m1+a1]
                                            )
    return z

def kappa3_12(mat11,mat12,mat22):
    mat21=transpose(mat12).copy()
    k3=zeros((4*(lmax-lmin+1),4*(lmax-lmin+1),4*(lmax-lmin+1)))
    for p in range(lmin,lmax+1):
        for q in range(lmin,lmax+1):
            for r in range(lmin,lmax+1):  
                m23=matpart(p,q,mat21)
                m24=matpart(p,q,mat22)
                m25=matpart(p,r,mat21)
                m26=matpart(p,r,mat22)
                m31=matpart(q,p,mat11)
                m35=matpart(q,r,mat11)
                m36=matpart(q,r,mat12)
                m41=matpart(q,p,mat21)
                m45=matpart(q,r,mat21)
                m46=matpart(q,r,mat22)
                m51=matpart(r,p,mat11)
                m53=matpart(r,q,mat11)
                m54=matpart(r,q,mat12)
                m61=matpart(r,p,mat21)
                m63=matpart(r,q,mat21)
                m64=matpart(r,q,mat22)
                k3[4*(p-lmin):4*(p+1-lmin),4*(q-lmin):4*(q+1-lmin),4*(r-lmin):4*(r+1-lmin)] = (
                    k3contrib(p,q,r,m23,m24,m25,m26, m31 ,m35,m36, m41,m45,m46, m51,m53,m54, m61,m63,m64)
                    )
                print(p,q,r,flush=True)
    return k3

# (probably) working; do not edit...
def kappa4_12(mat11,mat12,mat22):
    mat21=transpose(mat12).copy()
    mats=zeros((2,2,mat11.shape[0],mat11.shape[1]))
    mats[0,0,:,:]=mat11
    mats[0,1,:,:]=mat12
    mats[1,0,:,:]=mat21
    mats[1,1,:,:]=mat22
#
    k4=zeros((4*(lmax-lmin+1),4*(lmax-lmin+1),4*(lmax-lmin+1),4*(lmax-lmin+1)))
    outlen=2*((lmax+1)**2-lmin**2)
    # l, inside freqs, inside t/e, outside freqs, outside t/e by ::2 or 1::2  
    m=zeros((lmax-lmin+1,2,2,2,2,2,2,outlen,outlen))
    for p in range(lmin,lmax+1):
        for i in itertools.product([0,1],repeat=6):
            m1=mats[i[4],i[0],:,i[2]+2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2].copy()
            m2=mats[i[1],i[5],i[3]+2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:].copy()
            m[p-lmin,i[0],i[1],i[2],i[3],i[4],i[5],:,:]=dot(m1,m2)
        print(p,flush=True)
#
    for p in range(lmin,lmax+1):
        for q in range(p,lmax+1):
            for r in range(q,lmax+1):
                for s in range(r,lmax+1):
                    z=zeros((4,4,4,4))
                    count=0
                    for inds in itertools.product([0,1],repeat=8):
                        count=0
                        biginds=tuple(2*array(inds[::2])+array(inds[1::2]))
                        for changingmatinds in itertools.permutations(zip([q,r,s],range(2,8,2),range(3,8,2))):
                            for perms in itertools.product([1,-1],repeat=3):
                                terms=[(p,0,1)]+list(changingmatinds)
                                fullperms=[1]+list(perms)
                                freqinds=list()
                                lterms=list()
                                for i in range(len(terms)):
                                    freqinds+=list(array(terms[i][1:3])[::fullperms[i]])
                                    #print terms[i][0],array(terms[i][1:3])[::fullperms[i]],terms[i][0],' ',
                                    lterms+=[terms[i][0]]
                                    lterms+=[terms[i][0]]
                                fis=freqinds[-1:]+freqinds[:-1]
                                fls=lterms[-1:]+lterms[:-1]
                                #print fls[0],fls[3],fls[1]-lmin,fis[1]%2,fis[2]%2,biginds[0],biginds[1],fis[0]%2,fis[3]%2,biginds[2],biginds[3]
                                mp1=matpart(fls[0],fls[3],m[fls[1]-lmin,fis[1]%2,fis[2]%2,inds[0],inds[1],fis[0]%2,fis[3]%2,:,:])[inds[2]::2,inds[3]::2]
                                mp2=matpart(fls[4],fls[7],m[fls[5]-lmin,fis[5]%2,fis[6]%2,inds[4],inds[5],fis[4]%2,fis[7]%2,:,:])[inds[6]::2,inds[7]::2]
                                z[biginds] += pairtrace(mp1,mp2)
                                count+=1
                    k4[4*(p-lmin):4*(p+1-lmin),4*(q-lmin):4*(q+1-lmin),4*(r-lmin):4*(r+1-lmin),4*(s-lmin):4*(s+1-lmin)]=z     
                    print(p,q,r,s,':',count,flush=True)
#
    for ells in (itertools.product(range(lmin,lmax+1),repeat=4)):
        p,q,r,s=list(ells)
        sargs=argsort(ells)
        psort,qsort,rsort,ssort=sorted(list(ells))
        zz1 = k4[4*(psort-lmin):4*(psort+1-lmin),4*(qsort-lmin):4*(qsort+1-lmin),4*(rsort-lmin):4*(rsort+1-lmin),4*(ssort-lmin):4*(ssort+1-lmin)].copy()
        zz2 = zeros_like(zz1)
        for inds in itertools.product(range(0,4),repeat=4):
            tinds=list(inds)
            sinds=array(tinds)[sargs]
            zz2[tuple(tinds)] = zz1[tuple(sinds)]
        k4[4*(p-lmin):4*(p+1-lmin),4*(q-lmin):4*(q+1-lmin),4*(r-lmin):4*(r+1-lmin),4*(s-lmin):4*(s+1-lmin)] = zz2
#
    return k4

def kappa4_12_fast(mat11,mat12,mat22):
    mat21=transpose(mat12).copy()
    mats=zeros((2,2,mat11.shape[0],mat11.shape[1]))
    mats[0,0,:,:]=mat11
    mats[0,1,:,:]=mat12
    mats[1,0,:,:]=mat21
    mats[1,1,:,:]=mat22
#
    count=0
#   
    k4=zeros((4*(lmax-lmin+1),4*(lmax-lmin+1),4*(lmax-lmin+1),4*(lmax-lmin+1)))
    outlen=2*((lmax+1)**2-lmin**2)
    # l, inside freqs, inside t/e, outside freqs, outside t/e by ::2 or 1::2  
    m=zeros((lmax-lmin+1,2,2,2,2,2,2,outlen,outlen))
    for p in range(lmin,lmax+1):
        for i in itertools.product([0,1],repeat=6):
            m1=mats[i[4],i[0],:,i[2]+2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2].copy()
            m2=mats[i[1],i[5],i[3]+2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:].copy()
            m[p-lmin,i[0],i[1],i[2],i[3],i[4],i[5],:,:]=dot(m1,m2)
        print(p,flush=True)
#
    for p in range(lmin,lmax+1):
        for q in range(p,lmax+1):
            for r in range(q,lmax+1):
                for s in range(r,lmax+1):
                    #count+=1
                    #continue
                    z=zeros((4,4,4,4))
                        #                        count=0
                    for changingmatinds in itertools.permutations(zip([q,r,s],range(2,8,2),range(3,8,2))):
                        for perms in itertools.product([1,-1],repeat=3):
                            terms=[(p,0,1)]+list(changingmatinds)
                            fullperms=[1]+list(perms)
                            freqinds=list()
                            lterms=list()
                            for i in range(len(terms)):
                                freqinds+=list(array(terms[i][1:3])[::fullperms[i]])
                                #freqinds+=terms[i][1:3]
                                #print terms[i][0],array(terms[i][1:3])[::fullperms[i]],terms[i][0],' ',
                                lterms+=[terms[i][0]]
                                lterms+=[terms[i][0]]
                            fis=freqinds[-1:]+freqinds[:-1]
                            fls=lterms[-1:]+lterms[:-1]
                            for inds1 in itertools.product([0,1],repeat=4):
                                mp1full=matpart(fls[0],fls[3],m[fls[1]-lmin,fis[1]%2,fis[2]%2,inds1[0],inds1[1],fis[0]%2,fis[3]%2,:,:])
                                mp2full=matpart(fls[4],fls[7],m[fls[5]-lmin,fis[5]%2,fis[6]%2,inds1[2],inds1[3],fis[4]%2,fis[7]%2,:,:])
                                for inds2 in itertools.product([0,1],repeat=4):
                                #count+=1
                                #biginds=tuple(2*array(inds[::2])+array(inds[1::2]))
                                #print fls[0],fls[3],fls[1]-lmin,fis[1]%2,fis[2]%2,biginds[0],biginds[1],fis[0]%2,fis[3]%2,biginds[2],biginds[3]
                                    mp1=mp1full[inds2[0]::2,inds2[1]::2]
                                    mp2=mp2full[inds2[2]::2,inds2[3]::2]
                                    biginds=(2*inds1[0]+inds1[1],2*inds2[0]+inds2[1],2*inds1[2]+inds1[3],2*inds2[2]+inds2[3])
                                    z[biginds] += pairtrace(mp1,mp2)
                                #count+=1
                    k4[4*(p-lmin):4*(p+1-lmin),4*(q-lmin):4*(q+1-lmin),4*(r-lmin):4*(r+1-lmin),4*(s-lmin):4*(s+1-lmin)]=z     
                    print(p,q,r,s,':',count,flush=True)
    print (count)
#
    for ells in (itertools.product(range(lmin,lmax+1),repeat=4)):
        p,q,r,s=list(ells)
        sargs=argsort(ells)
        psort,qsort,rsort,ssort=sorted(list(ells))
        zz1 = k4[4*(psort-lmin):4*(psort+1-lmin),4*(qsort-lmin):4*(qsort+1-lmin),4*(rsort-lmin):4*(rsort+1-lmin),4*(ssort-lmin):4*(ssort+1-lmin)].copy()
        zz2 = zeros_like(zz1)
        for inds in itertools.product(range(0,4),repeat=4):
            tinds=list(inds)
            #sinds=tinds
            sinds2=list()
            for i in sargs:
                sinds2.append(tinds[i])
            #sinds=array(tinds)[sargs]
            zz2[tuple(tinds)] = zz1[tuple(sinds2)]
            k4[4*(p-lmin):4*(p+1-lmin),4*(q-lmin):4*(q+1-lmin),4*(r-lmin):4*(r+1-lmin),4*(s-lmin):4*(s+1-lmin)] = zz2
        #
    return k4



# mess with this one...
def kappa4_12_test(mat11,mat12,mat22):
    mat21=transpose(mat12).copy()
    mats=zeros((2,2,mat11.shape[0],mat11.shape[1]))
    mats[0,0,:,:]=mat11
    mats[0,1,:,:]=mat12
    mats[1,0,:,:]=mat21
    mats[1,1,:,:]=mat22
#
    k4=zeros((4*(lmax-lmin+1),4*(lmax-lmin+1),4*(lmax-lmin+1),4*(lmax-lmin+1)))
    outlen=2*((lmax+1)**2-lmin**2)
    # l, inside freqs, inside t/e, outside freqs, outside t/e by ::2 or 1::2  
    m=zeros((lmax-lmin+1,2,2,2,2,2,2,outlen,outlen))
    for p in range(lmin,lmax+1):
        for i in itertools.product([0,1],repeat=6):
            m1=mats[i[4],i[0],:,i[2]+2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2].copy()
            m2=mats[i[1],i[5],i[3]+2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:].copy()
            m[p-lmin,i[0],i[1],i[2],i[3],i[4],i[5],:,:]=dot(m1,m2)
        print (p)
#
    for p in range(lmin,lmax+1):
        for q in range(p,lmax+1):
            for r in range(q,lmax+1):
                for s in range(r,lmax+1):
                    z=zeros((4,4,4,4))
                    count=0
                    for inds in itertools.product([0,1],repeat=8):
                        #                        count=0
                        biginds=tuple(2*array(inds[::2])+array(inds[1::2]))
                        for changingmatinds in itertools.permutations(zip([q,r,s],range(2,8,2),range(3,8,2))):
                            for perms in itertools.product([1,-1],repeat=3):
                                terms=[(p,0,1)]+list(changingmatinds)
                                fullperms=[1]+list(perms)
                                freqinds=list()
                                lterms=list()
                                for i in range(len(terms)):
                                    freqinds+=list(array(terms[i][1:3])[::fullperms[i]])
                                    #print terms[i][0],array(terms[i][1:3])[::fullperms[i]],terms[i][0],' ',
                                    lterms+=[terms[i][0]]
                                    lterms+=[terms[i][0]]
                                fis=freqinds[-1:]+freqinds[:-1]
                                fls=lterms[-1:]+lterms[:-1]
                                #print fls[0],fls[3],fls[1]-lmin,fis[1]%2,fis[2]%2,biginds[0],biginds[1],fis[0]%2,fis[3]%2,biginds[2],biginds[3]
                                #                                mp1=matpart(fls[0],fls[3],m[fls[1]-lmin,fis[1]%2,fis[2]%2,inds[0],inds[1],fis[0]%2,fis[3]%2,:,:])[inds[2]::2,inds[3]::2]
                                #                                mp2=matpart(fls[4],fls[7],m[fls[5]-lmin,fis[5]%2,fis[6]%2,inds[4],inds[5],fis[4]%2,fis[7]%2,:,:])[inds[6]::2,inds[7]::2]
                                #                                z[biginds] += pairtrace(mp1,mp2)
                                count+=1
                                #                    k4[4*(p-lmin):4*(p+1-lmin),4*(q-lmin):4*(q+1-lmin),4*(r-lmin):4*(r+1-lmin),4*(s-lmin):4*(s+1-lmin)]=z     
                    print (p,q,r,s,':',count)
#
    for ells in (itertools.product(range(lmin,lmax+1),repeat=4)):
        p,q,r,s=list(ells)
        sargs=argsort(ells)
        psort,qsort,rsort,ssort=sorted(list(ells))
        zz1 = k4[4*(psort-lmin):4*(psort+1-lmin),4*(qsort-lmin):4*(qsort+1-lmin),4*(rsort-lmin):4*(rsort+1-lmin),4*(ssort-lmin):4*(ssort+1-lmin)].copy()
        zz2 = zeros_like(zz1)
        for inds in itertools.product(range(0,4),repeat=4):
            tinds=list(inds)
            sinds=array(tinds)[sargs]
            zz2[tuple(tinds)] = zz1[tuple(sinds)]
        k4[4*(p-lmin):4*(p+1-lmin),4*(q-lmin):4*(q+1-lmin),4*(r-lmin):4*(r+1-lmin),4*(s-lmin):4*(s+1-lmin)] = zz2
#
    return k4



'''
for changingmatinds in itertools.permutations(zip([13,17,23],range(2,8,2),range(3,8,2))):
    print changingmatinds
    for perms in itertools.product([1,-1],repeat=3):
        terms=[(11,0,1)]+list(changingmatinds)
        fullperms=[1]+list(perms)
        freqinds=list()
        lterms=list()
        print terms
        for i in range(len(terms)):
            freqinds+=list(array(terms[i][1:3])[::fullperms[i]])
            print terms[i][0],array(terms[i][1:3])[::fullperms[i]],terms[i][0],' ',
            lterms+=[terms[i][0]]
            lterms+=[terms[i][0]]
        finalfreqinds=freqinds[-1:]+freqinds[:-1]
        finallterms=lterms[-1:]+lterms[:-1]
        print finalfreqinds, finallterms
        print
'''

        
def kappa3_12_fast(mat11,mat12,mat22):
    mat21=transpose(mat12).copy()
    k3=zeros((4*(lmax-lmin+1),4*(lmax-lmin+1),4*(lmax-lmin+1)))
    outlen=2*((lmax+1)**2-lmin**2)
    m_11_21=zeros((2,2,outlen,outlen))
    m_11_22=zeros((2,2,outlen,outlen))
    m_21_21=zeros((2,2,outlen,outlen))
    m_21_22=zeros((2,2,outlen,outlen))
#
    for p in range(lmin,lmax+1):
        m_11_21[0,0,:,:]=dot(mat11[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_11_21[0,1,:,:]=dot(mat11[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])
        m_11_21[1,0,:,:]=dot(mat11[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_11_21[1,1,:,:]=dot(mat11[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])
#
        m_11_22[0,0,:,:]=dot(mat11[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_11_22[0,1,:,:]=dot(mat11[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])
        m_11_22[1,0,:,:]=dot(mat11[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_11_22[1,1,:,:]=dot(mat11[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])
#
        m_21_21[0,0,:,:]=dot(mat21[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_21_21[0,1,:,:]=dot(mat21[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])
        m_21_21[1,0,:,:]=dot(mat21[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_21_21[1,1,:,:]=dot(mat21[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])
#
        m_21_22[0,0,:,:]=dot(mat21[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_21_22[0,1,:,:]=dot(mat21[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])
        m_21_22[1,0,:,:]=dot(mat21[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_21_22[1,1,:,:]=dot(mat21[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])
#
        for q in range(p,lmax+1):
            for r in range(q,lmax+1):
                z=zeros((4,4,4))
                for inds in itertools.product([0,1],repeat=6):
                    biginds=tuple(2*array(inds[::2])+array(inds[1::2]))
                    z[biginds] += (
                        + pairtrace(matpart(r,q,m_21_21[inds[0],inds[1],:,:])[inds[5]::2,inds[2]::2],matpart(q,r,mat21)[inds[3]::2,inds[4]::2])
                        + pairtrace(matpart(r,q,m_11_21[inds[0],inds[1],:,:])[inds[4]::2,inds[2]::2],matpart(q,r,mat22)[inds[3]::2,inds[5]::2])
                        + pairtrace(matpart(r,q,m_21_22[inds[0],inds[1],:,:])[inds[5]::2,inds[3]::2],matpart(q,r,mat11)[inds[2]::2,inds[4]::2])
                        + pairtrace(matpart(r,q,m_11_22[inds[0],inds[1],:,:])[inds[4]::2,inds[3]::2],matpart(q,r,mat12)[inds[2]::2,inds[5]::2])
                        + pairtrace(matpart(q,r,m_21_21[inds[0],inds[1],:,:])[inds[3]::2,inds[4]::2],matpart(r,q,mat21)[inds[5]::2,inds[2]::2])
                        + pairtrace(matpart(q,r,m_11_21[inds[0],inds[1],:,:])[inds[2]::2,inds[4]::2],matpart(r,q,mat22)[inds[5]::2,inds[3]::2])
                        + pairtrace(matpart(q,r,m_21_22[inds[0],inds[1],:,:])[inds[3]::2,inds[5]::2],matpart(r,q,mat11)[inds[4]::2,inds[2]::2])
                        + pairtrace(matpart(q,r,m_11_22[inds[0],inds[1],:,:])[inds[2]::2,inds[5]::2],matpart(r,q,mat12)[inds[4]::2,inds[3]::2])
                        )
                k3[4*(p-lmin):4*(p+1-lmin),4*(q-lmin):4*(q+1-lmin),4*(r-lmin):4*(r+1-lmin)]=z     
                print (p,q,r)
#
    for ells in (itertools.product(range(lmin,lmax+1),repeat=3)):
        p,q,r=list(ells)
        sargs=argsort(ells)
        psort,qsort,rsort=sorted(list(ells))
        zz1 = k3[4*(psort-lmin):4*(psort+1-lmin),4*(qsort-lmin):4*(qsort+1-lmin),4*(rsort-lmin):4*(rsort+1-lmin)].copy()
        zz2 = zeros_like(zz1)
        for inds in itertools.product(range(0,4),repeat=3):
            tinds=list(inds)
            sinds=array(tinds)[sargs]
            zz2[tuple(tinds)] = zz1[tuple(sinds)]
        k3[4*(p-lmin):4*(p+1-lmin),4*(q-lmin):4*(q+1-lmin),4*(r-lmin):4*(r+1-lmin)] = zz2
#        
    return k3


'''
    for p in range(lmin,lmax+1):
        for q in range(lmin,lmax+1):
            for r in range(lmin,lmax+1):
                psort,qsort,rsort=sorted(list((p,q,r)))
                zz1=k3[4*(psort-lmin):4*(psort+1-lmin),4*(qsort-lmin):4*(qsort+1-lmin),4*(rsort-lmin):4*(rsort+1-lmin)].copy()

                k3[4*(p-lmin):4*(p+1-lmin),4*(q-lmin):4*(q+1-lmin),4*(r-lmin):4*(r+1-lmin)] = (
                    k3[4*(psort-lmin):4*(psort+1-lmin),4*(qsort-lmin):4*(qsort+1-lmin),4*(rsort-lmin):4*(rsort+1-lmin)]
                    )
'''

def dec_kap1(ci,k1):
    n=k1.shape[0]
    k1t=k1.copy()
    dk1=zeros_like(k1)
    for i in range(n):
        for s in range(n):
            dk1[i]+=ci[i,s]*k1t[s]
    return dk1

def dec_derivkap1(ci,k1):
    n=mu.shape[0]
    k1t=k1.copy()
    dk1=zeros_like(k1)
    for i in range(n):
        for x in range(n):
            for s in range(n):
                dk1[i,x]+=ci[i,s]*k1t[s,x]
    return dk1

def dec_kap2(ci,k2):
    n=k2.shape[0]
#
    k2t=k2.copy()
    dk2=zeros_like(k2)
    for p in range(n):
        for j in range(n):
            for s in range(n):
                dk2[p,j]+=ci[j,s]*k2t[p,s]
#
    k2t=dk2.copy()
    dk2=zeros_like(k2)
    for i in range(n):
        for j in range(n):
            for s in range(n):
                dk2[i,j]+=ci[i,s]*k2t[s,j]
#
    return dk2
    

def dec_derivkap2(ci,k2):
    n=k2.shape[0]
#
    k2t=k2.copy()
    dk2=zeros_like(k2)
    for p in range(n):
        for j in range(n):
            for x in range(n):
                for s in range(n):
                    dk2[p,j,x]+=ci[j,s]*k2t[p,s,x]
#
    k2t=dk2.copy()
    dk2=zeros_like(k2)
    for i in range(n):
        for j in range(n):
            for x in range(n):
                for s in range(n):
                    dk2[i,j,x]+=ci[i,s]*k2t[s,j,x]
#
    return dk2
    



def dec_kap3(ci,k3):
    n=k3.shape[0]
    k3t=k3.copy()
    dk3=zeros_like(k3)
    for p in range(n):
        for q in range(n):
            for k in range(n):
                for s in range(n):
                    dk3[p,q,k]+=ci[k,s]*k3t[p,q,s]
#
    k3t=dk3.copy()
    dk3=zeros_like(k3)
    for p in range(n):
        for j in range(n):
            for k in range(n):
                for s in range(n):
                    dk3[p,j,k]+=ci[j,s]*k3t[p,s,k]
#    
    k3t=dk3.copy()
    dk3=zeros_like(k3)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for s in range(n):
                    dk3[i,j,k]+=ci[i,s]*k3t[s,j,k]
#
    return dk3

def dec_kap4_old(ci,k4):
    n=k4.shape[0]
#
    k4t=k4.copy()
    dk4=zeros_like(k4)
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for k in range(n):
                    for s in range(n):
                        dk4[p,q,r,k]+=ci[k,s]*k4t[p,q,r,s]
#
    k4t=dk4.copy()
    dk4=zeros_like(k4)
    for p in range(n):
        for q in range(n):
            for j in range(n):
                for k in range(n):
                    for s in range(n):
                        dk4[p,q,j,k]+=ci[j,s]*k4t[p,q,s,k]
    #
    k4t=dk4.copy()
    dk4=zeros_like(k4)
    for p in range(n):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for s in range(n):
                        dk4[p,i,j,k]+=ci[i,s]*k4t[p,s,j,k]
#
    k4t=dk4.copy()
    dk4=zeros_like(k4)
    for h in range(n):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for s in range(n):
                        dk4[h,i,j,k]+=ci[h,s]*k4t[s,i,j,k]
#
    return dk4

def dec_kap4(ci,k4):
    dk4=k4.copy()
    dk4=tensordot(ci,dk4,axes=([1],[3]))
    dk4=transpose(dk4,(1,2,3,0))
    dk4=tensordot(ci,dk4,axes=([1],[2]))
    dk4=transpose(dk4,(1,2,0,3))
    dk4=tensordot(ci,dk4,axes=([1],[1]))
    dk4=transpose(dk4,(1,0,2,3))
    dk4=tensordot(ci,dk4,axes=([1],[0]))
    return dk4

def dec_diag_kap3(ci,k3):
    n=k3.shape[0]
    k3t=zeros_like(k3)
    for l in range(lmin,lmax+1):
        k3t[4*(l-lmin):4*(l+1-lmin),4*(l-lmin):4*(l+1-lmin),4*(l-lmin):4*(l+1-lmin)] = (
            k3[4*(l-lmin):4*(l+1-lmin),4*(l-lmin):4*(l+1-lmin),4*(l-lmin):4*(l+1-lmin)]
        )
#    
    dk3=zeros_like(k3)
    for p in range(n):
        for q in range(n):
            for k in range(n):
                for s in range(n):
                    dk3[p,q,k]+=ci[k,s]*k3t[p,q,s]
#    
    k3t=dk3.copy()
    dk3=zeros_like(k3)
    for p in range(n):
        for j in range(n):
            for k in range(n):
                for s in range(n):
                    dk3[p,j,k]+=ci[j,s]*k3t[p,s,k]
#    
    k3t=dk3.copy()
    dk3=zeros_like(k3)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for s in range(n):
                    dk3[i,j,k]+=ci[i,s]*k3t[s,j,k]
#
    return dk3
    
#Nb squashes should only be done before decoupling!
def squash_mat(c):
    cov=zeros((3*(lmax-lmin+1),3*(lmax-lmin+1)))
    for p in range(0,(lmax-lmin+1)):
        for q in range(0,(lmax-lmin+1)):
            cov[3*p,3*q]=c[4*p,4*q]
            cov[3*p,3*q+1]=c[4*p,4*q+1]+c[4*p,4*q+2]
            cov[3*p,3*q+2]=c[4*p,4*q+3]
            cov[3*p+1,3*q]=c[4*p+1,4*q]+c[4*p+2,4*q]
            cov[3*p+1,3*q+1]=c[4*p+1,4*q+1]+c[4*p+1,4*q+2]+c[4*p+2,4*q+1]+c[4*p+2,4*q+2]
            cov[3*p+1,3*q+2]=c[4*p+1,4*q+3]+c[4*p+2,4*q+3]
            cov[3*p+2,3*q]=c[4*p+3,4*q]
            cov[3*p+2,3*q+1]=c[4*p+3,4*q+1]+c[4*p+3,4*q+2]
            cov[3*p+2,3*q+2]=c[4*p+3,4*q+3]
    return cov

def squash_vec(v):
    y=zeros(3*(lmax-lmin+1))
    for p in range(0,(lmax-lmin+1)):
        y[3*p]=v[4*p]
        y[3*p+1]=v[4*p+1]+v[4*p+2]
        y[3*p+2]=v[4*p+3]
    return y

def test_deck2(ci,cls):
    cllong=makecllong_from_cls(cls)
    cov_11=make_m_slow(f_11, cllong, f_11, yn_11)
    cov_22=make_m_slow(f_22, cllong, f_22, yn_22)
    cov_12=make_m_slow(f_11, cllong, f_22, yn_12)
    return dec_kap2(ci,symcov12_short(cov_11,cov_12,cov_22))

def test_deck2_off(ci,cls):
    cllong=makecllong_from_cls(cls)
    cov_11=make_m_slow(f_11, cllong, f_11, yn_11)
    cov_22=make_m_slow(f_22, cllong, f_22, yn_22)
    cov_12=make_m_slow(f_11, cllong, f_22, yn_12)
    return dec_kap2(ci,symcov12_inc_offset(cov_11,cov_12,cov_22,ym_1,ym_2))

def test_k2(cls):
    cllong=makecllong_from_cls(cls)
    cov_11=make_m_slow(f_11, cllong, f_11, yn_11)
    cov_22=make_m_slow(f_22, cllong, f_22, yn_22)
    cov_12=make_m_slow(f_11, cllong, f_22, yn_12)
    return symcov12_short(cov_11,cov_12,cov_22)

def test_k3(cls):
    cllong=makecllong_from_cls(cls)
    cov_11=make_m_slow(f_11, cllong, f_11, yn_11)
    cov_22=make_m_slow(f_22, cllong, f_22, yn_22)
    cov_12=make_m_slow(f_11, cllong, f_22, yn_12)
    return symkappa3_12(cov_11,cov_12,cov_22)

def test_k4(cls):
    cllong=makecllong_from_cls(cls)
    cov_11=make_m_slow(f_11, cllong, f_11, yn_11)
    cov_22=make_m_slow(f_22, cllong, f_22, yn_22)
    cov_12=make_m_slow(f_11, cllong, f_22, yn_12)
    return symkappa4_12(cov_11,cov_12,cov_22)

def test_derivdeck2(ci,cls):
    cllong=makecllong_from_cls(cls)
    cov_11=make_m_slow(f_11, cllong, f_11, yn_11)
    cov_22=make_m_slow(f_22, cllong, f_22, yn_22)
    cov_12=make_m_slow(f_11, cllong, f_22, yn_12)
    return dec_derivkap2(ci,d_symcov12(cov_11,cov_12,cov_22,f_11,f_22))    


def test_derivk2(cls):
    cllong=makecllong_from_cls(cls)
    cov_11=make_m_slow(f_11, cllong, f_11, yn_11)
    cov_22=make_m_slow(f_22, cllong, f_22, yn_22)
    cov_12=make_m_slow(f_11, cllong, f_22, yn_12)
    return d_symcov12(cov_11,cov_12,cov_22,f_11,f_22)    

def test_kfunc2(kfunc,cls):
    cllong=makecllong_from_cls(cls)
    cov_11=make_m_slow(f_11, cllong, f_11, yn_11)
    cov_22=make_m_slow(f_22, cllong, f_22, yn_22)
    cov_12=make_m_slow(f_11, cllong, f_22, yn_12)
    return kfunc(cov_11,cov_12,cov_22, f_11, f_22)

def test_kfunc(kfunc,cls):
    cllong=makecllong_from_cls(cls)
    cov_11=make_m_slow(f_11, cllong, f_11, yn_11)
    cov_22=make_m_slow(f_22, cllong, f_22, yn_22)
    cov_12=make_m_slow(f_11, cllong, f_22, yn_12)
    t0=time.time()
    zz1=kfunc(cov_11,cov_12,cov_22)
    t1=time.time()
    print (t1-t0)
    return zz1


def fullskyanalyticEES(cldat,cls):
    return sum((ells+.5)*(cldat[2::3]/cls[2::3]+log(cls[2::3]/cldat[2::3])-1))

def grad_s_part(ci,cls):

    print ('generating pixel covmats...',flush=True)
    cllong=makecllong_from_cls(cls)

    cov_11=make_m_slow(f_11, cllong, f_11, yn_11)
    cov_22=make_m_slow(f_22, cllong, f_22, yn_22)
    cov_12=make_m_slow(f_11, cllong, f_22, yn_12)

    nells=lmaxlike-lminlike+1

    nmin=3*(lminlike-lmin)
    nmax=3*(lmaxlike-lmin+1)
    
    pars=cls[nmin:nmax].copy()

    n1=int(shape(pars)[0])
    n2=int(n1*(n1+1)/2)
    n=int(n1+n2)

    print ('generating cumulants...',flush=True)

    k1=pars.copy()
    k2=dec_kap2(ci,symcov12_short(cov_11,cov_12,cov_22))[nmin:nmax,nmin:nmax].copy()
    if do_linear:
        if use_offset_k2:
            k2=dec_kap2(ci,symcov12_inc_offset(cov_11,cov_12,cov_22,ym_1,ym_2))[nmin:nmax,nmin:nmax].copy()
    # testing returning the "linear" result...
        tmpdat=dec_kap1(ci,squash_vec(yp_12-n_12_p))[nmin:nmax].copy()
        dooutlier=False
        if dooutlier:
            sdk2=sqrt(diag(k2))
            for i in range(len(tmpdat[2::3])):
                if tmpdat[3*i+2]<-sdk2[3*i+2]:
                    tmpdat[3*i+2]=-sdk2[3*i+2]
        dodiagonalize=True
        if dodiagonalize:
            k2=diag(diag(k2))
        if doEEonly:
            return -dot(dot(transpose(tmpdat-k1)[2::3],inv(k2[2::3,2::3])),identity(int(n1/3)))
    #tmpdat=fidcls
        return -dot(dot(transpose(tmpdat-k1),inv(k2)),identity(n1))    
    #tmpdiff=zeros_like(cls)
    #tmpdiff[2::3]=(ells+.5)*(-fidcls[2::3]/k1[2::3]**2+1./k1[2::3])
    #return tmpdiff

    #testing zeroing the higher cumulants...
    k3=dec_kap3(ci,symkappa3_12(cov_11,cov_12,cov_22))[nmin:nmax,nmin:nmax,nmin:nmax].copy()
    #udk4=symkappa4_12(cov_11,cov_12,cov_22)
    #k4=dec_kap4(ci,udk4)
    udk4=symkappa4_12(cov_11,cov_12,cov_22)
    print ('deconvolving k4...',flush=True)

    k4=dec_kap4(ci,udk4)[nmin:nmax,nmin:nmax,nmin:nmax,nmin:nmax].copy()
    print ('done deconvolving k4...',flush=True)
    #k3=zeros((n1,n1,n1))
    #k4=zeros((n1,n1,n1,n1))

    #print k1
    #print k2
    #print k3
    #print k4
    dk1=identity(n1)
    print ('generating d_k2...',flush=True)
    uddk2=d_symcov12(cov_11,cov_12,cov_22,f_11,f_22)
    print ('deconvolving d_k2...',flush=True)
    dk2=dec_derivkap2(ci,uddk2)[nmin:nmax,nmin:nmax,nmin:nmax].copy()
    print ('done deconvolving d_k2...',flush=True)

    #print dk1
    #print dk2


    deriv=zeros((n,n1))
    diff=zeros(n)
    mat=zeros((n,n))

    print ('filling matrix...',flush=True)


    mat[:n1,:n1]=k2

    p=n1
    for p1 in range(0,n1):
        for p2 in range(p1,n1):
            for q in range(n1):
                mat[p,q] = k3[p1,p2,q]
            p+=1

    for p in range(0,n1):
        q=n1
        for q1 in range(0,n1):
            for q2 in range(q1,n1):
                mat[p,q] = (
                    + k3[p,q1,q2]
                    + k2[p,q1]*k1[q2] + k2[p,q2]*k1[q1]
                    )
                q+=1

    #zeroing k3 for xx-xx block...
    #k3=zeros((n1,n1,n1))            
    p=n1
    for p1 in range(0,n1):
        for p2 in range(p1,n1):
            q=n1
            for q1 in range(0,n1):
                for q2 in range(q1,n1):
                    mat[p,q] = (
                        + k4[p1,p2,q1,q2]
                        + k3[p1,p2,q1]*k1[q2] + k3[p1,p2,q2]*k1[q1]
                        + k2[p1,q1]*k2[q2,p2] + k2[p1,q2]*k2[q1,p2]
                        )
                    q+=1
            p+=1

            #print mat

    print ('inverting matrix...',flush=True)
    
    matinv=inv(mat)

    #print matinv

    #print dot(mat,matinv)

    print ('filling diff and deriv...',flush=True)

    tk1=dec_kap1(ci,squash_vec(yp_12-n_12_p))[nmin:nmax].copy()

    for p in range(0,n1):
        diff[p]=tk1[p]-pars[p]
    p=n1
    for p1 in range(0,n1):
        for p2 in range(p1,n1):
            diff[p]=tk1[p1]*tk1[p2]-k2[p1,p2]-k1[p1]*k1[p2]
            p+=1

    deriv[:n1,:n1]=dk1
    p=n1
    for p1 in range(0,n1):
        for p2 in range(p1,n1):
            deriv[p,:]=dk2[p1,p2,:]
            p+=1

            #assert(False)

    print ('computing likelihood...',flush=True)
            
    return -dot(dot(transpose(diff),matinv),deriv)

def grad_s(ci,cls):

    print ('generating pixel covmats...',flush=True)
    cllong=makecllong_from_cls(cls)

    cov_11=make_m_slow(f_11, cllong, f_11, yn_11)
    cov_22=make_m_slow(f_22, cllong, f_22, yn_22)
    cov_12=make_m_slow(f_11, cllong, f_22, yn_12)

    nells=lmax-lmin+1
    pars=cls.copy()

    n1=int(shape(pars)[0])
    n2=int(n1*(n1+1)/2)
    n=int(n1+n2)

    print ('generating cumulants...',flush=True)

    k1=cls.copy()
    k2=dec_kap2(ci,symcov12_short(cov_11,cov_12,cov_22))
    # testing returning the "linear" result...
    tmpdat=dec_kap1(ci,squash_vec(yp_12-n_12_p))
    #tmpdat=fidcls
    #return -dot(dot(transpose(tmpdat-k1),inv(k2)),identity(len(cls)))    
    #tmpdiff=zeros_like(cls)
    #tmpdiff[2::3]=(ells+.5)*(-fidcls[2::3]/k1[2::3]**2+1./k1[2::3])
    #return tmpdiff

    #testing zeroing the higher cumulants...
    k3=dec_kap3(ci,symkappa3_12(cov_11,cov_12,cov_22))
    #udk4=symkappa4_12(cov_11,cov_12,cov_22)
    #k4=dec_kap4(ci,udk4)
    udk4=symkappa4_12(cov_11,cov_12,cov_22)
    print ('deconvolving k4...',flush=True)


    k4=dec_kap4(ci,udk4)
    print ('done deconvolving k4...',flush=True)

    #k3=zeros((n1,n1,n1))
    #k4=zeros((n1,n1,n1,n1))

    #    print k1
    #print k2
    #print k3
    #print k4


    dk1=identity(n1)
    print ('generating d_k2...',flush=True)

    uddk2=d_symcov12(cov_11,cov_12,cov_22,f_11,f_22)
    print ('deconvolving d_k2...',flush=True)
    dk2=dec_derivkap2(ci,uddk2)
    print ('done deconvolving d_k2...',flush=True)


    #print dk1
    #print dk2


    deriv=zeros((n,n1))
    diff=zeros(n)
    mat=zeros((n,n))

    print ('filling matrix...',flush=True)

    mat[:n1,:n1]=k2

    p=n1
    for p1 in range(0,n1):
        for p2 in range(p1,n1):
            for q in range(n1):
                mat[p,q] = k3[p1,p2,q]
            p+=1

    for p in range(0,n1):
        q=n1
        for q1 in range(0,n1):
            for q2 in range(q1,n1):
                mat[p,q] = (
                    + k3[p,q1,q2]
                    + k2[p,q1]*k1[q2] + k2[p,q2]*k1[q1]
                    )
                q+=1

    #zeroing k3 for xx-xx block...
    #k3=zeros((n1,n1,n1))            
    p=n1
    for p1 in range(0,n1):
        for p2 in range(p1,n1):
            q=n1
            for q1 in range(0,n1):
                for q2 in range(q1,n1):
                    mat[p,q] = (
                        + k4[p1,p2,q1,q2]
                        + k3[p1,p2,q1]*k1[q2] + k3[p1,p2,q2]*k1[q1]
                        + k2[p1,q1]*k2[q2,p2] + k2[p1,q2]*k2[q1,p2]
                        )
                    q+=1
            p+=1

            #print mat

    print ('inverting matrix...',flush=True)
    
    matinv=inv(mat)

    #print matinv

    #print dot(mat,matinv)

    print ('filling diff and deriv...',flush=True)


    tk1=dec_kap1(ci,squash_vec(yp_12-n_12_p))

    for p in range(0,n1):
        diff[p]=tk1[p]-pars[p]
    p=n1
    for p1 in range(0,n1):
        for p2 in range(p1,n1):
            diff[p]=tk1[p1]*tk1[p2]-k2[p1,p2]-k1[p1]*k1[p2]
            p+=1

    deriv[:n1,:n1]=dk1
    p=n1
    for p1 in range(0,n1):
        for p2 in range(p1,n1):
            deriv[p,:]=dk2[p1,p2,:]
            p+=1

            #assert(False)

    print ('computing likelihood...',flush=True)

    return -dot(dot(transpose(diff),matinv),deriv)


    
# for now assuming M_AB^T=M_BA...
@jit(nopython=True)
def cov_12_34(mat23,mat41,mat24,mat31):
    cov=zeros((4*(lmax-lmin+1),4*(lmax-lmin+1)))
    for p in range(lmin,lmax+1):
        for q in range(lmin,lmax+1):
            m23=matpart(p,q,mat23)
            m41=matpart(q,p,mat41)
            m24=matpart(p,q,mat24)
            m31=matpart(q,p,mat31)
            for m in range(0,2*p+1):
                for n in range(0,2*q+1):
                    cov[4*(p-lmin)+0,4*(q-lmin)+0]+=m23[2*m+0,2*n+0]*m41[2*n+0,2*m+0]+m24[2*m+0,2*n+0]*m31[2*n+0,2*m+0]
                    cov[4*(p-lmin)+0,4*(q-lmin)+1]+=m23[2*m+0,2*n+0]*m41[2*n+1,2*m+0]+m24[2*m+0,2*n+1]*m31[2*n+0,2*m+0]
                    cov[4*(p-lmin)+0,4*(q-lmin)+2]+=m23[2*m+0,2*n+1]*m41[2*n+0,2*m+0]+m24[2*m+0,2*n+0]*m31[2*n+1,2*m+0]
                    cov[4*(p-lmin)+0,4*(q-lmin)+3]+=m23[2*m+0,2*n+1]*m41[2*n+1,2*m+0]+m24[2*m+0,2*n+1]*m31[2*n+1,2*m+0]
                    cov[4*(p-lmin)+1,4*(q-lmin)+0]+=m23[2*m+1,2*n+0]*m41[2*n+0,2*m+0]+m24[2*m+1,2*n+0]*m31[2*n+0,2*m+0]
                    cov[4*(p-lmin)+1,4*(q-lmin)+1]+=m23[2*m+1,2*n+0]*m41[2*n+1,2*m+0]+m24[2*m+1,2*n+1]*m31[2*n+0,2*m+0]
                    cov[4*(p-lmin)+1,4*(q-lmin)+2]+=m23[2*m+1,2*n+1]*m41[2*n+0,2*m+0]+m24[2*m+1,2*n+0]*m31[2*n+1,2*m+0]
                    cov[4*(p-lmin)+1,4*(q-lmin)+3]+=m23[2*m+1,2*n+1]*m41[2*n+1,2*m+0]+m24[2*m+1,2*n+1]*m31[2*n+1,2*m+0]
                    cov[4*(p-lmin)+2,4*(q-lmin)+0]+=m23[2*m+0,2*n+0]*m41[2*n+0,2*m+1]+m24[2*m+0,2*n+0]*m31[2*n+0,2*m+1]
                    cov[4*(p-lmin)+2,4*(q-lmin)+1]+=m23[2*m+0,2*n+0]*m41[2*n+1,2*m+1]+m24[2*m+0,2*n+1]*m31[2*n+0,2*m+1]
                    cov[4*(p-lmin)+2,4*(q-lmin)+2]+=m23[2*m+0,2*n+1]*m41[2*n+0,2*m+1]+m24[2*m+0,2*n+0]*m31[2*n+1,2*m+1]
                    cov[4*(p-lmin)+2,4*(q-lmin)+3]+=m23[2*m+0,2*n+1]*m41[2*n+1,2*m+1]+m24[2*m+0,2*n+1]*m31[2*n+1,2*m+1]
                    cov[4*(p-lmin)+3,4*(q-lmin)+0]+=m23[2*m+1,2*n+0]*m41[2*n+0,2*m+1]+m24[2*m+1,2*n+0]*m31[2*n+0,2*m+1]
                    cov[4*(p-lmin)+3,4*(q-lmin)+1]+=m23[2*m+1,2*n+0]*m41[2*n+1,2*m+1]+m24[2*m+1,2*n+1]*m31[2*n+0,2*m+1]
                    cov[4*(p-lmin)+3,4*(q-lmin)+2]+=m23[2*m+1,2*n+1]*m41[2*n+0,2*m+1]+m24[2*m+1,2*n+0]*m31[2*n+1,2*m+1]
                    cov[4*(p-lmin)+3,4*(q-lmin)+3]+=m23[2*m+1,2*n+1]*m41[2*n+1,2*m+1]+m24[2*m+1,2*n+1]*m31[2*n+1,2*m+1]
    return cov       

# WARNING: Provisionally adding a b_l'^2 to the coupling matrix...
# note an arbitrary tranpose in i[2] and i[3] to match cov formula...
def d_symmean(mat11_fid,mat22_fid):
    coup=zeros((3*(lmax-lmin+1),3*(lmax-lmin+1)))
    for l in itertools.product(range(lmin,lmax+1),repeat=2):
        m22=matpart(l[0],l[1],mat22_fid)
        m11=matpart(l[1],l[0],mat11_fid)
        zz1=zeros((3,3))
        for m in itertools.product(range(0,2*l[0]+1),range(0,2*l[1]+1)):
            for i in itertools.product([0,1],repeat=4):
                zz1[i[0]+i[1],i[2]+i[3]]+=m22[2*m[0]+i[1],2*m[1]+i[3]]*m11[2*m[1]+i[2],2*m[0]+i[0]]
        coup[3*(l[0]-lmin):3*(l[0]+1-lmin),3*(l[1]-lmin):3*(l[1]+1-lmin)]=zz1*b_l[l[1]]**2
    return coup

# warning: checking order of i[] terms...
def symcov12(mat11,mat12,mat22):
    mat21=transpose(mat12).copy()
    cov=zeros((3*(lmax-lmin+1),3*(lmax-lmin+1)))
    for l in itertools.product(range(lmin,lmax+1),repeat=2):
        zz1=zeros((3,3))
        m23=matpart(l[0],l[1],mat21)
        m41=matpart(l[1],l[0],mat21)
        m24=matpart(l[0],l[1],mat22)
        m31=matpart(l[1],l[0],mat11)
        for m in itertools.product(range(0,2*l[0]+1),range(0,2*l[1]+1)):
            for i in itertools.product([0,1],repeat=4):
                zz1[i[0]+i[1],i[2]+i[3]] += (
                    + m23[2*m[0]+i[1],2*m[1]+i[2]]*m41[2*m[1]+i[3],2*m[0]+i[0]]
                    + m24[2*m[0]+i[1],2*m[1]+i[3]]*m31[2*m[1]+i[2],2*m[0]+i[0]]
                    )
        cov[3*(l[0]-lmin):3*(l[0]+1-lmin),3*(l[1]-lmin):3*(l[1]+1-lmin)]=zz1
    return cov

def symcov12_inc_offset(mat11,mat12,mat22,ym1,ym2):
    mat21=transpose(mat12).copy()
    cov=zeros((3*(lmax-lmin+1),3*(lmax-lmin+1)))
    for l in itertools.product(range(lmin,lmax+1),repeat=2):
        zz1=zeros((3,3))
        v1=vecpart(l[0],ym1)
        v2=vecpart(l[0],ym2)
        v3=vecpart(l[1],ym1)
        v4=vecpart(l[1],ym2)
        m23=matpart(l[0],l[1],mat21)+outer(v2,v3)
        m41=matpart(l[1],l[0],mat21)+outer(v4,v1)
        m24=matpart(l[0],l[1],mat22)+outer(v2,v4)
        m31=matpart(l[1],l[0],mat11)+outer(v3,v1)
        for m in itertools.product(range(0,2*l[0]+1),range(0,2*l[1]+1)):
            for i in itertools.product([0,1],repeat=4):
                zz1[i[0]+i[1],i[2]+i[3]] += (
                    + m23[2*m[0]+i[1],2*m[1]+i[2]]*m41[2*m[1]+i[3],2*m[0]+i[0]]
                    + m24[2*m[0]+i[1],2*m[1]+i[3]]*m31[2*m[1]+i[2],2*m[0]+i[0]]
                    - 2.*v1[2*m[0]+i[0]]*v2[2*m[0]+i[1]]*v3[2*m[1]+i[2]]*v4[2*m[1]+i[3]]
                    )
        cov[3*(l[0]-lmin):3*(l[0]+1-lmin),3*(l[1]-lmin):3*(l[1]+1-lmin)]=zz1
    return cov



testcls=getclsmulti(tau_fid_file,(1,4,2))
testclsp=testcls.copy()
testclsm=testcls.copy()
testclsp[0]+=1.
testclsm[0]-=1.

def shiftcls(cls,q,x):
    newcls=cls.copy()
    newcls[q]+=x
    return newcls
    

def symcov12_short(mat11,mat12,mat22):
    mat21=transpose(mat12).copy()
    cov=zeros((3*(lmax-lmin+1),3*(lmax-lmin+1)))
    for l in itertools.product(range(lmin,lmax+1),repeat=2):
        zz1=zeros((3,3))
        m23=matpart(l[0],l[1],mat21)
        m41=matpart(l[1],l[0],mat21)
        m24=matpart(l[0],l[1],mat22)
        m31=matpart(l[1],l[0],mat11)
        for i in itertools.product([0,1],repeat=4):
            zz1[i[0]+i[1],i[2]+i[3]] += (
                + pairtrace(m23[i[1]::2,i[2]::2],m41[i[3]::2,i[0]::2])
                + pairtrace(m24[i[1]::2,i[3]::2],m31[i[2]::2,i[0]::2])
                )
        cov[3*(l[0]-lmin):3*(l[0]+1-lmin),3*(l[1]-lmin):3*(l[1]+1-lmin)]=zz1
    return cov

def symkappa3_12(mat11,mat12,mat22):
    mat21=transpose(mat12).copy()
    k3=zeros((3*(lmax-lmin+1),3*(lmax-lmin+1),3*(lmax-lmin+1)))
    outlen=2*((lmax+1)**2-lmin**2)
    m_11_21=zeros((2,2,outlen,outlen))
    m_11_22=zeros((2,2,outlen,outlen))
    m_21_21=zeros((2,2,outlen,outlen))
    m_21_22=zeros((2,2,outlen,outlen))

    for p in range(lmin,lmax+1):
        m_11_21[0,0,:,:]=dot(mat11[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_11_21[0,1,:,:]=dot(mat11[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])
        m_11_21[1,0,:,:]=dot(mat11[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_11_21[1,1,:,:]=dot(mat11[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])

        m_11_22[0,0,:,:]=dot(mat11[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_11_22[0,1,:,:]=dot(mat11[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])
        m_11_22[1,0,:,:]=dot(mat11[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_11_22[1,1,:,:]=dot(mat11[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])

        m_21_21[0,0,:,:]=dot(mat21[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_21_21[0,1,:,:]=dot(mat21[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])
        m_21_21[1,0,:,:]=dot(mat21[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_21_21[1,1,:,:]=dot(mat21[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat21[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])

        m_21_22[0,0,:,:]=dot(mat21[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_21_22[0,1,:,:]=dot(mat21[:,2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])
        m_21_22[1,0,:,:]=dot(mat21[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:])
        m_21_22[1,1,:,:]=dot(mat21[:,2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2],mat22[2*(p**2-lmin**2)+1:2*(2*p+1+p**2-lmin**2):2,:])

        for q in range(p,lmax+1):
            for r in range(q,lmax+1):
                z=zeros((3,3,3))
                for inds in itertools.product([0,1],repeat=6):
                    biginds=tuple(array(inds[::2])+array(inds[1::2]))
                    z[biginds] += (
                        + pairtrace(matpart(r,q,m_21_21[inds[0],inds[1],:,:])[inds[5]::2,inds[2]::2],matpart(q,r,mat21)[inds[3]::2,inds[4]::2])
                        + pairtrace(matpart(r,q,m_11_21[inds[0],inds[1],:,:])[inds[4]::2,inds[2]::2],matpart(q,r,mat22)[inds[3]::2,inds[5]::2])
                        + pairtrace(matpart(r,q,m_21_22[inds[0],inds[1],:,:])[inds[5]::2,inds[3]::2],matpart(q,r,mat11)[inds[2]::2,inds[4]::2])
                        + pairtrace(matpart(r,q,m_11_22[inds[0],inds[1],:,:])[inds[4]::2,inds[3]::2],matpart(q,r,mat12)[inds[2]::2,inds[5]::2])
                        + pairtrace(matpart(q,r,m_21_21[inds[0],inds[1],:,:])[inds[3]::2,inds[4]::2],matpart(r,q,mat21)[inds[5]::2,inds[2]::2])
                        + pairtrace(matpart(q,r,m_11_21[inds[0],inds[1],:,:])[inds[2]::2,inds[4]::2],matpart(r,q,mat22)[inds[5]::2,inds[3]::2])
                        + pairtrace(matpart(q,r,m_21_22[inds[0],inds[1],:,:])[inds[3]::2,inds[5]::2],matpart(r,q,mat11)[inds[4]::2,inds[2]::2])
                        + pairtrace(matpart(q,r,m_11_22[inds[0],inds[1],:,:])[inds[2]::2,inds[5]::2],matpart(r,q,mat12)[inds[4]::2,inds[3]::2])
                        )
                k3[3*(p-lmin):3*(p+1-lmin),3*(q-lmin):3*(q+1-lmin),3*(r-lmin):3*(r+1-lmin)]=z     
            #print p,q,r

    for ells in (itertools.product(range(lmin,lmax+1),repeat=3)):
        p,q,r=list(ells)
        sargs=argsort(ells)
        psort,qsort,rsort=sorted(list(ells))
        zz1 = k3[3*(psort-lmin):3*(psort+1-lmin),3*(qsort-lmin):3*(qsort+1-lmin),3*(rsort-lmin):3*(rsort+1-lmin)].copy()
        zz2 = zeros_like(zz1)
        for inds in itertools.product(range(0,3),repeat=3):
            tinds=list(inds)
            sinds=array(tinds)[sargs]
            zz2[tuple(tinds)] = zz1[tuple(sinds)]
        k3[3*(p-lmin):3*(p+1-lmin),3*(q-lmin):3*(q+1-lmin),3*(r-lmin):3*(r+1-lmin)] = zz2
        
    return k3

@jit(nopython=True)
def jitcheck(x):
    return 3.*x


@jit(nopython=True)
def intracheck():
    for changingmatinds in itertools.permutations(zip([11,13,17],range(2,8,2),range(3,8,2))):
        for perms in itertools.product([1,-1],repeat=3):
            terms=[(23,0,1)]+list(changingmatinds)
            fullperms=[1]+list(perms)
            freqinds=list()
            lterms=list()
            for i in range(len(terms)):
                if fullperms[i]==1:
                    freqinds.append(terms[i][1])
                    freqinds.append(terms[i][2])
                else:
                    freqinds.append(terms[i][2])
                    freqinds.append(terms[i][1])
                #freqinds+=list(array(terms[i][1:3])[::fullperms[i]])
                #freqinds+=terms[i][1:3]
                #print terms[i][0],array(terms[i][1:3])[::fullperms[i]],terms[i][0],' ',
                lterms+=[terms[i][0]]
                lterms+=[terms[i][0]]
            fls=lterms[-1:]+lterms[:-1]
            print (freqinds)

def sk4_ordercheck():
    for p in range(lmin,lmax+1):
        for q in range(p,lmax+1):
            for r in range(q,lmax+1):
                for s in range(r,lmax+1):
                    print ("(",p,q,r,s,"):")
                    for changingmatinds in itertools.permutations(zip([q,r,s],range(2,8,2),range(3,8,2))):
                        for perms in itertools.product([1,-1],repeat=3):
                            terms=[(p,0,1)]+list(changingmatinds)
                            fullperms=[1]+list(perms)
                            freqinds=list()
                            lterms=list()
                            for i in range(len(terms)):
                                freqinds+=list(array(terms[i][1:3])[::fullperms[i]])
                                #freqinds+=terms[i][1:3]
                                #print terms[i][0],array(terms[i][1:3])[::fullperms[i]],terms[i][0],' ',
                                lterms+=[terms[i][0]]
                                lterms+=[terms[i][0]]
                            fis=freqinds[-1:]+freqinds[:-1]
                            fls=lterms[-1:]+lterms[:-1]
                            for inds1 in itertools.product([0,1],repeat=4):
                                print (fls[0],fls[3],'of ',fls[1]-lmin,fis[1]%2,fis[2]%2,inds1[0],inds1[1],fis[0]%2,fis[3]%2)
                                print (fls[4],fls[7],'of ',fls[5]-lmin,fis[5]%2,fis[6]%2,inds1[2],inds1[3],fis[4]%2,fis[7]%2)



    return

# removing the array operation shaves a couple of seconds off for l=2-6
# reordering didn't help
#["array(float64, 4d, A)(array(float64, 2d, A),array(float64, 2d, A),array(float64, 2d, A))"]
@jit(nopython=True)
def symkappa4_12_reorder_noarray(mat11,mat12,mat22):
    mat21=mat12.transpose().copy()
    mats=zeros((2,2,mat11.shape[0],mat11.shape[1]))
    mats[0,0,:,:]=mat11
    mats[0,1,:,:]=mat12
    mats[1,0,:,:]=mat21
    mats[1,1,:,:]=mat22

    count=0
    
    k4=zeros((3*(lmax-lmin+1),3*(lmax-lmin+1),3*(lmax-lmin+1),3*(lmax-lmin+1)))
    outlen=2*((lmax+1)**2-lmin**2)
    # l, inside freqs, outside freqs, inside t/e,  then outside t/e by ::2 or 1::2  
    m=zeros((lmax-lmin+1,2,2,2,2,2,2,outlen,outlen))
    for p in range(lmin,lmax+1):
        for i in itertools.product([0,1],repeat=6):
            m1=mats[i[4],i[0],:,i[2]+2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2].copy()
            m2=mats[i[1],i[5],i[3]+2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:].copy()
            m[p-lmin,i[0],i[1],i[4],i[5],i[2],i[3],:,:]=dot(m1,m2)
        #print p

    for p in range(lmin,lmax+1):
        for q in range(p,lmax+1):
            for r in range(q,lmax+1):
                for s in range(r,lmax+1):
                    #count+=1
                    #continue
                    z=zeros((3,3,3,3))
                        #                        count=0
                    for changingmatinds in itertools.permutations(zip([q,r,s],range(2,8,2),range(3,8,2))):
                        for perms in itertools.product([1,-1],repeat=3):
                            terms=[(p,0,1)]+list(changingmatinds)
                            fullperms=[1]+list(perms)
                            freqinds=list()
                            lterms=list()
                            for i in range(len(terms)):
                                t1=terms[i][1]
                                t2=terms[i][2]
                                if fullperms[i]==1:
                                    freqinds.append(t1)
                                    freqinds.append(t2)
                                else:
                                    freqinds.append(terms[i][2])
                                    freqinds.append(terms[i][1])
                                #freqinds+=list(array(terms[i][1:3])[::fullperms[i]])
                                #freqinds+=terms[i][1:3]
                                #print terms[i][0],array(terms[i][1:3])[::fullperms[i]],terms[i][0],' ',
                                lterms+=[terms[i][0]]
                                lterms+=[terms[i][0]]
                            fis=freqinds[-1:]+freqinds[:-1]
                            fls=lterms[-1:]+lterms[:-1]
                            for inds1 in itertools.product([0,1],repeat=4):
                                mp1full=matpart(fls[0],fls[3],m[fls[1]-lmin,fis[1]%2,fis[2]%2,fis[0]%2,fis[3]%2,inds1[0],inds1[1],:,:])
                                mp2full=matpart(fls[4],fls[7],m[fls[5]-lmin,fis[5]%2,fis[6]%2,fis[4]%2,fis[7]%2,inds1[2],inds1[3],:,:])
                                for inds2 in itertools.product([0,1],repeat=4):
                                #count+=1
                                #biginds=tuple(2*array(inds[::2])+array(inds[1::2]))
                                #print fls[0],fls[3],fls[1]-lmin,fis[1]%2,fis[2]%2,biginds[0],biginds[1],fis[0]%2,fis[3]%2,biginds[2],biginds[3]
                                    mp1=mp1full[inds2[0]::2,inds2[1]::2]
                                    mp2=mp2full[inds2[2]::2,inds2[3]::2]
                                    biginds=(inds1[0]+inds1[1],inds2[0]+inds2[1],inds1[2]+inds1[3],inds2[2]+inds2[3])
                                    z[biginds] += pairtrace(mp1,mp2)
                                #count+=1
                    k4[3*(p-lmin):3*(p+1-lmin),3*(q-lmin):3*(q+1-lmin),3*(r-lmin):3*(r+1-lmin),3*(s-lmin):3*(s+1-lmin)]=z     
#        print "(",p,q,r,s,")",

    for ells in (itertools.product(range(lmin,lmax+1),repeat=4)):
        p,q,r,s=list(ells)
        sargs=numpy.argsort(ells)
        psort,qsort,rsort,ssort=sorted(list(ells))
        zz1 = k4[3*(psort-lmin):3*(psort+1-lmin),3*(qsort-lmin):3*(qsort+1-lmin),3*(rsort-lmin):3*(rsort+1-lmin),3*(ssort-lmin):3*(ssort+1-lmin)].copy()
        zz2 = zeros_like(zz1)
        for inds in itertools.product(range(0,3),repeat=4):
            tinds=list(inds)
            #sinds=tinds
            sinds2=list()
            for i in sargs:
                sinds2.append(tinds[i])
            #sinds=array(tinds)[sargs]
            zz2[tinds[0],tinds[1],tinds[2],tinds[3]] = zz1[sinds2[0],sinds2[1],sinds2[2],sinds2[3]]
            #            zz2[tuple(tinds)] = zz1[tuple(sinds2)]
            k4[3*(p-lmin):3*(p+1-lmin),3*(q-lmin):3*(q+1-lmin),3*(r-lmin):3*(r+1-lmin),3*(s-lmin):3*(s+1-lmin)] = zz2


    return k4

def symkappa4_12_nowork(mat11,mat12,mat22):
    mat21=transpose(mat12).copy()
    mats=zeros((2,2,mat11.shape[0],mat11.shape[1]))
    mats[0,0,:,:]=mat11
    mats[0,1,:,:]=mat12
    mats[1,0,:,:]=mat21
    mats[1,1,:,:]=mat22

    count=0
    
    k4=zeros((3*(lmax-lmin+1),3*(lmax-lmin+1),3*(lmax-lmin+1),3*(lmax-lmin+1)))
    outlen=2*((lmax+1)**2-lmin**2)
    # l, inside freqs, inside t/e, outside freqs, outside t/e by ::2 or 1::2  
    m=zeros((lmax-lmin+1,2,2,2,2,2,2,outlen,outlen))
    for p in range(lmin,lmax+1):
        for i in itertools.product([0,1],repeat=6):
            m1=mats[i[4],i[0],:,i[2]+2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2].copy()
            m2=mats[i[1],i[5],i[3]+2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:].copy()
            m[p-lmin,i[0],i[1],i[2],i[3],i[4],i[5],:,:]=dot(m1,m2)
        #print p

    for p in range(lmin,lmax+1):
        for q in range(p,lmax+1):
            for r in range(q,lmax+1):
                for s in range(r,lmax+1):
                    #count+=1
                    #continue
                    z=zeros((3,3,3,3))
                        #                        count=0
                    for changingmatinds in itertools.permutations(zip([q,r,s],range(2,8,2),range(3,8,2))):
                        for perms in itertools.product([1,-1],repeat=3):
                            terms=[(p,0,1)]+list(changingmatinds)
                            fullperms=[1]+list(perms)
                            freqinds=list()
                            lterms=list()
                            for i in range(len(terms)):
                                freqinds+=list(array(terms[i][1:3])[::fullperms[i]])
                                #freqinds+=terms[i][1:3]
                                #print terms[i][0],array(terms[i][1:3])[::fullperms[i]],terms[i][0],' ',
                                lterms+=[terms[i][0]]
                                lterms+=[terms[i][0]]
                            fis=freqinds[-1:]+freqinds[:-1]
                            fls=lterms[-1:]+lterms[:-1]
                            #                            for inds1 in itertools.product([0,1],repeat=4):
                                #mp1full=matpart(fls[0],fls[3],m[fls[1]-lmin,fis[1]%2,fis[2]%2,inds1[0],inds1[1],fis[0]%2,fis[3]%2,:,:])
                                #mp2full=matpart(fls[4],fls[7],m[fls[5]-lmin,fis[5]%2,fis[6]%2,inds1[2],inds1[3],fis[4]%2,fis[7]%2,:,:])
                                #  for inds2 in itertools.product([0,1],repeat=4):
                                #count+=1
                                #biginds=tuple(2*array(inds[::2])+array(inds[1::2]))
                                #print fls[0],fls[3],fls[1]-lmin,fis[1]%2,fis[2]%2,biginds[0],biginds[1],fis[0]%2,fis[3]%2,biginds[2],biginds[3]
                                #mp1=mp1full[inds2[0]::2,inds2[1]::2]
                                #mp2=mp2full[inds2[2]::2,inds2[3]::2]
                                #biginds=(inds1[0]+inds1[1],inds2[0]+inds2[1],inds1[2]+inds1[3],inds2[2]+inds2[3])
                                #z[biginds] += pairtrace(mp1,mp2)
                                #count+=1
                                #  pass
                                #z+=innersk4(m,fls,fis) 
                    k4[3*(p-lmin):3*(p+1-lmin),3*(q-lmin):3*(q+1-lmin),3*(r-lmin):3*(r+1-lmin),3*(s-lmin):3*(s+1-lmin)]=z    
        print ("(",p,q,r,s,")",)

    for ells in (itertools.product(range(lmin,lmax+1),repeat=4)):
        p,q,r,s=list(ells)
        sargs=argsort(ells)
        psort,qsort,rsort,ssort=sorted(list(ells))
        zz1 = k4[3*(psort-lmin):3*(psort+1-lmin),3*(qsort-lmin):3*(qsort+1-lmin),3*(rsort-lmin):3*(rsort+1-lmin),3*(ssort-lmin):3*(ssort+1-lmin)].copy()
        zz2 = zeros_like(zz1)
        for inds in itertools.product(range(0,3),repeat=4):
            tinds=list(inds)
            #sinds=tinds
            sinds2=list()
            for i in sargs:
                sinds2.append(tinds[i])
            #sinds=array(tinds)[sargs]
            zz2[tuple(tinds)] = zz1[tuple(sinds2)]
            k4[3*(p-lmin):3*(p+1-lmin),3*(q-lmin):3*(q+1-lmin),3*(r-lmin):3*(r+1-lmin),3*(s-lmin):3*(s+1-lmin)] = zz2


    return k4




def symkappa4_12(mat11,mat12,mat22):
    mat21=transpose(mat12).copy()
    mats=zeros((2,2,mat11.shape[0],mat11.shape[1]))
    mats[0,0,:,:]=mat11
    mats[0,1,:,:]=mat12
    mats[1,0,:,:]=mat21
    mats[1,1,:,:]=mat22

    count=0
    
    k4=zeros((3*(lmax-lmin+1),3*(lmax-lmin+1),3*(lmax-lmin+1),3*(lmax-lmin+1)))
    outlen=2*((lmax+1)**2-lmin**2)
    # l, inside freqs, inside t/e, outside freqs, outside t/e by ::2 or 1::2  
    m=zeros((lmax-lmin+1,2,2,2,2,2,2,outlen,outlen))
    for p in range(lmin,lmax+1):
        for i in itertools.product([0,1],repeat=6):
            m1=mats[i[4],i[0],:,i[2]+2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2].copy()
            m2=mats[i[1],i[5],i[3]+2*(p**2-lmin**2):2*(2*p+1+p**2-lmin**2):2,:].copy()
            m[p-lmin,i[0],i[1],i[2],i[3],i[4],i[5],:,:]=dot(m1,m2)
        #print p

    for p in range(lmin,lmax+1):
        for q in range(p,lmax+1):
            for r in range(q,lmax+1):
                for s in range(r,lmax+1):
                    #count+=1
                    #continue
                    z=zeros((3,3,3,3))
                        #                        count=0
                    for changingmatinds in itertools.permutations(zip([q,r,s],range(2,8,2),range(3,8,2))):
                        for perms in itertools.product([1,-1],repeat=3):
                            terms=[(p,0,1)]+list(changingmatinds)
                            fullperms=[1]+list(perms)
                            freqinds=list()
                            lterms=list()
                            for i in range(len(terms)):
                                freqinds+=list(array(terms[i][1:3])[::fullperms[i]])
                                #freqinds+=terms[i][1:3]
                                #print terms[i][0],array(terms[i][1:3])[::fullperms[i]],terms[i][0],' ',
                                lterms+=[terms[i][0]]
                                lterms+=[terms[i][0]]
                            fis=freqinds[-1:]+freqinds[:-1]
                            fls=lterms[-1:]+lterms[:-1]
                            for inds1 in itertools.product([0,1],repeat=4):
                                mp1full=matpart(fls[0],fls[3],m[fls[1]-lmin,fis[1]%2,fis[2]%2,inds1[0],inds1[1],fis[0]%2,fis[3]%2,:,:])
                                mp2full=matpart(fls[4],fls[7],m[fls[5]-lmin,fis[5]%2,fis[6]%2,inds1[2],inds1[3],fis[4]%2,fis[7]%2,:,:])
                                for inds2 in itertools.product([0,1],repeat=4):
                                #count+=1
                                #biginds=tuple(2*array(inds[::2])+array(inds[1::2]))
                                #print fls[0],fls[3],fls[1]-lmin,fis[1]%2,fis[2]%2,biginds[0],biginds[1],fis[0]%2,fis[3]%2,biginds[2],biginds[3]
                                    mp1=mp1full[inds2[0]::2,inds2[1]::2]
                                    mp2=mp2full[inds2[2]::2,inds2[3]::2]
                                    biginds=(inds1[0]+inds1[1],inds2[0]+inds2[1],inds1[2]+inds1[3],inds2[2]+inds2[3])
                                    z[biginds] += pairtrace(mp1,mp2) 
                                #count+=1
                    k4[3*(p-lmin):3*(p+1-lmin),3*(q-lmin):3*(q+1-lmin),3*(r-lmin):3*(r+1-lmin),3*(s-lmin):3*(s+1-lmin)]=z     
        print ("(",p,q,r,s,")",)

    for ells in (itertools.product(range(lmin,lmax+1),repeat=4)):
        p,q,r,s=list(ells)
        sargs=argsort(ells)
        psort,qsort,rsort,ssort=sorted(list(ells))
        zz1 = k4[3*(psort-lmin):3*(psort+1-lmin),3*(qsort-lmin):3*(qsort+1-lmin),3*(rsort-lmin):3*(rsort+1-lmin),3*(ssort-lmin):3*(ssort+1-lmin)].copy()
        zz2 = zeros_like(zz1)
        for inds in itertools.product(range(0,3),repeat=4):
            tinds=list(inds)
            #sinds=tinds
            sinds2=list()
            for i in sargs:
                sinds2.append(tinds[i])
            #sinds=array(tinds)[sargs]
            zz2[tuple(tinds)] = zz1[tuple(sinds2)]
            k4[3*(p-lmin):3*(p+1-lmin),3*(q-lmin):3*(q+1-lmin),3*(r-lmin):3*(r+1-lmin),3*(s-lmin):3*(s+1-lmin)] = zz2


    return k4

def dummyfunc():
    return 0

# WARNING: need to think about the beam when differentiating...
# CORRECTION: Here I first used "fiducial matrices" in place of Q^t Y.  This is wrong when the
# noise has been reshaped in the Q's, because the true noise, not the reshaped noise, appears
# fiducial covariance

def d_symcov12_slow(mat11,mat12,mat22,qty11,qty22):
    mat21=transpose(mat12).copy()
    dcov=zeros((3*(lmax-lmin+1),3*(lmax-lmin+1),3*(lmax-lmin+1)))
    for l in itertools.product(range(lmin,lmax+1),repeat=3):
        zz1=zeros((3,3,3))
        m21_ij=matpart(l[0],l[1],mat21)
        m21_ji=matpart(l[1],l[0],mat21)
        m22_ij=matpart(l[0],l[1],mat22)
        m11_ji=matpart(l[1],l[0],mat11)
        f22_ik=matpart(l[0],l[2],qty22)
        f11_ki=matpart(l[2],l[0],qty11)
        f22_jk=matpart(l[1],l[2],qty22)
        f22_kj=matpart(l[2],l[1],qty22)
        f11_jk=matpart(l[1],l[2],qty11)
        f11_kj=matpart(l[2],l[1],qty11)
        print ('made parts for l='+str(l),flush=True)


        for m in itertools.product(range(0,2*l[0]+1),range(0,2*l[1]+1),range(0,2*l[2]+1)):
            for i in itertools.product([0,1],repeat=6):
                zz1[i[0]+i[1],i[2]+i[3],i[4]+i[5]] += (
                    + f22_ik[2*m[0]+i[1],2*m[2]+i[4]] * f11_kj[2*m[2]+i[5],2*m[1]+i[2]] * m21_ji[2*m[1]+i[3],2*m[0]+i[0]]
                    + m21_ij[2*m[0]+i[1],2*m[1]+i[2]] * f22_jk[2*m[1]+i[3],2*m[2]+i[5]] * f11_ki[2*m[2]+i[4],2*m[0]+i[0]]
                    + f22_ik[2*m[0]+i[1],2*m[2]+i[4]] * f22_kj[2*m[2]+i[5],2*m[1]+i[3]] * m11_ji[2*m[1]+i[2],2*m[0]+i[0]]
                    + m22_ij[2*m[0]+i[1],2*m[1]+i[3]] * f11_jk[2*m[1]+i[2],2*m[2]+i[5]] * f11_ki[2*m[2]+i[4],2*m[0]+i[0]]
                    )
                    #*b_l[l]**2

        dcov[3*(l[0]-lmin):3*(l[0]+1-lmin),3*(l[1]-lmin):3*(l[1]+1-lmin),3*(l[2]-lmin):3*(l[2]+1-lmin)]=zz1
        print ('and updated matrix entries',flush=True)
    return dcov

# WARNING: Provisionally adding a b_l[2]^2 to match that added to the coupling matrix...
def d_symcov12(mat11,mat12,mat22,qty11,qty22):
    mat21=transpose(mat12).copy()
    dcov=zeros((3*(lmax-lmin+1),3*(lmax-lmin+1),3*(lmax-lmin+1)))
    for l in itertools.product(range(lmin,lmax+1),repeat=3):
        zz1=zeros((3,3,3))
        m21_ij=matpart(l[0],l[1],mat21)
        m21_ji=matpart(l[1],l[0],mat21)
        m22_ij=matpart(l[0],l[1],mat22)
        m11_ji=matpart(l[1],l[0],mat11)
        f22_ik=matpart(l[0],l[2],qty22)
        f11_ki=matpart(l[2],l[0],qty11)
        f22_jk=matpart(l[1],l[2],qty22)
        f22_kj=matpart(l[2],l[1],qty22)
        f11_jk=matpart(l[1],l[2],qty11)
        f11_kj=matpart(l[2],l[1],qty11)

        for i in itertools.product([0,1],repeat=6):
            zz1[i[0]+i[1],i[2]+i[3],i[4]+i[5]] += (
                + pairtrace(dot(f22_ik[i[1]::2,i[4]::2], f11_kj[i[5]::2,i[2]::2]), m21_ji[i[3]::2,i[0]::2])
                + pairtrace(dot(m21_ij[i[1]::2,i[2]::2], f22_jk[i[3]::2,i[5]::2]), f11_ki[i[4]::2,i[0]::2])
                + pairtrace(dot(f22_ik[i[1]::2,i[4]::2], f22_kj[i[5]::2,i[3]::2]), m11_ji[i[2]::2,i[0]::2])
                + pairtrace(dot(m22_ij[i[1]::2,i[3]::2], f11_jk[i[2]::2,i[5]::2]), f11_ki[i[4]::2,i[0]::2])
                )*b_l[l[2]]**2

        dcov[3*(l[0]-lmin):3*(l[0]+1-lmin),3*(l[1]-lmin):3*(l[1]+1-lmin),3*(l[2]-lmin):3*(l[2]+1-lmin)]=zz1
    return dcov



    
'''
nmaps=2
intermatrices=zeros((nmaps,nmaps,nmaps,nmaps,4*(lmax-lmin+1),4*(lmax-lmin+1),4*(lmax-lmin+1)))

class CovMat:
    def  __init__(self,d1,d2):
        self.d1=d1
        self.d2=d2
#        self.c=zeros(((lmax+1)**2-lmin**2),((lmax+1)**2-lmin**2))
    def __str__(self):
        return 'M_'+str(self.d1)+str(self.d2)


class CombinedCovMat:
    def __init__(self,l,da,db,d1,d2):
        self.d1=d1
        self.d2=d2
        self.da=da
        self.db=db
        self.l=l
#        self.cm=Covmat(t1,m1,t2,m2)
    def __str__(self):
        return 'M('+str(self.l)+')^'+str(self.da)+str(self.db)+'_'+str(self.d1)+str(self.d2)


cms=[CovMat(1,2),CovMat(3,4)]
for i in list(itertools.permutations(cms[:-1])):
    for flip in itertools.product(['','^T'],repeat=len(i)):
        j=list(i)
        j.append(cms[-1])
        fl=list(flip)
        fl.append('')
        jz=zip(j,fl)
        for k in jz:
            print str(k[0])+k[1],
        print



cms=[CovMat(1,2),CovMat(3,4),CovMat(5,6)]
for i in list(itertools.permutations(cms[1:])):
    for flip in itertools.product(['','^T'],repeat=len(i)):
        j=[cms[0]]+list(i)
        fl=['']+list(flip)
        jz=zip(j,fl)
        for k in jz:
            print str(k[0])+k[1],
        for p in range(len(jz[:-1])):
            print str(jz[p][0].d2)+str(jz[p+1][0].d1),
        print str(jz[p+1][0].d2)+str(jz[0][0].d1)

nterms=dict()
#cms=[CovMat(1,2),CovMat(3,4),CovMat(5,6)]
cms=list(itertools.repeat(CovMat(1,2),4))
for i in itertools.permutations(cms[1:]):
    for flip in itertools.product(['','T'],repeat=len(i)):
        inds=[]
        inds.append(cms[0].d1)
        inds.append(cms[0].d2)
        for k in zip(i,flip):
            if k[1]=='':
                inds.append(k[0].d1)
                inds.append(k[0].d2)
            elif k[1]=='T':
                inds.append(k[0].d2)
                inds.append(k[0].d1)
        newinds=inds[1:]+inds[:1]
        for k in range(0,len(newinds),2):
            print str(newinds[k])+str(newinds[k+1]),
        if tuple(newinds) not in nterms:
            nterms[tuple(newinds)]=1
        else:
            nterms[tuple(newinds)]+=1
        print
for i in nterms:
    for k in range(0,len(i),2):
            print str(i[k])+str(i[k+1]),
    print ':', nterms[i]


for inds in [(x,y,z,w) for x in range(nmaps) for y in range(nnmaps) for z in range(nmaps) for w in range(nmaps)]:
    intermatrices[inds[0],inds[1],inds[2],inds[3],:,:,:]=make3term(inds)

def make3term(i,j,p,q):
    global intermatrices
    for (l,l1,2) in itertools.product(range(lmin,lmax+1),repeat=3):
        m_ip=matpart(l1,l,covs[i,p])
        m_qj=matpart(l,l2,covs[q,j])
        intermatrices[i,j,p,q,4*(l-lmin):4*(l+1-lmin):4,4*(l1-lmin):4*(l1+1-lmin):4,4*(l2-lmin):4*(l2+1-lmin):4]+= (
                dot(m_ip[4*(l1-lmin):4*(l1+1-lmin),4*(l-lmin):4*(l+1-lmin)],m_qj[4*(l-lmin):4*(l+1-lmin),4*(l2-lmin):4*(l2+1-lmin)])
            )

for i in itertools.starmap(make3term,itertools.product(range(nmaps),repeat=4)):
    pass

'''

   #def cov_12_12(mat11, mat22, mat12):
   #covs=zeros((3,3,(lmax-lmin+1),(lmax-lmin+1)))
   #covs_ee=zeros((2*(lmax-lmin+1),2*(lmax-lmin+1)))
   #covs_eb=zeros((2*(lmax-lmin+1),2*(lmax-lmin+1)))
   #covs_be=zeros((2*(lmax-lmin+1),2*(lmax-lmin+1)))
   #covs_bb=zeros((2*(lmax-lmin+1),2*(lmax-lmin+1)))
   #for p in range(lmin,lmax+1):
   #     for q in range(lmin,lmax+1):
   #         m11_qp=matpart(q,p,mat11)
   #         m22_pq=matpart(p,q,mat22)
   #         m12_qp=matpart(q,p,mat12)
   #         m12_pq=matpart(p,q,mat12)
   #         covs_ee[p-2*lmin,q-2*lmin]=pairtrace(m12_qp,m12_pq)[0]+pairtrace(m11_qp,m22_pq)[0]    
   #         covs_eb[p-2*lmin,q-2*lmin]=pairtrace(m12_qp,m12_pq)[1]+pairtrace(m11_qp,m22_pq)[1]    
   #         covs_be[p-2*lmin,q-2*lmin]=pairtrace(m12_qp,m12_pq)[2]+pairtrace(m11_qp,m22_pq)[2]    
   #         covs_bb[p-2*lmin,q-2*lmin]=pairtrace(m12_qp,m12_pq)[3]+pairtrace(m11_qp,m22_pq)[3]    
   # return covs

n_11_p=sum_mat(yn_11)
n_22_p=sum_mat(yn_22)
n_12_p=sum_mat(yn_12)

ells=arange(lmin,lmax+1.0)

def d_from_c(c):
    return c*ells*(ells+1.0)/2.0/pi

# DANGER TO CHECK: THE COUPLING MATRIX SHOULDN'T HAVE THE FIDUCIAL COVARIANCE, RATHER IT SHOULD
# HAVE Q^TY; THEY DIFFER IN YN_11 ETC... 
# coupling matrix needs reshaped covariance...
cov_11_fid=make_m_slow(f_11, clforweighting, f_11, yn_11)
cov_22_fid=make_m_slow(f_22, clforweighting, f_22, yn_22)
cov_12_fid=make_m_slow(f_11, clforweighting, f_22, yn_12)
cov_21_fid=transpose(cov_12_fid).copy()
#couple_12=cov_12_34(zeros_like(cov_21_fid),zeros_like(cov_21_fid),cov_22_fid,cov_11_fid)
couple_12=cov_12_34(zeros_like(cov_21_fid),zeros_like(cov_21_fid),f_22,f_11)
coupleinv_12=inv(couple_12)

# errors need full covariance...
fidlongcls=makecllong(fidcldatfile)
cov_11_fid=make_m_slow(f_11, fidlongcls, f_11, yn_11)
cov_22_fid=make_m_slow(f_22, fidlongcls, f_22, yn_22)
cov_12_fid=make_m_slow(f_11, fidlongcls, f_22, yn_12)
cov_21_fid=transpose(cov_12_fid).copy()
cov_12_pq_fid=cov_12_34(cov_21_fid,cov_21_fid,cov_22_fid,cov_11_fid)
dec_cov_12_pq_fid=dot(coupleinv_12,dot(cov_12_pq_fid,transpose(coupleinv_12)))

# further use needs to match the coupling matrix...
cov_11_fid=make_m_slow(f_11, clforweighting, f_11, yn_11)
cov_22_fid=make_m_slow(f_22, clforweighting, f_22, yn_22)
cov_12_fid=make_m_slow(f_11, clforweighting, f_22, yn_12)
cov_21_fid=transpose(cov_12_fid).copy()

ci=inv(d_symmean(f_11,f_22))
fidcls=getclsmulti(fidcldatfile,(1,4,2))

cs_11=d_symmean(f_11,f_11)
cs_22=d_symmean(f_22,f_22)


def s_simp(cls):
    return dot(transpose(cls-fidcls),fidgrads+grad_s(ci,cls)+4.*grad_s(ci,.5*(cls+fidcls)))/6.

def s_romb(cls):
    grads=zeros(5)
    dcls=cls-fidcls
    grads[0]=dot(transpose(dcls),fidgrads)
    grads[1]=dot(transpose(dcls),grad_s(ci,fidcls+.25*dcls))
    grads[2]=dot(transpose(dcls),grad_s(ci,fidcls+.50*dcls))
    grads[3]=dot(transpose(dcls),grad_s(ci,fidcls+.75*dcls))
    grads[4]=dot(transpose(dcls),grad_s(ci,cls))
    # delete following lines to revert to romberg integration
    if int_method == 'romberg':
        return scipy.integrate.romb(grads,.25,show=False)
    elif int_method == 'quad':
        x_int = np.array((0., .25,.5,.75,1.))
        return scipy.integrate.quad(scipy.interpolate.CubicSpline(x_int,grads), 0., 1.)[0]
    #print('s_romb: integration romb = {}'.format(int1))
    #print('s_romb: integration quad = {}'.format(int2))
    #return scipy.integrate.romb(grads,.25,show=True)

def gradterm(cls,x):
    dcls=cls-fidcls
    return dot(transpose(dcls),grad_s(ci,fidcls+x*dcls))

def s_romb_parallel(cls):
    grads=zeros(5)
    grads[0]=dot(transpose(cls-fidcls),fidgrads)
    spacings=array((.25,.5,.75,1.))
    workargs=zip(itertools.repeat(cls),spacings)
    pool=multiprocessing.Pool(processes=no_processes)
    result=pool.starmap(gradterm,workargs)
    pool.close()
    pool.join()
    grads[1:]=result
    # delete following lines to revert to romberg integration
    if int_method == 'romberg':
        return scipy.integrate.romb(grads,.25,show=True)
    elif int_method == 'quad':
        x_int = np.array((0., .25,.5,.75,1.))
        return scipy.integrate.quad_vec(scipy.interpolate.CubicSpline(x_int,grads), a=0., b=1., workers=no_processes+1)[0]
    #return scipy.integrate.romb(grads,.25,show=True)

def gradterm_part(cls,x):
    dcls=cls-fidcls
    partdcls=dcls[3*(lminlike-lmin):3*(lmaxlike-lmin+1)].copy()
    if doEEonly:
        partdcls=partdcls[2::3].copy()
        if not do_linear:
            # RB: originally remove the next few lines
            print('about to multiply (gradterm_part)')
            print(partdcls.shape, grad_s_part(ci,fidcls+x*dcls)[2::3].shape)
            return dot(transpose(partdcls),grad_s_part(ci,fidcls+x*dcls)[2::3])
            print('success (gradterm_part)')
            # RB: originally delete up to here
    return dot(transpose(partdcls),grad_s_part(ci,fidcls+x*dcls))

def s_romb_part_parallel(cls):
    grads=zeros(5)
    dcls=cls-fidcls
    partdcls=dcls[3*(lminlike-lmin):3*(lmaxlike-lmin+1)].copy()
    if doEEonly:
        if not do_linear:
            print('about to multiply (s_romb_part_parallel)')
            print(partdcls[2::3].shape, fidgrads_part[2::3].shape)
            grads[0]=dot(transpose(partdcls)[2::3],fidgrads_part[2::3])
            print('success (s_romb_part_parallel)')
        # RB: originally it was that way
        elif do_linear:
            grads[0]=dot(transpose(partdcls)[2::3],fidgrads_part)
    else:
        grads[0]=dot(transpose(partdcls),fidgrads_part)
    spacings=array((.25,.5,.75,1.))
    workargs=zip(itertools.repeat(cls),spacings)
    pool=multiprocessing.Pool(processes=no_processes)
    result=pool.starmap(gradterm_part,iterable=workargs)
    pool.close()
    pool.join()
    grads[1:]=result
    # delete following lines to revert to romberg integration
    if int_method == 'romberg':
        return scipy.integrate.romb(grads,.25,show=True)
    elif int_method == 'quad':
        x_int = np.array((0., .25,.5,.75,1.))
        return scipy.integrate.quad_vec(scipy.interpolate.CubicSpline(x_int,grads), a=0., b=1., workers=no_processes+1)[0]
    #print('s_romb_part_parallel: integration romb = {}'.format(int1))
    #print('s_romb_part_parallel: integration quad = {}'.format(int2))
    #return scipy.integrate.romb(grads,.25,show=True)

def s_romb_part(cls):
    grads=zeros(5)
    dcls=(cls-fidcls)
    partdcls=dcls[3*(lminlike-lmin):3*(lmaxlike-lmin+1)].copy()
    grads[0]=dot(transpose(partdcls),fidgrads_part)
    grads[1]=dot(transpose(partdcls),grad_s_part(ci,fidcls+.25*dcls))
    grads[2]=dot(transpose(partdcls),grad_s_part(ci,fidcls+.50*dcls))
    grads[3]=dot(transpose(partdcls),grad_s_part(ci,fidcls+.75*dcls))
    grads[4]=dot(transpose(partdcls),grad_s_part(ci,cls))
    if int_method == 'romberg':
        return scipy.integrate.romb(grads,.25,show=False)
    elif int_method == 'quad':
        x_int = np.array((0., .25,.5,.75,1.))
        return scipy.integrate.quad(scipy.interpolate.CubicSpline(x_int,grads), 0., 1.)[0]


def s_romb_diff_part(cls1,cls2):
    grads=zeros(5)
    dcls=(cls2-cls1)
    partdcls=dcls[3*(lminlike-lmin):3*(lmaxlike-lmin+1)].copy()
    grads[0]=dot(transpose(partdcls),grad_s_part(ci,cls1))
    grads[1]=dot(transpose(partdcls),grad_s_part(ci,cls1+.25*dcls))
    grads[2]=dot(transpose(partdcls),grad_s_part(ci,cls1+.50*dcls))
    grads[3]=dot(transpose(partdcls),grad_s_part(ci,cls1+.75*dcls))
    grads[4]=dot(transpose(partdcls),grad_s_part(ci,cls2))
    if int_method == 'romberg':
        return scipy.integrate.romb(grads,.25,show=False)
    elif int_method == 'quad':
        x_int = np.array((0., .25,.5,.75,1.))
        return scipy.integrate.quad(scipy.interpolate.CubicSpline(x_int,grads), 0., 1.)[0]


def gradtermdiff_part(cls1,cls2,x):
    dcls=cls2-cls1
    partdcls=dcls[3*(lminlike-lmin):3*(lmaxlike-lmin+1)].copy()
    return dot(transpose(partdcls),grad_s_part(ci,cls1+x*dcls))


def s_romb_diff_part_parallel(cls1,cls2):
    grads=zeros(5)
    dcls=cls2-cls1
    partdcls=dcls[3*(lminlike-lmin):3*(lmaxlike-lmin+1)].copy()
    spacings=array((0.,.25,.5,.75,1.))
    workargs=zip(itertools.repeat(cls1),itertools.repeat(cls2),spacings)
    pool=multiprocessing.Pool(processes=no_processes)
    result=pool.starmap(gradtermdiff_part,workargs)
    pool.close()
    pool.join()
    grads=result
    # delete following lines to revert to romberg integration
    if int_method == 'romberg':
        return scipy.integrate.romb(grads,.25,show=True)
    elif int_method == 'quad':
        x_int = np.array((0., .25,.5,.75,1.))
        return scipy.integrate.quad_vec(scipy.interpolate.CubicSpline(x_int,grads), a=0., b=1., workers=no_processes)[0]


def s_simp_diff(cls1,cls2):
    gr=(
        + grad_s(ci,cls1)
        + grad_s(ci,cls2)
        + grad_s(ci,.5*(cls1+cls2))*4.
        )/6.
    return dot(transpose(cls2-cls1),gr)

def s_romb_diff(cls1,cls2):
    grads=zeros(5)
    dcls=cls2-cls1
    grads[0]=dot(transpose(dcls),grad_s(ci,cls1))
    grads[1]=dot(transpose(dcls),grad_s(ci,cls1+.25*dcls))
    grads[2]=dot(transpose(dcls),grad_s(ci,cls1+.50*dcls))
    grads[3]=dot(transpose(dcls),grad_s(ci,cls1+.75*dcls))
    grads[4]=dot(transpose(dcls),grad_s(ci,cls2))
    print (grads)
    if int_method == 'romberg':
        return scipy.integrate.romb(grads,.25,show=False)
    elif int_method == 'quad':
        x_int = np.array((0., .25,.5,.75,1.))
        return scipy.integrate.quad(scipy.interpolate.CubicSpline(x_int,grads), 0., 1.)[0]

def d_from_c_mat(m):
    scal=ells*(ells+1.0)/2.0/pi
    return mul_cols(mul_rows(m,scal),scal)

destroy_ee_pcov=False
if destroy_ee_pcov:
    yp_12[0::4]=0.0

destroy_eb_be_bb_pcov=False
if destroy_eb_be_bb_pcov:
    yp_12[1::4]=0.0
    yp_12[2::4]=0.0
    yp_12[3::4]=0.0

dl_12_fid_ee=d_from_c(dot(coupleinv_12,yp_12-n_12_p)[0::4])
dl_12_fid_eb=d_from_c(dot(coupleinv_12,yp_12-n_12_p)[1::4])
dl_12_fid_be=d_from_c(dot(coupleinv_12,yp_12-n_12_p)[2::4])
dl_12_fid_bb=d_from_c(dot(coupleinv_12,yp_12-n_12_p)[3::4])
d_cov_12_fid_ee=d_from_c_mat(dec_cov_12_pq_fid[0::4,0::4])
d_cov_12_fid_eb=d_from_c_mat(dec_cov_12_pq_fid[1::4,1::4])
d_cov_12_fid_be=d_from_c_mat(dec_cov_12_pq_fid[2::4,2::4])
d_cov_12_fid_bb=d_from_c_mat(dec_cov_12_pq_fid[3::4,3::4])

clee_fid=getcls(fidcldatfile,1)
clbb_fid=getcls(fidcldatfile,2)
cleb_fid=getcls(fidcldatfile,4)
clbe_fid=getcls(fidcldatfile,4)

dl_theory_ee=d_from_c(clee_fid)
dl_theory_eb=d_from_c(cleb_fid)
dl_theory_be=d_from_c(clbe_fid)
dl_theory_bb=d_from_c(clbb_fid)

dl_err_fid_ee=sqrt(diag(d_cov_12_fid_ee))
dl_err_fid_eb=sqrt(diag(d_cov_12_fid_eb))
dl_err_fid_be=sqrt(diag(d_cov_12_fid_be))
dl_err_fid_bb=sqrt(diag(d_cov_12_fid_bb))

#figure(1);clf();plot(ells,dl_11,'r.',label="1x1");plot(ells,dl_22,'g.',label="2x2");plot(ells,dl_12,'bo-',label="1x2");errorbar(ells,dl_theory,yerr=dl_err,fmt='k-',label="theory");xlim(0,lmax+1);title(r"Power Spectra");legend();xlabel(r'multipole $l$');ylabel(r"$D_l / \mu K^2$")
#figure(1);clf();axhline(0.,color='r');errorbar(ells,dl_theory,yerr=dl_err,fmt='k-',label="fiducial theory");plot(ells,dl_12,'bo-',label="1x2");xlim(0,lmax+1);title(r"Power Spectra");legend();xlabel(r'multipole $l$');ylabel(r"$D_l / \mu K^2$");ylim(-.11,.25)
#savefig(basedir+'plots/lowlpower.pdf')

figure(1)
clf()
suptitle(r'{}{}x{}{} GHz full Power Spectra ({})'.format(freq1,split1,freq2,split2,data_set))

pymin=-.05
pymax=.1

subplot(2,2,1)
axhline(0,color='r')
plot(ells,dl_theory_ee,'k--')
errorbar(ells,dl_12_fid_ee,yerr=dl_err_fid_ee,fmt='b.')
xlim(0,lmax+1)
ylim(0,2000)
title(r'$D_l^\mathrm{TT}$')

subplot(2,2,2)
plot(ells,dl_theory_eb, 'k--')
errorbar(ells,dl_12_fid_eb,yerr=dl_err_fid_eb,fmt='b.')
xlim(0,lmax+1)
ylim(-5,10)
title(r'$D_l^\mathrm{TE}$')

subplot(2,2,3)
plot(ells,dl_theory_be,'k--')
errorbar(ells,dl_12_fid_be,yerr=dl_err_fid_be,fmt='b.')
xlim(0,lmax+1)
ylim(-5,10)
title(r'$D_l^\mathrm{ET}$')


subplot(2,2,4)
plot(ells,dl_theory_bb, 'k--')
errorbar(ells,dl_12_fid_bb,yerr=dl_err_fid_bb,fmt='b.')
savetxt(basedir+'qml/ps_{}_{}{}x{}{}_ell_{}-{}_tau_0.{:03d}-0.{:03d}{}{}.txt'.format(data_set,freq1,split1,freq2,split2,lminlike,lmaxlike,imin,imax, linear_approx,pol_approx), np.c_[ells,dl_12_fid_ee,dl_err_fid_ee,dl_12_fid_eb,dl_err_fid_eb,dl_12_fid_be,dl_err_fid_be,dl_12_fid_bb,dl_err_fid_bb])
xlim(0,lmax+1)
ylim(pymin,pymax)
title(r'$D_l^\mathrm{EE}$')

savefig(basedir+'plots/ps_{}_{}{}x{}{}_ell_{}-{}_tau_0.{:03d}-0.{:03d}{}{}.pdf'.format(data_set,freq1,split1,freq2,split2,lminlike,lmaxlike,imin,imax,linear_approx,pol_approx))

if show_plot:
    show()

close()
'''
figure(3)
clf()
title('Checks of noise power spectra')
plot(ells,(d_from_c(dec_kap1(inv(cs_11),squash_vec(n_11_p))[2::3])),'ro-',label='100 GHz')
plot(ells,(d_from_c(dec_kap1(inv(cs_22),squash_vec(n_22_p))[2::3])),'go-',label='143 GHz')
plot(ells,(d_from_c(dec_kap1(ci,squash_vec(n_12_p))[2::3])),'bo-',label='100x143 GHz')
axhline(15e-3,color='red')
axhline(8e-3,color='green')
ylim(0,.1)
legend()
'''

'''
def doTTscatter(mn1,mn2):
    lmaxhere=20
    m1=healpy.read_map(mn1)
    m2=healpy.read_map(mn2)
    ytemp=y_full[:3072,0:2*lmaxhere:2].copy()
    yt1=y_full[mn1==1,:].copy()
    yt2=y_full[mn2==1,:].copy()

testTTscatter=False
if testTTscatter:
    doTTscatter('molinari/mask_TQU_T_COMM_DX11D2_P_R1.50_DX1170_R1.50_RING_T.fits','molinari/commander_dx12_mask_temp_likelihood_n0016_v01.fits')
'''

###
if only_compute_ps:
    assert(False)
#
##

#show()
print ("About to start scan...")
#pause(5)

print ("Starting scan...")


def mlnlike(d,m,cov):
    return 0.5*dot(transpose(d-m),dot(inv(cov),d-m))+0.5*slogdet(cov)[1]



#fidgrads=grad_s(ci,fidcls)
#print('fidgrads:', fidgrads.shape)
fidgrads_part=grad_s_part(ci,fidcls)
print('fidgrads_part:', fidgrads_part.shape)

print ('lmin='+str(lmin)+', lmax='+str(lmax)+'.')
print ('tau scan using lmin='+str(lminlike)+', '+str(lmaxlike)+'.')

if(lminlike<lmin):
    print ('error: lminlike less than lmin')
    assert(False)

if(lmaxlike>lmax):
    print ('error: lmaxlike greater than lmax')
    assert(False)
    
#imin=20
#imax=70
#imax=int(sys.argv[1])
#print('new imax', imax)
#istep=2
taus=zeros(int((imax-imin+istep-1)/istep+1))
mlls=zeros(int((imax-imin+istep-1)/istep+1))
fmlls=zeros(int((imax-imin+istep-1)/istep+1))
newmlls=zeros(int((imax-imin+istep-1)/istep+1))
pos=0
cipart=coupleinv_12[3::4,:][(lminlike-lmin):(lmaxlike+1-lmin),:].copy()
#cipart=coupleinv_12.copy()
cipart_t=transpose(cipart).copy()
dee=dot(cipart,yp_12)

fidcovar=dot(cipart,dot(cov_12_pq_fid,cipart_t))


def summarizetau(mlls):    
    mllmin=amin(mlls)
    taumin=taus[argmin(mlls)]
    norm=1.e-3*istep*sum(exp(-(mlls-mllmin)))
    p=exp(-(mlls-mllmin))/norm
    taumean=sum(taus*p)/sum(p)
    tausq=sum(taus**2*p)/sum(p)
    tausd=sqrt(tausq-taumean*taumean)
    print ('==================================')
    print ('Posterior peaks at: {:1.5f}'.format(taumin))
    print ('tau mean is at:     {:1.5f}'.format(taumean))
    print ('tau sd is:          {:1.5f}'.format(tausd))
    print ('==================================')
    return p, taumean, tausd

def plot_posterior(tau_array, p_array, tau_mean, tau_err, save_file, title, save=True):
    figure(2)
    clf()
    plot(tau_array,p_array,label='{} $\tau={:1.4f}\pm{:1.4f}$'.format(title,tau_mean,tau_err))
    xlabel(r'$\tau$');ylabel(r'$P$')
    title(r'Posterior for $\tau$ ({})'.format(data_set))
    xlim(0.02, 0.08);legend(loc='upper right');
    if save:
        savefig(basedir+'plots/posterior_{}_{}{}x{}{}_tau_{}_like_glass_ell_{}-{}_tau_0.{:03d}-0.{:03d}{}{}.pdf'.format(data_set,freq1,split1,freq2,split2,save_file,lminlike,lmaxlike,imin,imax,linear_approx,pol_approx))    
        print('save posterior under')
        print('plots/posterior_{}_{}{}x{}{}_tau_{}_like_glass_ell_{}-{}_tau_0.{:03d}-0.{:03d}{}{}.pdf'.format(data_set,freq1,split1,freq2,split2,save_file,lminlike,lmaxlike,imin,imax,linear_approx,pol_approx))


def save_posterior(tau_array, p_array1,p_array2,p_array3):
    savetxt(basedir+'posterior/posterior_samples_{}_{}{}x{}{}_tau_like_glass_ell_{}-{}_tau_0.{:03d}-0.{:03d}{}{}.dat'.format(data_set,freq1,split1,freq2,split2,lminlike,lmaxlike,imin,imax,linear_approx,pol_approx), np.c_[tau_array, p_array1,p_array2,p_array3])


pos=0
print('Do computation for ncm for freq={}'.format(freq1))
if freq1==100:
    ncm_pixel = ncm100
    data = d_1[tl:]
if freq1==143:
    ncm_pixel = ncm143
    data = d_2[tl:]

for i in range(imin,imax+1,istep):
    fname=tau_dir+'tau_scan_0_{:03d}_lensedCls.dat'.format(i)
    print (i,fname)
    cllong=makecllong_EEEBBB(fname)
    t0 = time.time()
    ycytncm = dot(yqumask, dot(cllong, yqumaskt)) + ncm_pixel
    #print(yqumask.shape, cllong.shape, yqumaskt.shape, ncm_pixel.shape, ycytncm.shape)
    ycytncminv = inv(ycytncm)
    tmp1 = dot(data,dot(ycytncminv,data.transpose()))
    sign, logdet = slogdet(ycytncm)
    print(i/1000., tmp1, logdet,sign*1, 0.5*(tmp1 + logdet))
    t1 = time.time()
    print('time execution took:', t1-t0)
    newmlls[pos]= 0.5*(tmp1 + logdet)
    pos+=1

pnew, taunew, taunew_err=summarizetau(newmlls)
assert(False)

'''
if dofidlike:
    print ('For fid gauss:')
    pfid, taufid, taufid_err=summarizetau(fmlls)
if dogdlike:
    print ('For gauss det:')
    pgd, taugd, taugd_err=summarizetau(mlls)
if donewlike:
    print ('For new likelihood:')
    pnew, taunew, taunew_err=summarizetau(newmlls)

if dofidlike and dogdlike and donewlike:
    figure(2);clf()
    plot(taus, pfid, label=r'Fiducial Gaussian $ \tau ={:1.4f}\pm{:1.4f}$'.format(taufid,taufid_err))
    plot(taus, pgd,  label=r'Gaussian Determinant $\tau={:1.4f}\pm{:1.4f}$'.format(taugd,taugd_err))
    plot(taus, pnew, label=r'Simpson $\tau={:1.4f}\pm{:1.4f}$'.format(taunew,taunew_err))
    xlim(0.02, 0.08);xlabel(r'$\tau$');ylabel(r'$P$');title(r'Posteriors for $\tau$ for $l='+str(lminlike)+'-'+str(lmaxlike)+'$ for {}{}x{}{} ({})'.format(freq1,split1,freq2,split2,data_set))
    legend(loc='upper right');
    savefig(basedir+'plots/posterior_{}_{}{}x{}{}_tau_tot_like_glass_ell_{}-{}_tau_0.{:03d}-0.{:03d}{}{}.pdf'.format(data_set,freq1,split1,freq2,split2,lminlike,lmaxlike,imin,imax,linear_approx,pol_approx))
    print('save posterior under')
    print('plots/posterior_{}_{}{}x{}{}_tau_tot_like_glass_ell_{}-{}_tau_0.{:03d}-0.{:03d}{}{}.pdf'.format(data_set,freq1,split1,freq2,split2,lminlike,lmaxlike,imin,imax,linear_approx,pol_approx))
    save_posterior(taus, pfid, pgd, pnew)
else:
    if dofidlike:
        plot_posterior(taus, pfid, taufid, taufid_err, save_file='fid', title='Fiducial Gaussian')
    elif dogdlike:
        plot_posterior(taus, pgd,  taugd,  taugd_err,  save_file='gd',  title='Gaussian Determinant')
    elif donewlike:
        plot_posterior(taus, pnew, taunew, taunew_err, save_file='new', title='Simpson')

'''