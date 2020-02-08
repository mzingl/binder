import warnings; warnings.simplefilter('ignore')
import numpy as np
import copy
from triqs_tprf.wannier90 import *
from pytriqs.lattice.tight_binding import energies_on_bz_path
from matplotlib import cm
from triqs_tprf.tight_binding import *
from matplotlib import pylab as plt
import skimage.measure
import ipywidgets as widgets
from TB_functions import *
import matplotlib

matplotlib.rcParams.update({'font.size':35})
matplotlib.rcParams['figure.figsize'] = (25,10)
matplotlib.rcParams['xtick.major.pad']='10'
matplotlib.rcParams['ytick.major.pad']='10'
matplotlib.rc('xtick', labelsize=35) 
matplotlib.rc('ytick', labelsize=35)

# Define K-Points and PATH
G = np.array([ 0.00, 0.00, 0.00])
X = np.array([+0.50, 0.00, 0.00])
Y = np.array([+0.00,+0.50, 0.00])
M = np.array([+0.50,-0.50, 0.00])
Z = np.array([+0.00,+0.00,+0.50])
R = np.array([+0.50, 0.00,+0.50])
A = np.array([+0.50,+0.50,+0.50])
paths = [(G, X), (X, M), (M, G), (G, Z), (Z, R), (R, A), (A, Z)]

# Load Wannier model
TBL = get_TBL('/home/jovyan/LNO_ap3_863_W90/')
#TBL = get_TBL('/home/manuel/TESTS/binder/data/LNO_ap3_863_W90/')

# Plot function
def func(d_cf_z2 = 0.0, d_cf_x2y2 = 0.0, hopping_x = 1.0, hopping_z=1.0, 
         kz = 0.0, Nk_path = 20, Nk_FS = 10):

    lw = 10
    sign = [1,-1]
    
    # Modify hopping
    hopping = copy.deepcopy(TBL._hop)
    for h in hopping:
        if not any(h):
            hopping[(0,0,0)] += np.array([[d_cf_z2,0],[0,d_cf_x2y2]])
       # elif not any(h[0:2]):
       #hopping[h] *= hopping_z
        elif h[2] == 0:
            hopping[h] *= hopping_x
        else:
            hopping[h] *= hopping_z

    tbl = TBLattice(units = TBL.Units, hopping = hopping, orbital_positions = TBL.OrbitalPositions,
                    orbital_names = TBL.OrbitalNames)
    matplotlib.rcParams.update({'font.size':35})
    matplotlib.rcParams['figure.figsize'] = (25,10)
    matplotlib.rcParams['xtick.major.pad']='10'
    matplotlib.rcParams['ytick.major.pad']='10'
    matplotlib.rc('xtick', labelsize=35) 
    matplotlib.rc('ytick', labelsize=35)
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.subplots_adjust(wspace=0.4)
    
    # Band structure plot
    k, K, E = energies_on_bz_paths(paths, tbl, Nk_path)
    for bidx in xrange(E.shape[0]):
        plt.plot(k, np.real(E[bidx,:]),'ko',ms=4)
    plt.plot([0,300],[0,0],'k-',label=None)
    plt.grid()
    plt.xticks(K,[r'$\Gamma$',r'X',r'M',r'$\Gamma$',r'Z',r'R',r'A',r'Z'])
    plt.xlim([K.min(), K.max()])
    plt.ylabel(r"$\epsilon_{\mathbf{k}}$ (eV)")
    plt.ylim(-2,4)
    
    # FS plots
    
    plt.subplot(1,2,2)

    FS_kx_ky, char = get_kx_ky_FS(X,Y,Z,tbl,k_trans_back=np.eye(3),N_kxy=Nk_FS,kz=2.0*kz)

    for i in range(len(FS_kx_ky)):
            for n in range(len(FS_kx_ky[i][:,0])):
                dummy, v = get_E_and_v_at_k(tbl,FS_kx_ky[i][n,:])
                color = cm.plasma(np.real((v['x']**2+v['y']**2)/2.0))
                for s1 in sign:
                    for s2 in sign:
                        plt.plot(s1*FS_kx_ky[i][n:n+2,0],s2*FS_kx_ky[i][n:n+2,1], '-', 
                                 lw=lw, color=color, solid_capstyle='round')
            i += 1

   
    plt.title(r'$k_z$ = '+str(round(kz,3)))
    plt.xlabel(r'$k_x$')
    plt.ylabel(r'$k_y$')
    plt.xticks([-0.5,0.0,0.5])
    plt.yticks([-0.5,0.0,0.5])
    plt.axis('equal') 
    plt.xlim(-0.55,0.55)
    plt.ylim(-0.55,0.55)
    
    im = plt.scatter([0,0], [0,0], s = 1, c = [0,1], cmap = cm.plasma)
    cbaxes = fig.add_axes([0.92, 0.30, 0.01, 0.4])
    cb = plt.colorbar(im, cax = cbaxes,ticks=[0,1])
    cb.ax.set_yticklabels(['min','max'])  # vertically oriented colorbar
    cb.ax.set_title(r' $v^2$')
    plt.show()

# widgets
style = {'description_width': '100px'}
d_cf_z2_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.01, value=0.0, 
                                     description=r'$\Delta \epsilon_{z^2}$', 
                                     continuous_update=False)
d_cf_x2y2_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.01, value=0.0, 
                                       description=r'$\Delta \epsilon_{x^2-y^2}$', 
                                       continuous_update=False)
s_h_ab_slider = widgets.FloatSlider(min=0.0, max=2.0, step=0.01, value=1.0, 
                                    description=r'$t_{ab}$ scaling', 
                                    continuous_update=False)
s_h_c_slider = widgets.FloatSlider(min=0.0, max=2.0, step=0.01, value=1.0, 
                                   description=r'$t_{c}$ scaling', 
                                   continuous_update=False)
kz_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.0, 
                                description=r'$k_z$', 
                                continuous_update=False)
Nk_path_slider = widgets.IntSlider(min=5, max=50, step=5, value=10, 
                                   description=r'$N_k$', 
                                   continuous_update=False)
Nk_FS_slider = widgets.IntSlider(min=5, max=50, step=5, value=10, 
                                 description=r'$N_{FS}$', 
                                 continuous_update=False)
reset_button = widgets.Button(description = "Reset")

def reset_values(b):
    ui.children[1].children[2].value = 10
    ui.children[1].children[1].value = 10
    ui.children[0].children[0].value = 0.0
    ui.children[0].children[1].value = 0.0
    ui.children[0].children[2].value = 1.0
    ui.children[0].children[3].value = 1.0
    ui.children[1].children[0].value = 0.0

reset_button.on_click(reset_values)


out = widgets.interactive_output(func, {'d_cf_z2':d_cf_z2_slider, 'd_cf_x2y2':d_cf_x2y2_slider, 
                                        'hopping_x':s_h_ab_slider, 'hopping_z':s_h_c_slider,
                                        'kz':kz_slider, 'Nk_path':Nk_path_slider, 'Nk_FS':Nk_FS_slider});
out.layout.height = '800px'
out.layout.width = '800px'
    
ui = widgets.HBox([widgets.VBox([d_cf_z2_slider,d_cf_x2y2_slider,s_h_ab_slider,s_h_c_slider]), 
                         widgets.VBox([kz_slider, Nk_path_slider, Nk_FS_slider]),  widgets.VBox([reset_button])] )
