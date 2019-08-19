import numpy as np
from TB_functions import *
import copy
from triqs_tprf.wannier90 import *
from matplotlib import cm
from triqs_tprf.tight_binding import *
from pytriqs.lattice.tight_binding import energies_on_bz_path, energy_matrix_on_bz_path
from matplotlib import pylab as plt 
import skimage.measure
import matplotlib.patheffects as pe
import matplotlib

# Define K-Points and PATH
G = np.array([ 0.00, 0.00, 0.00])
X = np.array([+0.50, 0.00, 0.00])
Y = np.array([+0.00,+0.50, 0.00])

M = np.array([+0.50,-0.50, 0.00])
Z = np.array([+0.00,+0.00,+0.50])
R = np.array([+0.50, 0.00,+0.50])
A = np.array([+0.50,+0.50,+0.50])

paths = [(G, X), (X, M), (M, G), (G, Z), (Z, R), (R, A), (A, Z)]

TBL = load_data()

def func(model = TBL['LNO unstrained'], crystal_field = 0.0, hopping_x = 1.0, hopping_z=1.0, kz_plot1 = 0.0,
         kz_plot2 = 0.5, Nk_path = 20, Nk_FS = 10):
    
    lw = 5
    hopping = copy.deepcopy(model._hop)
    hopping[(0,0,0)] += np.array([[-crystal_field/2.0,0],[0,crystal_field/2.0]])
    hopping[(1,0,0)][0,0] *= hopping_x
    hopping[(0,1,0)][0,0] *= hopping_x
    hopping[(-1,0,0)][0,0] *= hopping_x
    hopping[(0,-1,0)][0,0] *= hopping_x
    hopping[(0,0,1)][0,0] *= hopping_z
    hopping[(0,0,-1)][0,0] *= hopping_z
    
    hopping[(1,0,0)][1,1] *= hopping_x
    hopping[(0,1,0)][1,1] *= hopping_x
    hopping[(-1,0,0)][1,1] *= hopping_x
    hopping[(0,-1,0)][1,1] *= hopping_x
    hopping[(0,0,1)][1,1] *= hopping_z
    hopping[(0,0,-1)][1,1] *= hopping_z
   
    hopping[(1,0,0)][0,1] *= hopping_x
    hopping[(0,1,0)][0,1] *= hopping_x
    hopping[(-1,0,0)][0,1] *= hopping_x
    hopping[(0,-1,0)][0,1] *= hopping_x
    hopping[(0,0,1)][0,1] *= hopping_z
    hopping[(0,0,-1)][0,1] *= hopping_z
    
    hopping[(1,0,0)][1,0] *= hopping_x
    hopping[(0,1,0)][1,0] *= hopping_x
    hopping[(-1,0,0)][1,0] *= hopping_x
    hopping[(0,-1,0)][1,0] *= hopping_x
    hopping[(0,0,1)][1,0] *= hopping_z
    hopping[(0,0,-1)][1,0] *= hopping_z
    
    tbl = TBLattice(units = model.Units, hopping = hopping, orbital_positions = model.OrbitalPositions,
                    orbital_names = model.OrbitalNames)
    
    beta = 5.0
    kx = np.linspace(0.0,0.5,Nk_FS)
    KX,KY,KZ = np.meshgrid(kx,kx,kx)
    k_points = np.vstack([KX.ravel(), KY.ravel(),KZ.ravel()]).T.reshape(-1,3)
    sigma = np.zeros(3)

    Vol = tbl.Units[0][0]*tbl.Units[1][1]*tbl.Units[2][2]
    E, V = get_E_and_v_for_kgrid(tbl, k_points)
    for ik in range(len(E)):
        sigma[0] += V[ik,0]**2.0 *fermi_dis(E[ik],beta=beta)
        sigma[1] += V[ik,1]**2.0 *fermi_dis(E[ik],beta=beta)
        sigma[2] += V[ik,2]**2.0 *fermi_dis(E[ik],beta=beta)

    sigma = sigma/(len(E)*Vol)*1e3

    #print calc_N(tbl,nk=NN)

    k, K, E_mat = energy_matrix_on_bz_paths(paths, tbl, Nk_path)
    E = np.zeros((2,np.shape(E_mat)[2]),dtype=complex)
    v = np.zeros((2,2,np.shape(E_mat)[2]),dtype=complex)
    for ik in range(np.shape(E_mat)[2]):
        E[:,ik], v[:,:,ik] = np.linalg.eig(E_mat[:,:,ik])
    
    fig = plt.figure()
    plt.subplot(1,3,1)

    for bidx in xrange(E.shape[0]):
        color = v[bidx,1]**2
        plt.scatter(k, np.real(E[bidx]), c=cm.seismic(np.real(color)), edgecolor='none',linewidth=2)
        plt.plot([0,300],[0,0],'k-',label=None)

    plt.grid()
    plt.xticks(K,[r'$\Gamma$',r'$X$',r'$M$',r'$\Gamma$',r'$Z$',r'$R$',r'$A$',r'$Z$'])
    plt.xlim([K.min(), K.max()])
    plt.ylabel(r"$\epsilon_{\mathbf{k}}$")
    plt.ylim(-2,4)

    kz_values = [kz_plot1,kz_plot2]
    for ikz, kz in enumerate(kz_values):
        plt.subplot(1,1+len(kz_values),ikz+2)
        
        FS_kx_ky, char = get_kx_ky_FS(X,Y,Z,tbl,N_kxy=Nk_FS,kz=kz)

        for i in range(len(FS_kx_ky)):
                for n in range(len(FS_kx_ky[i][:,0])):
                    plt.plot(FS_kx_ky[i][n:n+2,0],FS_kx_ky[i][n:n+2,1], '-', lw=lw, solid_capstyle='round', color = char[i][n])
                    plt.plot(-FS_kx_ky[i][n:n+2,0],FS_kx_ky[i][n:n+2,1], '-', lw=lw, solid_capstyle='round', color = char[i][n])
                    plt.plot(-FS_kx_ky[i][n:n+2,0],-FS_kx_ky[i][n:n+2,1], '-', lw=lw, solid_capstyle='round', color = char[i][n])
                    plt.plot(FS_kx_ky[i][n:n+2,0],-FS_kx_ky[i][n:n+2,1], '-', lw=lw, solid_capstyle='round', color = char[i][n])
                i += 1
        plt.xlim(-0.5,0.5)
        plt.ylim(-0.5,0.5)
        plt.title(r'$k_z$ = '+str(round(kz/2.0,3)))
        plt.xlabel(r'$k_x$')
        plt.ylabel(r'$k_y$')
        
    im = plt.scatter([0,0], [0,0], s = 1, c = [0,1], cmap = cm.seismic)
    cbaxes = fig.add_axes([0.42, 0.18, 0.01, 0.2]) 
    cb = plt.colorbar(im, cax = cbaxes,ticks=[0,1])
    cb.ax.set_yticklabels(['z2', 'x2y2'])  # vertically oriented colorbar
    plt.tight_layout()
    plt.show()
    print 'sigma_x ~', np.round(sigma[0],3), ';  sigma_z ~', np.round(sigma[2],3), 

