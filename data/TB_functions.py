from triqs_tprf.wannier90 import *
from triqs_tprf.tight_binding import *
from pytriqs.lattice.tight_binding import energies_on_bz_path, energy_matrix_on_bz_path
import skimage.measure
import copy
from matplotlib import cm

def energies_on_bz_paths(paths, tb_lattice, n_pts=50):

    """ Given a list of k-point paths compute the eigen energies along
    the paths using n_pts discrete points for each sub-path. """

    # -- Get the reciprocal lattice vectors
    bz = BrillouinZone(tb_lattice.bl)
    k_mat = np.array(bz.units())

    n_paths = len(paths)
    n_orb = tb_lattice.NOrbitalsInUnitCell

    k = np.zeros(n_pts * n_paths)
    E = np.zeros((n_orb, n_pts * n_paths))

    k_length = 0. # accumulative k-path length

    for pidx, (ki, kf) in enumerate(paths):

        s, e = pidx * n_pts, (pidx+1) * n_pts
        E[:, s:e] = energies_on_bz_path(tb_lattice.tb, ki, kf, n_pts)

        dk = np.dot(k_mat.T, (ki - kf))
        a = np.linspace(0., 1., num=n_pts, endpoint=False)
        k_vec = a[:, None] * dk[None, :]

        k[s:e] = np.linalg.norm(k_vec, axis=1) + k_length
        k_length += np.linalg.norm(dk)

    K = np.concatenate((k[::n_pts], [2 * k[-1] - k[-2]])) # add last point for K-grid

    return k, K, E


def energy_matrix_on_bz_paths(paths, TBL, n_pts=50):

    """ Given a list of k-point paths compute the eigen energies along
    the paths using n_pts discrete points for each sub-path. """

    # -- Get the reciprocal lattice vectors
    bz = BrillouinZone(TBL.bl)
    k_mat = np.array(bz.units())

    n_paths = len(paths)
    n_orb = TBL.NOrbitalsInUnitCell

    k = np.zeros(n_pts * n_paths)
    E = np.zeros((n_orb, n_orb, n_pts * n_paths),dtype=complex)

    k_length = 0. # accumulative k-path length

    for pidx, (ki, kf) in enumerate(paths):

        s, e = pidx * n_pts, (pidx+1) * n_pts
        E[:,:, s:e] = energy_matrix_on_bz_path(TBL.tb, ki, kf, n_pts)

        dk = np.dot(k_mat.T, (ki - kf))
        a = np.linspace(0., 1., num=n_pts, endpoint=False)
        k_vec = a[:, None] * dk[None, :]

        k[s:e] = np.linalg.norm(k_vec, axis=1) + k_length
        k_length += np.linalg.norm(dk)

    K = np.concatenate((k[::n_pts], [2 * k[-1] - k[-2]])) # add last point for K-grid

    return k, K, E

def reg(k) : return tuple( int(x) for x in k)

def fract_ind_to_val(x,ind):
    ind[ind == len(x)-1] = len(x)-1-1e-6
    int_ind = [int(indi) for indi in ind]
    int_ind_p1 = [int(indi)+1 for indi in ind]
    return x[int_ind] + (x[int_ind_p1] - x[int_ind])*(np.array(ind)-np.array(int_ind))

def get_kx_ky_FS(X,Y,Z,tbl,N_kxy=10,kz=0.0):

    kx = np.linspace(0,0.5,N_kxy)
    ky = np.linspace(0,0.5,N_kxy)


    E_FS = np.zeros((2,N_kxy,N_kxy))
    for kyi in range(N_kxy):
        path_FS = [(Y/(N_kxy-1)*kyi +kz*Z, X+Y/(N_kxy-1)*kyi+kz*Z)]
        du, du, E_FS[:,:,kyi] = energies_on_bz_paths(path_FS, tbl, n_pts=N_kxy)
        
    contours = {}
    FS_kx_ky = {}
    char = {}
    for ib in range(np.shape(E_FS)[0]):
        contours[ib] = skimage.measure.find_contours(E_FS[ib,:,:],0.0)

    i = 0  
    for cb in contours:
        for ci in range(np.shape(contours[cb])[0]):
            FS_kx_ky[i] = np.vstack([fract_ind_to_val(kx,contours[cb][ci][:,0]),fract_ind_to_val(ky,contours[cb][ci][:,1]),kz*Z[2]*np.ones(len(contours[cb][ci][:,0]))]).T.reshape(-1,3)
            char[i] = {}
            for n in range(len(FS_kx_ky[i][:,0])):
                MAT = energy_matrix_on_bz_path(tbl.tb, FS_kx_ky[i][n,:],FS_kx_ky[i][n,:], n_pts=1)
                E, v = np.linalg.eig(MAT[:,:,0])
                idx = np.argmin(np.abs(E))
                char[i][n] = cm.seismic(np.real(v[idx,idx])**2)
            i += 1
    return FS_kx_ky, char

def get_E_and_v_for_kgrid(tbl, kpoints):
    hopping = tbl._hop
    hopping_vx = copy.deepcopy(hopping)
    for element in hopping:
        hopping_vx[element] = hopping_vx[element]*element[0]*(-1j)

    hopping_vy = copy.deepcopy(hopping)
    for element in hopping:
        hopping_vy[element] = hopping_vy[element]*element[1]*(-1j)  

    hopping_vz = copy.deepcopy(hopping)
    for element in hopping:
        hopping_vz[element] = hopping_vz[element]*element[2]*(-1j)

    tbl_vx = TBLattice(units = tbl.Units, hopping = hopping_vx,
                       orbital_positions = tbl.OrbitalPositions,
                       orbital_names = tbl.OrbitalNames)
    tbl_vy = TBLattice(units = tbl.Units, hopping = hopping_vy,
                       orbital_positions = tbl.OrbitalPositions,
                       orbital_names = tbl.OrbitalNames)
    tbl_vz = TBLattice(units = tbl.Units, hopping = hopping_vz,
                       orbital_positions = tbl.OrbitalPositions,
                       orbital_names = tbl.OrbitalNames)
    Nk = np.shape(kpoints)[0]
    Ek = np.zeros(Nk)
    Vk = np.zeros((Nk,3))
    for ik in range(Nk):
        Hmat = energy_matrix_on_bz_path(tbl.tb,kpoints[ik,:],kpoints[ik,:], n_pts=1)
        E, U = np.linalg.eig(Hmat[:,:,0])
        idx = np.argmin(np.abs(E))
        Ek[ik] = np.real(E[idx])

        Vxmat = energy_matrix_on_bz_path(tbl_vx.tb,kpoints[ik,:],kpoints[ik,:], n_pts=1)
        Vymat = energy_matrix_on_bz_path(tbl_vy.tb,kpoints[ik,:],kpoints[ik,:], n_pts=1)
        Vzmat = energy_matrix_on_bz_path(tbl_vz.tb,kpoints[ik,:],kpoints[ik,:], n_pts=1)
        Vx = np.dot(np.transpose(U),np.dot(Vxmat[:,:,0],U))
        Vy = np.dot(np.transpose(U),np.dot(Vymat[:,:,0],U))
        Vz = np.dot(np.transpose(U),np.dot(Vzmat[:,:,0],U))
        Vk[ik,0] = np.real(Vx[idx,idx])
        Vk[ik,1] = np.real(Vy[idx,idx])
        Vk[ik,2] = np.real(Vz[idx,idx])
    return Ek, Vk


def get_E_and_v_at_k(tbl, K):
    hopping = tbl._hop
    hopping_vx = copy.deepcopy(hopping)
    for element in hopping:
        hopping_vx[element] = hopping_vx[element]*element[0]*(-1j)

    hopping_vy = copy.deepcopy(hopping)
    for element in hopping:
        hopping_vy[element] = hopping_vy[element]*element[1]*(-1j)  

    hopping_vz = copy.deepcopy(hopping)
    for element in hopping:
        hopping_vz[element] = hopping_vz[element]*element[2]*(-1j)

    tbl_vx = TBLattice(units = tbl.Units, hopping = hopping_vx,
                       orbital_positions = tbl.OrbitalPositions,
                       orbital_names = tbl.OrbitalNames)
    tbl_vy = TBLattice(units = tbl.Units, hopping = hopping_vy,
                       orbital_positions = tbl.OrbitalPositions,
                       orbital_names = tbl.OrbitalNames)
    tbl_vz = TBLattice(units = tbl.Units, hopping = hopping_vz,
                       orbital_positions = tbl.OrbitalPositions,
                       orbital_names = tbl.OrbitalNames)


    du, du, Hmat = energy_matrix_on_bz_paths([(K,K)], tbl, n_pts=2)
    E, U = np.linalg.eig(Hmat[:,:,0])
    idx = np.argmin(np.abs(E))
    Vxmat = energy_matrix_on_bz_path(tbl_vx.tb,kpoints[ik,:],kpoints[ik,:], n_pts=1)
    Vymat = energy_matrix_on_bz_path(tbl_vy.tb,kpoints[ik,:],kpoints[ik,:], n_pts=1)
    Vzmat = energy_matrix_on_bz_path(tbl_vz.tb,kpoints[ik,:],kpoints[ik,:], n_pts=1)
    V = {}
    V['x'] = np.dot(np.transpose(U),np.dot(Vxmat[:,:,0],U))
    V['y'] = np.dot(np.transpose(U),np.dot(Vymat[:,:,0],U))
    V['z'] = np.dot(np.transpose(U),np.dot(Vzmat[:,:,0],U))
    V['x'] = V['x'][idx,idx]
    V['y'] = V['y'][idx,idx]
    V['z'] = V['z'][idx,idx]
    return E[idx], V

def fermi_dis(E,beta=40.0):
    return (beta / (np.exp(E * beta) + 1))  * (1.0 / (np.exp(-E * beta) + 1))

def calc_N(tbl, nk=100):
    Nk = np.array([nk])
    E = energies_on_bz_grid(tbl.tb,nk)
    return 2.0*np.sum(E < 0.0)/(nk**3.0)

# Load data and construct tight-binding objects
def load_data():
    TBL = {}

    fn = ['775','863','956']
    modelname = ['LNO -2.3%','LNO unstrained','LNO +2.4%']
    for ifni, fni in enumerate(fn):

        # -- Read Wannier90 results
        path = '/home/manuel/TESTS/LaNiO3_new_FS/LNO_ap3_'+fni+'_W90/'
        hopping, num_wann = parse_hopping_from_wannier90_hr_dat(path + 'w2w_hr.dat')
        units = parse_lattice_vectors_from_wannier90_wout(path + 'w2w.wout')
        E_ref, w_ref = parse_band_structure_from_wannier90_band_dat(path + 'w2w_band.dat')
        TBL[modelname[ifni]] = TBLattice(units = units, hopping = hopping, orbital_positions = [(0,0,0)]*num_wann,
                             orbital_names = [str(i) for i in xrange(num_wann)])

    path = '/home/manuel/TESTS/LaNiO3_new_FS/LNO_ap3_863_W90/'
    hopping, num_wann = parse_hopping_from_wannier90_hr_dat(path + 'w2w_hr.dat')
    units = parse_lattice_vectors_from_wannier90_wout(path + 'w2w.wout')    

    hopping_c = copy.deepcopy(hopping)
    hopping_c[(0,0,0)] = TBL[modelname[0]]._hop[(0,0,0)]
    TBL['LNO unstrained with crystal field from -2.3% strain'] = TBLattice(units = units, hopping = hopping_c, orbital_positions = [(0,0,0)]*num_wann,
                             orbital_names = [str(i) for i in xrange(num_wann)])

    hopping_t = copy.deepcopy(hopping)
    hopping_t[(0,0,0)] = TBL[modelname[2]]._hop[(0,0,0)]
    TBL['LNO unstrained with crystal field from +2.3% strain'] = TBLattice(units = units, hopping = hopping_t, orbital_positions = [(0,0,0)]*num_wann,
                             orbital_names = [str(i) for i in xrange(num_wann)])

    return TBL

