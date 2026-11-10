import os, sys
from pyscf import gto
import numpy as np
topdir = os.path.abspath (os.path.join (__file__, '..'))

def structure( distance, basis):

    mol = gto.Mole()

    with open (os.path.join (topdir, "me2n2_geom.gjf"), 'r') as f:
        for i in range(6):
            f.readline()    

        number_of_atom = 10 #Modify this accordingly yo your molecule
        lines = []	
        for atom in range(number_of_atom):	
            lines.append(f.readline().split())
		
    lines[1][2] = distance  #where is the variable in the gjf
	
	#tranform Z-matrix to Cartesian
    atomlist = []
    for line in lines:
        if (len(line) == 1): atomlist.append([line[0], 0, 0, 0])  #'%s %10.8f %10.8f %10.8f'
        if (len(line) == 3): atomlist.append([line[0], float(line[2]), 0, 0])  
        if (len(line) == 5):
            if (int(line[1]) == 1): sign = 1
            if (int(line[1]) == 2): sign = -1
            x = sign*np.cos(float(line[4])* np.pi / 180.0)*float(line[2]) + float(atomlist[int(line[1])- 1][1])
            y = sign*np.sin(float(line[4])* np.pi / 180.0)*float(line[2]) + float(atomlist[int(line[1])- 1][2])	
            atomlist.append([line[0], x, y, 0]) 
        if (len(line) == 8): 
            avec = np.array(atomlist[int(line[1])- 1] [1:4])
            bvec = np.array(atomlist[int(line[3])- 1] [1:4])
            cvec = np.array(atomlist[int(line[5])- 1] [1:4])
		
            dst = float(line[2])
            ang = float(line[4]) * np.pi / 180.0
            tor = float(line[6]) * np.pi / 180.0
		
            v1 = avec - bvec
            v2 = avec - cvec
		
            n = np.cross(v1, v2)
            nn = np.cross(v1, n)
		
            n /= np.linalg.norm(n)
            nn /= np.linalg.norm(nn)
		
            n *= -np.sin(tor)
            nn *= np.cos(tor)
		
            v3 = n + nn
            v3 /= np.linalg.norm(v3)
            v3 *= dst * np.sin(ang)
		
            v1 /= np.linalg.norm(v1)
            v1 *= dst * np.cos(ang)
            position = avec + v3 - v1	
		
            atomlist.append([line[0], position[0], position[1], position[2]]) 

    #reorder
    #order =[0,14,15,16,1,6,2,7,8,3,9,10,4,11,5,12,13]
    #atomlist = [ atomlist[i] for i in order]
	
    mol.atom = atomlist
    mol.basis = { 'N': basis, 'C': basis, 'H': basis }
    mol.charge = 0
    mol.spin = 0
    mol.build()
    return mol
	
#Structure test:
'''
import sys
sys.path.append('../../QC-DMET/src')
import localintegrals, dmet, qcdmet_paths
from pyscf import gto, scf, symm, future
import numpy as np
import ME2N2_struct

basis = 'sto-6g' 
distance = 3.2
mol = ME2N2_struct.structure( distance, basis)
xyz = np.asarray(mol.atom)
for atom in xyz:
	print(atom[0],atom[1],atom[2],atom[3])
'''
