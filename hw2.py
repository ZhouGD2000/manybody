import numpy as np
# !/ usr / bin / env python
from pyscf import gto , scf, cc ,mp
from matplotlib import pyplot as plt
import pyscf

mol = gto.M (unit = 'Bohr',atom = '''H 0 0 0 ;H 1.2 0 0 ''',basis = 'sto-3g')
rhf_h2 = scf.RHF( mol ).run()
ccsd_h2=cc.CCSD(rhf_h2)

e_ccsd=ccsd_h2.kernel()

e_h2 = rhf_h2.kernel ()
print(e_h2)

mp_h2=mp.MP2(rhf_h2)
e_mp=mp_h2.kernel()
#rhf_h2.analyze () # Orbital analysis


mol1 = pyscf.M(unit = 'Bohr',
        atom = 'H 0 0 0; H 1.2 0 0', 
        basis = 'sto-3g'
    )

mf = mol1.KS()
mf.xc = 'pbe'
mf.kernel()

mf.xc = 'lda'
mf.kernel()


mf.xc = 'pbe0'
mf.kernel()

'''
A simple example to run DFT calculation.

See also pyscf/dft/libxc.py and pyscf/dft/xcfun.py for the complete list of
available XC functionals.
'''


erhf=[]
epbe=[]
elda=[]
epbe0=[]

ra=np.arange(0.5,20,0.2)
e=[[] for i in range(len(ra))]
for x in ra:
    mol = gto.M (unit = 'Bohr',atom = '''H 0 0 0 ;H %f 0 0 '''%x,basis = 'sto-3g')
    rhf_h2 = scf.RHF ( mol )
    erhf.append(rhf_h2.kernel())
    print(x)
    mol1 = pyscf.M(unit = 'Bohr',
        atom = 'H 0 0 0; H %f 0 0'%x, 
        basis = 'sto-3g'
    )

    mf = mol1.KS()
    mf.xc = 'pbe'
    epbe.append(mf.kernel())
    mf.xc = 'pbe0'
    epbe0.append(mf.kernel())
    mf.xc = 'lda'
    elda.append(mf.kernel())
    
plt.plot(ra,erhf,label="rhf")
plt.plot(ra,epbe,label="pbe")
plt.plot(ra,epbe0,label="pbe0")
plt.plot(ra,elda,label="lda")
plt.legend()
plt.savefig("compare1.png")



# Orbital energies, Mulliken population etc.
mf.analyze()