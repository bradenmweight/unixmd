from __future__ import division
from build.cioverlap import wf_overlap 
from qm.gaussian09.gaussian09 import Gaussian09
from misc import au_to_A, eV_to_au, call_name
import os, shutil, re, textwrap, subprocess
import numpy as np
import subprocess as sp

class DFT_G16(Gaussian09):
    """ Class for the (TD)DFT method of Gaussian 09

        :param object molecule: Molecule object
        :param string functional: Exchange-correlation functional information
        :param string basis_set: Basis set information
        :param string memory: Allocatable memory
        :param string guess: Initial guess for SCF iterations
        :param string guess_file: Initial guess file
        :param string root_path: Path for Gaussian 09 root directory
        :param integer nthreads: Number of threads in the calculations
        :param string version: Version of Gaussian 09
    """
    def __init__(self, molecule, nthreads=1, memory="1gb", functional="BLYP", basis_set="STO-3G", \
        guess="Harris", guess_file="./g16.chk", root_path="./", version="Revision A.02", dielectric_material=None):
        # Initialize Gaussian09 common variables
        super(DFT_G16, self).__init__(basis_set, memory, nthreads, root_path, version)

        # Initialize Gaussian09 DFT variables
        self.functional = functional
        self.dielectric_material = dielectric_material

        # Set initial guess for DFT calculation
        self.guess = guess.lower()
        self.guess_file = os.path.abspath(guess_file)
        if not (self.guess in ["harris", "read"]):
            error_message = "Invalid initial guess for DFT!"
            error_vars = f"guess = {self.guess}"
            raise ValueError (f"( {self.qm_method}.{call_name()} ) {error_message} ( {error_vars} )")

        # Set 'l_nacme' with respect to the computational method
        molecule.l_nacme = True

        # Re-calculation of excited state forces is not needed for ground state dynamics
        if (molecule.nst > 1):
            self.re_calc = True
        else:
            self.re_calc = False

        # MO dimension, initialized later by reading Gaussian09 log
        self.nbasis = 0
        self.norb = 0
        self.nfc = 0
        self.nocc = 0
        self.nvirt = 0
 
        # Temporaries for NACME calculation, also initialized later if not allocated
        # ao_overlap - the number of AOs, the number of AOs
        # mo_coef - the number of MOs, the number of AOs
        # ci_coef - the number BO states, the number of occ, the number of virt
        self.pos_old = []
        self.ao_overlap = []
        self.mo_coef_old = []
        self.mo_coef_new = []
        self.ci_coef_old = []
        self.ci_coef_new = []

    def get_data(self, molecule, base_dir, bo_list, dt, istep, calc_force_only):
        """ Extract energy, gradient from (TD)DFT method

            :param object molecule: Molecule object
            :param string base_dir: Base directory
            :param integer,list bo_list: List of BO states for BO calculation
            :param double dt: Time interval
            :param integer istep: Current MD step
            :param boolean calc_force_only: Logical to decide whether calculate force only
        """
        self.copy_files(molecule, istep, calc_force_only)
        super().get_data(base_dir, calc_force_only)
        self.get_input(molecule, istep, bo_list, calc_force_only)
        self.run_QM(base_dir, istep, bo_list)
        self.extract_QM(molecule, istep, bo_list, dt, calc_force_only)
        self.move_dir(base_dir)

    def copy_files(self, molecule, istep, calc_force_only):
        """ Copy necessary scratch files in previous step

            :param object molecule: Molecule object
            :param integer istep: Current MD step
            :param boolean calc_force_only: Logical to decide whether calculate force only
        """
        #print ( "Current Path:", os.path.cwd() )
        
        # Copy required files for NACME
        if (molecule.nst > 1 and not calc_force_only and istep >= 0):
            if (istep == 0):
                print ( "Gaussian Scratch Sapce:", self.scr_qm_dir)
                shutil.copy( f"{self.scr_qm_dir}/g16.rwf", f"{self.scr_qm_dir}/../g16.rwf.pre" )
                #shutil.copy(os.path.join(self.scr_qm_dir, "g16.rwf"), \
                #    os.path.join(self.scr_qm_dir, "../g16.rwf.pre"))

        # Copy required files to read initial guess
        if (self.guess == "read" and istep >= 0):
            # After T = 0.0 s
            shutil.copy( f"{self.scr_qm_dir}/g16.chk", f"{self.scr_qm_dir}/../g16.chk.pre" )
            #shutil.copy(os.path.join(self.scr_qm_dir, "g16.chk"), \
            #    os.path.join(self.scr_qm_dir, "../g16.chk.pre"))

    def get_input(self, molecule, istep, bo_list, calc_force_only):
        """ Generate Gaussian 09 input files: g16.inp

            :param object molecule: Molecule object
            :param integer istep: Current MD step
            :param integer,list bo_list: List of BO states for BO calculation
            :param boolean calc_force_only: Logical to decide whether calculate force only
        """
        # Read check-point file from previous step
        if (self.guess == "read"):
            if (istep == -1):
                if (os.path.isfile(self.guess_file)):
                    # Copy guess file to currect directory
                    shutil.copy( self.guess_file, f"{self.scr_qm_dir}/g16.chk" )  # os.path.join(self.scr_qm_dir, "g16.chk"))
                    restart = True
                else:
                    # TODO : Printout about reading a checkpoint file for the initial guess
                    # print(f"( {self.qm_method}.{call_name()} ) Make the initial guess of density only for the 1st step.\n", flush=True)
                    restart = False
            elif (istep >= 0):
                # Move previous file to currect directory
                os.rename("../g16.chk.pre", "./g16.chk")
                restart = True
        elif (self.guess == "harris"):
            restart = False

        if (calc_force_only):
            restart = True


        def getSplitBasis(basis_set):
            """
            Looking for input as string with '//' separating "atom/basis" items
            Example: "Ti/LANL2DZ//C/3-21G//O/3-21G//F/6-31G*"
            """
            atom_basis = [ j.split('/') for j in basis_set.split("//") ]
            NAtoms_unique = len(atom_basis)
            QECP = False # Check if we need another section for ECP (e.g., LANL2DZ)
            ECP_list = ['LANL2DZ'] # Manually add here to search for more. Otherwise, an error will occur.
            ECP_out = ''

            basis_list = "\n"
            for at in atom_basis:
                if ( at[1].upper() in ECP_list ):
                    QECP = True
                    ECP_list = f'{at[0]} 0\n{at[1]}\n'
                    ECP_out = f'pseudo=read'
                basis_list += "  " + at[0] + " 0\n"
                basis_list += "  " + at[1] + " \n****\n"
            basis_set = "GEN"

            return basis_set, basis_list, ECP_out, ECP_list

        # Indicates that the user wants a split basis set. Need to specify for all atoms.
        # Example: Ti/LANL2DZ//C/3-21G//O/3-21G//F/6-31G*
        QSplitBasis = False
        basis_header = self.basis_set
        ECP_out = ''
        basis_list = ''
        ECP_list = ''
        if ( "//" in self.basis_set ):
            if (istep == 0): print ("Found split basis set.")
            QSplitBasis = True
            basis_header, basis_list, ECP_out, ECP_list = getSplitBasis(self.basis_set)


        # Indicates that user wants a solvent cavity. Only CPCM is implemented with
        #   choice of dielectric constant.
        solvent_method = ""
        if ( self.dielectric_material != None ):
            solvent_method = f"scrf=(smd,solvent={self.dielectric_material},read)"
            solvent_list = "surface=sas"

        # Make 'g16.inp' file
        input_g16 = ""

        # Ground-state calculation
        input_route = textwrap.dedent(f"""\
        %nproc={self.nthreads}
        %mem={self.memory}
        %chk=g16.chk\n""")
 
        input_route += textwrap.dedent(f"""\
        # {self.functional}/{basis_header} {ECP_out} \n# {solvent_method} nosymm""")

        if (restart):
            input_route += f" guess=read"

        if (bo_list[0] == 0):
            input_route += f" force"

        if (calc_force_only):
            input_route += f" geom=allcheck"
        input_route += "\n"

        input_g16 += input_route

        if (not calc_force_only):
            # Title section block
            input_g16 += f"\ng16 Ground State Calculation\n\n"

            # Molecule specification block
            input_molecule = textwrap.dedent(f"""\
            {int(molecule.charge)} 1
            """)
            for iat in range(molecule.nat_qm):
                list_pos = list(molecule.pos[iat] * au_to_A)
                input_molecule += \
                    f"{molecule.symbols[iat]}{list_pos[0]:15.8f}{list_pos[1]:15.8f}{list_pos[2]:15.8f}\n"
            input_g16 += input_molecule #+ "\n"
            
        if( basis_header == 'GEN' ):
            input_g16 += basis_list + "\n"
        else:
            input_g16 += "\n"
        if( ECP_out != '' ): input_g16 += ECP_list + "\n"
        if ( solvent_method != '' ): input_g16 += solvent_list + "\n"
        input_g16 += "\n\n\n"

        # Excited-state calculation
        if (molecule.nst > 1 and not (calc_force_only and bo_list[0] == 0)):
            input_route = textwrap.dedent(f"""\
            --Link1--
            %nproc={self.nthreads}
            %mem={self.memory}
            %chk=g16.chk\n""")

            if (not calc_force_only):
                input_route += f"""%rwf=g16.rwf\n"""

            input_route += f"""# {self.functional}/{basis_header} {ECP_out} {solvent_method}\n# td(Root={bo_list[0]}, Nstates={molecule.nst - 1})"""\
            """ geom=allcheck guess=read nosymm"""

            if (bo_list[0] > 0):
                input_route += " force"
            input_g16 += input_route + "\n"
            
            if( basis_header == 'GEN' ):
                input_g16 += basis_list + "\n"
            else:
                input_g16 += "\n"
            if( ECP_out != '' ): input_g16 += ECP_list + "\n"
            if ( solvent_method != '' ): input_g16 += solvent_list + "\n"
            input_g16 += "\n\n\n"

        # Write "doubled molecule" input
        if (self.calc_coupling and molecule.nst > 1 and not calc_force_only and istep >= 0):
            if (istep == 0):
                os.rename('../g16.rwf.pre', './g16.rwf.pre')
 
            # Stop the run after L302 calculating overlap
            # Keep running the job regardless of interatomic distances; IOp(2/12=3)
            input_route = textwrap.dedent(f"""\
--Link1--
%kjob l302
%rwf=g16_double.rwf
# {self.functional}/{basis_header} {ECP_out} \n# {solvent_method} IOp(2/12=3) {"IOp(3/41=2000)" * (self.functional in ['AM1', 'PM3', 'PM6', 'PM7'])} nosymm\n\n""")
 
            input_g16 += input_route
 
            # Title section block
            input_title = f"g16 double input\n\n"
            input_g16 += input_title
 
            # Molecule specification block
            input_molecule = textwrap.dedent(f"""\
            {2 * int(molecule.charge)} 1
            """)
            for iat in range(molecule.nat_qm):
                list_pos = list(self.pos_old[iat] * au_to_A)
                input_molecule += \
                    f"{molecule.symbols[iat]}{list_pos[0]:15.8f}{list_pos[1]:15.8f}{list_pos[2]:15.8f}\n"
 
            for iat in range(molecule.nat_qm):
                list_pos = list(molecule.pos[iat] * au_to_A)
                input_molecule += \
                    f"{molecule.symbols[iat]}{list_pos[0]:15.8f}{list_pos[1]:15.8f}{list_pos[2]:15.8f}\n"
            #input_molecule += "\n"
            input_g16 += input_molecule

            if( basis_header == 'GEN' ):
                input_g16 += basis_list + "\n"
            else:
                input_g16 += "\n"
            if( ECP_out != '' ): input_g16 += ECP_list + "\n"
            if ( solvent_method != '' ): input_g16 += solvent_list + "\n"
            input_g16 += "\n\n\n"

        file_name = "g16.inp"
        with open(file_name, "w") as f:
            f.write(input_g16)

    def run_QM(self, base_dir, istep, bo_list):
        """ Run (TD)DFT calculation and save the output files to qm_log directory

            :param string base_dir: Base directory
            :param integer istep: Current MD step
            :param integer,list bo_list: List of BO states for BO calculation
        """
        # Set environment variables
        if (istep == -1):
            os.environ["g16root"] = self.root_path
            os.environ["GAUSS_SCDIR"] = self.scr_qm_dir
            path_profile =  f"{self.root_path}/bsd/g16.profile"  #os.path.join(self.root_path, "/bsd/g16.profile") # BMW, OLD = "g16/bsd/g16.profile"
            command = f'env -i sh -c "source {path_profile} && env"' # BMW
            #print ("!!!!", command)
            #print ("!!!!", subprocess.getoutput(command).split('\n'))
            
            # BMW, I commented this out because it was causing issues. Seems to work fine without it ...
            """
            for line in subprocess.getoutput(command).split("\n"):
                if ( "command not found" in line ): continue
                print ("LINE = ", line.split("="))
                key, value = line.split("=")
                os.environ[key] = value
            """
        # Set run command
        qm_command = f"{self.root_path}/g16"    # os.path.join(self.root_path, "g16/g16") # BMW
        command = f"{qm_command} < g16.inp > log"

        # Run Gaussian09
        os.system(command)

        # Copy the output file to 'qm_log' directory
        tmp_dir = f"{base_dir}/qm_log"  #os.path.join(base_dir, "qm_log")
        if (os.path.exists(tmp_dir)):
            log_step = f"log.{istep + 1}.{bo_list[0]}"
            shutil.copy("log", f"{tmp_dir}/log_step" )  # os.path.join(tmp_dir, log_step))

    def extract_QM(self, molecule, istep, bo_list, dt, calc_force_only):
        """ Read the output files to get BO information

            :param object molecule: Molecule object
            :param integer istep: Current MD step
            :param integer,list bo_list: List of BO states for BO calculation
            :param double dt: Time interval
            :param boolean calc_force_only: Logical to decide whether calculate force only
        """
        file_name = "log"
        with open(file_name, "r") as f:
            log = f.read()

        # Check the convergence of the calculation
        if ("Convergence failure" in log):
            error_message = "SCF iteration not converged, please see the output carefully!"
            error_vars = f"output file = {self.scr_qm_dir}/{file_name}"
            raise Exception (f"( {self.qm_method}.{call_name()} ) {error_message} ( {error_vars} )")

        # Energy
        if (not calc_force_only):
            # Read ground energy
            if ( self.functional in ['AM1', 'PM3', 'PM6', 'PM7'] ):
                energy_lines = sp.check_output( "grep 'SCF Done' log" , shell=True).decode("utf-8").split('\n')[:-1]
                energy = [ j.split()[4] for j in energy_lines ]
            else:
                energy = re.findall('SCF Done:\s+E\(\S+\)\s+=\s+([-]\S+)\s+A.U.', log)
            energy = np.array(energy[0], dtype=np.float64)
            molecule.states[0].energy = energy

            if (molecule.nst > 1):
                energy = re.findall('Excited\sState\s+\w+:\s+\w+-\S+\s+(\S+)\s+eV', log)
                energy = np.array(energy, dtype=np.float64)
                energy *= eV_to_au
                for ist in range(1, molecule.nst):
                    molecule.states[ist].energy = molecule.states[0].energy + energy[ist - 1]

        # Force
        tmp_f = "Forces\s+\(Hartrees\/Bohr\)\n.+\n.+" \
            + "\n\s+\d*\s+\d*\s+([-]*\S+)\s+([-]*\S+)\s+([-]*\S+)" * molecule.nat_qm
        force = re.findall(tmp_f, log)
        force = np.array(force[0], dtype=np.float64)
        force = force.reshape(molecule.nat_qm, 3, order='C')
        molecule.states[bo_list[0]].force = np.copy(force)

        # NACME
        if (self.calc_coupling and molecule.nst > 1 and not calc_force_only):
            if (istep == -1):
                self.init_buffer(molecule)
                self.orb_ini = np.zeros(1, dtype=np.int32)
                self.orb_final = np.zeros(1, dtype=np.int32)
                self.orb_final[0] = self.norb
            else:
                self.CI_overlap(molecule, istep, dt)
 
            # Save geometry in the buffer
            self.pos_old = np.copy(molecule.pos)

    def init_buffer(self, molecule):
        """ Initialize buffer variables to get NACME

            :param object molecule: Molecule object
        """
        file_name = "log"
        with open(file_name, "r") as f:
            log = f.read()

        self.nbasis = re.findall('NBasis=\s+(\d+)\s+', log)
        self.nbasis = int(self.nbasis[0])
        self.nfc = re.findall('NFC=\s+(\d+)\s+', log)
        self.nfc = int(self.nfc[0])
        self.nocc = re.findall('NOA=\s+(\d+)\s+', log)
        self.nocc = int(self.nocc[0])
        self.nvirt = re.findall('NVA=\s+(\d+)\s+', log)
        self.nvirt = int(self.nvirt[0])
        self.norb = self.nocc + self.nvirt

        self.pos_old = np.zeros((molecule.nat_qm, molecule.ndim))
        self.ao_overlap = np.zeros((self.nbasis, self.nbasis))
        self.mo_coef_old = np.zeros((self.norb, self.nbasis))
        self.mo_coef_new = np.zeros((self.norb, self.nbasis))
        self.ci_coef_old = np.zeros((molecule.nst, self.nocc, self.nvirt))
        self.ci_coef_new = np.zeros((molecule.nst, self.nocc, self.nvirt))

    def CI_overlap(self, molecule, istep, dt):
        """ Read the necessary files and calculate NACME from tdnac.c routine
            note that only reading of several files is required in this method

            :param object molecule: Molecule object
            :param integer istep: Current MD step
            :param double dt: Time interval
        """
        path_rwfdump = f"{self.root_path}/rwfdump" # os.path.join(self.root_path, "/rwfdump") # BMW, OLD = "g16/rwfdump"
 
        # Read overlap
        self.ao_overlap = self.read_ao_overlap(path_rwfdump, "g16_double.rwf")

        # Read mo coefficients
        if (istep == 0):
            self.mo_coef_old = self.read_mo_coef(path_rwfdump, "g16.rwf.pre") 

        self.mo_coef_new = self.read_mo_coef(path_rwfdump, "g16.rwf") 

        # Read CI coefficients
        if (istep == 0):
            self.ci_coef_old[1:] = self.read_xy_coef(molecule, path_rwfdump, "g16.rwf.pre")

        self.ci_coef_new[1:] = self.read_xy_coef(molecule, path_rwfdump, "g16.rwf") 

        # Calculate wavefunction overlap with orbital scheme
        wf_overlap(self, molecule, istep, dt)

    def read_ao_overlap(self, path_rwfdump, fn_rwf):
        """ Read a rwf file to obtain ao_overlap data

            :param string path_rwfdump: The path for rwfdump binary
            :param string fn_rwf: The name of the rwf file
        """
        os.system(path_rwfdump + f" {fn_rwf} ao_overlap.dat 514R")

        with open('ao_overlap.dat', "r") as f:
            log = f.read()

        tmp = re.findall('[-]?\d+\.\d+D[+-]\d\d', log)
        tmp = [float(x.replace('D', 'e')) for x in tmp]
 
        tmp_ovr = np.zeros((self.nbasis * 2, self.nbasis * 2))
 
        cnt = 0
        for ibasis in range(self.nbasis * 2):
            for jbasis in range(ibasis + 1):
                tmp_ovr[ibasis, jbasis] = tmp[cnt]
                cnt += 1

        tmp_ovr += np.transpose(tmp_ovr) - np.diag(np.diag(tmp_ovr))

        # Slicing the components between t and t+dt
        return tmp_ovr[:self.nbasis, self.nbasis:]

    def read_mo_coef(self, path_rwfdump, fn_rwf):
        """ Read a rwf file to obtain mo_coef data

            :param string path_rwfdump: The path for rwfdump binary
            :param string fn_rwf: The name of the rwf file
        """
        os.system(path_rwfdump + f" {fn_rwf} mo_coef.dat 524R")

        with open('mo_coef.dat', "r") as f:
            log = f.read()

        tmp = re.findall('[-]?\d+\.\d+D[+-]\d\d', log)
        tmp = np.array([x.replace('D','e') for x in tmp], dtype=np.float64)

        tmp_mo = tmp.reshape(self.nbasis, self.nbasis)

        return tmp_mo[self.nfc:self.nbasis]

    def read_xy_coef(self, molecule, path_rwfdump, fn_rwf):
        """ Read a rwf file to obtain xy_coef data

            :param object molecule: Molecule object
            :param string path_rwfdump: The path for rwfdump binary
            :param string fn_rwf: The name of the rwf file
        """
        os.system(path_rwfdump + f" {fn_rwf} xy_coef.dat 635R")

        with open(f'xy_coef.dat', "r") as f:
            log = f.read()

        tmp = re.findall('[-]?\d+\.\S+[+-]\d+', log)

        # Drop the first 12 dummy elements
        tmp = tmp[12:]
 
        # Gaussian09 deals with 4 times as much roots as the input NStates value.
        # the nr. of excitation function => nocc \times nvirt
        # spin degrees of freedom => 2
        # X+Y, X-Y => 2
        roots = (molecule.nst - 1) * 4
        num_coef = 4 * (self.nocc * self.nvirt) * roots
 
        tmp = tmp[:num_coef]
        tmp = np.array([x.replace('D','e') for x in tmp], dtype=np.float64)
        xpy, xmy = tmp.reshape(2, roots, 2, -1)
        x = 0.5 * (xpy + xmy)
 
        # Drop beta part and unrequested excited states
        x = x[:(molecule.nst - 1), 0, :]
 
        return x.reshape(-1, self.nocc, self.nvirt)
 

