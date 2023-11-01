

import numpy as np 
import time 
#from numba import jit

try:
    from dolfin import *
    have_dolfin = True 
except ModuleNotFoundError: 
    have_dolfin = False
    print("Dolfin is not installed, cant use the Fenics EIT solver.")



class EIT():
    def __init__(self, Inj, z, mesh="dense"):
        """
        Inj: current injection pattern (32, N) matrix with the number of patterns N

        """
        assert mesh in ["dense", "sparse"], print("mesh has to be dense/sparse")

        self.mesh_name = mesh

        ### Load mesh and marker
        self.omega = Mesh()
        self.mvc = MeshValueCollection("size_t", self.omega, 1)

        if self.mesh_name == "dense":
            with XDMFFile("src/forward_operator/EIT_disk.xdmf") as infile:
                infile.read(self.omega)
            with XDMFFile("src/forward_operator/EIT_boundary.xdmf") as infile:  
                infile.read(self.mvc, "name_to_read")
        elif self.mesh_name == "sparse":
            with XDMFFile("src/forward_operator/EIT_disk_coarse.xdmf") as infile:
                infile.read(self.omega)
            with XDMFFile("src/forward_operator/EIT_boundary_coarse.xdmf") as infile:  
                infile.read(self.mvc, "name_to_read")            

        elektrode_marker = cpp.mesh.MeshFunctionSizet(self.omega, self.mvc)

        self.Inj = Inj 

        self.z = z

        ### Functionspace stuff
        V = FiniteElement("CG", self.omega.ufl_cell(), 1)
        R = FiniteElement("R", self.omega.ufl_cell(), 0)
        fn_space_list = [V, R]
        for i in range(32):
            fn_space_list.append(FiniteElement("R", self.omega.ufl_cell(), 0))
        W = MixedElement(fn_space_list)
        self.W = FunctionSpace(self.omega, W)

        self.dx = Measure("dx", domain=self.omega)
        self.ds = Measure("ds", domain=self.omega, subdomain_data=elektrode_marker) # boundary measure

        self.electrode_len = assemble(1 * self.ds(1))

        self.psi = TestFunction(self.W)
        
        self.assemble_boundary_condition()

    def assemble_boundary_condition(self):
        bs = []
        for inj_idx in range(self.Inj.shape[-1]):
            non_zero_idx = np.where(self.Inj[:,inj_idx] != 0.)[0] # 
            # [0, 2]
            L = 0
            for idx in non_zero_idx:
                el_idx = idx + 1 # [1,3]
                L += 1/self.electrode_len * inner(self.Inj[idx, inj_idx], self.psi[int(el_idx+1)]) * self.ds(int(el_idx))

            bs.append(assemble(L))

        self.bs = bs 

    def forward_solve(self, sigma, return_list_of_solution=False):
        u = TrialFunction(self.W)

        a = inner(sigma * grad(u[0]), grad(self.psi[0])) * self.dx
        u_sum = 0
        psi_sum = 0
        for i in range(32):
            a += (1/self.z[i] * inner(u[0] - u[i+2], self.psi[0] - self.psi[i+2])) * self.ds(i+1)
            u_sum += u[i+2]
            psi_sum += self.psi[i+2]

        a += u_sum * self.psi[1] * self.ds + psi_sum * u[1] * self.ds

        self.A = assemble(a) 
        

        u_all = []
        U_all = []
        solver = LUSolver(self.A)
        solution_list = []
        for inj_idx in range(self.Inj.shape[-1]):
            w = Function(self.W)

            #time_s = time.time()
            solver.solve(w.vector(), self.bs[inj_idx])
            #time_e = time.time() 
            #print("Solve forward linear system: ", time_e - time_s, " s")

            sol_tuple = split(w)

            u = sol_tuple[0]
            u_all.append(u)

            sol_vector = w.vector().get_local()

            U = sol_vector[-32:]
            #for k in range(32):
            #    U.append(sol_vector[-32+k])
            #    #rint("Elektrode ", k+1, sol_vector[-32+k])
                
            #print("Sum is:", sum(sol_vector[-32:]))

            U_all.append(U)

            solution_list.append(w)
            # For the values of the function:
            #print("solution values")
            #print(sol_tuple[0].vector().get_local()[:-33]) # values at the degrees of freedom
            #print(sol_tuple[0].compute_vertex_values()[:-33]) # values corresponding to the vertex ordering of the mesh
            # above outputs are not equal, but the indexing can be transformed:
            #v2d = vertex_to_dof_map(self.W.sub(0).collapse())
            #print(sol_tuple[0].vector().get_local()[v2d][:-33]) # now its the same

        if return_list_of_solution:
            return u_all, U_all, solution_list
        return u_all, U_all

    def solve_adjoint(self, deltaU, sigma):
        """
        deltaU: NxM matrix with N number of current patterns, M number of electrodes

        """
        u = TrialFunction(self.W)

        a = inner(sigma * grad(u[0]), grad(self.psi[0])) * self.dx
        u_sum = 0
        psi_sum = 0
        for i in range(32):
            a += (1/self.z[i] * inner(u[0] - u[i+2], self.psi[0] - self.psi[i+2])) * self.ds(i+1)
            u_sum += u[i+2]
            psi_sum += self.psi[i+2]

        a += u_sum * self.psi[1] * self.ds + psi_sum * u[1] * self.ds

        self.A = assemble(a) 
        
        bs = []
        for idx in range(32): # electrodes 
            L = 0
            el_idx = idx + 1
            L += 1/self.electrode_len * inner(1, self.psi[int(el_idx+1)]) * self.ds(int(el_idx))

            bs.append(assemble(L))       


        p_all = [] 
        solver = LUSolver(self.A)
        for inj_idx in range(self.Inj.shape[-1]): # pattern
            w2 = Function(self.W).vector()

            # only the boundary is different 
            for j in range(32):
                diffU = float(deltaU[inj_idx, j])
                w2 += bs[j]*diffU

            w = Function(self.W)

            #time_s = time.time()
            solver.solve(w.vector(), w2)
            #time_e = time.time() 

            #print("Solve adjoint linear system: ", time_e - time_s, " s")

            sol_tuple = split(w) #w.split()

            p = sol_tuple[0]
            p_all.append(p)

        return p_all
    

class EITContactImp(EIT):
    def __init__(self, Inj):
        """
        Inj: current injection pattern (32, N) matrix with the number of patterns N

        """
        super().__init__(Inj, np.ones(32))
        self.prepare_matrix()
        self.rhs_placeholer = Function(self.W)
        self.solution_list = [] # saves all solutions from the last forward call

    def assemble_boundary_condition(self):
        bs = []
        for inj_idx in range(self.Inj.shape[-1]):
            non_zero_idx = np.where(self.Inj[:,inj_idx] != 0.)[0] # 
            # [0, 2]
            L = 0
            for idx in non_zero_idx:
                el_idx = idx + 1 # [1,3]
                L += 1/self.electrode_len * inner(self.Inj[idx, inj_idx], self.psi[int(el_idx+1)]) * self.ds(int(el_idx))

            bs.append(assemble(L))

        self.bs = bs 

    def prepare_matrix(self):
        u = TrialFunction(self.W)
        a_diff = inner(grad(u[0]), grad(self.psi[0])) * self.dx
        self.A_diff = assemble(a_diff)
        self.A_electrode = []
        u_sum = 0
        psi_sum = 0
        for i in range(32):
                a = (inner(u[0] - u[i+2], self.psi[0] - self.psi[i+2])) * self.ds(i+1)
                self.A_electrode.append(assemble(a))
                u_sum += u[i+2]
                psi_sum += self.psi[i+2]

        a_average = u_sum * self.psi[1] * self.ds + psi_sum * u[1] * self.ds
        self.A_average = assemble(a_average)

    def forward_solve(self, sigma, y_list):
        u = TrialFunction(self.W)

        self.A = sigma * self.A_diff + self.A_average
        for i in range(32):
            self.A += y_list[i] * self.A_electrode[i] 
        

        u_all = []
        U_all = []
        self.lu_solver = LUSolver(self.A)
        self.solution_list = []
        for inj_idx in range(self.Inj.shape[-1]):
            w = Function(self.W)

            time_s = time.time()
            self.lu_solver.solve(w.vector(), self.bs[inj_idx])
            #solve(self.A, w.vector(), self.bs[inj_idx]) 
            time_e = time.time() 
            #print("Solve linear system: ", time_e - time_s, " s")
            sol_tuple = w.split()

            u = sol_tuple[0]
            u_all.append(u)

            sol_vector = w.vector().get_local()

            U = []
            for k in range(32):
                U.append(sol_vector[-32+k])

            U_all.append(U)

            self.solution_list.append(w)

        return u_all, U_all

    def compute_sigma_and_y_gradient(self):
        U_sigma_list = []
        U_y_list = []
        for w in self.solution_list:
            self.rhs_placeholer.assign(w)
            ### First the gradient with respect to sigma:
            w_sigma = Function(self.W)
            # right hand side:
            L = -inner(grad(self.rhs_placeholer[0]), grad(self.psi[0])) * self.dx
            b = assemble(L)
            
            self.lu_solver.solve(w_sigma.vector(), b)

            # we only need the U values:
            U_sigma_list.append(w_sigma.vector().get_local()[-32:])

            ### Now the gradients w.r.t to y_k
            U_y_grad = []
            w_y_i = Function(self.W)
            for i in range(32):
                # rhs:
                L = -inner(self.rhs_placeholer[0] - self.rhs_placeholer[i+2], 
                        self.psi[0] - self.psi[i+2]) * self.ds(i+1)
                b = assemble(L)
                
                self.lu_solver.solve(w_y_i.vector(), b)

                U_y_grad.append(w_y_i.vector().get_local()[-32:])

            U_y_list.append(U_y_grad)

        return np.array(U_sigma_list), np.array(U_y_list)

