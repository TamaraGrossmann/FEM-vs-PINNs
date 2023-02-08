from __future__ import print_function
from fenics import *
import numpy as np
import time
import json,os

########################################################################
# Load Solutions from fine grid FEM (GT) 
########################################################################
# Load GT solution
save_dir = './Eval_Points/1D_Allen-Cahn/'
with open(os.path.join(save_dir,'eval_coordinates.json'), 'r') as f:
    eval_coordinates= json.load(f)
mesh_coord = eval_coordinates['mesh_coord']['0']
dt_coord = eval_coordinates['dt_coord']['0']
with open(os.path.join(save_dir,'eval_solution_mat.json'), 'r') as f:
    eval_solution_mat= json.load(f)
eval_solution_mat = np.asarray(eval_solution_mat)

########################################################################
# Solve PDE with FEM
########################################################################
tol = 1E-14
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain"
    def inside(self, x, on_boundary):
        return bool(x[0] < 0.0 + tol and x[0] > 0.0 -tol and on_boundary)

    # Map right boundary 'H' to left boundary 'G' 
    def map(self, x, y):
        y[0] = x[0] - 1.0

# Create periodic boundary condition
pbc = PeriodicBoundary()

dt = 1e-3 
T = 0.05
num_steps = int(T/dt)
nums = [32,128,512,2048] 
timer1=[]
eps = 0.01
av_iter_sol = 10 # Over how many iterations we want to average

all_times = [dt*(n+1) for n in range(int(num_steps))] # List of all times for which we get the solution, will be useful for evaluation. We do not start from t = 0
results, solution=dict({}),dict({})

for num in nums:
  print('Start solving', num)
  mesh = IntervalMesh(int(num), 0, 1)
  V = FunctionSpace(mesh, 'CG', 1, constrained_domain = pbc) # Periodic BC are included in the definition of the function space 

  results[num] = dict({})
  time_solving = 0
  for i in range(av_iter_sol):
    t=0
    time_solving = 0
    u_0 = Expression('0.5*(0.5*sin(x[0]*2*pi) + 0.5*sin(x[0]*16*pi)) + 0.5', degree = 1) #initial value. Has the real part only
    u_n = interpolate(u_0, V)
    u = Function(V)
    v = TestFunction(V)

    # weak form of the pde
    F = (u - u_n)*v*dx + eps * dot(grad(u),grad(v))*dt*dx + (1/eps) * 2*u_n*(1-u_n)*(1-2*u_n)*v*dt*dx 
    Jac = derivative(F, u)

    t0 = time.time()
    for n in range(int(num_steps)):
        # Update current time
        t += dt
        # Compute solution        
        solve(F == 0, u, bcs=None, J = Jac) 
        # Update previous solution
        u_n.assign(u)

        filepath = './1D-Allen-Cahn-FEM/Approx-Solution-semiimplicit/'+str(num)+"iter_"+ str(n) 
        hdf = HDF5File(MPI.comm_world, filepath, "w")
        hdf.write(u, "/f")  
        hdf.close()
    t1 = time.time()
    time_solving += t1 - t0
    
  tot_solve = (time_solving) / av_iter_sol
  results[num]['time_solve'] = tot_solve # Save solution time for each mesh size

  ########################################################################
  # Compare results to GT solutions
  ########################################################################
  print('Start comparing to GT', num)
  ## Load FEM solution
  mesh = IntervalMesh(int(num), 0, 1)# For each mesh, the solution belongs to different V, hence it must be declared again
  V = FunctionSpace(mesh, 'CG', 1, constrained_domain = pbc)
  u_load = []
  # Load function
  for n in range(int(num_steps)):
    filepath = './1D-Allen-Cahn-FEM/Approx-Solution-semiimplicit/' + str(num)+"iter_" + str(n)
    hdf = HDF5File(MPI.comm_world, filepath, "r")
    fun_load = Function(V)
    hdf.read(fun_load, "/f") 
    u_load.append(fun_load)
    hdf.close() 

  l2_errors_u = []
  mat_approx = np.zeros((len(dt_coord),len(mesh_coord)))
  ## Interpolate for each GT time step
  for n in range(len(dt_coord)):
    n = int(n)
    tot_eval = 0
    differences = np.abs( np.subtract(all_times, dt_coord[n]) ) # Differences between times at which we got the solution and the eval time
    closest_idx = np.argpartition(differences, 2)
    coeff = (-differences[closest_idx]+dt)/dt
    
    u1 = u_load[closest_idx[0]] # Solutions between which we interpolate in the temporal domain
    u2 = u_load[closest_idx[1]]
    u_approx = [] # List of amplitudes at evaluation points in the spatial domain

    t0 = time.time()
    u_inter = project(u1*coeff[0] + u2*coeff[1], V) # Interpolated solution between two times
    for eval_point in mesh_coord: 
      u_approx.append(u_inter(eval_point))
    t1 = time.time()
    mat_approx[n,:] = np.array(u_approx)
    # Calulate the relative L2 norm
    u_l2 = np.linalg.norm(eval_solution_mat[n,:] - np.array(u_approx))/np.linalg.norm(eval_solution_mat[n,:])

    l2_errors_u.append(u_l2)
    tot_eval += t1 - t0

  solution[num] = mat_approx.tolist()
  results[num]['time_eval'] = tot_eval/len(dt_coord)
  results[num]['l2_errors_u'] = np.mean(l2_errors_u)
  results[num]['var_u'] = np.var(l2_errors_u)

  save_dir = './1D-Allen-Cahn-FEM/'
  with open(os.path.join(save_dir,'FEM_semiimplicit_results.json'), "w") as write_file:
    json.dump(solution, write_file)

  with open(os.path.join(save_dir,'FEM_semiimplicit_evaluation.json'), "w") as write_file:
    json.dump(results, write_file)

  print(json.dumps(results, indent=4))