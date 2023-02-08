from __future__ import print_function
from fenics import *
import numpy as np
import time, json, os

########################################################################
# Load Solutions from fine grid FEM (GT) 
########################################################################
# Load GT solution
save_dir = './Eval_Points/2D_Schroedinger/'
with open(os.path.join(save_dir,'eval_coordinates.json'), 'r') as f:
    eval_coordinates= json.load(f)
mesh_coord = eval_coordinates['mesh_coord']['0']
dt_coord = eval_coordinates['dt_coord']['0']
num_eval_steps = len(dt_coord)
with open(os.path.join(save_dir,'eval_solution_mat.json'), 'r') as f:
    eval_solution_mat= json.load(f)
eval_solution_mat = np.asarray(eval_solution_mat)
true_u = eval_solution_mat[0]
true_v = eval_solution_mat[1]
true_h = eval_solution_mat[2]

########################################################################
# Solve PDE with FEM
########################################################################
#Definitions of mesh functions, loading eval points etc
tol = 1E-14
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((near(x[0], -5) or near(x[1], -5)) and 
                (not ((near(x[0], -5) and near(x[1], 5)) or 
                        (near(x[0], 5) and near(x[1], -5)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1): # Top right corner
            y[0] = x[0] - 10.
            y[1] = x[1] - 10.
        elif near(x[0], 1): # Right edge
            y[0] = x[0] - 10.
            y[1] = x[1]
        else:   # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 10.

# Create periodic boundary condition
pbc = PeriodicBoundary()

dt = 1e-3
T = np.pi/2
num_steps = int(T/dt)
nums = [(16,16),(32,32),(40,40),(64,64),(128,128)]
av_iter_sol = 10 # Over how many iterations we want to average

results, solution = dict({}),dict({}) # Save eval times, solution times, errors

for num in nums:
  numx = num[0]
  numy = num[1]
  print('Start solving', numx)
  mesh = RectangleMesh(Point(-5,-5), Point(5, 5), numx, numy)
  V = VectorFunctionSpace(mesh, 'CG', 1, dim = 2, constrained_domain = pbc) # periodic BC are included in the definition of the function space
  # Here vector space is used because we must write separate equations for real and imaginary parts of h, and h is [h_re , h_im]

  all_times = [dt*(n+1) for n in range(int(num_steps))] # List of all times for which we get the solution, will be useful for evaluation. We do not start from t = 0
  results[numx] = dict({})
  
  for i in range(av_iter_sol):
    t=0
    time_solving = 0
    u_0 = Expression(  ( 'pow(cosh(x[0]), -1) + 0.5 * ( pow(cosh(x[1]-2), -1) + pow(cosh(x[1]+2), -1) )', '0'), degree = 1) # Initial value. Has the real part only
    u_n = interpolate(u_0, V)
    u = TrialFunction(V)

    v = TestFunction(V)
    F_Re = (-u[1]+u_n[1])*v[0]*dx - 0.5 * dot(grad(u[0]),grad(v[0]))*dt*dx + (u_n[0]**2+u_n[1]**2)*u[0]*v[0]*dt*dx # 
    F_Im = (u[0]-u_n[0])*v[1]*dx - 0.5 * dot(grad(u[1]),grad(v[1]))*dt*dx + (u_n[0]**2+u_n[1]**2)*u[1]*v[1]*dt*dx

    a_Re, L_Re = lhs(F_Re) , rhs(F_Re)
    a_Im, L_Im = lhs(F_Im) , rhs(F_Im)
    a = a_Re + a_Im
    L = L_Re + L_Im
    u = Function(V)

    save_dir = os.path.join('./2D-Schroedinger-FEM/Approx-Solution-semiimplicit/','Mesh_%03d' %numx)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    t0 = time.time()
    for n in range(int(num_steps)):
        # Update current time
        t += dt
        # Compute solution        
        solve(a == L, u, bcs = None, solver_parameters={'linear_solver':'gmres'}) 
        # Update previous solution
        u_n.assign(u)

        filepath = os.path.join(save_dir,'iter_%05d' %n)
        hdf = HDF5File(MPI.comm_world, filepath, "w")
        hdf.write(u, "/f")  
        hdf.close()

    t1 = time.time()
    time_solving += t1 - t0
  tot_solve = (time_solving) / av_iter_sol
  results[numx]['time_solve'] = tot_solve
    

  ########################################################################
  # Compare results to GT solutions
  ########################################################################
  print('Start comparing to GT', num)
  ## Load FEM solution
  mesh = RectangleMesh(Point(-5,-5), Point(5, 5), numx, numy)
  V = VectorFunctionSpace(mesh, 'CG', 1, dim = 2, constrained_domain = pbc)
  u_load = []
  # Load function
  save_dir = os.path.join('./2D-Schroedinger-FEM/Approx-Solution-semiimplicit/','Mesh_%03d' %numx)
  for n in range(int(num_steps)):
    filepath = os.path.join(save_dir,'iter_%05d' %n)
    hdf = HDF5File(MPI.comm_world, filepath, "r")
    fun_load = Function(V)
    hdf.read(fun_load, "/f") 
    u_load.append(fun_load)
    hdf.close() 

  l2_errors_u = []
  l2_errors_v = []
  l2_errors_h = []
  mat_approx_u = np.zeros((len(dt_coord),len(mesh_coord)))
  mat_approx_v = np.zeros((len(dt_coord),len(mesh_coord)))
  mat_approx_h = np.zeros((len(dt_coord),len(mesh_coord)))
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
    v_approx = []
    h_approx = []
    t0 = time.time()
    u_inter = project(u1*coeff[0] + u2*coeff[1], V) # Interpolated solution between two times
    for eval_point in mesh_coord: 
      u_approx.append(u_inter(eval_point)[0])
      v_approx.append(u_inter(eval_point)[1])
      h_approx.append( np.sqrt(u_inter(eval_point)[0]**2+u_inter(eval_point)[1]**2) )
    t1 = time.time()

    mat_approx_u[n,:] = np.array(u_approx)
    mat_approx_v[n,:] = np.array(v_approx)
    mat_approx_h[n,:] = np.array(h_approx)
    mat_approx = np.concatenate((mat_approx_u[None],mat_approx_v[None],mat_approx_h[None]),axis=0)
    # Calulate the relative L2 norm
    u_l2 = np.linalg.norm(true_u[n] - np.array(u_approx))/np.linalg.norm(true_u[n])
    v_l2 = np.linalg.norm(true_v[n] - np.array(v_approx))/np.linalg.norm(true_v[n])
    h_l2 = np.linalg.norm(true_h[n] - np.array(h_approx))/np.linalg.norm(true_h[n])

    l2_errors_u.append(u_l2)
    l2_errors_v.append(v_l2)
    l2_errors_h.append(h_l2)
    tot_eval += t1 - t0
  solution[numx] = mat_approx.tolist()
  results[numx]['time_eval_GT'] = tot_eval/len(dt_coord)
  results[numx]['lr2_errors_u'] = np.mean(l2_errors_u)
  results[numx]['lr2_errors_v'] = np.mean(l2_errors_v)
  results[numx]['lr2_errors_h'] = np.mean(l2_errors_h)

  save_dir = './2D-Schroedinger-FEM/'
  with open(os.path.join(save_dir,'FEM_semiimplicit_results.json'), "w") as write_file:
    json.dump(solution, write_file)

  with open(os.path.join(save_dir,'FEM_semiimplicit_evaluation.json'), "w") as write_file:
    json.dump(results, write_file)

  print(json.dumps(results, indent=4))
