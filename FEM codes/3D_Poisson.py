from __future__ import print_function
from fenics import *
import numpy as np
import time
import json, os

def boundary_L(x, on_boundary):
  return  on_boundary and np.isclose(x[0], 0) 

def boundary_R(x, on_boundary):
  return  on_boundary and np.isclose(x[0], 1) 

def boundary_U(x, on_boundary):
  return  on_boundary and np.isclose(x[2], 1)

def boundary_D(x, on_boundary):
  return  on_boundary and np.isclose(x[2], 0)  

def boundary_F(x, on_boundary): #front
  return  on_boundary and np.isclose(x[1], 0)

def boundary_B(x, on_boundary): #back
  return  on_boundary and np.isclose(x[1], 1)  

def u_true(x):
  return np.sin(x[0]*np.pi)*np.sin(x[1]*np.pi)*np.sin(x[2]*np.pi)
  
# Define boundary condition
u_F = Expression('0', degree = 1) #y = 0
u_B = Expression('0', degree = 1) #y = 1
u_U = Expression('0', degree = 1) #z=1
u_D = Expression('0', degree = 1) #z = 0
u_L = Expression('0', degree = 1) #x = 0
u_R = Expression('0', degree = 1) #x = 1

with open("./Eval_Points/3D_Poisson_eval-points.json", 'r') as f:
  eval_points = json.load(f)
  eval_points = np.array(eval_points['mesh_coord']['0'])


nums = [16, 32, 64, 128]#Mesh spacings that will be investigated

#Storing values for each mesh size
y_results, times_solve, times_eval, l2_rel = dict({}), dict({}), dict({}), dict({})

for num in nums:
  # Create mesh and define function space. 
  mesh = UnitCubeMesh(num,num,num)
  V = FunctionSpace(mesh, 'CG', 1)

  bc_L = DirichletBC(V, u_L, boundary_L)
  bc_R = DirichletBC(V, u_R, boundary_R)
  bc_U = DirichletBC(V, u_U, boundary_U)
  bc_D = DirichletBC(V, u_D, boundary_D)
  bc_F = DirichletBC(V, u_F, boundary_F)
  bc_B = DirichletBC(V, u_B, boundary_B)
  bcs = [bc_L, bc_R, bc_U, bc_D, bc_F, bc_B ]

  tot_solve = 0
  tot_eval = 0
  for count in range(0,10):
    u = TrialFunction(V)  
    v = TestFunction(V)
    f = Expression('(3*(pow(pi,2)))*sin(x[0]*pi)*sin(x[1]*pi)*sin(x[2]*pi)', degree = 1)
    F = dot(grad(u), grad(v))*dx - f*v*dx


    a, L = lhs(F), rhs(F)
    u = Function(V)

    t0 = time.time()
    solve(a == L, u, bcs, solver_parameters={'linear_solver':'cg','preconditioner':'ilu'})
    t1 = time.time()
    time_solving = t1 -t0
    tot_solve += time_solving

    t0 = time.time()
    y_approx = [u(eval_point) for eval_point in eval_points]
    t1 = time.time()
    time_evaluation = t1 - t0 
    tot_eval += time_evaluation

  time_solving = tot_solve / 10
  time_evaluation = tot_eval /10

  ########################################################################
  # Compare results to GT solutions
  ########################################################################
  print('Start comparing to GT', num)

  y_approx = np.array(y_approx)
  y_true = np.array([u_true(eval_point) for eval_point in eval_points])

  print(y_approx.shape,y_true.shape)

  l2 = np.linalg.norm(y_approx -y_true)
  y_true_norm = np.linalg.norm(y_true)
  l2_rel_single = l2 / y_true_norm

  print('Average solution time', time_solving)
  print('Average evaluation time', time_evaluation)
  print('Average accuracy: ', l2_rel_single)

  y_results[num], times_solve[num], times_eval[num], l2_rel[num] = y_approx.tolist(), time_solving, time_evaluation, l2_rel_single


  results = dict({'y_results': y_results,
                  'times_solve': times_solve,
                  'times_eval': times_eval,
                  'l2_rel': l2_rel})

  save_dir = './3D-Poisson-FEM'
  if not os.path.exists(save_dir):
      os.mkdir(save_dir)

  with open(os.path.join(save_dir,'FEM_results.json'), "w") as write_file:
    json.dump(results, write_file)
