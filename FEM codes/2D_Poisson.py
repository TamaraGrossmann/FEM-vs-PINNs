from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import time
import json, os

def boundary_L(x, on_boundary):
  return  on_boundary and np.isclose(x[0], 0) 

def boundary_R(x, on_boundary):
  return  on_boundary and np.isclose(x[0], 1) 

def boundary_U(x, on_boundary):
  return  on_boundary and np.isclose(x[1], 1)

def boundary_D(x, on_boundary):
  return  on_boundary and np.isclose(x[1], 0)  

# Define boundary condition
u_D = Expression('0', degree = 1)

def u_true(x):
  return (x[0]**2)*((x[0]-1)**2)*(x[1])*((x[1]-1)**2)

# Load evaluation points
with open("./Eval_Points/2D_Poisson_eval-points.json", 'r') as f:
  eval_points = json.load(f)
  eval_points = np.array(eval_points['mesh_coord']['0'])

nums = [100, 200 , 300, 400 , 500, 600, 700, 800, 900, 1000] # Mesh spacings that will be investigated

#Storing values for each mesh size
y_results, times_solve, times_eval, l2_rel = dict({}), dict({}), dict({}), dict({})

for num in nums:

  # Create mesh and define function space. 
  mesh = UnitSquareMesh(num,num)
  V = FunctionSpace(mesh, 'CG', 1)

  bc_D = DirichletBC(V, u_D, boundary_D)
  bcs = [bc_D]

  tot_solve = 0
  tot_eval = 0
  for count in range(0,10):
    u = TrialFunction(V)  
    v = TestFunction(V)
    f = Expression('- 2*( pow(x[0],4)*(3*x[1] - 2) + pow(x[0],3)*(4 - 6*x[1]) + pow(x[0],2)*(6*pow(x[1],3) - 12*pow(x[1],2) + 9*x[1] - 2) - 6*x[0]*pow((x[1]-1),2)*x[1] + pow((x[1]-1),2)*x[1])', degree = 1)
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

  l2 = np.linalg.norm(y_true - y_approx)
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


  save_dir = './2D-Poisson-FEM'
  if not os.path.exists(save_dir):
      os.mkdir(save_dir)

  with open(os.path.join(save_dir,'FEM_results.json'), "w") as write_file:
    json.dump(results, write_file)
