from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import time
import json, os

# Load evaluation points
with open("./Eval_Points/1D_Poisson_eval-points.json", 'r') as f:
  ordered_points = json.load(f)
  ordered_points = np.array(ordered_points)

def boundary_R(x, on_boundary):
  return  on_boundary and np.isclose(x[0], 1) 

def boundary_L(x, on_boundary):
  return  on_boundary and np.isclose(x[0], 0)

def u_e(x):
  return x*np.exp(-x*x)

u_L = Expression('0', degree = 1)
u_R = Expression('exp(-1)', degree = 1)

nums =  [64,128,256,512,1024,2048,4096]
n=0

#Storing values for each mesh size
y_results, times_solve, times_eval, l2_rel = dict({}), dict({}), dict({}), dict({})


for num in nums:
  # Create mesh and define function space. 
  mesh = IntervalMesh(int(num),0,1)
  V = FunctionSpace(mesh, 'CG', 1)

  # Define boundary condition
  bc_L = DirichletBC(V, u_L, boundary_L)
  bc_R = DirichletBC(V, u_R, boundary_R)
  bcs = [bc_L, bc_R]

  u = TrialFunction(V)
  v = TestFunction(V)

  f = Expression('6 * x[0] * exp(-x[0]*x[0]) - 4*(x[0]*x[0]*x[0]) * exp(-x[0]*x[0]) ', degree = 1)
  F = dot(grad(u), grad(v))*dx - f*v*dx
  a, L = lhs(F), rhs(F)
  u = Function(V)
  tot_solve = 0
  tot_eval = 0
  for count in range(0,10):
    t0 = time.time()
    solve(a == L, u, bcs, solver_parameters={'linear_solver':'cg','preconditioner': 'ilu'})
    t1 = time.time()
    tot_solve += t1 -t0
    t0 = time.time()
    u_approx = [u(point) for point in ordered_points]
    t1 = time.time()
    tot_eval += t1 - t0 
    u_approx = np.array(u_approx) 
  time_solving = tot_solve / 10
  time_evaluation = tot_eval /10
  u_true = np.array( [u_e(point) for point in ordered_points] )

  # Calculate relative L2 error
  l2 = np.linalg.norm(u_approx - u_true.squeeze())
  u_true_norm = np.linalg.norm(u_true)
  l2_rel_single = l2 / u_true_norm

  print('Average solution time', time_solving)
  print('Average evaluation time', time_evaluation)
  print('Average accuracy on a random testset of 512 points: ', l2_rel_single)


  y_results[num], times_solve[num], times_eval[num], l2_rel[num] = u_approx.tolist(), time_solving, time_evaluation, l2_rel_single

  results = dict({'y_results': y_results,
                  'times_solve': times_solve,
                  'times_eval': times_eval,
                  'l2_rel': l2_rel})

save_dir = './1D-Poisson-FEM'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

with open(os.path.join(save_dir,'FEM_results.json'), "w") as write_file:
  json.dump(results, write_file)

