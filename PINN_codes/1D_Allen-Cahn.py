import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import jax, optax, time, pickle
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import numpy as onp
from functools import partial
import json
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)

import numpy.random as npr

from Allen_Cahn_1D.model import PDESolution
from Allen_Cahn_1D.util_gt import ImportData, CompareGT
from Allen_Cahn_1D.util import sample_points

#----------------------------------------------------
# Hyperparameters
#----------------------------------------------------
architecture_list = [[20,20,20,1],[100,100,100,1],[500,500,500,1],[20,20,20,20,1],[100,100,100,100,1],[500,500,500,500,1],[20,20,20,20,20,1],[100,100,100,100,100,1],[500,500,500,500,500,1],[20,20,20,20,20,20,1],[100,100,100,100,100,100,1],[500,500,500,500,500,500,1],[20,20,20,20,20,20,20,1],[100,100,100,100,100,100,100,1]]
lr = 1e-4
num_epochs = 50000 
eps = 0.01

#----------------------------------------------------
# Define Loss Function
#----------------------------------------------------
# PDE residual for 1D Allen-Cahn
@partial(jax.vmap, in_axes=(None, 0, 0, None), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def residual(u, t, x, eps):
    u_t = jax.jvp(u, (t, x), (1., 0.))[1]
    u_xx = jax.hessian(u,argnums=1)(t,x)
    return u_t - eps*u_xx  + (1/eps)*2*u(t,x)*(1-u(t,x))*(1-2*u(t,x))


# Inital condition
@partial(jax.vmap, in_axes=0)
def u_init(xs):
    return jnp.array([0.5*(0.5*jnp.sin(xs*2*jnp.pi) + 0.5*jnp.sin(xs*16*jnp.pi)) + 0.5])


# Loss functionals
@jax.jit
def pde_residual(params, points):
    return jnp.mean(residual(lambda t, x: model.apply(params, jnp.stack((t, x))), points[:, 0], points[:, 1],eps)**2)

@partial(jax.jit, static_argnums=0)
def init_residual(u_init,params, xs):
    ini_approx = model.apply(params, jnp.stack((jnp.zeros_like(xs[:,0]), xs[:,0]), axis=1))
    ini_true = u_init(xs[:,0])
    return jnp.mean((ini_approx - ini_true)**2) 

@jax.jit
def boundary_residual(params, ts): #periodic bc
    return jnp.mean((model.apply(params, jnp.stack((ts[:,0], jnp.zeros_like(ts[:,0])), axis=1)) - 
                                  model.apply(params, jnp.stack((ts[:,0], jnp.ones_like(ts[:,0])), axis=1)))**2)


#----------------------------------------------------
# Define Training Step
#----------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def training_step_ini(params, opt, opt_state, key):
    domain_points, boundary, init = sample_points([0.,0.],[0.05,1.],20000,250,500)

    loss_val, grad = jax.value_and_grad(lambda params: init_residual(u_init,params, init))(params)
    
    update, opt_state = opt.update(grad, opt_state, params)
    params = optax.apply_updates(params, update)
    return params, opt_state, key, loss_val

@partial(jax.jit, static_argnums=(1,))
def training_step(params, opt, opt_state, key):#, u_init):
    domain_points, boundary, init = sample_points([0.,0.],[0.05,1.],20000,250,500)

    loss_val, grad = jax.value_and_grad(lambda params: pde_residual(params, domain_points) + 
                                                    1000*init_residual(u_init,params, init) +
                                                    boundary_residual(params, boundary))(params)
    
    pde_val, ini_val, bound_val = pde_residual(params, domain_points), init_residual(u_init,params, init), boundary_residual(params, boundary)
    update, opt_state = opt.update(grad, opt_state, params)
    params = optax.apply_updates(params, update)
    return params, opt_state, key, loss_val
def train_loop(params, adam, opt_state, key):
    losses = []
    for i in range(7000):
        params, opt_state, key, loss_val = training_step_ini(params, adam, opt_state, key)
        losses.append(loss_val.item())

    for _ in range(num_epochs):
        params, opt_state, key, loss_val = training_step(params, adam, opt_state, key)
        losses.append(loss_val.item())
    return losses, params, opt_state, key, loss_val


#----------------------------------------------------
# Define Helper Functions for L-BFGS wrapper
#----------------------------------------------------
def concat_params(params):
    params, tree = jax.tree_util.tree_flatten(params)
    shapes = [param.shape for param in params]
    return jnp.concatenate([param.reshape(-1) for param in params]), tree, shapes

def unconcat_params(params, tree, shapes):
    split_vec = jnp.split(params, onp.cumsum([onp.prod(shape) for shape in shapes]))
    split_vec = [vec.reshape(*shape) for vec, shape in zip(split_vec, shapes)]
    return jax.tree_util.tree_unflatten(tree, split_vec)


#----------------------------------------------------
# Load GT solution
#----------------------------------------------------
GTloader = ImportData('./Eval_Points/1D_Allen-Cahn/')
mesh_coord, dt_coord = GTloader.get_FEM_coordinates()
FEM = GTloader.get_FEM_results()

#----------------------------------------------------
# Train PINN
#----------------------------------------------------
# Train model 10 times and average over the times
u_results = dict({})
times_adam, times_lbfgs, times_total, times_eval, l2_rel, var, arch  = dict({}), dict({}), dict({}), dict({}), dict({}), dict({}), dict({})
n=0
print('Start training')
for feature in architecture_list:
    print('Architecture: ', feature)
    times_adam_temp = []
    times_lbfgs_temp = []
    times_total_temp = []
    times_eval_temp = []
    l2_errors = []
    for _ in range(10):
        #----------------------------------------------------
        # Initialise Model
        #----------------------------------------------------
        model = PDESolution(feature)
        key, key_ = jax.random.split(jax.random.PRNGKey(17))
        batch_dim = 4
        feature_dim = 2
        params = model.init(key_, jnp.ones((batch_dim, feature_dim)))

        #----------------------------------------------------
        # Initialise Optimiser
        #----------------------------------------------------
        adam = optax.adam(lr)
        opt_state = adam.init(params)

        #----------------------------------------------------
        # Start Training with Adam optimiser
        #----------------------------------------------------
        start_time = time.time() 
        losses, params, opt_state, key, loss_val = jax.block_until_ready(train_loop(params, adam, opt_state, key))  #this is only available from jax 0.2.27, but I have 0.2.24 installed
        adam_time = time.time()-start_time
        times_adam_temp.append(adam_time)
        print("Adam training time: ", adam_time)

        #----------------------------------------------------
        # Start Training with L-BFGS optimiser
        #----------------------------------------------------
        init_point, tree, shapes = concat_params(params)
        domain_points, boundary, init = sample_points([0.,0.],[0.05,1.],20000,250,500)

        print('Starting L-BFGS Optimisation')
        start_time2 = time.time()
        results = tfp.optimizer.lbfgs_minimize(jax.value_and_grad(lambda params: pde_residual(unconcat_params(params, tree, shapes), domain_points) + 
                                                            1000*init_residual(u_init,unconcat_params(params, tree, shapes), init) +
                                                            boundary_residual(unconcat_params(params, tree, shapes), boundary)),
                                    init_point,
                                    max_iterations=50000,
                                    num_correction_pairs=50,
                                    f_relative_tolerance=1.0 * jnp.finfo(float).eps)
       
        lbfgs_time = time.time()-start_time2
        times_total_temp.append(time.time()-start_time)
        times_lbfgs_temp.append(lbfgs_time)

        # Evaluation
        tuned_params = unconcat_params(results.position, tree, shapes)
        
        l2, times_temp, approx, gt_fem, domain_pt = CompareGT.get_FEM_comparison(mesh_coord,dt_coord,FEM,model,tuned_params)
        times_eval_temp.append(times_temp)
        l2_errors.append(jnp.mean(jnp.array(l2)))

    u_gt = gt_fem.tolist()
    domain_pts = domain_pt.tolist()
    u_results[n] = approx.tolist()
    times_adam[n], times_lbfgs[n], times_total[n], times_eval[n], l2_rel[n], var[n], arch[n] = onp.mean(times_adam_temp), onp.mean(times_lbfgs_temp), onp.mean(times_total_temp), onp.mean(times_eval_temp), onp.mean(jnp.array(l2_errors)).tolist(), onp.var(jnp.array(l2_errors)).tolist(), feature
    n+=1
    results = dict({'domain_pts': domain_pts,
                    'u_results': u_results,
                    'u_gt': u_gt})

    evaluation = dict({'arch': arch,
        'times_adam': times_adam,
        'times_lbfgs': times_lbfgs,
        'times_total': times_total,
        'times_eval': times_eval,
        'l2_rel': l2_rel,
        'var_u': var})

    save_dir = './1D-Allen-Cahn-PINN'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir,'PINNs_results_smalleps.json'), "w") as write_file:
        json.dump(results, write_file)

    with open(os.path.join(save_dir,'PINNs_evaluation_smalleps.json'), "w") as write_file:
        json.dump(evaluation, write_file)
    
    print(json.dumps(evaluation, indent=4))
