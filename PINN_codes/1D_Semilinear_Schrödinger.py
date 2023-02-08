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

from Schroedinger_1D.model import PDESolution
from Schroedinger_1D.util_gt import ImportData, CompareGT
from Schroedinger_1D.util import sample_points


#----------------------------------------------------
# Hyperparameters
#----------------------------------------------------
architecture_list = [[20,20,20,2],[100,100,100,2],[20,20,20,20,2],[100,100,100,100,2],[20,20,20,20,20,2],[100,100,100,100,100,2],[20,20,20,20,20,20,2],[100,100,100,100,100,100,2]]
lr = 1e-4
num_epochs = 50000  

#----------------------------------------------------
# Define Loss Function
#----------------------------------------------------
# PDE residual for 1D Semilinear Schrödinger
@partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def residual(u, t, x):
    u_t = jax.jvp(u, (t, x), (1., 0.))[1]
    u_real_t = u_t[0]
    u_imag_t = u_t[1]
    u_xx = jax.hessian(u, argnums=1)(t, x)
    u_real_xx = u_xx[0]
    u_imag_xx = u_xx[1]

    h = (u(t,x)[0])**2 + (u(t,x)[1])**2

    f_real = -u_imag_t + 0.5*u_real_xx + h*u(t,x)[0] 
    f_imag = u_real_t + 0.5*u_imag_xx + h*u(t,x)[1]
    return f_real, f_imag

# Inital condition
@partial(jax.vmap, in_axes=0)
def u_init(xs):
    return jnp.array([2./jnp.cosh(xs), 0.])

# Loss functionals
@jax.jit
def pde_residual(params, points):
    f_real, f_imag = residual(lambda t, x: model.apply(params, jnp.stack((t, x))), points[:, 0], points[:, 1])
    return jnp.mean(f_real**2) + jnp.mean(f_imag**2)

@partial(jax.jit, static_argnums=0)
def init_residual(u_init, params, xs):
    return jnp.mean((model.apply(params, jnp.stack((jnp.zeros_like(xs[:,0]), xs[:,0]), axis=1)) - u_init(xs[:,0]))**2)

@jax.jit
def boundary_residual(params, ts):
    return jnp.mean((model.apply(params, jnp.stack((ts[:,0], 5 * jnp.ones_like(ts[:,0])), axis=1)) - 
                                  model.apply(params, jnp.stack((ts[:,0], -5 * jnp.ones_like(ts[:,0])), axis=1)))**2)

#----------------------------------------------------
# Define Training Step
#----------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def training_step(params, opt, opt_state, key):
    domain_points, boundary, init = sample_points([0.,-5.],[1.,5.],20000,50,50)

    loss_val, grad = jax.value_and_grad(lambda params: pde_residual(params, domain_points) + 
                                                    init_residual(u_init, params, init) +
                                                    boundary_residual(params, boundary))(params)
    
    update, opt_state = opt.update(grad, opt_state, params)
    params = optax.apply_updates(params, update)
    return params, opt_state, key, loss_val

def train_loop(params, adam, opt_state, key):
    losses = []
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
GTloader = ImportData('./Eval_Points/1D_Schroedinger')
mesh_coord, dt_coord = GTloader.get_FEM_coordinates()
FEM_real,FEM_imag,FEM_sq = GTloader.get_FEM_results()

#----------------------------------------------------
# Train PINN
#----------------------------------------------------
# Train model 10 times and average over the times
u_results, v_results, h_results = dict({}), dict({}), dict({})
times_adam, times_lbfgs, times_total, times_eval, l2_rel_u, l2_rel_v, l2_rel_h, var_u, var_v, var_h, arch  =  dict({}), dict({}), dict({}), dict({}), dict({}), dict({}), dict({}), dict({}), dict({}), dict({}), dict({})
n=0
print('Start training')
for feature in architecture_list:
    print('Architecture: ', feature)
    times_adam_temp = []
    times_lbfgs_temp = []
    times_total_temp = []
    times_eval_temp = []

    l2_errors_u = []
    l2_errors_v = []
    l2_errors_h = []
    l2_errors_u_R = []
    l2_errors_v_R = []
    l2_errors_h_R = []
    for _ in range(10):
        #----------------------------------------------------
        # Initialise Model
        #----------------------------------------------------
        model = PDESolution(feature)
        key, key_ = jax.random.split(jax.random.PRNGKey(17))
        params = model.init(key_, jnp.ones((4, 2)))

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
        domain_points, boundary, init = sample_points([0.,-5.],[1.,5.],10000,50,50)

        print('Starting L-BFGS Optimisation')
        start_time2 = time.time()
        results = tfp.optimizer.lbfgs_minimize(jax.value_and_grad(lambda params: pde_residual(unconcat_params(params, tree, shapes), domain_points) + 
                                                            init_residual(u_init, unconcat_params(params, tree, shapes), init) +
                                                            boundary_residual(unconcat_params(params, tree, shapes), boundary)),
                                    init_point,
                                    max_iterations=50000,
                                    num_correction_pairs=50,
                                    f_relative_tolerance=1.0 * jnp.finfo(float).eps)
        lbfgs_time = time.time()-start_time2
        times_total_temp.append(time.time()-start_time)
        times_lbfgs_temp.append(lbfgs_time)

        tuned_params = unconcat_params(results.position, tree, shapes)
        
        # Evaluation
        real_l2, imag_l2, sq_l2, times_temp, approx, h_approx, true_u, true_v, true_h, domain_pt = CompareGT.get_FEM_comparison(mesh_coord,dt_coord,FEM_real,FEM_imag,FEM_sq,model,tuned_params) #dt_coord_100,
        times_eval_temp.append(times_temp)
        l2_errors_u.append(jnp.mean(jnp.array(real_l2)))#real_l2)
        l2_errors_v.append(jnp.mean(jnp.array(imag_l2)))#imag_l2)
        l2_errors_h.append(jnp.mean(jnp.array(sq_l2)))#sq_l2)
        
        print('L2 rel errors this: ',jnp.mean(jnp.array(real_l2)),jnp.mean(jnp.array(imag_l2)),jnp.mean(jnp.array(sq_l2)))
        print('L2 rel errors overall: ',jnp.mean(jnp.array(l2_errors_u)),jnp.mean(jnp.array(l2_errors_v)),jnp.mean(jnp.array(l2_errors_h)))
        print('eval time: ', times_temp)


    u_gt, v_gt, h_gt, domain_pts = true_u.tolist(), true_v.tolist(), true_h.tolist(), domain_pt.tolist()
    u_results[n], v_results[n], h_results[n] = approx[:,:,0].tolist(), approx[:,:,1].tolist(), h_approx.tolist()
    times_adam[n], times_lbfgs[n], times_total[n], times_eval[n], l2_rel_u[n], l2_rel_v[n], l2_rel_h[n], var_u[n], var_v[n], var_h[n], arch[n] = onp.mean(times_adam_temp), onp.mean(times_lbfgs_temp), onp.mean(times_total_temp), onp.mean(times_eval_temp), onp.mean(jnp.array(l2_errors_u)).tolist(), onp.mean(jnp.array(l2_errors_v)).tolist(), onp.mean(jnp.array(l2_errors_h)).tolist(), onp.var(jnp.array(l2_errors_u)).tolist(), onp.var(jnp.array(l2_errors_v)).tolist(), onp.var(jnp.array(l2_errors_h)).tolist(), feature
    n+=1
    results = dict({'domain_pts': domain_pts,
                    'u_results': u_results,
                    'v_results': v_results,
                    'h_results': h_results,
                    'u_gt': u_gt,
                    'v_gt': v_gt,
                    'h_gt': h_gt})

    evaluation = dict({'arch': arch,
        'times_adam': times_adam,
        'times_lbfgs': times_lbfgs,
        'times_total': times_total,
        'times_eval': times_eval,
        'l2_rel': l2_rel_u,
        'l2_rel_v': l2_rel_v,
        'l2_rel_h': l2_rel_h,
        'var_u': var_u,
        'var_v': var_v,
        'var_h': var_h})

    save_dir = './1D-Schroedinger-PINN'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir,'PINNs_results-update.json'), "w") as write_file:
        json.dump(results, write_file)

    with open(os.path.join(save_dir,'PINNs_evaluation-update.json'), "w") as write_file:
        json.dump(evaluation, write_file)
    
    print(json.dumps(evaluation, indent=4))
