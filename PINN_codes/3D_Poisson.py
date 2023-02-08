import jax, flax, optax, time
import os
import jax.numpy as jnp
from functools import partial
import json, pickle
from pyDOE import lhs
from typing import Sequence
from tensorflow_probability.substrates import jax as tfp
import numpy as onp

os.environ["CUDA_VISIBLE_DEVICES"]="0"
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)

#----------------------------------------------------
# Hyperparameters
#----------------------------------------------------
architecture_list = [[20,20,1],[60,60,1],[20,20,20,1],[60,60,60,1],[20,20,20,20,1],[60,60,60,60,1],[20,20,20,20,20,1],[60,60,60,60,60,1]]
lr = 1e-3
num_epochs = 20000

#----------------------------------------------------
# Define Neural Network Architecture
#----------------------------------------------------
class PDESolution(flax.linen.Module):
    features: Sequence[int]

    @flax.linen.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = flax.linen.tanh(flax.linen.Dense(feat)(x))
        x = flax.linen.Dense(self.features[-1])(x)
        return x

#----------------------------------------------------
# Define Loss Function
#----------------------------------------------------
# Analytic solution of the 3D Poisson equation
@partial(jax.vmap, in_axes=(0, 0, 0), out_axes=0)
@jax.jit
def analytic_sol(xs,ys,zs):
    out = jnp.sin(xs*jnp.pi)*jnp.sin(ys*jnp.pi)*jnp.sin(zs*jnp.pi)
    return out

# PDE residual for 3D Poisson
@partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def residual(u, x, y, z):
    Hx = jax.hessian(u, argnums=0)(x,y,z)
    Hy = jax.hessian(u, argnums=1)(x,y,z)
    Hz = jax.hessian(u, argnums=2)(x,y,z)
    lhs = Hx+Hy+Hz
    rhs = (-3*(jnp.pi**2))*jnp.sin(x*jnp.pi)*jnp.sin(y*jnp.pi)*jnp.sin(z*jnp.pi)
    return lhs - rhs

# Loss functionals
@jax.jit
def pde_residual(params, points):
    return jnp.mean(residual(lambda x, y, z: model.apply(params, jnp.stack((x, y, z))), points[:, 0], points[:, 1], points[:, 2])**2)

@jax.jit
def boundary_dirichlet(params, points): 
    u_x0 = jnp.mean((model.apply(params, jnp.stack((jnp.zeros(*points[:, 0].shape), points[:, 1], points[:, 2]), axis=1)))**2) # u(0,y,z) = 0
    u_x1 = jnp.mean((model.apply(params, jnp.stack((jnp.ones(*points[:, 0].shape), points[:, 1], points[:, 2]), axis=1)))**2) # u(1,y,z) = 0
    u_y0 = jnp.mean((model.apply(params, jnp.stack((points[:, 0], jnp.zeros(*points[:, 1].shape), points[:, 2]), axis=1)))**2) # u(x,0,z) = 0
    u_y1 = jnp.mean((model.apply(params, jnp.stack((points[:, 0], jnp.ones(*points[:, 1].shape), points[:, 2]), axis=1)))**2) # u(x,1,z) = 0
    u_z0 = jnp.mean((model.apply(params, jnp.stack((points[:, 0], points[:, 1], jnp.zeros(*points[:, 2].shape)), axis=1)))**2) # u(x,y,0) = 0
    u_z1 = jnp.mean((model.apply(params, jnp.stack((points[:, 0], points[:, 1], jnp.ones(*points[:, 2].shape)), axis=1)))**2) # u(x,y,1) = 0
    return u_x0 + u_x1 + u_y0 + u_y1 + u_z0 + u_z1

#----------------------------------------------------
# Define Training Step
#----------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def training_step(params, opt, opt_state, key):
    lb = jnp.array([0.,0.,0.])
    ub = jnp.array([1.,1.,1.])
    domain_points = lb + (ub-lb)*lhs(3, 1000)
    boundary = lb + (ub-lb)*lhs(3, 100)
    
    loss_val, grad = jax.value_and_grad(lambda params: pde_residual(params, domain_points) + 
                                                    boundary_dirichlet(params, boundary))(params)
    
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
# Train PINN
#----------------------------------------------------
# Train model 10 times and average over the times
y_results, times_adam, times_lbfgs, times_total, times_eval, l2_rel, var, arch = dict({}), dict({}), dict({}), dict({}), dict({}), dict({}), dict({}), dict({})
n=0
for feature in architecture_list:
    times_adam_temp = []
    times_lbfgs_temp = []
    times_total_temp = []
    times_eval_temp = []
    accuracy_temp = []
    for _ in range(10):
        #----------------------------------------------------
        # Initialise Model
        #----------------------------------------------------
        model = PDESolution(feature)
        key, key_ = jax.random.split(jax.random.PRNGKey(17))
        batch_dim = 8
        feature_dim = 3
        params = model.init(key, jnp.ones((batch_dim, feature_dim)))

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

        init_point, tree, shapes = concat_params(params)
        # Generate data
        lb = jnp.array([0.,0.,0.])
        ub = jnp.array([1.,1.,1.])
        domain_points = lb + (ub-lb)*lhs(3, 4000)
        boundary = lb + (ub-lb)*lhs(3, 400)

        print('Starting L-BFGS Optimisation')
        start_time2 = time.time()
        results = tfp.optimizer.lbfgs_minimize(jax.value_and_grad(lambda params: pde_residual(unconcat_params(params, tree, shapes), domain_points) + 
                                                            boundary_dirichlet(unconcat_params(params, tree, shapes), boundary)) , 
                                    init_point,
                                    max_iterations=50000,
                                    num_correction_pairs=50,
                                    f_relative_tolerance=1.0 * jnp.finfo(float).eps)
        lbfgs_time = time.time()-start_time2
        times_total_temp.append(time.time()-start_time)
        times_lbfgs_temp.append(lbfgs_time)

        # Evaluation
        tuned_params = unconcat_params(results.position, tree, shapes)

        with open("./Eval_Points/3D_Poisson_eval-points.json", 'r') as f:
            domain_points = json.load(f)
            domain_points = jnp.array(domain_points['mesh_coord']['0'])

        start_time3 = time.time()
        u_approx = model.apply(tuned_params, jnp.stack((domain_points[:, 0], domain_points[:, 1], domain_points[:,2]), axis=1)).squeeze()
        times_eval_temp.append(time.time()-start_time3)

        u_true = analytic_sol(domain_points[:,0],domain_points[:,1],domain_points[:,2]).squeeze()
        run_accuracy = onp.linalg.norm(u_true - u_approx)/onp.linalg.norm(u_true)
        accuracy_temp.append(run_accuracy)

    y_gt = u_true.tolist()
    domain_pts = domain_points.tolist()
    y_results[n], times_adam[n], times_lbfgs[n], times_total[n], times_eval[n], l2_rel[n], var[n], arch[n] = u_approx.tolist(), onp.mean(times_adam_temp), onp.mean(times_lbfgs_temp), onp.mean(times_total_temp), onp.mean(times_eval_temp), onp.mean(accuracy_temp).tolist(), onp.var(accuracy_temp).tolist(), architecture_list[n]
    n+=1
    results = dict({'domain_pts': domain_pts,
                    'y_results': y_results,
                    'y_gt': y_gt})

    evaluation = dict({'arch': arch,
        'times_adam': times_adam,
        'times_lbfgs': times_lbfgs,
        'times_total': times_total,
        'times_eval': times_eval,
        'l2_rel': l2_rel,
        'var': var})

    save_dir = './3D-Poisson-PINN'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir,'PINNs_results.json'), "w") as write_file:
        json.dump(results, write_file)

    with open(os.path.join(save_dir,'PINNs_evaluation.json'), "w") as write_file:
        json.dump(evaluation, write_file)
    
    print(json.dumps(evaluation, indent=4))

