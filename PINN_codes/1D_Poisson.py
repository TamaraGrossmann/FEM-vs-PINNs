import jax, flax, optax, time, pickle
import os
import jax.numpy as jnp
from functools import partial
from pyDOE import lhs
from typing import Sequence
import json
from tensorflow_probability.substrates import jax as tfp
import numpy as onp

# To only run on one GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)

#----------------------------------------------------
# Hyperparameters
#----------------------------------------------------
architecture_list = [[1,1],[2,1],[5,1],[10,1],[20,1],[40,1],[5,5,1],[10,10,1],[20,20,1],[40,40,1],[5,5,5,1],[10,10,10,1],[20,20,20,1],[40,40,40,1]]
lr = 1e-4
num_epochs = 15000
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
# Hessian-vector product
def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(lambda x: f(x)[0]), primals, tangents)[1]

# PDE residual for 1D Poisson
@partial(jax.vmap, in_axes=(None, 0), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def residual(u, x):
    v = jax.numpy.ones(x.shape)
    lhs = hvp(u,(x,),(v,))
    rhs = (-6*x + 4*x**3)*jax.numpy.exp(-x**2)
    return lhs - rhs

# Loss functionals
@jax.jit
def pde_residual(params, points):
    return jnp.mean(residual(lambda x: model.apply(params, x), points)**2)

@jax.jit
def boundary_residual0(params, xs):
    return jnp.mean((model.apply(params, jnp.zeros_like(xs)))**2)

@jax.jit
def boundary_residual1(params, xs):
    return jnp.mean((model.apply(params, jnp.ones_like(xs))-jnp.exp(-1.))**2)

#----------------------------------------------------
# Define Training Step
#----------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def training_step(params, opt, opt_state, key):
    lb = onp.array(0.)
    ub = onp.array(1.)
    domain_xs = lb + (ub-lb)*lhs(1, 256) #latin hybercube sampling
    boundary_xs = lb + (ub-lb)*lhs(1, 2)

    loss_val, grad = jax.value_and_grad(lambda params: pde_residual(params, domain_xs) + 
                                                    boundary_residual0(params, boundary_xs) +
                                                    boundary_residual1(params, boundary_xs))(params)
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
        return onp.concatenate([param.reshape(-1) for param in params]), tree, shapes

def unconcat_params(params, tree, shapes):
        split_vec = onp.split(params, onp.cumsum([onp.prod(shape) for shape in shapes]))
        split_vec = [vec.reshape(*shape) for vec, shape in zip(split_vec, shapes)]
        return jax.tree_util.tree_unflatten(tree, split_vec)

#----------------------------------------------------
# Train PINN
#----------------------------------------------------
# Train model 10 times and average over the times
y_results, domain_pts, times_adam, times_lbfgs, times_total, times_eval, l2_rel, var, arch = dict({}), dict({}), dict({}), dict({}), dict({}), dict({}), dict({}), dict({}), dict({})
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
        feature_dim = 1
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

        # Generate data
        lb = onp.array(0.)
        ub = onp.array(1.)
        domain_xs = lb + (ub-lb)*lhs(1, 256)
        boundary_xs = lb + (ub-lb)*lhs(1, 2)

        init_point, tree, shapes = concat_params(params)

        # L-BFGS optimisation
        print('Starting L-BFGS Optimisation')
        start_time2 = time.time()
        results = tfp.optimizer.lbfgs_minimize(jax.value_and_grad(lambda params: pde_residual(unconcat_params(params, tree, shapes), domain_xs) + 
                                                            boundary_residual0(unconcat_params(params, tree, shapes), boundary_xs) +
                                                            boundary_residual1(unconcat_params(params, tree, shapes), boundary_xs)), 
                                    init_point,
                                    max_iterations=50000,
                                    num_correction_pairs=50,
                                    f_relative_tolerance=1.0 * onp.finfo(float).eps)
        lbfgs_time = time.time()-start_time2
        times_total_temp.append(time.time()-start_time)
        times_lbfgs_temp.append(lbfgs_time)

        # Evaluation and comparison to ground truth
        tuned_params = unconcat_params(results.position, tree, shapes)

        with open("./Eval_Points/1D_Poisson_eval-points.json", 'r') as f:
            domain_points = json.load(f)
            domain_points = jnp.array(domain_points)

        start_time3 = time.time()
        u_approx = jax.block_until_ready(model.apply(tuned_params, domain_points).squeeze()) 
        times_eval_temp.append(time.time()-start_time3)

        u_true = (domain_points*jnp.exp(-domain_points**2)).squeeze()
        run_accuracy = (onp.linalg.norm(u_approx - u_true))/onp.linalg.norm(u_true)
        accuracy_temp.append(run_accuracy)

    y_gt = (domain_points*jnp.exp(-domain_points**2)).tolist()
    y_results[n], domain_pts[n], times_adam[n], times_lbfgs[n], times_total[n], times_eval[n], l2_rel[n], var[n], arch[n] = u_approx.tolist(),  domain_points.tolist(), onp.mean(times_adam_temp), onp.mean(times_lbfgs_temp), onp.mean(times_total_temp), onp.mean(times_eval_temp), onp.mean(accuracy_temp).tolist(), onp.var(accuracy_temp).tolist(), architecture_list[n]
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

    save_dir = './1D-Poisson-PINN'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir,'PINNs_results.json'), "w") as write_file:
        json.dump(results, write_file)

    with open(os.path.join(save_dir,'PINNs_evaluation.json'), "w") as write_file:
        json.dump(evaluation, write_file)
    
    print(json.dumps(evaluation, indent=4))