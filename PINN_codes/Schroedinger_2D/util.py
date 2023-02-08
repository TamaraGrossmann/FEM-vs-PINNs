from pyDOE import lhs
import jax.numpy as jnp

def sample_points(low_b,up_b,num_domain,num_bound,num_ini):
    lb = jnp.array(low_b)
    ub = jnp.array(up_b)
    domain_points = lb + (ub-lb)*lhs(3, num_domain)
    boundary = lb + (ub-lb)*lhs(3, num_bound)
    init = lb[1:] + (ub[1:]-lb[1:])*lhs(2, num_ini)

    return domain_points, boundary, init