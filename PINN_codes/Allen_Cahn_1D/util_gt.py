import jax.numpy as jnp
import time, jax, os, json
import scipy.io

class ImportData:
    def __init__(self,save_dir = './Eval_Points/1D_Allen-Cahn/'):
        self.save_dir = save_dir

    def get_FEM_coordinates(self):
        with open(os.path.join(self.save_dir,'eval_coordinates.json'), 'r') as f:
            eval_coordinates= json.load(f)
        mesh_coord = eval_coordinates['mesh_coord']['0']
        dt_coord = eval_coordinates['dt_coord']['0']
        return mesh_coord, dt_coord

    def get_FEM_results(self):
        with open(os.path.join(self.save_dir,'eval_solution_mat.json'), 'r') as f:
            eval_solution_mat= json.load(f)
        eval_solution_mat = jnp.asarray(eval_solution_mat)
        return eval_solution_mat

        
def get_relative_error(u,v):
        l2 = jnp.linalg.norm(u - v)/jnp.linalg.norm(u)
        return l2

class CompareGT:

    def get_FEM_comparison(mesh_coord,dt_coord,FEM,model,tuned_params):
        dom_mesh = jnp.asarray(mesh_coord).squeeze()
        dom_mesh_ = jnp.tile(dom_mesh,len(dt_coord)) #repeating the dom_mesh, dt_coord_100.shape-times
        dom_ts = jnp.repeat(jnp.array(dt_coord),len(mesh_coord)) #repeating ts, len(mesh_coord)-times
        domain_pt = jnp.stack((dom_ts,dom_mesh_),axis=1) #stacking them together, meaning for each mesh coordinate we look at every time instance in ts
        
        start_time = time.time()
        approx = jax.block_until_ready(model.apply(tuned_params, domain_pt).squeeze())
        times_eval = time.time()-start_time
        
        approx = approx.reshape(len(dt_coord),len(mesh_coord)) 
        l2 = []

        for l in range(len(dt_coord)):
            l2.append(get_relative_error(FEM[int(l)],approx[int(l),:]))

        return l2, times_eval, approx, FEM, domain_pt
