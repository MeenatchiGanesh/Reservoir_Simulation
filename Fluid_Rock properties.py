from two_phase_utils import *
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import io
from numpy.linalg import norm
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Initialize simulation parameters
tstep = 2
timeint = np.arange(0, 10, tstep)
sim_time = prop_time(tstep=tstep, timeint=timeint)
rock = prop_rock(kx=np.array([50, 50, 50, 150, 150, 150]),
                 ky=np.array([200, 200, 200, 300, 300, 300]),
                 por=0.22, cr=0)
fluid = prop_fluid(c_o=0.8e-5, mu_o=2.5, rho_o=49.1, p_bub=3500, p_atm=14.7)
grid = prop_grid(Nx=3, Ny=2, Nz=1)
res = prop_res(Lx=1500, Ly=1500, Lz=200,
               press_n=np.array([2500, 2525, 2550, 2450, 2475, 2500]),
               sg_n=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
               press_n1_k=np.array([2505, 2530, 2555, 2455, 2480, 2505]),
               sg_n1_k=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
well1 = prop_well(loc=(1, 1), q=0)
props = {'rock': rock, 'fluid': fluid, 'grid': grid, 'res': res, 'well': [well1], 'time': sim_time}

# Visualize Fluid and Rock Properties
fluid.plot_all()
rock.plot_all()

# Distribute properties and values to the grid
b_o = np.zeros(grid.Ny * grid.Nx)
db_o = np.zeros(grid.Ny * grid.Nx)
# ... (Continue with the rest of your code for distributing properties and values)

# Construct matrix T (containing transmissibility terms)
T = construct_T(mat, params)
params.update({'T': T})
p_n1_k = res.stack_ps(res.press_n1_k, res.sg_n1_k)

# Construct matrix D (containing accumulation terms)
D = construct_D(mat, params, props)
p_n = res.stack_ps(res.press_n, res.sg_n)


# Print Error Metrics
print('Mean absolute error in R =', mean_abs_error_R)
print('Mean absolute error in J =', mean_abs_error_J)
print('Relative error in R =', relative_error_R)
print('Relative error in J =', relative_error_J)

# Save Flipped Residual and Jacobian into .csv Files
np.savetxt('flipped_R.csv', R_flipped, delimiter=',')
np.savetxt('flipped_J.csv', J_flipped, delimiter=',')

# Convert and Save Matrices to Sparse Format
J_ref_sparse = csr_matrix(J_reference)
io.mmwrite('sparse_J_ref.mtx', J_ref_sparse)
J_result_sparse = csr_matrix(J_flipped)
io.mmwrite('sparse_J_result.mtx', J_result_sparse)

R_ref_sparse = csr_matrix(R_reference)
io.mmwrite('sparse_R_ref.mtx', R_ref_sparse)
R_result_sparse = csr_matrix(R_flipped)
io.mmwrite('sparse_R_result.mtx', R_result_sparse)


# Calculate Residual Matrix
R = np.dot(T, p_n1_k) - D

# Calculate Jacobian Matrix
J = construct_J(mat, params, props)

# Flip Variables to Match Reference Format
R_flipped = flip_variables(R, 0)
J_flipped = flip_variables(J, 1)



# Load Reference Data for Reservoir Assam Basin
j_dir_ref = 'https://example.com/reservoir_assam_basin_reference_J.csv'
r_dir_ref = 'https://example.com/reservoir_assam_basin_reference_R.csv'
J_reference = pd.read_csv(j_dir_ref, header=None)
R_reference = pd.read_csv(r_dir_ref, header=None)


# Evaluate Mean Absolute Error and Relative Error
mean_abs_error_R = mean_absolute_error(R_flipped, R_reference)
mean_abs_error_J = mean_absolute_error(J_flipped, J_reference)

relative_error_R = norm(np.reshape(R_flipped, (-1, 1)) - R_reference) / norm(R_reference)
relative_error_J = norm(J_flipped - J_reference) / norm(J_reference)

# Print Error Metrics
print('Mean absolute error in R =', mean_abs_error_R)
print('Mean absolute error in J =', mean_abs_error_J)
print('Relative error in R =', relative_error_R)
print('Relative error in J =', relative_error_J)

# Save Flipped Residual and Jacobian into .csv Files
np.savetxt('flipped_R.csv', R_flipped, delimiter=',')
np.savetxt('flipped_J.csv', J_flipped, delimiter=',')

# Convert and Save Matrices to Sparse Format
J_ref_sparse = csr_matrix(J_reference)
io.mmwrite('sparse_J_ref.mtx', J_ref_sparse)
J_result_sparse = csr_matrix(J_flipped)
io.mmwrite('sparse_J_result.mtx', J_result_sparse)

R_ref_sparse = csr_matrix(R_reference)
io.mmwrite('sparse_R_ref.mtx', R_ref_sparse)
R_result_sparse = csr_matrix(R_flipped)
io.mmwrite('sparse_R_result.mtx', R_result_sparse)
# Save Flipped Residual and Jacobian into .csv Files
np.savetxt('flipped_R.csv', R_flipped, delimiter=',')
np.savetxt('flipped_J.csv', J_flipped, delimiter=',')

# Convert and Save Matrices to Sparse Format
J_ref_sparse = csr_matrix(J_reference)
io.mmwrite('sparse_J_ref.mtx', J_ref_sparse)
J_result_sparse = csr_matrix(J_flipped)
io.mmwrite('sparse_J_result.mtx', J_result_sparse)

R_ref_sparse = csr_matrix(R_reference)
io.mmwrite('sparse_R_ref.mtx', R_ref_sparse)
R_result_sparse = csr_matrix(R_flipped)
io.mmwrite('sparse_R_result.mtx', R_result_sparse)