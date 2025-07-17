# put this file in the same folder as the tutorial ComparingNetworks.ipynb file
# in the same folder there should be ComputationThruDynamicsBenchmark/
# under the ComputationThruDynamicsBenchmark/ folder, there should be models_GRU_128/ and models_NODE_3/
import pickle
import numpy as np
from sklearn.decomposition import PCA
from DSA.dsa import DSA

# Set up paths to saved models (adjust HOME_DIR as needed)
HOME_DIR = "ComputationThruDynamicsBenchmark/"
fpath_GRU_128 = HOME_DIR + "models_GRU_128/"
fpath_NODE = HOME_DIR + "models_NODE_3/"

# Import the analysis class (assumes ctd is installed and in PYTHONPATH)
from ctd.comparison.analysis.tt.tt import Analysis_TT

# Create analysis objects (run_name can be any string)
analysis_GRU_128 = Analysis_TT(run_name="GRU_128_3bff", filepath=str(fpath_GRU_128))
analysis_NODE = Analysis_TT(run_name="NODE_3_3bff", filepath=str(fpath_NODE))

# Extract latent activities from validation set
latents_gru = analysis_GRU_128.get_latents(phase='val').detach().cpu().numpy()
latents_node = analysis_NODE.get_latents(phase='val').detach().cpu().numpy()

# Reshape to (n_samples, n_latents)
traj_gru = latents_gru.reshape(-1, latents_gru.shape[-1])
traj_node = latents_node.reshape(-1, latents_node.shape[-1])

########################################################
# if we directly run the DSA on the latents, it will fail because the latents' dimensions are not aligned
# so we need to reduce the dimensions of the latents to the same number of dimensions for the time being
# should discuss later if we have the dimension mismatch for our models
########################################################
# traj_gru = latents_gru.reshape(-1, latents_gru.shape[-1])  # shape: (n_trials*n_timesteps, n_latents)
# traj_node = latents_node.reshape(-1, latents_node.shape[-1])

# # Run DSA
# n_delays = 20
# delay_interval = 10

# dsa = DSA(X=traj_gru, Y=traj_node, n_delays=n_delays, delay_interval=delay_interval)
# similarity = dsa.fit_score()

####################PCA reduction ###################################

# Reduce both to the same number of dimensions (e.g., 3)
n_components = 3
pca_gru = PCA(n_components=n_components)
pca_node = PCA(n_components=n_components)
traj_gru_pca = pca_gru.fit_transform(traj_gru)
traj_node_pca = pca_node.fit_transform(traj_node)

# Run DSA on PCA-reduced latents
n_delays = 20
delay_interval = 10
dsa = DSA(
    X=traj_gru_pca,
    Y=traj_node_pca,
    n_delays=n_delays,
    delay_interval=delay_interval
)
similarity = dsa.fit_score()

print(f"Standalone DSA similarity score between GRU_128 and NODE latents (PCA-reduced): {similarity}")