
from .distance_matrix import euclidean_distances_streamlined as pairwise_distances

from pdb import set_trace as pb
from tqdm import tqdm
import numpy as np

class ComputeDistMatrix:
    def __init__(self, embeddings, labeled_idxs, budget, save_matrix = True, metric='euclidean'):
        super().__init__()
        self.embeddings = embeddings
        self.labeled_idxs = np.zeros((embeddings.shape[0]), dtype=bool)
        self.budget = budget
        self.metric = metric
        
        dist_mat_diag = np.sum(embeddings**2, axis=1)
        self.dist_mat_diag = dist_mat_diag[:, None]

        self.save_matrix = save_matrix
        if save_matrix:
            self.dist_mat = np.zeros((dist_mat_diag.shape[0], budget))

        self.compute_matrix(labeled_idxs)
        

    def compute_matrix(self, new_labeled_idxs):
        use_idxs = (new_labeled_idxs.astype(np.float32) - self.labeled_idxs.astype(np.float32)).astype(bool)

        dist_mat_lab = pairwise_distances(self.embeddings, self.embeddings[use_idxs, :], n_jobs=1, metric=self.metric)

        # if hasattr(self, 'dist_mat'):
        #     self.dist_mat = np.concatenate((self.dist_mat, dist_mat_lab), axis=1)
        # else:
        #     self.dist_mat = dist_mat_lab
        if self.save_matrix:
            self.dist_mat[:, np.sum(self.labeled_idxs):(np.sum(self.labeled_idxs+use_idxs))] = dist_mat_lab
        
        if hasattr(self, 'mat_min'):
            dist_mat_lab_min = np.hstack((self.mat_min, dist_mat_lab))
        else:
            dist_mat_lab_min = dist_mat_lab

        current_min_center = np.argmin(dist_mat_lab_min, axis=1)
        if hasattr(self, 'mat_min_center'):
            self.mat_min_center[current_min_center > 0] = \
                current_min_center[current_min_center > 0]+np.max(self.mat_min_center)
        else:
            self.mat_min_center = current_min_center

        self.mat_min = np.take_along_axis(dist_mat_lab_min, current_min_center[:,None], axis=1)


        self.labeled_idxs = self.labeled_idxs + use_idxs

    def return_matrix(self):
        return self.dist_mat[:, 0:np.sum(self.labeled_idxs)]
        # return self.dist_mat

def coreset_k_centres_greedy(data, budget, indices=None, silent=False, save_matrix=True, metric='euclidean'):

    embeddings = data
    if indices is None:
        labeled_idxs = np.zeros((embeddings.shape[0]), dtype=bool)
        starting = np.random.choice(np.arange(embeddings.shape[0]))
        labeled_idxs[starting] = True
        number_of_samples = budget - 1
    else: 
        labeled_idxs = np.zeros((embeddings.shape[0]), dtype=bool)
        labeled_idxs[indices] = True
        number_of_samples = budget - indices.shape[0]

    # ======================================
    mat_helper = ComputeDistMatrix(embeddings, labeled_idxs, budget, save_matrix=save_matrix, metric=metric)
    # ======================================
    index_pool = np.arange(len(labeled_idxs))
    # ======================================
    index_list = np.argwhere(labeled_idxs == True).squeeze(axis=1).tolist()
    # ======================================
    for i in tqdm(range(number_of_samples), ncols=100, disable=silent):
        # mat = mat_helper.return_matrix()
        # mat = mat[~labeled_idxs, :]
        # mat_min = mat.min(axis=1)
        mat_min = mat_helper.mat_min[~labeled_idxs]
        q_idx_ = mat_min.argmax()
        # print(mat_min.values.max())
        # torch.sum(mat_helper.dist_mat[np.array(index_list), :] < mat_min.values.max())
        q_idx = index_pool[~labeled_idxs][q_idx_]
        index_list.append(q_idx)
        labeled_idxs[q_idx] = True
        mat_helper.compute_matrix(labeled_idxs)

    # picked_indices = np.argwhere(labeled_idxs == True).squeeze()
    picked_indices = np.array(index_list)
    if save_matrix:
        dist_mat = mat_helper.dist_mat
    else:
        dist_mat = None
    mat_min = mat_helper.mat_min
    mat_min_center = mat_helper.mat_min_center

    # np.sum(dist_mat[np.array(index_list), :] < dist_mat.min(axis=1).max())
    return picked_indices, dist_mat, mat_min, mat_min_center
