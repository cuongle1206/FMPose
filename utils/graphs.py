
import numpy as np

BONES16j_MPII = [(0, 1), (1, 2), (2, 6), (6, 3), (3, 4), (4, 5), (6, 7), (7, 8),
                 (8, 9), (10, 11), (11, 12), (12, 8), (8, 13), (13, 14), (14, 15)]

BONES17j_H36M = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
                 (8, 9), (9, 10), (8, 1), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]

def create_adjacency_matrix(edges, num_nodes):
    A           = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges: A[edge] = 1.
    return A

def normalize_adjacency_matrix(A):
    node_degrees        = A.sum(-1)
    degs_inv_sqrt       = np.power(node_degrees, -0.5)
    norm_degs_matrix    = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)

def get_adjacency_matrix(num_nodes: int=16, inward: list=BONES16j_MPII):
    outward     = [(j, i) for (i, j) in inward]
    neighbor    = inward + outward
    self_loops  = [(i, i) for i in range(num_nodes)]
    # A           = create_adjacency_matrix(neighbor, num_nodes)
    A           = create_adjacency_matrix((neighbor+self_loops), num_nodes)
    A_norm      = normalize_adjacency_matrix(A)
    return A, A_norm

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    A, A_norm   = get_adjacency_matrix(16)
    print(A.shape, A_norm.shape)
    fig         = plt.figure()
    ax1         = fig.add_subplot(121)
    ax1.imshow(A, cmap='Blues')
    ax1.set_title('A')
    ax2         = fig.add_subplot(122)
    ax2.imshow(A_norm, cmap='Blues')
    ax2.set_title('A_norm')
    plt.savefig(f'./utils/adjacency.png', dpi=200, format='png')
    plt.close()
    