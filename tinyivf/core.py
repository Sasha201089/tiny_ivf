import pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import faiss
import scipy
import os

class TinyIVF:

    def __init__(self, space='l2', dim=768, nprobe=3, folder_name="/start-vector", n_clusters=-1):
        if space != "l2":
            raise ValueError("На данный момент поддерживается только евклидово расстояние")
        self.space = space
        self.dim = dim
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        self.folder_name = folder_name
        self.clusters_index = None
        self.local_indices = {}
        self.local_dims = {}
        self.projections = {}
        self.local_ids = {}
        self.global_to_local = np.zeros((0, 2), dtype=np.int32)
        self.next_global_id = 0

    def _make_flat(self, dim):
        return faiss.IndexFlatL2(dim)

    def _effective_rank(self, vectors):
        if vectors.shape[0] == 1:
            p = np.eye(vectors.shape[1], dtype=np.float32)
            return p, vectors.shape[1]
        u, s, vt = scipy.linalg.svd(vectors, full_matrices=False, overwrite_a=True, check_finite=False)
        total = s.sum()
        if total == 0:
            p = np.eye(vectors.shape[1], dtype=np.float32)
            return p, vectors.shape[1]
        probs = s / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        eff_rank = int(round(np.exp(entropy)))
        eff_rank = max(1, min(eff_rank, vt.shape[0]))
        return vt[:eff_rank].astype(np.float32), eff_rank

    def train(self, vectors):
        vectors = np.array(vectors, dtype=np.float32)

        model = faiss.Kmeans(d=self.dim, k=self.n_clusters, niter=20, gpu=False)
        model.train(vectors)
        distances, labels = model.index.search(vectors, 1)
        labels = labels.ravel()
        centroids = model.centroids

        clusters_index = self._make_flat(self.dim)
        clusters_index.add(centroids)
        self.clusters_index = clusters_index

        self.local_indices = {}
        self.local_dims = {}
        self.projections = {}
        self.local_ids = {}
        self.next_global_id = 0
        self.global_to_local = np.zeros((0, 2), dtype=np.int32)

        for cid in range(self.n_clusters):
            cluster_vectors = vectors[labels == cid]
            if cluster_vectors.shape[0] == 0:
                continue
            p, eff_rank = self._effective_rank(cluster_vectors)
            self.projections[cid] = p
            self.local_dims[cid] = eff_rank
            self.local_indices[cid] = self._make_flat(eff_rank)
            self.local_ids[cid] = np.empty(0, dtype=np.int32)

    def add_items(self, vectors):
        vectors = np.array(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        _, id_centroids = self.clusters_index.search(vectors, 1)
        id_centroids = id_centroids[:, 0]

        num_new = len(vectors)
        row_global_ids = self.next_global_id + np.arange(num_new, dtype=np.int32)
        self.next_global_id += num_new

        new_g2l = np.empty((num_new, 2), dtype=np.int32)

        centroid_to_rows = {}
        for row, cid in enumerate(id_centroids):
            cid = int(cid)
            centroid_to_rows.setdefault(cid, []).append(row)

        for cid, rows in centroid_to_rows.items():
            if cid not in self.local_indices:
                continue
            local_vectors = vectors[rows]
            projected = local_vectors @ self.projections[cid].T
            projected = np.ascontiguousarray(projected, dtype=np.float32)

            start_local_id = self.local_indices[cid].ntotal
            self.local_indices[cid].add(projected)

            added_gids = row_global_ids[rows]

            self.local_ids[cid] = np.append(self.local_ids[cid], added_gids)

            for offset, row in enumerate(rows):
                new_g2l[row] = [cid, start_local_id + offset]

        self.global_to_local = np.vstack([self.global_to_local, new_g2l])

        return True

    def query(self, vectors, k=3):
        vectors = np.array(vectors, dtype=np.float32)
        n_vectors = len(vectors)

        _, id_centroids_batch = self.clusters_index.search(vectors, self.nprobe)

        centroid_to_vec_indices = {}
        for vec_idx, centroid_ids in enumerate(id_centroids_batch):
            for cid in centroid_ids:
                cid = int(cid)
                if cid == -1:
                    continue
                centroid_to_vec_indices.setdefault(cid, []).append(vec_idx)

        results = [[] for _ in range(n_vectors)]

        def process_centroid(centroid_id):
            if centroid_id not in self.local_indices or self.local_indices[centroid_id].ntotal == 0:
                return []
            vec_indices = centroid_to_vec_indices[centroid_id]
            local_vectors = vectors[vec_indices]
            projected = local_vectors @ self.projections[centroid_id].T
            projected = np.ascontiguousarray(projected, dtype=np.float32)
            local_index = self.local_indices[centroid_id]
            search_k = min(k, local_index.ntotal)
            dists_batch, ids_batch = local_index.search(projected, search_k)
            local_results = []
            for i, vec_idx in enumerate(vec_indices):
                for j in range(search_k):
                    local_id = int(ids_batch[i][j])
                    if local_id == -1:
                        continue
                    global_id = int(self.local_ids[centroid_id][local_id])
                    local_results.append((vec_idx, dists_batch[i][j], global_id))
            return local_results

        with ThreadPoolExecutor(max_workers=min(32, len(centroid_to_vec_indices))) as executor:
            futures = [executor.submit(process_centroid, cid) for cid in centroid_to_vec_indices]
            for future in futures:
                for vec_idx, dist, global_id in future.result():
                    results[vec_idx].append((dist, global_id))

        itog = []
        for vec_results in results:
            vec_results.sort(key=lambda x: x[0])
            top_k = vec_results[:k]
            top_k_ids = [gid for _, gid in top_k]
            top_k_distances = [dist for dist, _ in top_k]
            itog.append((top_k_ids, top_k_distances))

        return itog

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.clusters_index, os.path.join(path, "centroids.bin"))
        for cid in self.local_indices:
            os.makedirs(os.path.join(path, str(cid)), exist_ok=True)
            faiss.write_index(self.local_indices[cid], os.path.join(path, str(cid), "index.bin"))
            np.save(os.path.join(path, str(cid), "projection.npy"), self.projections[cid])
        state = {
            "n_clusters": self.n_clusters,
            "dim": self.dim,
            "space": self.space,
            "local_dims": self.local_dims,
            "local_ids": self.local_ids,
            "global_to_local": self.global_to_local,
            "next_global_id": self.next_global_id,
        }
        with open(os.path.join(path, "state.pkl"), "wb") as f:
            pickle.dump(state, f)

    def load(self, path):
        self.clusters_index = faiss.read_index(os.path.join(path, "centroids.bin"))
        with open(os.path.join(path, "state.pkl"), "rb") as f:
            state = pickle.load(f)
        self.n_clusters = state["n_clusters"]
        self.dim = state["dim"]
        self.space = state["space"]
        self.local_dims = state["local_dims"]
        self.local_ids = state["local_ids"]
        self.global_to_local = state["global_to_local"]
        self.next_global_id = state["next_global_id"]
        self.local_indices = {}
        self.projections = {}
        for cid in self.local_dims:
            index_path = os.path.join(path, str(cid), "index.bin")
            if os.path.exists(index_path):
                self.local_indices[cid] = faiss.read_index(index_path)
                self.projections[cid] = np.load(os.path.join(path, str(cid), "projection.npy"))