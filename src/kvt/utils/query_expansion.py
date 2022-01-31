import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class QueryExpansion:
    def __init__(
        self,
        alpha=1,
        k=2,
        similarity_threshold=None,
        normalize_similarity=False,
        strategy_to_deal_original="discard",
        n_query_update_iter=1,
        n_reference_update_iter=1,
        batch_size=10,
        use_cuda=True,
    ):
        self.alpha = alpha
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.normalize_similarity = normalize_similarity
        self.strategy_to_deal_original = strategy_to_deal_original
        self.n_query_update_iter = n_query_update_iter
        self.n_reference_update_iter = n_reference_update_iter
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        assert strategy_to_deal_original in ("add", "concat", "discard")

    def create_index(self, reference_embeddings):
        # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        index = faiss.IndexFlatIP(
            reference_embeddings.shape[1]
        )  # inner product
        if self.use_cuda:
            index = faiss.index_cpu_to_all_gpus(index)
        index.add(reference_embeddings)
        return index

    def search_index(
        self,
        index_or_reference_embeddings,
        query_embeddings,
        reference_ids=None,
        k=20,
        batch_size=1,
        normalize=True,
    ):
        if normalize:
            query_embeddings = self.normalize(query_embeddings)

        if isinstance(index_or_reference_embeddings, np.ndarray):
            if normalize:
                index_or_reference_embeddings = self.normalize(
                    index_or_reference_embeddings
                )
            index = self.create_index(index_or_reference_embeddings)
        else:
            index = index_or_reference_embeddings

        results_id, results_sim = [], []
        N = len(query_embeddings)
        n_iters = int(np.ceil(N / batch_size))
        for i in range(n_iters):
            D, indices = index.search(
                query_embeddings[batch_size * i : min(batch_size * (i + 1), N)],
                k,
            )
            if reference_ids is not None:
                indices = np.apply_along_axis(
                    lambda x: reference_ids[x], 1, indices
                )
            results_id.append(indices)
            results_sim.append(D)
        return np.vstack(results_id), np.vstack(results_sim)

    def normalize(self, embeddings):
        embeddings /= np.linalg.norm(embeddings, axis=1).reshape((-1, 1))
        return embeddings

    def expand(self, sims, ids, original_embeddings, reference_embeddings):
        assert len(sims) == len(original_embeddings)

        if self.normalize_similarity:
            sims /= np.linalg.norm(sims, axis=1).reshape((-1, 1))

        weights = np.expand_dims(sims ** self.alpha, axis=-1)
        if self.similarity_threshold is not None:
            mask = np.expand_dims(sims < self.similarity_threshold, axis=-1)
            weights[mask] = 0

        embeddings = (reference_embeddings[ids] * weights).sum(axis=1)

        if self.strategy_to_deal_original == "add":
            embeddings += original_embeddings
        elif self.strategy_to_deal_original == "concat":
            embeddings = self.normalize(embeddings)
            original_embeddings = self.normalize(original_embeddings)
            embeddings = np.concatenate(
                [original_embeddings, embeddings], axis=1
            )

        return self.normalize(embeddings)

    def query_expansion(
        self, original_embeddings, reference_embeddings, index=None
    ):
        if index is None:
            ids, sims = self.search_index(
                reference_embeddings,
                original_embeddings,
                reference_ids=None,
                k=self.k,
                batch_size=self.batch_size,
            )
        else:
            ids, sims = self.search_index(
                index,
                original_embeddings,
                reference_ids=None,
                k=self.k,
                batch_size=self.batch_size,
            )
        embeddings = self.expand(
            sims, ids, original_embeddings, reference_embeddings
        )
        return embeddings

    def __call__(self, query_embeddings, reference_embeddings):
        query_embeddings = self.normalize(query_embeddings)
        reference_embeddings = self.normalize(reference_embeddings)

        calculate_at_once = False
        if (self.n_reference_update_iter > 0) and (
            self.strategy_to_deal_original == "concat"
        ):
            reference_embeddings = np.vstack(
                [reference_embeddings, query_embeddings]
            )
            calculate_at_once = True

        for _ in range(self.n_reference_update_iter):
            reference_embeddings = self.query_expansion(
                reference_embeddings, reference_embeddings
            )

        if not calculate_at_once:
            for _ in range(self.n_query_update_iter):
                query_embeddings = self.query_expansion(
                    query_embeddings, reference_embeddings
                )
        else:
            query_embeddings = reference_embeddings[-len(query_embeddings) :]
            reference_embeddings = reference_embeddings[
                : -len(query_embeddings)
            ]

        return query_embeddings, reference_embeddings
