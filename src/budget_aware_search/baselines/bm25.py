from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm

from budget_aware_search.metrics import evaluate_predictions
from budget_aware_search.utils import write_json


@dataclass
class SparseBM25:
    vectorizer: CountVectorizer
    weighted_documents: sparse.csr_matrix
    product_ids: np.ndarray
    k1: float
    b: float

    @classmethod
    def fit(
        cls,
        product_ids: list[str] | np.ndarray,
        documents: list[str] | np.ndarray,
        k1: float = 1.5,
        b: float = 0.75,
        min_df: int = 1,
        max_features: int | None = None,
    ) -> "SparseBM25":
        if len(product_ids) != len(documents):
            raise ValueError("product_ids and documents must have the same length")
        if len(product_ids) == 0:
            raise ValueError("Cannot fit BM25 on an empty catalog")

        vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[\w-]+\b",
            min_df=min_df,
            max_features=max_features,
            dtype=np.float32,
        )
        counts = vectorizer.fit_transform(list(documents)).tocsr().astype(np.float32)
        n_documents = counts.shape[0]
        document_lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = float(document_lengths.mean())
        if average_length <= 0:
            raise ValueError("All catalog documents are empty after tokenization")

        document_frequency = np.diff(counts.tocsc().indptr).astype(np.float32)
        idf = np.log1p((n_documents - document_frequency + 0.5) / (document_frequency + 0.5))

        weighted = counts.copy()
        row_indices = np.repeat(np.arange(n_documents), np.diff(weighted.indptr))
        normalization = k1 * (
            1.0 - b + b * document_lengths[row_indices] / average_length
        )
        term_frequency = weighted.data.copy()
        weighted.data = (
            idf[weighted.indices]
            * term_frequency
            * (k1 + 1.0)
            / (term_frequency + normalization)
        ).astype(np.float32)
        weighted.eliminate_zeros()

        return cls(
            vectorizer=vectorizer,
            weighted_documents=weighted,
            product_ids=np.asarray(product_ids, dtype=str),
            k1=float(k1),
            b=float(b),
        )

    @property
    def product_to_row(self) -> dict[str, int]:
        if not hasattr(self, "_product_to_row"):
            self._product_to_row = {
                str(product_id): index for index, product_id in enumerate(self.product_ids)
            }
        return self._product_to_row

    def score_candidates(self, query: str, candidate_product_ids: list[str]) -> np.ndarray:
        query_vector = self.vectorizer.transform([query]).tocsr()
        query_terms = query_vector.indices
        scores = np.zeros(len(candidate_product_ids), dtype=np.float32)
        if query_terms.size == 0:
            return scores

        known_positions: list[int] = []
        document_rows: list[int] = []
        mapping = self.product_to_row
        for position, product_id in enumerate(candidate_product_ids):
            row = mapping.get(str(product_id))
            if row is not None:
                known_positions.append(position)
                document_rows.append(row)
        if not document_rows:
            return scores

        candidate_matrix = self.weighted_documents[document_rows][:, query_terms]
        candidate_scores = np.asarray(candidate_matrix.sum(axis=1)).ravel().astype(np.float32)
        scores[np.asarray(known_positions)] = candidate_scores
        return scores

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, directory / "vectorizer.joblib")
        sparse.save_npz(directory / "weighted_documents.npz", self.weighted_documents)
        np.save(directory / "product_ids.npy", self.product_ids)
        write_json(
            {
                "k1": self.k1,
                "b": self.b,
                "documents": int(len(self.product_ids)),
                "vocabulary": int(len(self.vectorizer.vocabulary_)),
            },
            directory / "metadata.json",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "SparseBM25":
        directory = Path(directory)
        vectorizer = joblib.load(directory / "vectorizer.joblib")
        matrix = sparse.load_npz(directory / "weighted_documents.npz").tocsr()
        product_ids = np.load(directory / "product_ids.npy", allow_pickle=False)
        metadata = pd.read_json(directory / "metadata.json", typ="series")
        return cls(
            vectorizer=vectorizer,
            weighted_documents=matrix,
            product_ids=product_ids,
            k1=float(metadata["k1"]),
            b=float(metadata["b"]),
        )


def score_candidate_pool(model: SparseBM25, frame: pd.DataFrame) -> pd.DataFrame:
    scored_parts: list[pd.DataFrame] = []
    for _, group in tqdm(
        frame.groupby("query_id", sort=False),
        total=frame["query_id"].nunique(),
        desc="BM25 candidate scoring",
    ):
        part = group.copy()
        part["score"] = model.score_candidates(
            query=str(group["query"].iloc[0]),
            candidate_product_ids=group["product_id"].astype(str).tolist(),
        )
        scored_parts.append(part)
    return pd.concat(scored_parts, ignore_index=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit/evaluate a sparse BM25 baseline.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--index-dir", default="models/bm25")
    parser.add_argument("--output", default="outputs/bm25_val.parquet")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    processed_dir = Path(args.processed_dir)
    index_dir = Path(args.index_dir)
    required_index_files = [
        index_dir / "vectorizer.joblib",
        index_dir / "weighted_documents.npz",
        index_dir / "product_ids.npy",
        index_dir / "metadata.json",
    ]

    if args.refit or not all(path.exists() for path in required_index_files):
        catalog = pd.read_parquet(processed_dir / "catalog.parquet")
        print(f"Fitting BM25 on {len(catalog):,} catalog products...")
        model = SparseBM25.fit(
            product_ids=catalog["product_id"].astype(str).to_numpy(),
            documents=catalog["product_text"].fillna("").astype(str).to_numpy(),
            k1=args.k1,
            b=args.b,
            min_df=args.min_df,
            max_features=args.max_features,
        )
        model.save(index_dir)
    else:
        model = SparseBM25.load(index_dir)

    frame = pd.read_parquet(processed_dir / f"{args.split}.parquet")
    scored = score_candidate_pool(model, frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output, index=False)

    summary, per_query = evaluate_predictions(
        scored, bootstrap_samples=args.bootstrap_samples
    )
    write_json(summary, output.with_suffix(".metrics.json"))
    per_query.to_parquet(output.with_suffix(".per_query.parquet"), index=False)
    print(f"BM25 nDCG@10: {summary['metrics']['ndcg@10']['mean']:.6f}")
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
