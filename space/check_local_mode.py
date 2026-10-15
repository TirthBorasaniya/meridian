"""Build-time check (SPEC S5.2): does Qdrant local mode support named sparse vectors with the IDF
modifier and the RRF Query API fusion used by Meridian? The result is written to a JSON file
in the image. The Space runs the Qdrant server binary either way, because releases are server
snapshots and local mode cannot restore them (DECISIONS.md D-007).
"""

import argparse
import json

from qdrant_client import QdrantClient, models


def check() -> dict:
    """Run the query pattern against an in-memory local-mode client."""
    client = QdrantClient(":memory:")
    client.create_collection(
        "probe",
        vectors_config={"dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.upsert(
        "probe",
        [
            models.PointStruct(
                id=index,
                vector={
                    "dense": [1.0, float(index), 0.0],
                    "sparse": models.SparseVector(indices=[index, 7], values=[1.0, 1.0]),
                },
            )
            for index in range(1, 6)
        ],
    )
    response = client.query_points(
        "probe",
        prefetch=[
            models.Prefetch(query=[1.0, 2.0, 0.0], using="dense", limit=50),
            models.Prefetch(
                query=models.SparseVector(indices=[2], values=[1.0]), using="sparse", limit=50
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=60)),
        limit=3,
    )
    return {
        "named_sparse_with_idf": True,
        "rrf_fusion": len(response.points) > 0,
        "top_id": response.points[0].id if response.points else None,
    }


def main() -> None:
    """Write the check result; fail the build if the pattern errors."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = check()
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
