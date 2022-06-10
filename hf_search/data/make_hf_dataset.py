import pandas as pd
import huggingface_hub as hf
from requests.exceptions import HTTPError
import datasets
import sentence_transformers


def make_hf_dataset_with_embeddings(
    sentence_transformer: sentence_transformers.SentenceTransformer,
    df: pd.DataFrame,
    embedded_col: str,
    batch_size: int = 128,
):
    df = df.dropna(subset=[embedded_col])
    ds = datasets.Dataset.from_pandas(df)
    ds_with_embeddings = ds.map(
        lambda rec: {
            **rec,
            f"{embedded_col}_embedding": sentence_transformer.encode(rec[embedded_col]),
        },
        batched=True,
        batch_size=batch_size,
    )
    return ds_with_embeddings
