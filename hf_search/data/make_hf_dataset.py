import pandas as pd
import datasets
import torch
torch.multiprocessing.set_start_method('spawn', force=True)
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


def make_hf_models_dataset(sentence_transformer_name, upstream, product, batch_size):
    model = sentence_transformers.SentenceTransformer(sentence_transformer_name)
    df = pd.read_csv(str(upstream["prepare_model_records_with_readmes"]))

    ds = make_hf_dataset_with_embeddings(model, df, "readme", batch_size=batch_size)
    ds.save_to_disk(str(product))
