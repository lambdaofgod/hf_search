import pandas as pd
import huggingface_hub as hf
from requests.exceptions import HTTPError
from tqdm.contrib.concurrent import thread_map


def get_readme(model_record):
    try:
        path = hf.hf_hub_download(model_record.modelId, "README.md")
        with open(path) as f:
            readme_contents = f.read()
    except (HTTPError, UnicodeDecodeError) as e:
        return None
    return readme_contents


def prepare_model_records_with_readmes(product):
    model_metadata = hf.list_models()
    datasets = hf.list_datasets()
    readmes = thread_map(get_readme, model_metadata)
    model_metadata_df = pd.DataFrame.from_records(
        {**record.__dict__, "readme": readme}
        for (record, readme) in zip(model_metadata, readmes)
    )
    model_metadata_df.to_csv(str(product), index=False)
