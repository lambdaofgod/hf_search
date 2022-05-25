import pandas as pd
import huggingface_hub as hf
from requests.exceptions import HTTPError
from tqdm.contrib.concurrent import thread_map


def get_readme(model_record):
    try:
        path = hf.hf_hub_download(model_record.modelId, "README.md")
        readme_contents = open(path).read()
    except (HTTPError, UnicodeDecodeError) as e:
        return None
    return readme_contents


def get_model_records_with_readmes(product):
    model_metadata = hf.list_models()
    datasets = hf.list_datasets()
    readmes = thread_map(get_readme, model_metadata)
    model_metadata_df = pd.DataFrame.from_records(
        {**record.__dict__, "readme": readme}
        for (record, readme) in zip(model_metadata, readmes)
    )
    model_metadata_df.to_csv(str(product))
