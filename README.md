# Leichte Sprache

This project aims to create an automated translator from standard German to Leichte Sprache, a ruleset for German that simplifies the language to make it widely accessible, for example to people who are functional illiterates or who comprehend only very basic German.

For this purpose, it contains code to:

-  crawl content in Leichte Sprache
- create a parallel dataset Leichte Sprache - standard German via artificial data training generation using an LLM
- finetune a model with the parallel dataset using PEFT 
- load the model and generate some examples

## Setup
To clone the repository and run the setup, run:
```
git clone https://github.com/nsaef/leichte-sprache.git
cd leichte-sprache
./install.sh
```
Notes:

- the repo is intended to be run on Linux or WSL
- creates a conda environment with python >=3.9
- installs the package leichte-sprache
- installs a pre-commit hook that runs black 
- installs German locale (sudo needed)
- creates a directory `data` in the repo's root directory

In order to use all functionalities, you need to create a `.env` file from `.env.template`. 

## Terminology

- Standard German: German as is taught in schools, used in media, spoken in everyday life
- [Leichte Sprache](https://en.wikipedia.org/wiki/Leichte_Sprache): specific ruleset to make German more comprehensible
- Singular dataset: Dataset with only one type of language, usually Leichte Sprache
- Parallel dataset: Dataset with both standard Germand and Leichte Sprache

## Usage


### Data Preparation
The package comes with multiple entrypoints. A complete workflow could look like this:
- `initialize_db`: Set up an SQLite database 
- `crawl_all`: Crawl all supported sources and store their contents in the SQLite DB
- `create_singular_dataset`: Process a dataset in Leichte Sprache from Konvens 2024 and store it in the SQLite DB in the same format as the crawled texts
- `translate_singular_dataset`: Translate the singular dataset from Leichte Sprache to standard German via an LLM. Intermediary saves to the DB are made regularly, and when re-running the command, only rows of the singular dataset that haven't been translated yet are loaded. Depending on your hardware, this step may take a while.
- `push_dataset_to_hub`: Remove undesirable rows from the dataset, then push it to the HuggingFace dataset hub. The HuggingFace repo name must be specified in the `.env` file.

All the above functionalities can also be run with the single command `run_data_pipeline`. Note that this is expected to run for at least a few hours.

### Model Training
To train a model with the data created above, run the following steps:

#### Set up MLFLow
The training script expects an MLFlow instance to which the training results are logged. To set it up, run:

```shell
docker run -d -v LOCAL_MLFLOW_PATH/mlruns:/mlruns -v /LOCAL_MLFLOW_PATH/mlartifacts:/mlartifacts --restart unless-stopped --name mlflow -p 0.0.0.0:5555:8080/tcp ghcr.io/mlflow/mlflow:v2.11.1 mlflow server --host 0.0.0.0 --port 8080
```
**Note for WSL usage**: MLFlow now runs within WSL. To connect to the GUI from Windows, run `ip a` to find the IP listed under `eth0`. 

Configure `.env` with MLFLow variables:

```
MLFLOW_EXPERIMENT_NAME = "leichte_sprache"
MLFLOW_TRACKING_URI = "http://IP:5555/"
```

#### Create a train config
Create a YAML file containing the training parameters. For an example, see `src/leichte_sprache/training/example_training_args.yaml`

#### Run the training
Run `python src/leichte_sprache/training/train.py PATH_TO_CONFIG.YAML` to finetune a model using PEFT. Adapt the parameters to your needs and your machine's capabilities.

#### Test the model
Run `python src/leichte_sprache/evaluation/run_model.py --base_model BASE_MODEL_NAME --peft_model CHECKPOINT_PATH` in order to generate a handful of example texts with the finetuned model.

Note: this is still rather rudimentary and will be extended in the future.

## Data Structures
### DB Structure
The SQLite DB created for this project is stored in the `data` directory. It will have the following tables after processing has run:

#### crawled_texts
This is where the crawled texts are stored. Table structure:

| source        | text                                      | url                               | crawl_timestamp    | title                              | release_date      | full_text                                      | id                                   |
|---------------|-------------------------------------------|-----------------------------------|--------------------|------------------------------------|-------------------|-----------------------------------------------|--------------------------------------|
| dlf | lorem             | http://example.com/article1       | 2024-06-10 10:15:00 | title                  | 2024-06-01 08:00:00 | title<br/>lorem   | d41d8cd98f00b204e9800998ecf8427e     |
| ndr | ipsum    | http://example.com/article2       | 2024-06-10 11:30:00 | None            | 2024-06-05 09:00:00 | ipsum     | 098f6bcd4621d373cade4e832627b4f6     |
| mdr  | foo | http://example.com/article3       | 2024-06-10 12:45:00 | title2                     | None | title2<br/>foo      | ad0234829205b9033196ba818f7a872b     |

Columns:
- `source`: name of the source website
- `text`: article text with basic processing (utf-8 encoding, strip spaces)
- `url`: URL of the website containing the content
- `crawl_timestamp`: date and time the content was crawled
- `title`: optional title of the content
- `release_date`: optional release date of the content
- `full_text`: concatenation of the title and the text
- `id`: MD5 hash of the full text

#### dataset_singular
Contains all available texts in Leichte Sprache (including datasets that were not crawled). Table structure:

| id                                   | text                                    | orig_ids                          |
|--------------------------------------|-----------------------------------------|-----------------------------------|
| d41d8cd98f00b204e9800998ecf8427e     | Title<br/>This is a short example text.           | [1, 2, 3]                         |
| 098f6bcd4621d373cade4e832627b4f6     | Another example with a different text.  | [4, 5, 6]                         |
| ad0234829205b9033196ba818f7a872b     | Title2<br/>More sample text for a different article.| http://example.com/article-3  |

Columns:
- `id`: MD5 hash of the full text/ID field from `crawled_texts`
- `text`: title + article text with basic processing (utf-8 encoding, strip spaces)/full_text from `crawled_texts`
- `orig_ids`: identifier(s) from the original source, i.e. IDs or URLs

#### dataset_singular_translated
This table contains the parallel dataset created via artificial data generation. Table structure:

| id                                   | text                                    | orig_ids                          | prompts                                                         | translated                               |
|--------------------------------------|-----------------------------------------|-----------------------------------|-----------------------------------------------------------------|------------------------------------------|
| d41d8cd98f00b204e9800998ecf8427e     | This is a short example text.           | [1, 2, 3]                         | [{"role": "user", "content": "prompt"}] | text in standard German.      |

Columns:
- `id`: ID field from `dataset_singular`
- `text`: text from `dataset_singular`
- `orig_ids`: orid_ids from `dataset_singular`
- `prompts`: prompt used to create the translated example (for documentation purposes)
- `translated`: text automatically translated to standard German

### Dataset format
The final parallel dataset is in the format:

| id | leichte_sprache    | standard_german              | source | url                        | release_date |
|--- |------------------- |------------------------------|--------|----------------------------|--------------|
| 2aa64159ff1108cbba73d89b9ed24a36 | Industrie-Gebiet<br/>Ein Gebiet ist ein Teil von einer Stadt:<br/>Oder es ist ein Teil von einem Land.<br/>In einem Industrie-Gebiet <br/>gibt es viele Fabriken und Betriebe.<br/>Zum Beispiel:<br/>    • Druckereien.<br/>       Da werden Bücher und Zeitungen gedruckt.<br/>    • Auto-Bauer<br/>    • oder Maschinen-Bauer.<br/>       Da werden große Maschinen gebaut.  | Das Industriegebiet ist eine geografische Einheit, die sich innerhalb einer Stadt oder eines Landes befindet und sich durch die Ansammlung von Fabriken und Betrieben auszeichnet. Beispielsweise umfasst ein Industriegebiet Druckereien, in denen Bücher und Zeitungen gedruckt werden, Automobilhersteller sowie Maschinenbauer, die große Maschinen produzieren. | mdr | https://www.mdr.de/nachrichten-leicht/woerterbuch/glossar-industrie-gebiet-100.html| 2018-03-16 09:13:00 |


## Next Steps
- add tests for crawler
- extend the example texts in `evaluation.run_model`
- create a proper evaluator (rule-based or classifier)
- create a DPO dataset
- train a DPO model
