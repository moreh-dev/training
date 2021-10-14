Moreh eval_mlm_accuracy debug

Data copy
=================================================================================
cp /home/share/dataset/mlperf/bert/cks/model.ckpt-28252.pt cks/

Run moreh
=================================================================================
for 1 gpu:
```
bash run_bert_ref_3072_load_test_1gpu.sh
```
for 4 gpu:
```
bash run_bert_ref_3072_load_test_4gpu.sh
```
for 8 gpu:
```
bash run_bert_ref_3072_load_test_8gpu.sh
```




BERT Benchmark with pure PyTorch
=================================================================================

Modified benchmark from https://github.com/mlperf/training_results_v0.7/tree/master/NVIDIA/benchmarks/bert/implementations/pytorch.

# Download dataset and files

Download files from the [google drive](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT).

Prepare the following files.

```shell
./cleanup_scripts/wiki/enwiki-20200101-pages-articles-multistream.xml.bz2 # Dataset
./cks/bert_config.json # Bert hyperparameter configuration
./cks/tf1_ckpt/model.ckpt-28252.data-00000-of-00001
./cks/tf1_ckpt/model.ckpt-28252.index
./cks/tf1_ckpt/model.ckpt-28252.meta
./vocab.txt # Vocab file to map WordPiece to word id.
```

# Preprocess dataset

```shell
bzip2 -d enwiki-20200101-pages-articles-multistream.xml.bz2 # At bert/cleanup_scripts/wiki

cd ..    # back to bert/cleanup_scripts 
pip install wikiextractor
python -m wikiextractor.WikiExtractor wiki/enwiki-20200101-pages-articles-multistream.xml # Results are placed in bert/cleanup_scripts/text
./process_wiki.sh 'text/*/wiki_??'
```

Run [WikiExtractor.py](https://github.com/attardi/wikiextractor) to extract the wiki pages from the XML
The generated wiki pages file will be stored as <data dir>/LL/wiki_nn; for example <data dir>/AA/wiki_00. Each file is ~1MB, and each sub directory has 100 files from wiki_00 to wiki_99, except the last sub directory.

Clean up
The clean up scripts (some references here) are in the scripts directory.
The following command will run the clean up steps, and put the results in ./results
./process_wiki.sh '<data dir>/*/wiki_??'

After running the process_wiki.sh script, for the 20200101 wiki dump, there will be 531 files, named part-00xxx-of-00500 in the ./results directory.

Original readme says 500 files should be generated, but actually 531 files are generated. Maybe it is because wikiextractor has been updated.

# Checkpoint conversion

Require tensorflow 1.x.

```
python convert_tf_checkpoint.py --tf_checkpoint cks/tf1_ckpt/model.ckpt-28252 --bert_config_path cks/bert_config.json --output_checkpoint cks/model.ckpt-28252.pt
```

# Generate the BERT input dataset

The create_pretraining_data.py script duplicates the input plain text, replaces different sets of words with masks for each duplication, and serializes the output into the TFRecord file format. 

We will use part 000 to 500 as training dataset, and remaining 501 to 531 as evaluation dataset.

```
./create_pretraining.sh
```

The script converts the dataset to tfrecord format. It then save the output to `train_dataset` or `eval_dataset`. 

The script is jsut a simple for loop to convert 531 parts of the dataset. It takes long time (about 30 hour if running sequentially), so consider splitting the for loop in multiple parts then running it parallel.

# Run the model training

```
./run.sh
```

It takes about 15 hours to train on single TITAN RTX GPU.