# Question Answering Corpus

This repository contains a script to generate question/answer pairs using
CNN and Daily Mail articles downloaded from the Wayback Machine.

For a detailed description of this corpus please read:
[Teaching Machines to Read and Comprehend][arxiv], Hermann et al., NIPS 2015.
Please cite the paper if you use this corpus in your work.

### Bibtex

```
@inproceedings{nips15_hermann,
author = {Karl Moritz Hermann and Tom\'a\v{s} Ko\v{c}isk\'y and Edward Grefenstette and Lasse Espeholt and Will Kay and Mustafa Suleyman and Phil Blunsom},
title = {Teaching Machines to Read and Comprehend},
url = {http://arxiv.org/abs/1506.03340},
booktitle = "Advances in Neural Information Processing Systems (NIPS)",
year = "2015",
}
```

## Download Processed Version

In case the script does not work you can also download the processed data sets from [http://cs.nyu.edu/~kcho/DMQA/]. This should help in situations where the underlying data is not accessible (Wayback Machine partially down).

## Running the Script

### Prerequisites

Python 2.7, `wget`, `libxml2`, `libxslt`, `python-dev` and `virtualenv`. `libxml2` must be version 2.9.1. 
You can install `libxslt` from here: [http://xmlsoft.org/libxslt/downloads.html](http://xmlsoft.org/libxslt/downloads.html)

```
sudo pip install virtualenv
sudo apt-get install python-dev
```

### Download Script

```
mkdir rc-data
cd rc-data
wget https://github.com/deepmind/rc-data/raw/master/generate_questions.py
```

### Download and Extract Metadata

```
wget https://storage.googleapis.com/deepmind-data/20150824/data.tar.gz -O - | tar -xz --strip-components=1
```

The news article metadata is ~1 GB.

### Enter Virtual Environment and Install Packages

```
virtualenv venv
source venv/bin/activate
wget https://github.com/deepmind/rc-data/raw/master/requirements.txt
pip install -r requirements.txt
```

You may need to install `libxml2` development packages to install `lxml`:

```
sudo apt-get install libxml2-dev libxslt-dev
```

### Download URLs

```
python generate_questions.py --corpus=[cnn/dailymail] --mode=download
```

This will download news articles from the Wayback Machine. Some URLs may be
unavailable. The script can be run again and will cache
URLs that already have been downloaded. Generation of questions can run
without all URLs downloaded successfully.

### Generate Questions

```
python generate_questions.py --corpus=[cnn/dailymail] --mode=generate
```

Note, this will generate ~1,000,000 small files for the Daily Mail so an SSD is
preferred.

Questions are stored in [cnn/dailymail]/questions/ in the following format:

```
[URL]

[Context]

[Question]

[Answer]

[Entity mapping]
```

### Deactivate Virtual Environment

```
deactivate
```

### Verifying Test Sets

```
wget https://github.com/deepmind/rc-data/raw/master/expected_[cnn/dailymail]_test.txt
comm -3 <(cat expected_[cnn/dailymail]_test.txt) <(ls [cnn/dailymail]/questions/test/)
```

The filenames of the questions in the first column are missing generated questions. No output means everything is downloaded and generated correctly.

[arxiv]: http://arxiv.org/abs/1506.03340
