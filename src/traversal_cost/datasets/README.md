## Dataset description

This folder stores generated datasets. A dataset used to train a Siamese network is a folder with the following structure:

```bash
dataset_{name}
├── features
│   ├── 000.npy
│   ├── 001.npy
│   └── ...
├── features_description.txt
├── labels.csv
├── pairs.csv
├── pairs_test.csv
└── pairs_train.csv
```

where:
- ***xxx*.npy** is a numpy array containing features extracted from the signal *xxx*
- **features_description.txt** is a text file containing information on how the features where extracted from the signals
- **labels.csv** is a csv file containing the labels of the signals (terrain class and linear velocity of the robot)
- **pairs.csv** is a csv file containing the pairs of signals to compare (the signal with id1 is supposed to be extracted from a terrain with a lower traversal cost than the associated signal with id2)
- **pairs_train.csv** contains the training set of pairs
- **pairs_test.csv** contains the test set of pairs
