import numpy as np
import pickle
import random

random.seed(1)
np.random.seed(1)

def gen_seq(slength: int, num_seq: int) -> list:
    """
    Generates a set of num_seq random DNA sequences of length slength, used for 
    generating negatives.
    Input: slength (int), num_seq (int)
    Output: list[str]
    """

    neg = []
    for i in range(num_seq):
        rstring = str()
        for j in range(slength):
            rstring += random.choice(
                ["A", "C", "T", "G"],
            )
        neg.append(rstring)
    return neg

def gen_negatives(length: int, num_negatives: int) -> list:
    """
    Generates a set of <num_negative> negative samples with <length> characters each.
    Input: length (int), num_negatives (int)
    Output: list[str]
    """
    # Generate negatives for test sets.

    return gen_seq(length, num_negatives)


def read_motif_file(fname: str) -> list:
    """
    Reads in all sequences listed in a FASTA file at fname.
    Input: fname (str)
    Output: list[str]
    """
    
    # Read motif file into a list of sequences.

    f = open(fname)

    lines = f.readlines()
    returns = []

    i = 1
    while i < len(lines):
        if lines[i][len(lines[i]) - 1] == '\n':
            returns += [lines[i][:len(lines[i]) - 1]]
        else:
            returns += [lines[i]]
        i += 2

    f.close()

    return returns

def preprocess_motif(motifs: list[str], motif_files: list[str]) -> None:
    """
    Preprocesses two lists of motif, file pairs into a training set (2/3 of the motif data) and a test set (1/3 of the motif data, and a matching size set of negatives). 
    Input: motifs (list[str]), motif_files (list[str])
    Output: none, two dumped pickle files
    """

    test_motif_dict = {}
    train_motif_dict = {}

    for name, mfile in zip(motifs, motif_files):
        ## Get FASTA Sequences for Motif Locations
        motif_fasta = read_motif_file(mfile)
        proportion = len(motif_fasta) // 3

        ## Convert FASTA Sequences to Numpy Arrays
        fasta_np = np.array(motif_fasta)

        # Randomly sample test and training data
        np.random.shuffle(fasta_np)
        fasta_np_train = fasta_np[proportion:]
        fasta_np_test = fasta_np[0:proportion]

        fasta_labels_test = np.ones(len(fasta_np_test))
        fasta_labels_train = np.ones(len(fasta_np_train))

        # Randomly sample negative test data
        negative_fasta_test = gen_negatives(len(motif_fasta[0]), proportion)
        neg_np = np.array(negative_fasta_test)
        neg_labels = np.zeros(len(negative_fasta_test))

        ## Append Motif and Negative Control Sequences and Shuffle for test
        fasta_np_test = np.append(fasta_np_test, neg_np)
        fasta_labels_test = np.append(fasta_labels_test, neg_labels)
        p = np.random.permutation(len(fasta_np_test))
        fasta_np_test = fasta_np_test[p]
        fasta_labels_test = fasta_labels_test[p]

        ## Create Dictionaries with Motif as Key and FASTA Sequences as Values
        test_motif_dict[name] = (fasta_np_test, fasta_labels_test)

        train_motif_dict[name] = (fasta_np_train, fasta_labels_train)

    # Dump motif_dicts to pickle files
    with open("train_motif_dict.pkl", "wb") as f:
        pickle.dump(train_motif_dict, f)
    with open("test_motif_dict.pkl", "wb") as f:
        pickle.dump(test_motif_dict, f)
