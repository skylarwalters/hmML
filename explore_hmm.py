import hmmlearn as hmm
import numpy as np
from preprocess import preprocess_motif
from hmm_traintest import train_hmm, test_hmms
import pickle

motifs = ["chinmo", "kenbarbie", "lameduck", "sugarbabe"]
bed_files = ["data/" + motif + "_motifs.fa" for motif in motifs]

preprocess_motif(motifs, bed_files)

train_data = "train_motif_dict.pkl"
test_data = "test_motif_dict.pkl"
mystery_bag = "data/mystery.pkl"


# Train HMM
trained_models = {}  # key: motif, value: trained HMM
for motif in motifs:
    trained_models[motif] = train_hmm(motif, train_data)
    

# Test HMM
results = test_hmms(trained_models, test_data, True)

print("Results for TEST data.")
for motif, result in zip(motifs, results):
    print(f"{motif}: {result}")

print()

