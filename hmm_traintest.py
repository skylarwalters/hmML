from hmmlearn import hmm
import numpy as np
import pickle

# create numerical relation between nucleotide base and number
def tideNum(nt):
    tideDict = {
        'A':0,
        'C':1,
        'T':2,
        'G':3
    }
    return tideDict[nt]

def train_hmm(motif, motif_pickle):
    """
    Train HMM on motif and negative control sequences and return trained HMM.
    Input: motif (str), motif_pickle (str)
    Output: hmm.CategoricalHMM
    """

    # Import Pickle File Here
    with open(motif_pickle, "rb") as f:
        motif_dict = pickle.load(f)

    states = ["background", "motif"]

    #emissions
    emissions = np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
    
    model = hmm.CategoricalHMM(random_state=1, init_params="", n_components=2, n_iter=100)

    model.startprob_ = np.array([.9, .1]) 
    model.transmat_ = np.array([[.9, .1], [.2, .8]])

    model.emissionprob_ = emissions

    # Prepare Data for Training and Fit HMM

    seqArr = []
    lengths = []
    for seq in motif_dict[motif][0]:
        lengths.append(len(seq))
        minilist = []

        for letter in seq:
            minilist.append(tideNum(letter))
        

        seqArr.extend(minilist)


    seqArr = np.asarray(seqArr).reshape(-1, 1)
    lengths = np.asarray(lengths).reshape(-1, 1)

    model.fit(X=seqArr, lengths=lengths)

    return model


def predict(sequence) -> bool:
    """
    The base predict function. Predicts true if any predicted state is "motif".
    """
    return True if 1 in sequence else False

def test_hmm(model, data, ground_truths=None):
    """
    Given a set of sequences, returns whether they are likely to contain the motif or only background.
    Input: hmm.CategoricalHMM, sequences (list[string]), labels (list[int])
    Output: float
    """
   

    # Prepare data for decoding and extract most likely sequence of states for each observation
    
    seqArr = []
    lengths = []
    
    for seq in data:

        lengths.append(len(seq))
        minilist = []

        for letter in seq:
            minilist.append(tideNum(letter))
        
        seqArr.extend(minilist)
    
    seqArr = np.asarray(seqArr).reshape(-1, 1)
    lengths = np.asarray(lengths).reshape(-1, 1)
      
    _, states_sequence = model.decode(X=seqArr, lengths=lengths, algorithm='viterbi')    
    

    # reshape out of 1D array
    evalSequences = []
    stateCounter = 0

    for length in lengths:
        currLength = length
        seq = []

        while currLength > 0:
            seq += list([states_sequence[stateCounter]])
            stateCounter += 1
            currLength -= 1
        
        
        evalSequences += list([seq])
    
    
    # For each sequence, use the predict function to make a list of 1s and 0s.
    # 1: sequence containing a motif, 0: sequence missing a motif.
    test_results = []

    for seq in evalSequences:
        if predict(seq):
            test_results += [1]
        else:
            test_results += [0]

    # Determine Model Performance
    test_results = np.asarray(test_results)
    if ground_truths is not None:
        i = 0
        correctCt = 0

        while i < len(ground_truths):
            if ground_truths[i] == test_results[i]:
                correctCt += 1
            i += 1
            
        # Percent of obersvations that match ground truth labels
        return correctCt/len(ground_truths)

    else:
        return np.sum(test_results)/len(test_results)
        # Check what fraction of sequences yielded positive predictions (label is 1)


def test_hmms(models, test_pickle, labeled):
    """
    Given multiple models for different motifs, tests a set of sequences and returns which motifs are likely to be present.
    Input: models (list[hmm.CategoricalHMM]), test_pickle (str), labeled (bool)
    Output: list[float]
    """
    with open(test_pickle, "rb") as f:
        motif_dict = pickle.load(f)

    test_results = []
    for motif in models:
        if labeled:
            observations, ground_truths = motif_dict[motif]
        else:
            # ground_truths will be None here
            observations, ground_truths = motif_dict["mystery"]
        test_results.append(test_hmm(models[motif], observations, ground_truths))
    return test_results
