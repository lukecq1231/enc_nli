import numpy
import os

from main import train

if __name__ == '__main__':
    model_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    train(
    saveto           = './{}.npz'.format(model_name),
    reload_          = False,
    dim_word         = 300,
    dim              = 600,
    patience         = 7,
    n_words          = 100140,
    decay_c          = 0.,
    clip_c           = 10.,
    lrate            = 0.0004,
    optimizer        = 'adam', 
    maxlen           = 450,
    batch_size       = 32,
    valid_batch_size = 32,
    dispFreq         = 100,
    validFreq        = int(392702/32+1),
    saveFreq         = int(392702/32+1),
    use_dropout      = True,
    verbose          = False,
    datasets         = ['../../data/word_sequence/premise_multinli_0.9_train.txt', 
                        '../../data/word_sequence/hypothesis_multinli_0.9_train.txt',
                        '../../data/word_sequence/label_multinli_0.9_train.txt'],
    valid_datasets   = ['../../data/word_sequence/premise_multinli_0.9_dev_matched.txt', 
                        '../../data/word_sequence/hypothesis_multinli_0.9_dev_matched.txt',
                        '../../data/word_sequence/label_multinli_0.9_dev_matched.txt'],
    test_datasets    = ['../../data/word_sequence/premise_multinli_0.9_dev_mismatched.txt', 
                        '../../data/word_sequence/hypothesis_multinli_0.9_dev_mismatched.txt',
                        '../../data/word_sequence/label_multinli_0.9_dev_mismatched.txt'],
    dictionary       = '../../data/word_sequence/vocab_cased.pkl',
    embedding        = '../../data/glove/glove.840B.300d.txt',
    )

