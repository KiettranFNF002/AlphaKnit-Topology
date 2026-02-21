"""
Shared configuration for AlphaKnit AI model.
"""

# ------------------------------------------------------------------ #
#  Vocabulary                                                          #
# ------------------------------------------------------------------ #

VOCAB = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
    "mr_6":  4,
    "sc":    5,
    "inc":   6,
    "dec":   7,
}

VOCAB_SIZE = len(VOCAB)
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}

PAD_ID = VOCAB["<PAD>"]
SOS_ID = VOCAB["<SOS>"]
EOS_ID = VOCAB["<EOS>"]
UNK_ID = VOCAB["<UNK>"]

# ------------------------------------------------------------------ #
#  Model hyperparameters                                               #
# ------------------------------------------------------------------ #

D_MODEL    = 128    # embedding / latent dimension
N_HEADS    = 4      # transformer attention heads
N_LAYERS   = 3      # transformer decoder layers
FFN_DIM    = 256    # feed-forward inner dimension
DROPOUT    = 0.1

# ------------------------------------------------------------------ #
#  Data                                                                #
# ------------------------------------------------------------------ #

N_POINTS     = 256   # fixed point cloud size (pad/sample to this)
MAX_SEQ_LEN  = 300   # max token sequence length (pad/truncate)

# ------------------------------------------------------------------ #
#  Training                                                            #
# ------------------------------------------------------------------ #

BATCH_SIZE = 32
LR         = 1e-3
EPOCHS     = 20
GRAD_CLIP  = 1.0
CHECKPOINT_DIR = "checkpoints"
