import torch

WORKING_FOLDER_PATH = "/content/drive/MyDrive/Ade-Colab-Notebooks/AI-Thursdays/Language-Models/CNN-Model-Solutions"

class Config:
    line_divider = "=" * 50
    next_line = "\n"
    random_state = 100
    test_fraction_split = 0.2
    vocab_size = 20000
    maxlen = 500
    batch_size = 64
    embedding_dim = 100
    num_filters = 100
    kernel_sizes = [3, 4, 5]  # Kim CNN
    hidden_dim = 128
    dropout = 0.5
    epochs = 20
    lr = 2e-3
    lr_schedule_mode = 'min'
    lr_schedule_factor = 0.5
    lr_schedule_is_verbose = True
    patience = 3  # early stopping patience
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_fraction_split = 0.8
    all_data_path = f"{WORKING_FOLDER_PATH}/data/IMDB Dataset.csv"
    train_valid_dataset_path = f"{WORKING_FOLDER_PATH}/data/imdb_reviews_dataset_train.csv"
    test_dataset_path = f"{WORKING_FOLDER_PATH}/data/imdb_reviews_dataset_test.csv"
    model_simple_cnn_path = f"{WORKING_FOLDER_PATH}/models/model_simple_cnn.pth"
    model_advanced_cnn_path = f"{WORKING_FOLDER_PATH}/models/model_advanced_cnn.pth"
    is_use_pretrained_model = True

config = Config()