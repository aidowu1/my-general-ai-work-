import torch

WORKING_FOLDER_PATH = "/content/drive/MyDrive/Ade-Colab-Notebooks/AI-Thursdays/Language-Models/CNN-Model-Solutions"

class Config:
    line_divider = "=" * 50
    next_line = "\n"
    random_state = 100
    test_fraction_split = 0.2
    vocab_size = 1000
    maxlen = 1000
    batch_size = 32
    embedding_dim = 10
    num_filters = 16
    kernel_size = 3
    hidden_dim = 250
    epochs = 5
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_fraction_split = 0.8
    all_data_path = f"{WORKING_FOLDER_PATH}/data/IMDB Dataset.csv"
    train_valid_dataset_path = f"{WORKING_FOLDER_PATH}/data/imdb_reviews_dataset_train.csv"
    test_dataset_path = f"{WORKING_FOLDER_PATH}/data/imdb_reviews_dataset_test.csv"
    model_simple_cnn_path = f"{WORKING_FOLDER_PATH}/models/model_simple_cnn.pth"
    model_advanced_cnn_path = f"{WORKING_FOLDER_PATH}/models/model_advanced_cnn.pth"
    is_use_pretrained_model = True

config = Config()