import os
import shutil

from DataLoading import DataGenerator
from SpineSeparation import SpineSeparation
from predictText import predict_books_names
from textModel import train_text_model, plot_model_results, load_trained_model


def split_data(data_folder):
    final_paths = []
    for path in os.listdir(data_folder):
        final_paths.append(path)

    train_final_paths = final_paths[: int(len(final_paths) * 0.90)]

    val_final_paths = final_paths[int(len(final_paths) * 0.90):]

    for path in train_final_paths:
        img_path = data_folder + "/" + path
        new_path = data_folder + "/train/"
        shutil.copy(img_path, new_path)

    print("train data done")

    for path in val_final_paths:
        img_path = data_folder + "/" + path
        new_path = data_folder + "/test/"
        shutil.copy(img_path, new_path)

    print("test data done")

if __name__ == "__main__":
    print("---- Program started ----")

    input_shape = (128, 32)
    batch_size = 128
    epochs = 20
    train_db_path = "images/text_data/train"
    test_db_path = "images/text_data/test"

    save_model_file_path = "saved_models/textRG_best_model_5.hdf5"

    history, pred_model, characters = train_text_model(input_shape, batch_size, epochs,train_db_path,test_db_path,
                                                       save_model_file_path)
    plot_model_results(history)

    load_trained_model(input_shape, train_db_path, test_db_path,save_model_file_path)

    # this is the data generator for the books spines images
    books_train_generator = DataGenerator('images/test', (400, 400), 15, shuffle=True)

    batch_x = books_train_generator[0]

    sp = SpineSeparation(batch_x)
    spines_paths = sp.spineSeparation()

    predict_books_names(pred_model, characters, spines_paths)

    print("---- Program ended ----")
