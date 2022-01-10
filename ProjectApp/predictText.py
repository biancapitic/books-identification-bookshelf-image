import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def pre_process_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ### actually returns h, w
    h, w = img.shape

    ### if height less than 32
    if h < 32:
        add_zeros = np.ones((32 - h, w)) * 255
        img = np.concatenate((img, add_zeros))
        h = 32

    ## if width less than 128
    if w < 128:
        add_zeros = np.ones((h, 128 - w)) * 255
        img = np.concatenate((img, add_zeros), axis=1)
        w = 128

    ### if width is greater than 128 or height greater than 32
    if w > 128 or h > 32:
        img = cv2.resize(img, (128, 32))

    img = np.expand_dims(img, axis=2)

    # Normalize each image
    img = img / 255.

    return img

def predict_output(model, characters,img):
    # predict outputs on validation images
    prediction = model.predict(np.array([img]))
    ## shape (batch_size, num_timesteps, vocab_size)

    # using CTC decoder
    out = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(prediction,
                                   input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                                   greedy=True)[0][0])
    text = []
    ## get the final text
    for x in out:
        for p in x:
            if int(p) != -1:
                text.append(characters[int(p)])

    final_text = ""
    for c in text:
        final_text += c
    return str(final_text)

def plot_imgs(all_imgs):
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 2
    columns = len(all_imgs/2)
    ind = 1
    for img_label in all_imgs.keys():

        fig.add_subplot(rows, columns, ind)

    # showing image
        plt.imshow(all_imgs[img_label], cmap='gray')
        plt.axis('off')
        plt.title(img)
        ind += 1
    plt.show()

def test_text_production(model, characters, img_paths):
    all_text_img = dict()
    for i in range(len(img_paths)):
        test_img = pre_process_image(img_paths[i])
        text = predict_output(model, characters, test_img)
        print(text)
        all_text_img[text] = test_img
    plot_imgs(all_text_img)

def look_for_book_in_line(words, line_words):
    res = []
    status = "not_done"
    for w in line_words:
        if words.find(str(w)) == 0:
            res.append(str(w))
            words = words[len(w):]
        else:
            return res, status
    status = "done"
    return res, status

def split_text_in_known_book_words(text):
    words = []
    file = open("books_titles/book_words.txt")
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        line_words = line.split(',')
        res, status = look_for_book_in_line(text, line_words)
        if len(res) > len(words):
            words = res
    return words


def predict_books_names(model, characters, img_paths):
    spines_text = []
    for i in range(len(img_paths)):
        test_img = pre_process_image(img_paths[i])
        text = predict_output(model, characters, test_img)
        spines_text.append(text)

    books = []
    unregognised = []
    for text in spines_text:
        book_title = split_text_in_known_book_words(text)
        if book_title == []:
            unregognised.append(text)
        books.append(book_title)
    print("These are the books: ")
    print(books)
    print(str(len(unregognised)) + " unrecognised books...")
    # print(unregognised)