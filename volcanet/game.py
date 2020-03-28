import os
import pickle
import pathlib
import numpy as np
import tkinter as tk
import tensorflow as tf
from PIL import Image, ImageTk
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sys import argv
from sklearn.metrics import f1_score

from volcanet import predict_ds
import data

GAME_DATA_PATH = os.path.join(os.path.dirname(__file__), 'game_data')


def store_predictions():
    # Load data
    ds = data.load_ds('test', 'class').shuffle(buffer_size=2000, seed=42).repeat()
    # Balance data
    class_datasets = [ds.filter(lambda x, y: y == c) for c in range(5)]
    ds = tf.data.experimental.sample_from_datasets(class_datasets, weights=[0.5 for _ in range(5)])

    ds = ds.take(15)

    img_ds = ds.map(lambda x, y: x)
    images = list(img_ds.as_numpy_iterator())
    label_ds = ds.map(lambda x, y: y)
    labels = list(label_ds.as_numpy_iterator())
    preds = predict_ds(img_ds)

    with open(GAME_DATA_PATH, 'wb') as fp:
        pickle.dump((images, labels, preds), fp)
    return images, labels, preds


def load_predictions():
    with open(GAME_DATA_PATH, 'rb') as fp:
        images, labels, preds = pickle.load(fp)
    return images, labels, preds


def human_training():
    top = tk.Tk()
    top.geometry("400x400")

    # loading_window = tk.Toplevel(top)

    load_var_setter = lambda x: "Loading: {}%".format(int(x/70.0))  # loaded x out of 6000 samples to load
    loading_var = tk.StringVar(top)
    loading_var.set(load_var_setter(0))
    loading_label = tk.Label(top, textvariable=loading_var)
    loading_label.pack()

    gameplay_label = tk.Text(top, wrap=tk.WORD)
    gameplay_label.insert(tk.INSERT, "Learn the volcano types then try to beat the AI.\n\n"
                                     "  Type 0: No volcano\n"
                                     "  Type 1: Definitely a volcano\n"
                                     "  Type 2: Probably\n"
                                     "  Type 3: Possibly\n"
                                     "  Type 4: Only a pit is visible")
    gameplay_label.pack()

    top.update_idletasks()
    top.update()

    # loading_window.update_idletasks()
    # loading_window.update()

    # Helper function to process images for tkinter
    def tk_image(image):
        image = np.uint8(np.reshape(image, image.shape[:2]) * 255)
        return ImageTk.PhotoImage(Image.fromarray(image))

    # Load training data
    examples_ds = data.load_ds('train', 'class').shuffle(buffer_size=2000, seed=42)
    # Balance data
    class_datasets = [examples_ds.filter(lambda x, y: y == c) for c in range(5)]
    examples_ds = tf.data.experimental.sample_from_datasets(class_datasets, weights=[0.5 for _ in range(5)])

    examples = []
    for id, (x, y) in enumerate(examples_ds.as_numpy_iterator()):
        loading_var.set(load_var_setter(id))
        top.update_idletasks()
        top.update()
        examples.append((tk_image(x), str(int(y))))
    examples_iter = iter(examples)

    loading_label.pack_forget()
    gameplay_label.pack_forget()

    def update_tkinter():
        try:
            image, label = next(examples_iter)
        except StopIteration:
            top.destroy()
            return
        gt_var.set("Type: " + label)
        image_label.configure(image=image)

    init_image, init_label = next(examples_iter)

    gt_var = tk.StringVar(top)
    gt_var.set("Type: " + init_label)

    image_label = tk.Label(top, image=init_image)

    gt_label = tk.Label(top, textvariable=gt_var)

    info_label = tk.Text(top, wrap=tk.WORD)
    info_label.insert(tk.INSERT, "Try to learn which images belong to which Type.\n\n"
                                 "  Type 0: No volcano\n"
                                 "  Type 1: Definitely a volcano\n"
                                 "  Type 2: Probably\n"
                                 "  Type 3: Possibly\n"
                                 "  Type 4: Only a pit is visible\n\n"
                                 "Close this window when you've finished learning.\n"
                                 "Then in the next stage you'll battle the AI.")

    button = tk.Button(top, text="Next", command=update_tkinter)
    button.pack()
    gt_label.pack()
    image_label.pack()
    info_label.pack()

    tk.mainloop()


def human_testing(images, labels, preds):
    top = tk.Tk()
    top.geometry("400x400")

    # Helper function to process images for tkinter
    def tk_image(image):
        image = np.uint8(np.reshape(image, image.shape[:2]) * 255)
        return ImageTk.PhotoImage(Image.fromarray(image))

    count = 15

    # Load training data
    # examples_ds = data.load_ds('test', 'class').shuffle(buffer_size=2000, seed=42).take(count)
    # examples = list((tk_image(x), str(int(y))) for x, y in examples_ds.as_numpy_iterator())
    labels = [str(int(label)) for label in labels]
    images = [tk_image(image) for image in images]
    preds = [str(int(pred)) for pred in preds]
    round_iter = enumerate(zip(images, labels, preds))
    # state = {'idx': 0}
    your_preds = []
    # your_scores
    # machine_scores = []

    def compute_score_results():
        your_f1 = f1_score(labels, your_preds, labels=[0, 1, 2, 3, 4], average='macro')
        machine_f1 = f1_score(labels, preds, labels=[0, 1, 2, 3, 4], average='macro')
        return int(your_f1 * 1000), int(machine_f1 * 1000)

    def finish_game():
        # Hide game elements
        count_label.pack_forget()
        button.pack_forget()
        entry.pack_forget()
        img_label.pack_forget()
        # Show final results
        your_score, machine_score = compute_score_results()
        score_message = "You beat the AI!" if your_score > machine_score else "The machines have won."
        final_results = tk.Text(top, wrap=tk.WORD)
        final_results.insert(tk.INSERT, "Your score: {}\n"
                                        "Machine score: {}\n\n"
                                        "{}".format(str(your_score), str(machine_score), score_message))
        final_results.pack()

    def submit_answer():
        input = entry.get()
        your_preds.append(input)
        gt = gt_var.get()
        pred = pred_var.get()
        your_score = input == gt
        machine_score = pred == gt
        img_label.pack_forget()
        entry.pack_forget()

        if your_score and machine_score:
            res = "It's a deuce!"
        elif your_score and not machine_score:
            res = "You win the round!"
        elif not your_score and machine_score:
            res = "Machine takes the round!"
        else:
            res = "Machine takes the round!"

        res_var.set(res)
        res_label.pack()
        button.configure(command=next_round)

    def next_round():
        try:
            idx, (image, label, pred) = next(round_iter)
        except StopIteration:
            finish_game()
            return
        # Hide unwanted elements
        res_label.pack_forget()
        # Configure new elements
        img_label.configure(image=image)
        gt_var.set(label)
        pred_var.set(pred)
        count_var.set(count_str(idx))
        button.configure(command=submit_answer)
        # Show new elements
        entry.pack()
        img_label.pack()

    init_idx, (init_image, init_label, init_pred) = next(round_iter)
    count_str = lambda i: "{}/{}".format(i, count)

    gt_var = tk.StringVar(top)
    gt_var.set(init_label)

    pred_var = tk.StringVar(top)
    pred_var.set(init_pred)

    count_var = tk.StringVar(top)
    count_var.set(count_str(init_idx))

    res_var = tk.StringVar(top)
    res_var.set("")

    count_label = tk.Label(top, textvariable=count_var)
    button = tk.Button(top, text="Next", command=submit_answer)
    img_label = tk.Label(top, image=init_image)
    res_label = tk.Label(top, textvariable=res_var)
    entry = tk.Entry(top)

    count_label.pack()
    button.pack()
    entry.pack()
    img_label.pack()

    tk.mainloop()


def main():
    if not pathlib.Path(GAME_DATA_PATH).is_file():
        images, labels, preds = store_predictions()
    else:
        images, labels, preds = load_predictions()

    human_training()
    print("Finished training")
    human_testing(images, labels, preds)


if __name__ == '__main__':

    main()
