import matplotlib.pyplot as plt
from DataLoading import DataGenerator
from matplotlib import cm
from SpineSeparation import SpineSeparation

if __name__ == '__main__':
    train_generator = DataGenerator('images/train', (400,400), 10, shuffle=True)

    batch_x = train_generator[0]

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=[16, 9])

    for i in range(len(axes)):
        axes[i].imshow(batch_x[i], cmap=cm.gray)
    plt.show()