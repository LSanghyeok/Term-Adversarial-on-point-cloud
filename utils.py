import matplotlib.pyplot as plt

idx2label = {0:"bathhub",
             1:"bed",
             2:"chair",
             3:"desk",
             4:"dresser",
             5:"monitor",
             6:"night_stand",
             7:"sofa",
             8:"table",
             9:"toilet"}

def visualize(pos,label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color='red', s=5)
    ax.text2D(0.87, 0.92, 'Label: {}'.format(
        idx2label[label]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    plt.draw()
    plt.pause(0.001)
    
