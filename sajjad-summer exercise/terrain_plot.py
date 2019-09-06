from imageio import imread
import matplotlib.pyplot as plt

def terrain_plot(terrain):

    plt.figure()
    plt.title('Terrain')
    plt.imshow(terrain, cmap = 'gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ =="__main__":
    terrain1 = imread('SRTM_data_Norway_1.tif')
    terrain_plot(terrain1)
    