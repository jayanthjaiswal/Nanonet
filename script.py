import keras
import imageio
import numpy as np
from scipy.misc import imresize


def crop_and_downsample(originalX, downsample_size=32):
    """
    Starts with a 250 x 250 image.
    Crops to 128 x 128 around the center.
    Downsamples the image to (downsample_size) x (downsample_size).
    Returns an image with dimensions (channel, width, height).
    """
    current_dim = 250
    target_dim = 128
    margin = int((current_dim - target_dim) / 2)
    left_margin = margin
    right_margin = current_dim - margin

    # newim is shape (6, 128, 128)
    newim = originalX[:, left_margin:right_margin, left_margin:right_margin]

    # resized are shape (feature_width, feature_height, 3)
    feature_width = feature_height = downsample_size
    resized1 = imresize(newim[0:3, :, :], (feature_width, feature_height), interp="bicubic", mode="RGB")
    resized2 = imresize(newim[3:6, :, :], (feature_width, feature_height), interp="bicubic", mode="RGB")

    # re-packge into a new X entry
    newX = np.concatenate([resized1, resized2], axis=2)

    # the next line is EXTREMELY important.
    # if you don't normalize your data, all predictions will be 0 forever.
    newX = newX / 255.0

    return newX


def extract_features(z):
    features = np.array([z[:,:,0],z[:,:,1],z[:,:,2]])
    return features


def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


def main():
    print("Input expects images with dimension 250x250")
    path1 = raw_input("Please enter path to 1st image ")
    path2 = raw_input("Please enter path to 2nd image ")
    model = load_model()
    img1 = imageio.imread(path1)
    img2 = imageio.imread(path2)
    testX = []
    testX.append(np.concatenate((extract_features(img1), extract_features(img2))))
    testX = np.asarray(testX).astype('float64')
    test = np.asarray([crop_and_downsample(x, downsample_size=32) for x in testX])
    result = model.predict(test)
    if result[0][1]>0.5:
        print("Similar")
    else:
        print("Not similar")
    print("Similarity Score", result[0][1])


if __name__ == "__main__":
    main()
