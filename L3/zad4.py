import time
import scipy.misc

from L3.k_means import KMeans


def find_best_colours(photo, target_colour_number):
    t0 = time.time()
    try:
        new_colours = KMeans(k=target_colour_number)
        new_colours.fit(photo.astype("float32"))
    except Exception:
        pass
    t1 = time.time()
    print("iterations: ", new_colours.iterations)
    print("time: ", t1 - t0)
    result_photo = new_colours.group_centers[new_colours.groups]
    return result_photo.astype("int64")


def reduce_photo_colours(photo_path, target_colour_number):
    photo = scipy.misc.imread(photo_path)
    width, height, colour_depth = photo.shape
    photo = photo.reshape([width * height, colour_depth])
    photo = find_best_colours(photo, target_colour_number)
    photo = photo.reshape([width, height, colour_depth])
    photo_path = photo_path.split(".")
    new_path = f"{'.'.join(photo_path[:-1])}_{target_colour_number}.{photo_path[-1]}"
    scipy.misc.imsave(new_path, photo)
    return photo_path, new_path


if __name__ == "__main__":
    colour_number = 20
    original_photo_path = "L3/photos/21074059_1535445439811862_946789196_n.jpg"
    reduce_photo_colours(original_photo_path, colour_number)