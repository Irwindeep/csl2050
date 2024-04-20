import numpy as np
from PIL import Image
from os import listdir
from os.path import join, isdir

# Function to load images from file-paths
def load_imgs(file_paths, slice_, color, resize):
    default_slice = (slice(0, 250), slice(0, 250))  # Setting the default slice to the size of original dataset i.e., 250x250

    if slice_ is None: slice_ = default_slice
    else: slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

    # Obtain the height and width of the image from slice
    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    # Resizing the image
    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)
        
    # Setting the dimensions for each image
    n_faces = len(file_paths)
    if not color: faces = np.zeros((n_faces, h, w), dtype=np.float32)
    else: faces = np.zeros((n_faces, h, w, 3), dtype=np.float32)

    # Loading images
    for i, file_path in enumerate(file_paths):
        pil_img = Image.open(file_path)
        pil_img = pil_img.crop((w_slice.start, h_slice.start, w_slice.stop, h_slice.stop))

        if resize is not None: pil_img = pil_img.resize((w, h))
        face = np.asarray(pil_img, dtype=np.float32)

        face /= 255.0
        if not color: face = face.mean(axis=2)
        faces[i, ...] = face

    return faces

# Function to fetch and load images from a certain directory
def fetch_lfw_deep_people(data_folder_path, slice_=None, color=False, resize=None, min_faces_per_person=0):
    person_names, file_paths = [], []

    # Fetching the names of the people and file paths
    for person_name in sorted(listdir(data_folder_path)):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path): continue

        paths = [join(folder_path, f) for f in sorted(listdir(folder_path))]
        n_pictures = len(paths)
        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace("_", " ")
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)

    n_faces = len(file_paths)
    if n_faces == 0: raise ValueError("min_faces_per_person=%d is too restrictive" % min_faces_per_person)

    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)
    file_paths = np.array(file_paths)

    # Loading the images of the found file-paths
    faces = load_imgs(file_paths, slice_, color, resize)

    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces, target, paths = faces[indices], target[indices], file_paths[indices]
    return faces, target, target_names, paths