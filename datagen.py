from PIL import Image
import numpy as np
import random
import os

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r %s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total:
        print()
        print()

class dataGenerator(object):
    def __init__(self, folders, im_size, mss=(1024 ** 3), flip=True, verbose=True):
        self.folders = folders
        self.im_size = im_size
        self.segment_length = mss // (im_size * im_size * 3)
        self.flip = flip
        self.verbose = verbose

        self.segments = []
        self.images = []
        self.update = 0

        if self.verbose:
            print("Importing images...")
            print("Maximum Segment Size: ", self.segment_length)

        try:
            os.mkdir("data")
        except FileExistsError:
            pass

        self.folder_to_npy()
        self.load_from_npy()

    def folder_to_npy(self):
        if self.verbose:
            print("Converting from images to numpy files...")

        for folder in self.folders:
            names = []

            for dirpath, _, filenames in os.walk(folder):
                for filename in [f for f in filenames if (f.endswith(".jpg") or f.endswith(".png") or f.endswith(".JPEG"))]:
                    fname = os.path.join(dirpath, filename)
                    names.append(fname)

            np.random.shuffle(names)

            if self.verbose:
                print(f"{len(names)} images in {folder}")

            kn = 0
            sn = 0
            segment = []

            for fname in names:
                if self.verbose:
                    print(f"\rProcessing {sn} // {kn}\t", end='\r')

                try:
                    temp = Image.open(fname).convert('RGB').resize((self.im_size, self.im_size), Image.BILINEAR)
                except:
                    print("Importing image failed on", fname)
                    continue

                temp = np.array(temp, dtype='uint8')
                segment.append(temp)
                kn += 1

                if kn >= self.segment_length:
                    np.save(f"data/data-{sn}.npy", np.array(segment))
                    segment = []
                    kn = 0
                    sn += 1

            if segment:
                np.save(f"data/data-{sn}.npy", np.array(segment))

    def load_from_npy(self):
        self.segments = []
        for dirpath, _, filenames in os.walk("data"):
            for filename in [f for f in filenames if f.endswith(".npy")]:
                self.segments.append(os.path.join(dirpath, filename))

        self.load_segment()

    def load_segment(self):
        if self.verbose:
            print("Loading segment")

        segment_num = random.randint(0, len(self.segments) - 1)
        self.images = np.load(self.segments[segment_num], allow_pickle=True)
        self.update = 0

    def get_batch(self, num):
        if self.update > len(self.images):
            self.load_segment()

        self.update += num
        idx = np.random.randint(0, len(self.images), num)
        out = []

        for i in idx:
            out.append(self.images[i])
            if self.flip and random.random() < 0.5:
                out[-1] = np.flip(out[-1], 1)

        return np.array(out).astype('float32') / 255.0
