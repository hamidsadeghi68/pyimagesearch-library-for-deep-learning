from imutils import paths
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor

imagePaths = list(paths.list_images("enter the dataset path here"))

sp = SimplePreprocessor(32, 32)

iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=100)

print(labels.shape)
print(data.shape)

