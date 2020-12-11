# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
 
# load an image
def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels
 
# load source image
import glob
a = glob.glob("bear_1.png")

sketch_idx = []

for i in range(len(a)):
    c = a[i].split("\\")
    c = c[0]+"/"+c[1]
    sketch_idx.append(c)

src_image=[]
for i in range(len(sketch_idx)):
    src_image.append(load_image(sketch_idx[i]))
# print(src_image)

# load model
model = load_model('model_087200.h5')

gen_image = []
for i in range(len(sketch_idx)):
    gen = model.predict(src_image[i])
    gen = (gen + 1) / 2.0 # scale from [-1,1] to [0,1]
    gen_image.append(gen)

n_samples = len(sketch_idx)
for i in range(n_samples):
    # pyplot.figure(figsize=(5,5))
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(src_image[i][0])
# plot generated target image
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(gen_image[i][0])

pyplot.show()
''''
a = glob.glob("./test_no_back/strawberry/*")
sketch_idx = []

for i in range(len(a)):
    c = a[i].split("\\")
    c = c[0]+"/"+c[1]
    sketch_idx.append(c)

src_image=[]
for i in range(len(sketch_idx)):
    src_image.append(load_image(sketch_idx[i]))
# print(src_image)

# load model
model = load_model('model_028500.h5')

gen_image = []
for i in range(len(sketch_idx)):
    gen = model.predict(src_image[i])
    gen = (gen + 1) / 2.0 # scale from [-1,1] to [0,1]
    gen_image.append(gen)

n_samples = len(sketch_idx)
for i in range(n_samples):
    # pyplot.figure(figsize=(5,5))
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(src_image[i][0])
# plot generated target image
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(gen_image[i][0])

pyplot.show()
'''