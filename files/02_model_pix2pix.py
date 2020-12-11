import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.layers import Concatenate, Dropout, BatchNormalization

import matplotlib.pyplot as plt



#1. 데이터 
def load_data(filename):

	data = np.load(filename)

	# unpack
	sketch, photo = data['arr_0'], data['arr_1']

	# scale: [0,255] -> [-1,1] 
	sketch = (sketch - 127.5) / 127.5
	photo = (photo - 127.5) / 127.5

	return [sketch, photo]


#2. 모델

# 판별자 (Discriminator)
def discriminator(image_shape):

	init = RandomNormal(stddev=0.02) #가중치 초기화

	src = Input(shape=image_shape)
	target = Input(shape=image_shape)
	merged = Concatenate()([src, target]) #input merge 

	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)

	# output 
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	out = Activation('sigmoid')(d) #fake or real (0 / 1)

	# 모델 정의
	model = Model([src, target], out)

	# optimizer
	opt = Adam(lr=0.0002, beta_1=0.5)

    # 컴파일
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])

	return model




# Encoder (downsampling)
def encoder_layer(layer_in, n_filters, batchnorm=True):

	# 가중치 초기화
	init = RandomNormal(stddev=0.02)

	# layer 추가 
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', 
			   kernel_initializer=init)(layer_in)

	# BatchNormalization이 True일 경우에만 추가 
	if batchnorm:
		g = BatchNormalization()(g, training=True)

	g = LeakyReLU(alpha=0.2)(g)

	return g


# Decoder (upsampling)
def decoder_layer(layer_in, skip_in, n_filters, dropout=True):

	# 가중치 초기화 
	init = RandomNormal(stddev=0.02)

	# layer 추가 
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', 
						kernel_initializer=init)(layer_in)
	g = BatchNormalization()(g, training=True)

	# Dropout이 True일 경우에만 추가 
	if dropout:
		g = Dropout(0.5)(g, training=True)

	# Encoder layer에서 Activation 거치기 전의 output 복사 & merge
	g = Concatenate()([g, skip_in])

	g = Activation('relu')(g)

	return g


# 생성자 (Generator)
def generator(image_shape=(256,256,3)):

	#가중치 초기화
	init = RandomNormal(stddev=0.02)

	# image input
	image_in = Input(shape=image_shape)

	# Encoder
	e1 = encoder_layer(image_in, 64, batchnorm=False)
	e2 = encoder_layer(e1, 128)
	e3 = encoder_layer(e2, 256)
	e4 = encoder_layer(e3, 512)
	e5 = encoder_layer(e4, 512)
	e6 = encoder_layer(e5, 512)
	e7 = encoder_layer(e6, 512)

    # 병목 현상 방지
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)

	# Decoder
	d1 = decoder_layer(b, e7, 512)
	d2 = decoder_layer(d1, e6, 512)
	d3 = decoder_layer(d2, e5, 512)
	d4 = decoder_layer(d3, e4, 512, dropout=False)
	d5 = decoder_layer(d4, e3, 256, dropout=False)
	d6 = decoder_layer(d5, e2, 128, dropout=False)
	d7 = decoder_layer(d6, e1, 64, dropout=False)

	# output 
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	image_out = Activation('tanh')(g) 

	# 모델 정의
	model = Model(image_in, image_out)
	return model
 


# GAN
def gan(g_model, d_model, image_shape):

    # 판별자 모델 훈련 동결
	d_model.trainable = False

    # input: photo
	photo_real = Input(shape=image_shape)

	# output_1: generator가 생성한 이미지 
	photo_generated = g_model(photo_real)

	# output_2: discriminator가 판별한 결과 
	r_or_f = d_model([photo_generated, photo_real])

	model = Model(photo_real, [r_or_f, photo_generated])

    # 모델 컴파일 
	opt = Adam(lr=0.0002, beta_1=0.5)

    #d_loss 1: binary_crossentropy / d_loss2: mae
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])

	return model
 

# data를 n_batch만큼 가져옴 
def generate_real(dataset, n_batch, patch_shape):

	sketch, photo = dataset

	# choose random instances
	ix = np.randint(0, sketch.shape[0], n_batch)

	# retrieve selected images
	sketch, photo = sketch[ix], photo[ix]

	# generate 'real' class labels (1)
	y = np.ones((n_batch, patch_shape, patch_shape, 1))

	return [sketch, photo], y


# fake image 생성
def generate_fake(g_model, n_batch, patch_shape):

	# generate fake instance
	fake = g_model.predict(n_batch)

	# create 'fake' class labels (0)
	y = np.zeros((len(fake), patch_shape, patch_shape, 1))

	return fake, y



# 중간 과정 plot / 모델 저장
def check_progress(step, g_model, dataset, n_samples=3):

	# photo
	[X_realA, X_realB], _ = generate_real(dataset, n_samples, 1)

	# 생성된 photo
	X_fakeB, _ = generate_fake(g_model, X_realA, 1)

	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0

	# plot real source images
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(X_realA[i])

	# plot generated target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(X_fakeB[i])

	# plot real target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples*2 + i)
		plt.axis('off')
		plt.imshow(X_realB[i])

	# plot 저장
	filename1 = 'plot_%06d.png' % (step+1)
	plt.savefig(filename1)
	plt.close()

	# 모델 저장
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))


# 3. 훈련 
def train(d_model, g_model, gan_model, dataset, n_epochs=150, n_batch=16):

	#discriminator의 output shape
	n_patch = d_model.output_shape[1]

	sketch, photo = dataset

	# 전체 데이터 개수를 n_batch만큼씩 줄 거니까 총 몇 번에 걸쳐서 줘야 하는가? (한 번의 epoch 동안)
	num_batch = int(len(sketch) / n_batch)

    # 전체 epoch 동안의 train data 분할 횟수(batch 횟수)
	iterations = num_batch * n_epochs
    
	# manually enumerate epochs
	for i in range(iterations):

		# batch 한 번만큼의 sketch, photo
		[X_realA, X_realB], y_real = generate_real(dataset, n_batch, n_patch)

		# batch 한 번만큼의 generated photo
		X_fakeB, y_fake = generate_fake(g_model, X_realA, n_patch)

		# discriminator: 진짜 사진을 '진짜'라고 판별
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)

		# discriminator: 가짜 사진을 '가짜'라고 판별
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

		# GAN 
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

        # 진행 상황 확인 
		if (i+1) % (num_batch * 10) == 0:
			check_progress(i, g_model, dataset)


# 호출
#1. 데이터
dataset = load_data('flower_berry.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)

# sketch와 photo의 shape
image_shape = dataset[0].shape[1:]

#2. 모델
d_model = discriminator(image_shape)
g_model = generator(image_shape)
gan_model = gan(g_model, d_model, image_shape)

#3. 컴파일 및 훈련 
train(d_model, g_model, gan_model, dataset)
