
from numpy.random import randint
import numpy as np
from keras.models import load_model
from keras.applications import VGG19, Xception
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import Model
import time
import numpy as np
from keras import Model
from keras.layers import Conv2D, PReLU, BatchNormalization, Flatten
from keras.layers import Dense, Input, add, Conv2DTranspose
from tqdm import tqdm
import numba

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def gener(input):  # generator
    layers = Conv2D(64, (9, 9), padding="same")(input)
    layers = PReLU(shared_axes=[1, 2])(layers)

    hold = layers  # hold this and pass it later with elementwise sum
    res_model = Conv2D(64, (3, 3), padding="same")(layers)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = PReLU(shared_axes=[1, 2])(res_model)
    res_model = Conv2D(64, (3, 3), padding="same")(res_model)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    layers = add([layers, res_model])

    layers = Conv2D(64, (3, 3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers, hold])

    # x8
    for i in range(3):
        up_model = Conv2D(128, (3, 3), padding="same")(layers)
        up_model = Conv2DTranspose(
            128, (3, 3), strides=(2, 2), padding='same')(up_model)
        layers = PReLU(shared_axes=[1, 2])(up_model)

    g = Conv2D(3, (9, 9), padding="same")(layers)

    return Model(inputs=input, outputs=g)


def discriminator(hr_shape):  # discriminator
    xceptionN = VGG19(weights="imagenet",
                      include_top=False, input_shape=hr_shape)
    y = Flatten()(xceptionN.output)
    y = Dense(128, activation='relu')(y)
    y = Dense(1, activation='sigmoid')(y)

    return Model(xceptionN.inputs, y)


def vgg_model(hr_shape):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
    # block 3 conv4 which will construct an image and we will compare it to the reference.. we cut at this layer and use the activations outcome as features
    # block4_conv4
    return Model(inputs=vgg.inputs, outputs=vgg.layers[15].output)


def combined_model(gen_model, disc_model, vgg, lr_ip):  # Combined model
    gen_img = gen_model(lr_ip)
    gen_features = vgg(gen_img)
    # freeze the discriminator and trick it and see if it will detect the generated image as real or fake
    disc_model.trainable = False
    output = disc_model(gen_img)

    return Model(inputs=[lr_ip], outputs=[output, gen_features])


@numba.jit(parallel=True)
def load_lr_images(main_dir, lr_list, imagesType):
    lr_images = []
    for img in lr_list:
        img_lr = cv2.imread(main_dir + "\data/" + imagesType + "/" + img)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        lr_images.append(img_lr)
        # to get some images not all of them
        # if len(lr_images) == 1000:
        #     break
    return lr_images


# %% params
lr_dims = 64
hr_dims = 512
# 1 to continue with models training .. or 0 to have a new model..
load_trained_models = 0
batch_size = 2
epochs = 10000
# %%
start = time.time()

main_dir = 'F:\scripts\Image enhancer\Back end\SRGAN'
lr_list = os.listdir(main_dir + "\data\lr_images")  # [:n]

lr_images = load_lr_images(main_dir, lr_list, "lr_images")

hr_list = os.listdir(main_dir + "\data/hr_images")  # [:n]
hr_images = load_lr_images(main_dir, hr_list, "hr_images")

lr_images = np.array(lr_images)
hr_images = np.array(hr_images)

lr_images = np.float16(lr_images)
hr_images = np.float16(hr_images)

# Scale values
lr_images = lr_images / 255.
hr_images = hr_images / 255.

# creating small part for testing and will use all images for training
lr_test = lr_images[0:50, :, :, :]
hr_test = hr_images[0:50, :, :, :]

hr_shape = (hr_images.shape[1], hr_images.shape[2], hr_images.shape[3])
lr_shape = (lr_images.shape[1], lr_images.shape[2], lr_images.shape[3])

lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)

if load_trained_models == 1:
    # new models or load the models and continue learning..
    generator = load_model(
        main_dir + '\data/Generator_model.h5', compile=False)
    discriminator = load_model(
        main_dir + '\data/Discriminator_model.h5', compile=False)
    vgg = load_model(main_dir + '\data/VGG_model.h5', compile=False)
    dat = np.loadtxt(
        main_dir + '\data/Generator_model_batch_and_epoch.txt')
    starting_batch = int(dat[0]) + 1
    starting_epoch = int(dat[1]) + 1

else:
    generator = gener(lr_ip)
    discriminator = discriminator((hr_dims, hr_dims, 3))
    vgg = vgg_model((hr_dims, hr_dims, 3))
    starting_epoch = 0
    starting_batch = 0

generator.summary()
discriminator.compile(loss="binary_crossentropy",
                      optimizer="adam", metrics=['accuracy'])
discriminator.summary()
print(vgg.summary())

vgg.trainable = False
gan_model = combined_model(generator, discriminator, vgg, lr_ip)
gan_model.compile(loss=["binary_crossentropy", "mse"],
                  loss_weights=[0.5, 1], optimizer="adam")  # the loss of recontruction with VGG is weighted 1 so it has a higher contribution
gan_model.summary()

# %%
trn_batch_lr = []
trn_batch_hr = []
for i in range(int(hr_images.shape[0] / batch_size)):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    trn_batch_hr.append(hr_images[start_idx:end_idx])
    trn_batch_lr.append(lr_images[start_idx:end_idx])
# clearing
del hr_images, lr_images

# if we completed the epoch then the batch will be at the end sample and we need to enter the next epoch with the first batch
if len(trn_batch_hr) == starting_batch:
    starting_batch = 0
    starting_epoch = starting_epoch + 1

for e in range(starting_epoch, epochs):  # epochs
    fake_lbls = np.zeros((batch_size, 1))
    real_lbls = np.ones((batch_size, 1))

    for b in tqdm(range(starting_batch, len(trn_batch_hr))):  # batches
        lr_imgs = trn_batch_lr[b]
        hr_imgs = trn_batch_hr[b]
        gen_imgs = generator.predict_on_batch(lr_imgs)

        discriminator.trainable = True
        discriminator.train_on_batch(gen_imgs, fake_lbls)
        discriminator.train_on_batch(hr_imgs, real_lbls)

        discriminator.trainable = False
        hr_imgs_features = vgg.predict(hr_imgs)
        gan_model.train_on_batch([lr_imgs], [real_lbls, hr_imgs_features])

    # saving after each epoch
    generator.save(main_dir + "\data/Generator_model.h5")
    discriminator.save(main_dir + "\data/Discriminator_model.h5")
    vgg.save(main_dir + "\data/VGG_model.h5")
    np.savetxt(
        main_dir + '\data/Generator_model_batch_and_epoch.txt', [b, e], fmt='%d')

end = time.time()
print(end - start)

# %% testing
# in case we have saved trained models we can load them here, so we could set load_trained_models = 1 and then place a breakpoint in the training loop (after: for e in range(starting_epoch, epochs) ) so we don't continue training and then test here..

[X1, X2] = [lr_test, hr_test]  # test were not included in training
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]


loaded_img = cv2.imread(main_dir + "/data/lr_images/1 (1).jpg")
loaded_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)
loaded_img = loaded_img / 255.
# to have the first dimension as 1 which was the no. samples or batches so we need it
loaded_img = loaded_img[np.newaxis, :, :, :]
gen_image = generator.predict(src_image)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Low resolution')
plt.imshow(np.float64(src_image[0, :, :, :]))
plt.subplot(232)
plt.title('G result')
plt.imshow(np.float64(gen_image[0, :, :, :]))
plt.subplot(233)
plt.title('High resolution')
plt.imshow(np.float64(tar_image[0, :, :, :]))

plt.show()
