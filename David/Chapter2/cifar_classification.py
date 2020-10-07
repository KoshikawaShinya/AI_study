from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam


def build_model(image_shape, num_classes):

    input_layer = Input(shape=image_shape)

    x = Flatten()(input_layer)

    x = Dense(200, activation='relu')(x)
    x = Dense(150, activation='relu')(x)

    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(input_layer, output_layer)

    return model



num_classes = 10
image_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = build_model(image_shape, num_classes)

optimizer = Adam()
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)