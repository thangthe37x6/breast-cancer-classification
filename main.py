import tensorflow as tf
from keras.layers import Input, Dense, TimeDistributed, GlobalAveragePooling2D, Reshape
from keras.applications import EfficientNetV2B0
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale = 1/255.,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True,
    validation_split=0.7

)
# bạn cần phải đổi đường dấn dữ liệu 
data_train = train_datagen.flow_from_directory("./ungthu/data",batch_size=32,target_size =(50,50),class_mode='categorical',subset='training')
data_test = train_datagen.flow_from_directory("./ungthu/data",batch_size=32,target_size =(50,50),class_mode='categorical',subset='validation')

# x_train, x_test,y_train,y_test = train_test_split(data,test_size=0.5,random_state=42)



efficientnet_v2_b0 = EfficientNetV2B0(
    input_shape=(50, 50,3), 
    include_top=False
)
efficientnet_v2_b0.trainable = False

initializer = tf.keras.initializers.GlorotNormal()
input_shape = (50, 50,3)

inputs = Input(shape=input_shape)
tdl = efficientnet_v2_b0(inputs)
z = Dense(units=128, activation='relu', kernel_initializer=initializer)(tdl)
outputs = Dense(units=2, activation='softmax', kernel_initializer=initializer)(z)
outputs = GlobalAveragePooling2D()(outputs)

model = Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# # Huấn luyện mô hình
history = model.fit(data_train, epochs=5, verbose=1) 
# loss, accuracy = model.evaluate(test_ds)
# print('Test accuracy:', accuracy)