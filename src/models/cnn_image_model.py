import tensorflow as tf
from tensorflow.keras import layers, models



def preprocess_image_input(input_shape=(224, 224, 3)):
    """دالة مسؤولة عن تهيئة وتعديل أبعاد وإضاءة الصورة قبل دخولها للشبكة."""
    inputs = layers.Input(shape=input_shape, name="raw_thumbnail_input")

    # معالجة وتعديل ألوان الصورة (Rescaling)
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    return models.Model(inputs=inputs, outputs=x, name="image_preprocessing")









