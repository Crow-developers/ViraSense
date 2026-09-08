import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0



def preprocess_image_input(input_shape=(224, 224, 3)):
    """دالة مسؤولة عن تهيئة وتعديل أبعاد وإضاءة الصورة قبل دخولها للشبكة."""
    inputs = layers.Input(shape=input_shape, name="raw_thumbnail_input")


    x = layers.Rescaling(1.0 / 255.0)(inputs)

    return models.Model(inputs=inputs, outputs=x, name="image_preprocessing")

def build_cnn_image_branch(input_shape=(224, 224, 3), embedding_dim=128):
    """دالة تحمّل EfficientNetB0 وتستخرج الـ Embeddings (شغل زميلتكِ)."""


    prep_model = preprocess_image_input(input_shape)


    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = False  # تجميد الأوزان الأساسية

    # 3. بناء شبكة التمرير
    image_input = layers.Input(shape=input_shape, name="thumbnail_input")
    preprocessed_image = prep_model(image_input)

    x = base_model(preprocessed_image, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(
        embedding_dim, activation="relu", name="image_embeddings"
    )(x)

    model = models.Model(
        inputs=image_input, outputs=output, name="cnn_image_branch"
    )
    return model



if __name__ == "__main__":
    cnn_branch = build_cnn_image_branch()
    cnn_branch.summary()