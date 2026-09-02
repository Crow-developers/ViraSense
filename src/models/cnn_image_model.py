import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def build_cnn_image_branch(input_shape=(224, 224, 3), embedding_dim=128):
    """فرع الصور الخاص بمشروع ViraSense باستخراج ميزات الـ Thumbnail."""

    image_input = layers.Input(shape=input_shape, name="thumbnail_input")

    # 2. تحميل موديل EfficientNetB0 بأوزان imagenet
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # 3. تجميد أوزان الموديل الأساسي
    base_model.trainable = False

    # 4. تمرير الصورة خلال الشبكة
    x = base_model(image_input, training=False)

    # 5. ضغط ميزات الصورة إلى Vector
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

    # 6. إضافة Dropout لمنع ה-Overfitting
    x = layers.Dropout(0.2)(x)

    # 7. طبقة كثيفة لتوحيد الأبعاد للدمج اللاحق
    output = layers.Dense(
        embedding_dim, activation="relu", name="image_embeddings"
    )(x)

    # 8. تكوين النموذج الفرعي
    model = models.Model(
        inputs=image_input, outputs=output, name="cnn_image_branch"
    )

    return model


if __name__ == "__main__":
    cnn_branch = build_cnn_image_branch()
    cnn_branch.summary()
    print("\n✅ The branch of cnn_image_model was built successfully")


