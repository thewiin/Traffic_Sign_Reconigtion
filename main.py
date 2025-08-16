import deeplake
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, SeparableConv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D,
    Reshape, Multiply
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

IMG_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 25

ds_train_val = deeplake.load("hub://activeloop/gtsrb-train")
ds_test = deeplake.load("hub://activeloop/gtsrb-test")

def process_dataset_to_numpy(dataset):
    x_data = []
    y_data = []
    for i in tqdm(range(len(dataset)), desc=f"Processing {dataset.path.split('/')[-1]}"):
        image = dataset.images[i].numpy()
        label = dataset.labels[i].numpy(aslist=True)[0]
        resized_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        x_data.append(resized_image)
        y_data.append(label)
    return np.array(x_data), np.array(y_data)

print("Đang xử lý và thay đổi kích thước ảnh...")
x_train_val, y_train_val_raw = process_dataset_to_numpy(ds_train_val)
x_test, y_test_raw = process_dataset_to_numpy(ds_test)

x_train_val = x_train_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
num_classes = len(np.unique(y_train_val_raw))
y_train_val = to_categorical(y_train_val_raw, num_classes)
y_test = to_categorical(y_test_raw, num_classes)

x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val_raw
)

print("\n Bắt đầu Phân tích dữ liệu khám phá (EDA)...")

num_classes_check = len(np.unique(y_train_val_raw))
plt.figure(figsize=(18, 7))
plt.hist(y_train_val_raw, bins=num_classes_check, rwidth=0.8, color='skyblue')
plt.title('Phân phối số lượng ảnh trên mỗi lớp')
plt.xlabel('ID Lớp (Loại biển báo)')
plt.ylabel('Số lượng ảnh')
plt.grid(axis='y', alpha=0.75)
plt.show()

print("Hiển thị 25 ảnh mẫu ngẫu nhiên từ bộ dữ liệu...")
plt.figure(figsize=(12, 12))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    random_index = np.random.randint(0, len(x_train_val))
    plt.imshow(x_train_val[random_index])
    plt.title(f'Loại: {y_train_val_raw[random_index]}')
    plt.axis('off')

plt.suptitle('Các ảnh mẫu và nhãn tương ứng', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("EDA hoàn tất!")

# Tăng cường dữ liệu
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)
datagen.fit(x_train)

# Mô hình CNN
def se_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    return Multiply()([input_tensor, se])

def build_simple_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = SeparableConv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = se_block(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_simple_model((IMG_SIZE, IMG_SIZE, 3), num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Huấn luyện mô hình
checkpoint = ModelCheckpoint('traffic_sign_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

print("\nBắt đầu quá trình huấn luyện...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint, early_stopping]
)

# Tải trọng số tốt nhất đã lưu
model.load_weights('traffic_sign_model.h5')

print("\n Đánh giá cuối cùng trên TẬP TEST CHÍNH THỨC:")
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Loss: {loss:.4f}")

# Ma trận nhầm lẫn
print("\n Đang tạo Ma trận nhầm lẫn...")
y_true = y_test_raw
y_prob = model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)
y_pred = y_prob.argmax(axis=1)
labels = np.arange(num_classes)
cm = confusion_matrix(y_true, y_pred, labels=labels)

NORMALIZE = True
if NORMALIZE:
    with np.errstate(invalid='ignore', divide='ignore'):
        cm_show = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_show = np.nan_to_num(cm_show)
    title_suffix = " — Normalized"
else:
    cm_show = cm
    title_suffix = " — Counts"

fig = plt.figure(figsize=(15, 15))
plt.imshow(cm_show, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix ({len(labels)} classes{title_suffix})')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(labels, labels, rotation=90)
plt.yticks(labels, labels)
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Biểu đồ Lịch sử Huấn luyện
def plot_history(history):
    best_epoch_idx = np.argmin(history.history['val_loss'])
    epochs_to_plot = best_epoch_idx + 1
    
    print(f"Quá trình huấn luyện dừng ở epoch: {len(history.history['val_loss'])}")
    print(f"Epoch tốt nhất (val_loss thấp nhất) là epoch: {epochs_to_plot}")
    
    acc = history.history['accuracy'][:epochs_to_plot]
    val_acc = history.history['val_accuracy'][:epochs_to_plot]
    loss = history.history['loss'][:epochs_to_plot]
    val_loss = history.history['val_loss'][:epochs_to_plot]
    
    epochs_range = range(1, epochs_to_plot + 1)

    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(epochs_to_plot, color='r', linestyle='--', label=f'Best Epoch: {epochs_to_plot}')
    plt.title('Model Accuracy (up to Best Epoch)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(epochs_to_plot, color='r', linestyle='--', label=f'Best Epoch: {epochs_to_plot}')
    plt.title('Model Loss (up to Best Epoch)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)