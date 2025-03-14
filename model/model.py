import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import cv2
import random
import time
import re
from sklearn.metrics import mean_absolute_error
import pandas as pd
import json

def create_progress_callback(epochs):
    start_time = [0] 
    def on_epoch_begin(epoch, logs=None):
        print(f"Epoch #{epoch+1}/{epochs}")
        start_time[0] = time.time()
    
    def on_epoch_end(epoch, logs=None):
        time_taken = time.time() - start_time[0]
        print(f"Epoch #{epoch+1}/{epochs} completed in {time_taken:.2f}s")
        print(f"Training loss: {logs.get('loss'):.4f}; MAE: {logs.get('mae'):.2f} years")
        print(f"Validation loss: {logs.get('val_loss'):.4f}; MAE: {logs.get('val_mae'):.2f} years")
        print(f"Overall progress: {(epoch+1)/epochs*100:.1f}% complete")
    
    callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=on_epoch_begin,
        on_epoch_end=on_epoch_end
    )
    return callback

def create_save_callback(history_data, model_ref, paths, start_epoch):
    def on_epoch_end(epoch, logs=None):
        if not logs:
            logs = {}
            
        for key, value in logs.items():
            if key not in history_data:
                history_data[key] = []
            history_data[key].append(float(value))
        
        training_state = {
            'completed_epochs': start_epoch + epoch + 1,
            'history': history_data,
            'last_lr': float(tf.keras.backend.get_value(model_ref.optimizer.learning_rate))
        }
        
        with open(paths['state'], 'w') as f:
            json.dump(training_state, f)
        model_ref.save_weights(paths['latest'])
    
    callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
    return callback

def build_model(img_size): # CNN
    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'), 
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    
    return model

def parse_utkface_filename(filename):
    parts = filename.split('_')
    if len(parts) >= 3:
        try:
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])
            return age, gender, race
        except:
            return None, None, None
    return None, None, None

def preprocess_image(img_path, img_size):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    
    filename = os.path.basename(img_path)
    age, _, _ = parse_utkface_filename(filename)
    if age is None:
        return None, None
    img = img.astype(np.float32) / 255.0

    return img, age

def create_dataset(data_dir, img_size, batch_size, val_split=0.2):
    image_paths = []
    ages = []
    total_files = 0
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            total_files += 1
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                age, _, _ = parse_utkface_filename(file)
                if age is not None:
                    image_paths.append(img_path)
                    ages.append(age)
    print(f"{len(image_paths)} images found")
    
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    split_idx = int(len(indices) * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    train_images = []
    train_ages = []
    val_images = []
    val_ages = []
    
    for i, idx in enumerate(train_indices):
        img, age = preprocess_image(image_paths[idx], img_size)
        if img is not None and age is not None:
            train_images.append(img)
            train_ages.append(age)

    for i, idx in enumerate(val_indices):
        img, age = preprocess_image(image_paths[idx], img_size)
        if img is not None and age is not None:
            val_images.append(img)
            val_ages.append(age)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((
        np.array(train_images), 
        np.array(train_ages)
    ))
    train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat()

    val_dataset = tf.data.Dataset.from_tensor_slices((
        np.array(val_images), 
        np.array(val_ages)
    ))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, len(train_images), len(val_images)

def train_model(model, data_dir, img_size, batch_size, epochs=10, save_dir="./model_ckpt", resume=True):
    os.makedirs(save_dir, exist_ok=True)
    
    ckpt_path = os.path.join(save_dir, "model_checkpoint.weights.h5")
    best_ckpt_path = os.path.join(save_dir, "best_model.weights.h5")  
    latest_ckpt_path = os.path.join(save_dir, "latest_model.weights.h5")
    state_path = os.path.join(save_dir, "training_state.json")
    start_epoch = 0
    history_data = {'loss': [], 'mae': [], 'val_loss': [], 'val_mae': []}
    
    if resume: # to pick up where it left off
        if os.path.exists(latest_ckpt_path):
            try:
                print(f"Loading {latest_ckpt_path}")
                model.load_weights(latest_ckpt_path)
                
                if os.path.exists(state_path):
                    with open(state_path, 'r') as f:
                        training_state = json.load(f)
                        start_epoch = training_state.get('completed_epochs', 0)
                        history_data = training_state.get('history', history_data)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                if os.path.exists(best_ckpt_path):
                    print(f"Trying best checkpoint instead")
                    model.load_weights(best_ckpt_path)
    
    train_dataset, val_dataset, train_size, val_size = create_dataset(data_dir, img_size, batch_size)
    train_steps = train_size // batch_size
    val_steps = max(1, val_size // batch_size)
    print(f"Training: {train_size} & {train_steps} batches. Testing: {val_size} & {val_steps} batches")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_ckpt_path, 
            save_best_only=True, 
            save_weights_only=True,
            monitor='val_mae',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae', 
            patience=3, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae', 
            factor=0.5, 
            patience=2,
            verbose=1
        ),
        create_save_callback(
            history_data,
            model,
            {
                'state': state_path,
                'latest': latest_ckpt_path
            },
            start_epoch
        ),
        create_progress_callback(epochs - start_epoch)
    ]
    
    remaining = epochs - start_epoch
    history = model.fit(
        train_dataset,
        epochs=remaining,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=0
    )
    
    for key, value in history.history.items():
        if key not in history_data:
            history_data[key] = []
        history_data[key].extend(value)
    
    class HistoryWrapper:
        def __init__(self, history):
            self.history = history
    
    combined_history = HistoryWrapper(history_data)
    model.save(f"{save_dir}/age_estimation_model.keras") 
    
    print(f"Final training MAE: {history_data['mae'][-1]:.2f} years")
    print(f"Final test MAE: {history_data['val_mae'][-1]:.2f} years")
    
    print("Now computing test results")
    evaluate_model(model, val_dataset, save_dir)
    
    return combined_history

def evaluate_model(model, val_dataset, save_dir): # for seeing performance
    true_ages = []
    pred_ages = []
    
    for images, ages in val_dataset:
        batch_preds = model.predict(images)
        true_ages.extend(ages.numpy())
        pred_ages.extend(batch_preds.flatten())
    
    mae = mean_absolute_error(true_ages, pred_ages)
    
    plt.figure()
    plt.scatter(true_ages, pred_ages, alpha=0.5)
    plt.plot([0, 100], [0, 100], 'r--')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title(f'Age Prediction Performance (MAE: {mae:.2f} years)')
    plt.grid(True)
    plt.savefig(f"{save_dir}/prediction_scatter.png")
    
    age_bins = [0, 15, 30, 45, 60, 75, 100]
    age_labels = ['0-15', '16-30', '31-45', '46-60', '61-75', '76+']
    results = []
    
    for i in range(len(age_bins) - 1):
        mask = (np.array(true_ages) >= age_bins[i]) & (np.array(true_ages) < age_bins[i+1])
        if np.sum(mask) > 0:
            group_mae = mean_absolute_error(
                np.array(true_ages)[mask], 
                np.array(pred_ages)[mask]
            )
            results.append({
                'Age Group': age_labels[i],
                'Count': np.sum(mask),
                'MAE': group_mae
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{save_dir}/age_group_performance.csv", index=False)
    
    plt.figure()
    plt.bar(results_df['Age Group'], results_df['MAE'])
    plt.xlabel('Age Group')
    plt.ylabel('Mean Absolute Error (years)')
    plt.title('Age Estimation Error by Age Group')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/age_group_mae.png")
    print(results_df)

def predict_age(model, image_path, img_size):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, "Could not load image"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        pred_age = model.predict(img)[0][0]
        
        return pred_age, None
    except Exception as e:
        return None, str(e)

def extract_namus_id(filename):
    patterns = [
        r'MP(\d+)',
        r'UP(\d+)',
        r'namus(\d+)',
        r'\/(\d+)-',
        r'_(\d+)[_\.]'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    return None

def create_age_visualization(image_path, predicted_age, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Predicted Age: {predicted_age:.1f} years", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_namus_data(model, img_size, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    
    total_images = len(image_paths)
    
    viz_dir = os.path.join(output_dir, "visualization")
    os.makedirs(viz_dir, exist_ok=True)
    
    for i, img_path in enumerate(image_paths):
        progress = (i + 1) / total_images * 100
        if i % max(1, total_images // 10) == 0 or (i+1) == total_images:
            print(f"{progress:.1f}% complete; ({i+1}/{total_images})")
        
        file_name = os.path.basename(img_path)
        namus_id = extract_namus_id(file_name)
        predicted_age, error = predict_age(model, img_path, img_size)
        
        if error:
            continue
        
        results.append({
            'File': file_name,
            'Path': img_path,
            'NamUs_ID': namus_id,
            'Predicted_Age': round(float(predicted_age), 1)
        })
        
        if i % 20 == 0:
            create_age_visualization(img_path, predicted_age, os.path.join(viz_dir, f"{namus_id or 'unknown'}_{i}.jpg"))
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "age_predictions.csv"), index=False)
    summary = results_df.groupby('NamUs_ID').agg({
        'Predicted_Age': ['min', 'max', 'mean', 'count']
    })

    summary.columns = ['Min_Age', 'Max_Age', 'Mean_Age', 'Image_Count']
    summary.reset_index().to_csv(os.path.join(output_dir, "age_summary.csv"), index=False)

def main():
    # in case there is a GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    img_size = 128
    batch_size = 32
    model = build_model(img_size)
    model.compile(optimizer=optimizers.Adam(1e-4), loss='mae', metrics=['mae'])
    utk_path = "./model/UTKFace"
    train_model(model, utk_path, img_size, batch_size, epochs=10, save_dir="./checkpoints")

    namus_dirs = ["./faces"]  # in future add other directories (Missing Faces, etc)
    print("Implementing model on NamUs data")

    for dir_path in namus_dirs:
        if os.path.exists(dir_path):
            output_dir = f"./output"
            process_namus_data(model, img_size, dir_path, output_dir)
    print("Age estimation on NamUS complete")

if __name__ == "__main__":
    main()