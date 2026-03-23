"""
Moodify — mood classifier for spotify audio features
started this during finals week which was probably a bad idea
but hey it works now so no regrets

uses tensorflow to classify tracks into moods based on 
danceability, energy, valence, tempo, etc.
trained on synthetic data bc spotify rate are limited...
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import json
import os

MOOD_LABELS = ["Happy", "Sad", "Energetic", "Calm", "Angry"]

FEATURE_NAMES = [
    "danceability", "energy", "valence", "tempo", "acousticness",
    "instrumentalness", "liveness", "speechiness", "loudness", "mode", "key"
]

NUM_FEATURES = len(FEATURE_NAMES)  # 11
NUM_CLASSES = len(MOOD_LABELS)     # 5

np.random.seed(42)
tf.random.set_seed(42)


# data generation:
# originally tried scraping spotify directly but the api kept
# throttling me so i just made synthetic profiles instead
# the distributions are based on ~200 real tracks i manually tagged

def make_mood_samples(mood, count):
    """generate fake spotify features that match a mood vibe"""
    
    # each tuple is (mean, std) for that feature
    # eyeballed from my own playlist
    mood_profiles = {
        "Happy": {
            "danceability": (0.70, 0.10),
            "energy": (0.72, 0.10),
            "valence": (0.80, 0.08),      # valence is happiness score
            "tempo": (120, 15),
            "acousticness": (0.20, 0.10),
            "instrumentalness": (0.05, 0.05),
            "liveness": (0.20, 0.10),
            "speechiness": (0.08, 0.05),
            "loudness": (-6, 2),
            "mode": (1, 0),                # major key = happy
            "key": (7, 3),
        },
        "Sad": {
            "danceability": (0.35, 0.10),  
            "energy": (0.30, 0.10),
            "valence": (0.20, 0.08),
            "tempo": (80, 15),
            "acousticness": (0.70, 0.12),
            "instrumentalness": (0.15, 0.10),
            "liveness": (0.12, 0.08),
            "speechiness": (0.05, 0.03),
            "loudness": (-12, 3),          # quiet and sad
            "mode": (0, 0),
            "key": (5, 3),
        },
        "Energetic": {
            "danceability": (0.80, 0.08),
            "energy": (0.90, 0.06),        # basically maxed out
            "valence": (0.65, 0.12),
            "tempo": (140, 15),
            "acousticness": (0.08, 0.05),
            "instrumentalness": (0.10, 0.08),
            "liveness": (0.30, 0.12),
            "speechiness": (0.10, 0.06),
            "loudness": (-4, 2),
            "mode": (1, 0),
            "key": (6, 4),
        },
        "Calm": {
            "danceability": (0.40, 0.10),
            "energy": (0.25, 0.10),
            "valence": (0.50, 0.15),       # calm and sad are NOT the same
            "tempo": (90, 15),
            "acousticness": (0.80, 0.10),
            "instrumentalness": (0.40, 0.20),
            "liveness": (0.10, 0.05),
            "speechiness": (0.04, 0.02),
            "loudness": (-15, 3),
            "mode": (1, 0),
            "key": (4, 3),
        },
        "Angry": {
            "danceability": (0.55, 0.12),
            "energy": (0.88, 0.06),
            "valence": (0.30, 0.10),       # high energy + low valence = angry
            "tempo": (135, 18),
            "acousticness": (0.05, 0.04),
            "instrumentalness": (0.03, 0.03),
            "liveness": (0.25, 0.10),
            "speechiness": (0.12, 0.07),
            "loudness": (-3, 2),            
            "mode": (0, 0),
            "key": (3, 4),
        },
    }

    profile = mood_profiles[mood]
    samples = np.zeros((count, NUM_FEATURES))

    for i, feature in enumerate(FEATURE_NAMES):
        mean, std = profile[feature]

        if feature == "mode":
            # mode is binary (major/minor) so binomial makes more sense
            samples[:, i] = np.random.binomial(1, mean, count).astype(float)
        elif feature == "key":
            # keys are integers 0-11, representing C through B
            vals = np.random.normal(mean, std, count)
            samples[:, i] = np.clip(np.round(vals), 0, 11)
        else:
            samples[:, i] = np.random.normal(mean, std, count)

    # clip everything to valid spotify ranges
    for i, feature in enumerate(FEATURE_NAMES):
        if feature == "tempo":
            samples[:, i] = np.clip(samples[:, i], 50, 220)
        elif feature == "loudness":
            samples[:, i] = np.clip(samples[:, i], -60, 0)
        elif feature == "key":
            samples[:, i] = np.clip(samples[:, i], 0, 11)
        elif feature not in ("mode",):
            samples[:, i] = np.clip(samples[:, i], 0.0, 1.0)

    return samples


def build_dataset(per_mood=2200):
    """
    make the full dataset. 2200 per mood = 11000 total
    tried 1000 first but accuracy was hovering around 90%
    bumping to 2200 got it past 97 so i stopped there
    """
    all_X = []
    all_y = []
    
    for mood_idx, mood in enumerate(MOOD_LABELS):
        data = make_mood_samples(mood, per_mood)
        all_X.append(data)
        labels = np.full(per_mood, mood_idx)
        all_y.append(labels)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # shuffle so the model doesn't just learn "first 2200 = happy"
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


# preprocessing:
# tried minmax scaling first but z-score worked better - bc of diff ranges

class Preprocessor:
    def __init__(self):
        self.means = None
        self.stds = None
    
    def fit(self, X):
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)
        # prevent divide by zero on constant features
        self.stds[self.stds == 0] = 1.0
        return self
    
    def transform(self, X):
        return (X - self.means) / self.stds
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def save(self, path):
        data = {
            "means": self.means.tolist(),
            "stds": self.stds.tolist()
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        p = cls()
        p.means = np.array(data["means"])
        p.stds = np.array(data["stds"])
        return p


# model:
# tried a CNN at one point which was bad in retrospect

def create_model():
    model = keras.Sequential([
        layers.Input(shape=(NUM_FEATURES,)),
        
        # first hidden layer - 128 felt right, 256 was overkill
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # second layer
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # third layer - considered removing this but it helped a tiny bit
        layers.Dense(32, activation="relu"),
        
        # output
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# training:

def run_training(num_epochs=60, batch_sz=64):
    print("=" * 55)
    print("  MOODIFY — training pipeline")
    print("=" * 55)

    # generate data
    print("\n[1/4] generating dataset...")
    X, y = build_dataset(per_mood=2200)
    total = len(X)
    print(f"       got {total} samples")

    # split — 70/15/15 train/val/test
    # considered 80/10/10 but wanted a beefier test set
    n_train = int(0.70 * total)
    n_val = int(0.15 * total)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]

    print(f"       train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # preprocess
    print("\n[2/4] normalizing features...")
    prep = Preprocessor()
    X_train_n = prep.fit_transform(X_train)
    X_val_n = prep.transform(X_val)
    X_test_n = prep.transform(X_test)

    # train
    print("\n[3/4] training model...")
    model = create_model()
    model.summary()

    # early stopping to stop overfitting
    stop_early = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )
    
    # reduce lr when stuck
    lr_reducer = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=0.000001
    )

    history = model.fit(
        X_train_n, y_train,
        validation_data=(X_val_n, y_val),
        epochs=num_epochs,
        batch_size=batch_sz,
        callbacks=[stop_early, lr_reducer],
        verbose=1,
    )

    # evaluate
    print("\n[4/4] testing...")
    test_loss, test_acc = model.evaluate(X_test_n, y_test, verbose=0)
    train_loss, train_acc = model.evaluate(X_train_n, y_train, verbose=0)

    print(f"\n{'─' * 40}")
    print(f"  train acc:  {train_acc * 100:.2f}%")
    print(f"  test acc:   {test_acc * 100:.2f}%")
    print(f"{'─' * 40}")

    # save everything
    os.makedirs("artifacts", exist_ok=True)
    model.save("artifacts/moodify_model.keras")
    prep.save("artifacts/preprocessor.json")

    # save history for the frontend charts
    hist_dict = {}
    for key, vals in history.history.items():
        hist_dict[key] = [float(v) for v in vals]
    hist_dict["test_accuracy"] = float(test_acc)
    hist_dict["train_accuracy"] = float(train_acc)

    with open("artifacts/training_history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    print("\nsaved to ./artifacts/")
    return model, prep, hist_dict


# inference:

def classify_track(features, model=None, preprocessor=None):
    """
    takes a dict of spotify audio features, returns predicted mood
    
    example:
        result = classify_track({
            "danceability": 0.75, "energy": 0.8, "valence": 0.9,
            "tempo": 128, ...
        })
        print(result["mood"])  # "Happy"
    """
    if model is None:
        model = keras.models.load_model("artifacts/moodify_model.keras")
    if preprocessor is None:
        preprocessor = Preprocessor.load("artifacts/preprocessor.json")

    # build feature vector in the right order
    x = []
    for f in FEATURE_NAMES:
        x.append(features[f])
    x = np.array([x])

    x_normalized = preprocessor.transform(x)
    probabilities = model.predict(x_normalized, verbose=0)[0]

    best_idx = int(np.argmax(probabilities))
    
    result = {
        "mood": MOOD_LABELS[best_idx],
        "confidence": float(probabilities[best_idx]),
        "all_probabilities": {}
    }
    
    for i, label in enumerate(MOOD_LABELS):
        result["all_probabilities"][label] = float(probabilities[i])
    
    return result


# spotify api helper:
# this would hit the actual spotify api in production
# keeping it as a stub for now bc oauth tokens expire every hour
def get_mood_recommendations(mood, num_tracks=10):
    """
    maps mood to spotify recommendation params
    in prod this calls GET /v1/recommendations
    """
    # these genre seeds are what worked best in my testing
    seed_map = {
        "Happy":     {"seed_genres": "pop,dance",        "target_valence": 0.8, "target_energy": 0.7},
        "Sad":       {"seed_genres": "acoustic,indie",    "target_valence": 0.2, "target_energy": 0.3},
        "Energetic": {"seed_genres": "edm,rock",          "target_valence": 0.6, "target_energy": 0.9},
        "Calm":      {"seed_genres": "ambient,classical",  "target_valence": 0.5, "target_energy": 0.2},
        "Angry":     {"seed_genres": "metal,punk",         "target_valence": 0.3, "target_energy": 0.9},
    }

    params = seed_map.get(mood, seed_map["Happy"])
    
    return {
        "mood": mood,
        "params": params,
        "endpoint": "GET /v1/recommendations",
        "limit": num_tracks,
        "note": "needs spotify oauth token to work"
    }


# main:

if __name__ == "__main__":
    model, prep, history = run_training()

    # quick demo
    print("\n\n" + "=" * 55)
    print("  demo predictions")
    print("=" * 55)

    test_tracks = [
        {
            "name": "some pop banger",
            "danceability": 0.75, "energy": 0.78, "valence": 0.85,
            "tempo": 125, "acousticness": 0.15, "instrumentalness": 0.02,
            "liveness": 0.18, "speechiness": 0.06, "loudness": -5,
            "mode": 1, "key": 7
        },
        {
            "name": "crying in the rain type beat",
            "danceability": 0.30, "energy": 0.25, "valence": 0.15,
            "tempo": 75, "acousticness": 0.78, "instrumentalness": 0.20,
            "liveness": 0.10, "speechiness": 0.04, "loudness": -14,
            "mode": 0, "key": 5
        },
        {
            "name": "gym playlist opener",
            "danceability": 0.85, "energy": 0.95, "valence": 0.70,
            "tempo": 150, "acousticness": 0.03, "instrumentalness": 0.08,
            "liveness": 0.35, "speechiness": 0.08, "loudness": -3,
            "mode": 1, "key": 9
        },
    ]

    for track in test_tracks:
        name = track.pop("name")
        result = classify_track(track, model, prep)
        recs = get_mood_recommendations(result["mood"])
        
        print(f"\n  >> {name}")
        print(f"     mood: {result['mood']} ({result['confidence']*100:.1f}%)")
        print(f"     recs: {recs['params']['seed_genres']}")
