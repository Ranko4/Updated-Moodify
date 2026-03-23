# Moodify

**[LIVE DEMO]("https://updated-moodify-changed.vercel.app/")**

Classifies the mood of music using a neural network trained on Spotify audio features. Feed it a track's danceability, energy, valence, tempo, etc. and it'll tell you if the vibe is happy, sad, energetic, calm, or angry - then recommend similar tracks through the Spotify API.

Started this as a project for Codeology at Berkeley. The original idea was just "what if I could auto-sort my playlists by mood" and now here we are.

## How it works

The model looks at 11 audio features that Spotify provides for every track:

- **Valence** — basically how happy/positive a track sounds (this one does the most heavy lifting)
- **Energy, Danceability, Tempo** — how intense and upbeat it feels
- **Acousticness, Instrumentalness** — whether it's acoustic/instrumental or produced
- **Loudness, Liveness, Speechiness, Mode, Key** — other characteristics

These get fed into a 4-layer neural network that outputs probabilities across 5 mood categories. The model was trained on ~11,000 synthetic tracks (2,200 per mood). The distributions are based on ~200 real tracks I manually tagged from my own playlists.

### Architecture

```
Input (11 features)
  → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
  → Dense(64)  → BatchNorm → ReLU → Dropout(0.2)
  → Dense(32)  → ReLU
  → Dense(5)   → Softmax
```

Gets ~98% training accuracy and ~97% validation. I tried a bunch of other architectures first before landing on this one.

## Project structure

```
├── src/
│   ├── App.jsx           -- react frontend + JS neural net
│   └── main.jsx          -- entry point
├── python/
│   ├── moodify_classifier.py  -- tensorflow training pipeline
│   └── requirements.txt       -- python deps
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

## Running it locally

### The web demo

```bash
npm install
npm run dev
```

This starts the frontend at `localhost:5173`. The neural network trains in your browser when the page loads — takes a few seconds. No backend needed.

### The Python version

```bash
cd python
pip install -r requirements.txt
python moodify_classifier.py
```

This trains the TensorFlow model from scratch, saves it to `./artifacts/`, and runs some demo predictions. Takes maybe a minute depending on your machine.

## Spotify API integration

The classifier is designed to plug into Spotify's Web API. Right now the API calls are stubbed out, but the pipeline works like this:

1. Get a track's audio features via `GET /v1/audio-features/{id}`
2. Normalize the features (z-score, same as training)
3. Run through the model → get a mood prediction
4. Map the mood to recommendation params (seed genres + target values)
5. Hit `GET /v1/recommendations` → get a mood-matched playlist

There's also support for `GET /v1/me/top/tracks` to analyze a user's listening history and build a mood profile, but I haven't fully built that out yet.

## Tech stuff

- **ML:** TensorFlow/Keras for the Python pipeline, hand-rolled DNN in JavaScript for the browser demo
- **Frontend:** React + Vite, SVG charts (tried recharts but it was 200kb for one chart)
- **Data:** NumPy for data generation, Spotify audio feature schema
- **Deployed on:** Vercel

## Things I'd do differently / future work

- Actually hook up Spotify OAuth so the recommendations are live
- Train on real labeled data instead of synthetic (need to figure out the rate limiting)
- Try a multi-label approach (a song can be both energetic and happy)
- Add a "analyze my playlist" feature using the top tracks endpoint
- The JS neural net doesn't have batch normalization or dropout — would be nice to add

## License

MIT
