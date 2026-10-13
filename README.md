# Can Trump's posts predict oil prices?

This study examines WTI price movements in the seconds after Donald Trump's oil-related Truth Social posts.
A small machine-learning model uses GPT's reading of each post to predict whether the price will go
**up**, **down** or **stay flat** over the next 30 seconds.

*Research only. Not investment advice.*

## Results

The model was tested on **212 oil-related posts from February to September 2026**. For each test month,
it trained only on earlier posts.

| Start of the 30-second measurement window | Up/down F1 |
|---|---:|
| At the post's timestamp | **0.440** |
| 1 second later | 0.401 |
| 2 seconds later | 0.355 |
| 3 seconds later | 0.328 |
| 4 seconds later | 0.347 |
| 5 seconds later | 0.341 |

The random-guessing benchmark—choosing up, flat or down with equal probability—is **0.294** on the same
test sample. The model scores highest with no delay; most of its improvement over the benchmark is lost
with a two- to three-second delay.

Each measurement window lasts a full 30 seconds. For example, a three-second delay measures the change
from 3 to 33 seconds after the post. These are prediction scores, not trading returns.

## How it works

1. **Collect the posts.** The dataset contains 8,259 feed items from 18 September 2025 to 25 September 2026,
   collected from trumpstruth.org. Of these, 4,138 are marked as authored by Trump; 3,861 posts have
   precomputed GPT labels.
2. **Read each post.** GPT-6 Sol answers nine questions about its content: whether it concerns oil, who
   is involved, how it could affect supply or demand, and whether an action is proposed, implemented or
   withdrawn. It identifies 365 oil-related posts.
3. **Build the inputs.** Fixed rules turn those answers into 17 numerical features for each post.
4. **Match the prices.** The price dataset contains 930,388 WTI broker quotes from Dukascopy. The post's
   timestamp is decoded from its Truth Social ID (`id >> 16`) at millisecond resolution. At each window
   boundary, the code uses the latest bid quote at or before that time, allowing a quote up to 60 seconds old.
5. **Assign an outcome.** The 30-second price change is labelled **up** above +0.075%, **down** below
   −0.075%, and **flat** otherwise. A total of 266 oil-related posts have usable prices at both boundaries.
6. **Test month by month.** A small gradient-boosted tree classifier with fixed settings trains on earlier
   posts and predicts the next month. Testing begins in February 2026, requires at least 30 training posts
   and two training classes, and produces predictions for 212 posts.
   The other 54 of the 266 posts, from September 2025 to January 2026, are not tested: there was not
   enough earlier data in those months to train a model on, so they are only used for training.
7. **Measure performance.** Compute up/down F1 on those predictions. Repeat the procedure with later
   window starts to measure how the result changes with delay.

## Run the analysis

Use **Python 3.11 or newer** and run the following from the repository root:

```sh
python3 -m pip install -r requirements.txt
python3 src/analyze.py
```

No OpenAI API key is needed to run the analysis: the GPT labels are already saved in `data/features.json`.

Expected output, apart from progress bars:

```text
(266, 17)
(266,)
0.44049239143882224
```

The first line describes 266 posts with 17 features each. The second is the shape of the prediction mask,
not the number of tested posts. The final line is the up/down F1 score with no measurement delay.

To reproduce the table for delays of zero to five seconds:

```sh
TQDM_DISABLE=1 PYTHONPATH=src python3 - <<'EOF'
from sklearn.metrics import f1_score

from analyze import read_features, read_price, walk_forward
from data_pre_processing import make_labels, pre_process

posts = read_features()
posts = posts[posts['oil_relevance'] != 'none']
features = pre_process(posts)
prices = read_price()

for delay in range(6):
    labels = make_labels(prices, posts, delay=f'{delay}s', window='30s')['sign']
    keep = labels.notna()
    X, y = features.loc[keep], labels.loc[keep]
    predictions = walk_forward(X, y)
    tested = predictions.notna()
    score = f1_score(y[tested], predictions[tested], labels=[-1, 1], average='macro')
    print(f'delay {delay} s: F1 {score:.3f}  ({tested.sum()} posts tested)')
EOF
```

## Files

| File | Purpose |
|---|---|
| `src/analyze.py` | Entry point: loads the data, tests the model and prints its score. |
| `src/data_pre_processing.py` | Builds the 17 features and the up/flat/down labels. |
| `src/data_reader.py` | Reads the posts and WTI prices. |
| `src/model.py` | Defines the classifier and training settings. |
| `src/feature_engineering.py` | Submits posts to GPT for labelling and retrieves batch results. Requires an OpenAI API key. |
| `data/features.json` | Precomputed GPT labels for 3,861 posts. |
| `data/posts.json` | The 8,259 posts without their text: id, link, time and author. Open a post's link to read it. |
| `data/oil_indices/wti_post_prices.csv` | The WTI bid the study reads: 0–5 s and 30–35 s after each of the 365 oil posts (blank where no quote in the previous 60 s). |
| `data/oil_indices/brent_1d.csv` | Daily Brent spot price (EIA), from earlier work; not used in the current analysis. |

The post texts and the full WTI tick data are not included. The ticks come from Dukascopy, whose terms do not allow republishing them.

## Limitations and version notes

- **Small, concentrated sample.** Only 212 posts were tested, and most concern the Iran war. The results
  may not generalise to other events or market conditions.
- **No significance or trading test in this version.** The reported F1 scores do not establish a reliable
  trading advantage or profitability after costs and execution delays.
- **Broker quotes.** Dukascopy quotes may differ from exchange prices. Matching allows quotes up to
  60 seconds old, so timestamp precision does not guarantee a fresh quote at each boundary.
