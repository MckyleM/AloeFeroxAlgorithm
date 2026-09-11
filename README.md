# Aloe Ferox phenology classifier

A PyTorch image classifier that predicts the phenology stage of an *Aloe ferox*
plant from a photograph, served through a Dash app with drag-and-drop upload.

Built for a machine-learning module (Milestone 2).

## The task

Eight single-label phenology classes:

`flowers` · `buds` · `fruits` · `No Evidence` · `flowers and fruits` ·
`flowers and buds` · `buds and fruits` · `flowers, fruits and buds`

Several classes are combinations of the others, which is what makes this harder
than it first looks — the model has to separate "flowers" from "flowers and
buds" on visual evidence that overlaps heavily.

## Model

Transfer learning on **ResNet-18** (`torchvision`, pretrained), with the final
fully-connected layer replaced to output the eight classes and a `log_softmax`
head — see `Dash/NeuralClass.py`:

```python
class SingleLabelCNN(nn.Module):
    def __init__(self, num_classes):
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
```

Images are resized to 224×224, converted to tensors and normalised to mean/std
0.5 per channel.

**Training** (`Mark2.ipynb`): Adam, learning rate 0.001, batch size 32,
`CrossEntropyLoss`, with a learning-rate scheduler, for 27 epochs. Training loss
fell to 0.3618.

## Results

| Evaluation | Accuracy |
| --- | --- |
| Over the evaluation dataloader (labelled "train" in the notebook) | 87.68% |
| Over the held-out `lastTest_set` | **67.36%** |

The held-out figure is the one that matters. For context, eight balanced classes
would put chance at 12.5%.

`Aloe.ipynb` is the earlier attempt — 15 epochs, 83.95% over its dataloader but
only 16.27% on its held-out set, which is essentially chance. `Mark2.ipynb`
is the working version and produced the deployed weights.

## Data

The dataset is built from an observation spreadsheet carrying taxon and
`field:phenology (foa)` columns. `Data.ipynb` filters to rows that actually have
a phenology label, removes duplicates, and downloads the corresponding images by
URL into local folders.

Before running anything, create two folders named `Test` and `Train` in the
repository root — the download script populates them.

## Layout

| Path | What it is |
| --- | --- |
| `Mark2.ipynb` | The working pipeline — data prep, training, evaluation |
| `Aloe.ipynb` | Earlier training attempt, kept for comparison |
| `Data.ipynb` | Dataset construction and image download |
| `Dash/dash_app.py` | The Dash application — **run this one** |
| `Dash/NeuralClass.py` | `SingleLabelCNN` model definition |
| `Dash/aloe_model2.pth` | Deployed weights |
| `AloeScript.py` | Notebook exported as a flat script |
| `app.py` | An early incomplete draft of the app — see the note below |

## Running it

```bash
pip install -r requirements.txt
python Dash/dash_app.py
```

Then open the local address Dash prints and upload a photograph.

## Repository notes

`app.py` in the repository root is an early draft: it calls `model(image)` but
never loads a model, so it will fail at prediction time. The working application
is `Dash/dash_app.py`, which loads `aloe_model2.pth` and imports the model class
from `Dash/NeuralClass.py`.
