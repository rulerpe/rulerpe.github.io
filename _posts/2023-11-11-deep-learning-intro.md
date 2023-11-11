# Deep Learning Introduction

## What is Deep Learning
Deep learning is a special type of machine learning modal that uses a architecture named Neuron Network.
It is very flexible that can solve almost any problem by varying it's weights. Using Stochasitc Gradient Desent(SDG), the model can automatilly learning and adjust it's weights to make better prediction.

*diagram of Deep Learning flow*
![image diagram of Deep Learning flow](https://github.com/rulerpe/rulerpe.github.io/assets/8906831/5bb6a1d8-6477-48d7-8bda-1da63a13864d)

### Deep Learning best at
- Computer Vision: Analyzing images at human level, and sometime does better then human. Example: face recognition.
- Natural Language Processing (NLP): Analyzing text based document, and generate human level text response. Example: spam detector.
- Combingin text and images: Gennerate text caption on images. Example: quick summary of a CT scan image.
- Tabular data: Analyzing data in table format. With library such as RAPIDS, that provides GPU acceleration, it will preform better them other machine learning model.
- Recommendation systems: A special type of tabular data, predict what user likes base on other users behavior. Example: Amazon, Netflix.

## The practice of Deep Learning
### Starting project
Iterate the model fast, dont spend too much time collecting data, you will never find the perfect data set. 
Start working on what ever you can find, and build your model, test the result. then you will find what you need. You might find out you dont needs much data to get a good result.
Iterate from end to end in the project, dont spend too much time fine-tuning or labelling data.
The end to end iteration apporach helps you better understand what you really need, you might find out you need more data, try out other alguratum. 
Create a quick prototype to show that your idea works.
Try using method from different deep learning area to solve your problem. 
Start off doing project that other people has already done, this way you can mesure how your model perform by compare the result with other peoples result. (Kaggle competition)
### The Drivetain Approch for building a Deep Learning model
The things to consider to before start to create a model
1. Defind Objective: What is the outcome are you tring to achieve ?
2. Levers: What are the inputs you have control of ?
3. Data: What data you can collect ?
4. Models: How the levers will influence the objective ?



## Model example: Bear Classifier 
App that can classify Grizzly, Black, and Teddy bear.
### import fastai library
```python
! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
```
### Gethering Data
Use Bing image search API to collect bear images
```python
key = os.environ.get('AZURE_SEARCH_KEY', '045a91f99ac8418586aba9f3dc01059b')
bear_types = 'grizzly','black','teddy'
path = Path('bears')

# create bear folder and loop thought bear type to get all bears.
if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} bear')
        download_images(dest, urls=results.attrgot('contentUrl'))
```
### Clean data
Remove croped images
```python
fns = get_image_files(path)
failed = verify_images(fns)
failed.map(Path.unlink);
```
### Data to DataLoaders
Provide our bear image data to the model by create DataLoader
Also do data augmentation on the data, by roatating, flipping, wraping the image.
```python
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
```
### Train model and clean data
```python
dls = bears.dataloaders(path)
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

# use confusion matrix to see how well the model does.
interp = ClassificationInterpretation.from_learner(learn)

# check the incrrect predection
interp.plot_top_losses(5, nrows=1)
interp.plot_confusion_matrix()

# use a GUI weight to relable or delete the data
cleaner = ImageClassifierCleaner(learn)
cleaner
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
```
### Export the trained model
Export the model so we can use it in production. the export model has the architecture and the trained parameters.
```python
learn.export()
path = Path()

# save as `export.pkl`
path.ls(file_exts='.pkl')

# we can make prediction using the model like this, as a function
learn_inf = load_learner(path/'export.pkl')
learn_inf.predict('images/grizzly.jpg')
```
### Create a simple application in notbook
```python
# upload button
btn_upload = widgets.FileUpload()
btn_upload

# output the upload image and label
out_pl = widgets.Output()
lbl_pred = widgets.Label()

# function triggers on upload that make a prediton of the image using the model we trained
def on_data_change(change):
  lbl_pred.value = ''
  img = PILImage.create(btn_upload.data[-1])
  out_pl.clear_output()
  with out_pl: display(img.to_thumb(128,128))
  pred,pred_idx,probs=learn_inf.predict(img)
  lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]: 04f}'
btn_upload.observe(on_data_change, names=['data'])

# use VBox to show the app in notbook
display(VBox([widgets.Label('Select your bear!'), btn_upload, out_pl, lbl_pred]))
```
