# EmbedSeg-napari

Here, we attempt to extend **`EmbedSeg`** for use with **`napari`**. This is in an early phase of development and useability should become better in the upcoming weeks. Please feel free to open an issue, suggest a feature request or change in the UI layout.

----------------------------------

## Getting started

Create a new python environment with a **`napari`** installation (referred here as **`napari-env`**). Next run the following commands in the terminal window:

```
git clone https://github.com/juglab/EmbedSeg-napari
cd EmbedSeg-napari
conda activate napari-env
python3 -m pip install -e .
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
napari
```

## Obtaining instance mask predictions on new images using pretrained models

Place all the images you wish to segment, under directories `test/images/`. We provide three 2D and four 3D pretrained models [here](https://github.com/juglab/EmbedSeg/tree/main/pretrained_models). One could follow the sequence of steps:

1. Select _Predict_ subpanel under _EmbedSeg-napari_ plugin.
2. Browse for the _test_ directory containing the images you wish to evaluate on. (Evaluation images should be present as _test/images/*.tif_).  
3. Next, browse for the pretrained-model weights. (Pretrained model weights have extension _*.pth_)
4. Then, browse for the _data_properties.json_ file which carries some dataset-specific properties
5. Check _test-time augmentation_, _Save Images_ and _Save Results_ checkboxes
6. Next, browse for a directory where you would like to export the predicted instance segmentation _tiff_ images
7. If everything went well so far, the paths to all the files specified by us, can be seen in the terminal window
8. Now we are ready to click the _Predict_ push button
9. All test images are processed one-by-one, and the original image and network prediction are loaded in the _napari_ visualizer window
10. If ground truth instance masks are available, then one can also calculate the accuracy of the predictions in terms of _mAP_
11. Toggle visibility of image back-and-forth to see the quality of the instance mask prediction
12. One could also drag and drop the images and predictions from the save directory into another viewer such as _Fiji_



https://user-images.githubusercontent.com/34229641/123522434-7f6dd700-d6bd-11eb-8d8b-595ae4c91481.mp4


## Training and Visualization

1. Select _Train_ subpanel under EmbedSeg-napari plugin
2. Browse for crops generated in the preprocessing stage. Single click on directory, one level above _train_ and _val_
3. Browse for _data_properties.json_ which carries some dataset-specific properties
4. Browse for directory where intermediate model weights and log files should be saved
5. Set the other parameters such as train and val size, train and val batch size etc
6. Click on _Begin training_ button
7. Note that internally visualization is updated every 5 training and validation steps 
8. Stop any time and resume from the last checkpoint by browsing to the last saved model weights (_checkpoint.pth_)




https://user-images.githubusercontent.com/34229641/123523355-5c462600-d6c3-11eb-8997-e0c1d7cee334.mp4


## TODOs
- [ ] Add visualization for virtual batch > 1
- [ ] Add code for displaying embedding 
- [ ] Fix the callback on `Stop training` button 
- [ ] Show visualizations for 3d as volumetric images and not as z-slices 
- [ ] Use `threadworker` while generating crops in `preprocessing` panel
- [ ] Remove `EmbedSeg` core code and include as a pip package




## Issues

If you encounter any problems, please **[file an issue]** along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/mlbyml/EmbedSeg-napari/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
