import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import shutil
import sys
from tifffile import imsave
import json
import numpy as np
import os
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog
from glob import glob
from napari.qt.threading import thread_worker
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QLabel, QWidget, QPushButton, QGridLayout, QVBoxLayout, QLineEdit, QComboBox, \
    QCheckBox, QProgressBar, QScrollArea, QRadioButton
from tqdm import tqdm

from embedseg_napari.criterions import get_loss
from embedseg_napari.datasets import get_dataset
from embedseg_napari.models import get_model
from embedseg_napari.utils.create_dicts import create_dataset_dict, create_model_dict, create_loss_dict, create_configs
from embedseg_napari.utils.generate_crops import process, process_one_hot, process_3d
from embedseg_napari.utils.preprocess_data import split_train_val, split_train_test, get_data_properties
from embedseg_napari.utils.create_dicts import create_test_configs_dict
from embedseg_napari.utils.utils2 import matching_dataset, obtain_AP_one_hot
from embedseg_napari.utils.test_time_augmentation import apply_tta_2d, apply_tta_3d
from embedseg_napari.utils.utils import AverageMeter, Logger
from embedseg_napari.utils.utils import Cluster_3d, Cluster
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


class Preprocess(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        # define components
        logo_path = 'embedseg_napari/resources/embedseg_napari_logo.png'
        self.logo_label = QLabel(f'<h1><img src="{logo_path}">EmbedSeg</h1>')
        self.method_description_label = QLabel(
            '<small>Embedding-based Instance Segmentation <br> in Microscopy.<br>If you are using this in your research please <a href="https://github.com/juglab/EmbedSeg#citation" style="color:gray;">cite us</a>.</small><br><small><tt><a href="https://github.com/juglab/EmbedSeg" style="color:gray;">https://github.com/juglab/EmbedSeg</a></tt></small>')

        self.download_data_label = QLabel('<h3>Download Data</h3>')
        self.data_dir_label = QLabel('Data Directory')
        self.data_dir_pushbutton = QPushButton('Browse')
        self.data_dir_pushbutton.setMaximumWidth(280)
        self.data_dir_pushbutton.clicked.connect(self._prepare_data_dir)

        self.data_type_label = QLabel('Data type')
        self.data_type_2d_radio_button = QRadioButton('2d')
        self.data_type_2d_radio_button.setChecked(True)
        self.data_type_2d_radio_button.toggled.connect(self._show_voxel_parameters)
        self.data_type_3d_radio_button = QRadioButton('3d')
        self.data_type_3d_radio_button.toggled.connect(self._show_voxel_parameters)

        self.split_data_label_1 = QLabel('<h3>Split Data into Train and Test</h3>')
        self.train_test_label = QLabel('Train-Test Name')
        self.train_test_lineedit = QLineEdit('train')
        self.train_test_lineedit.setAlignment(Qt.AlignCenter)
        self.train_test_lineedit.setMaximumWidth(280)

        self.train_test_subset_label = QLabel('Subset')
        self.train_test_subset_lineedit = QLineEdit('0.1')
        self.train_test_subset_lineedit.setMaximumWidth(280)
        self.train_test_subset_lineedit.setAlignment(Qt.AlignCenter)

        self.split_data_label_2 = QLabel('<h3>Split Data into Train and Val</h3>')
        self.train_val_label = QLabel('Train-Val Name')
        self.train_val_lineedit = QLineEdit('train')
        self.train_val_lineedit.setAlignment(Qt.AlignCenter)
        self.train_val_lineedit.setMaximumWidth(280)

        self.train_val_subset_label = QLabel('Subset')
        self.train_val_subset_lineedit = QLineEdit('0.15')
        self.train_val_subset_lineedit.setMaximumWidth(280)
        self.train_val_subset_lineedit.setAlignment(Qt.AlignCenter)

        self.specify_center_label = QLabel('<h3>Specify desired center</h3>')
        self.center_label = QLabel('Center')
        self.center_combobox = QComboBox(self)
        self.center_combobox.addItem('medoid')
        self.center_combobox.addItem('centroid')
        self.center_combobox.addItem('approximate-medoid')
        self.center_combobox.setEditable(True)
        self.center_combobox.lineEdit().setAlignment(Qt.AlignCenter)
        self.center_combobox.lineEdit().setReadOnly(True)
        self.center_combobox.setMaximumWidth(280)

        self.calculate_dataset_label = QLabel('<h3>Calculate some dataset specific properties</h3>')
        self.one_hot_checkbox = QCheckBox('One-hot encoded masks ?')
        self.image_datatype_label = QLabel('Data type')
        self.image_datatype_combobox = QComboBox(self)
        self.image_datatype_combobox.addItem('8 bit (u)')
        self.image_datatype_combobox.addItem('16 bit (u)')

        self.image_datatype_combobox.setEditable(True)
        self.image_datatype_combobox.lineEdit().setAlignment(Qt.AlignCenter)
        self.image_datatype_combobox.lineEdit().setReadOnly(True)
        self.image_datatype_combobox.setMaximumWidth(280)

        self.specify_cropping_label = QLabel('<h3>Specify Cropping Configuration</h3>')
        self.crops_dir_label = QLabel('Crops directory')
        self.crops_dir_pushbutton = QPushButton('Browse')
        self.crops_dir_pushbutton.clicked.connect(self._prepare_data_dir)
        self.crops_dir_pushbutton.setMaximumWidth(280)

        self.crop_size_x_label = QLabel('Crop Size [X]')
        self.crop_size_x_lineedit = QLineEdit('256')
        self.crop_size_y_label = QLabel('Crop Size [Y]')
        self.crop_size_y_lineedit = QLineEdit('256')
        self.crop_size_z_label = QLabel('Crop Size [Z]')
        self.crop_size_z_lineedit = QLineEdit('32')
        self.voxel_size_x_label = QLabel('Voxel Size [X] in microns')
        self.voxel_size_x_lineedit = QLineEdit('0.1733')
        self.voxel_size_y_label = QLabel('Voxel Size [Y] in microns')
        self.voxel_size_y_lineedit = QLineEdit('0.1733')
        self.voxel_size_z_label = QLabel('Voxel Size [Z] in microns')
        self.voxel_size_z_lineedit = QLineEdit('1.0')

        self.crop_size_x_lineedit.setAlignment(Qt.AlignCenter)
        self.crop_size_x_lineedit.setMaximumWidth(280)
        self.crop_size_y_lineedit.setAlignment(Qt.AlignCenter)
        self.crop_size_y_lineedit.setMaximumWidth(280)
        self.crop_size_z_lineedit.setAlignment(Qt.AlignCenter)
        self.crop_size_z_lineedit.setMaximumWidth(280)
        self.voxel_size_x_lineedit.setAlignment(Qt.AlignCenter)
        self.voxel_size_x_lineedit.setMaximumWidth(280)
        self.voxel_size_y_lineedit.setAlignment(Qt.AlignCenter)
        self.voxel_size_y_lineedit.setMaximumWidth(280)
        self.voxel_size_z_lineedit.setAlignment(Qt.AlignCenter)
        self.voxel_size_z_lineedit.setMaximumWidth(280)
        self.crop_size_y_label.setVisible(False)
        self.crop_size_y_lineedit.setVisible(False)
        self.crop_size_z_label.setVisible(False)
        self.crop_size_z_lineedit.setVisible(False)
        self.voxel_size_x_label.setVisible(False)
        self.voxel_size_x_lineedit.setVisible(False)
        self.voxel_size_y_label.setVisible(False)
        self.voxel_size_y_lineedit.setVisible(False)
        self.voxel_size_z_label.setVisible(False)
        self.voxel_size_z_lineedit.setVisible(False)
        self.preprocess_pushbutton = QPushButton('Generate Crops')
        self.preprocess_pushbutton.setMaximumWidth(280)
        self.preprocess_progress_bar = QProgressBar(self)
        self.preprocess_progress_bar.setMaximumWidth(280)
        self.preprocess_pushbutton.clicked.connect(self._preprocess)

        # outer layout
        outer_layout = QVBoxLayout()

        # inner layout 1
        grid_0 = QGridLayout()
        grid_0.addWidget(self.logo_label, 0, 0, 1, 1)
        grid_0.addWidget(self.method_description_label, 0, 1, 1, 1)
        grid_0.setSpacing(10)

        grid_1 = QGridLayout()
        grid_1.addWidget(self.download_data_label, 0, 0, 1, 2)
        grid_1.addWidget(self.data_dir_label, 1, 0)
        grid_1.addWidget(self.data_dir_pushbutton, 1, 1)
        grid_1.addWidget(self.data_type_label, 2, 0)
        grid_2 = QGridLayout()
        grid_2.addWidget(self.data_type_2d_radio_button, 0, 0)
        grid_2.addWidget(self.data_type_3d_radio_button, 0, 1)
        grid_2.setSpacing(10)
        grid_1.addLayout(grid_2, 2, 1)
        grid_1.setSpacing(10)

        # inner layout 3
        grid_3 = QGridLayout()
        grid_3.addWidget(self.split_data_label_1, 0, 0, 1, 2)
        grid_3.addWidget(self.train_test_label, 1, 0)
        grid_3.addWidget(self.train_test_lineedit, 1, 1)
        grid_3.addWidget(self.train_test_subset_label, 2, 0)
        grid_3.addWidget(self.train_test_subset_lineedit, 2, 1)
        grid_3.setSpacing(10)

        # inner layout 3
        grid_4 = QGridLayout()
        grid_4.addWidget(self.split_data_label_2, 0, 0, 1, 2)
        grid_4.addWidget(self.train_val_label, 1, 0)
        grid_4.addWidget(self.train_val_lineedit, 1, 1)
        grid_4.addWidget(self.train_val_subset_label, 2, 0)
        grid_4.addWidget(self.train_val_subset_lineedit, 2, 1)
        grid_4.setSpacing(10)

        # inner layout 4
        grid_5 = QGridLayout()
        grid_5.addWidget(self.specify_center_label, 0, 0, 1, 2)
        grid_5.addWidget(self.center_label, 1, 0)
        grid_5.addWidget(self.center_combobox, 1, 1)
        grid_5.setSpacing(10)

        # inner layout 5
        grid_6 = QGridLayout()
        grid_6.addWidget(self.calculate_dataset_label, 0, 0, 1, 2)
        grid_6.addWidget(self.one_hot_checkbox, 1, 0)
        grid_6.addWidget(self.image_datatype_label, 2, 0)
        grid_6.addWidget(self.image_datatype_combobox, 2, 1)
        grid_6.setSpacing(10)

        # inner layout 7
        grid_7 = QGridLayout()
        grid_7.addWidget(self.specify_cropping_label, 0, 0, 1, 2)
        grid_7.addWidget(self.crops_dir_label, 1, 0)
        grid_7.addWidget(self.crops_dir_pushbutton, 1, 1)
        grid_7.addWidget(self.crop_size_x_label, 2, 0)
        grid_7.addWidget(self.crop_size_x_lineedit, 2, 1)
        grid_7.addWidget(self.crop_size_y_label, 3, 0)
        grid_7.addWidget(self.crop_size_y_lineedit, 3, 1)
        grid_7.addWidget(self.crop_size_z_label, 4, 0)
        grid_7.addWidget(self.crop_size_z_lineedit, 4, 1)
        grid_7.addWidget(self.voxel_size_x_label, 5, 0)
        grid_7.addWidget(self.voxel_size_x_lineedit, 5, 1)
        grid_7.addWidget(self.voxel_size_y_label, 6, 0)
        grid_7.addWidget(self.voxel_size_y_lineedit, 6, 1)
        grid_7.addWidget(self.voxel_size_z_label, 7, 0)
        grid_7.addWidget(self.voxel_size_z_lineedit, 7, 1)
        grid_7.addWidget(self.preprocess_pushbutton, 8, 0)
        grid_7.addWidget(self.preprocess_progress_bar, 8, 1)
        grid_7.setSpacing(10)

        # add inner layouts to outer layout
        outer_layout.addLayout(grid_0)
        outer_layout.addLayout(grid_1)
        outer_layout.addLayout(grid_3)
        outer_layout.addLayout(grid_4)
        outer_layout.addLayout(grid_5)
        outer_layout.addLayout(grid_6)
        outer_layout.addLayout(grid_7)
        outer_layout.setSpacing(20)
        self.setLayout(outer_layout)
        self.setFixedWidth(560)

    def _show_voxel_parameters(self):
        if self.data_type_2d_radio_button.isChecked():
            self.crop_size_y_label.setVisible(False)
            self.crop_size_y_lineedit.setVisible(False)
            self.crop_size_z_label.setVisible(False)
            self.crop_size_z_lineedit.setVisible(False)
            self.voxel_size_x_label.setVisible(False)
            self.voxel_size_x_lineedit.setVisible(False)
            self.voxel_size_y_label.setVisible(False)
            self.voxel_size_y_lineedit.setVisible(False)
            self.voxel_size_z_label.setVisible(False)
            self.voxel_size_z_lineedit.setVisible(False)
        elif self.data_type_3d_radio_button.isChecked():
            self.crop_size_y_label.setVisible(True)
            self.crop_size_y_lineedit.setVisible(True)
            self.crop_size_z_label.setVisible(True)
            self.crop_size_z_lineedit.setVisible(True)
            self.voxel_size_x_label.setVisible(True)
            self.voxel_size_x_lineedit.setVisible(True)
            self.voxel_size_y_label.setVisible(True)
            self.voxel_size_y_lineedit.setVisible(True)
            self.voxel_size_z_label.setVisible(True)
            self.voxel_size_z_lineedit.setVisible(True)

    def _prepare_data_dir(self):
        if self.sender() == self.data_dir_pushbutton:
            self.data_dir = QFileDialog.getExistingDirectory(None, 'Open working directory', '/home/',
                                                             QFileDialog.ShowDirsOnly)
            print("=" * 25)
            print("Data directory chosen is {}".format(self.data_dir))
            if os.path.isdir(self.data_dir):
                self.data_dir_pushbutton.setStyleSheet("border :3px solid green")
        elif self.sender() == self.crops_dir_pushbutton:
            self.crops_dir = QFileDialog.getExistingDirectory(None, 'Open working directory', '/home/',
                                                              QFileDialog.ShowDirsOnly)
            print("=" * 25)
            print("Crops directory chosen is {}".format(self.crops_dir))
            if os.path.isdir(self.crops_dir):
                self.crops_dir_pushbutton.setStyleSheet("border :3px solid green")


    def _preprocess(self):
        if self.data_type_2d_radio_button.isChecked():
            print("=" * 25)
            print("Processing train-test split")
            if os.path.isdir(os.path.join(self.data_dir, '/test')):
                print("=" * 15)
                print("The `test` directory already exists at this location : {}".format(
                    os.path.join(self.data_dir, '/test')))
                print("No splitting of training data was performed")
            else:
                split_train_test(os.path.split(self.data_dir)[0], os.path.split(self.data_dir)[1],
                                 self.train_test_lineedit.text(), subset=float(self.train_test_subset_lineedit.text()))
            print("=" * 25)
            print("Processing train-val split")
            if os.path.isdir(os.path.join(self.data_dir, '/val')):
                print("=" * 15)
                print("The `val` directory already exists at this location : {}".format(
                    os.path.join(self.data_dir, '/val')))
                print("No splitting of training data was performed")
            else:
                split_train_val(os.path.split(self.data_dir)[0], os.path.split(self.data_dir)[1],
                                self.train_val_lineedit.text(), subset=float(self.train_val_subset_lineedit.text()))
            print("=" * 25)
            print("Calculating dataset-specific properties")

            data_properties_dir = get_data_properties(os.path.split(self.data_dir)[0], os.path.split(self.data_dir)[1],
                                                      train_val_name=['train'], test_name=['test'], mode='2d',
                                                      one_hot=self.one_hot_checkbox.isChecked())
            data_properties_dir[
                'data_type'] = '16-bit' if self.image_datatype_combobox.currentText() == '16 bit (u)' else '8-bit'

            with open('data_properties.json', 'w') as outfile:
                json.dump(data_properties_dir, outfile)
                print("=" * 15)
                print("Dataset properies of the `{}` dataset is saved to `data_properties.json`".format(
                    os.path.split(self.data_dir)[1]))
            print("=" * 25)
            print("Generating crops")
            for data_subset in ['train', 'val']:
                image_dir = os.path.join(self.data_dir, data_subset, 'images')
                instance_dir = os.path.join(self.data_dir, data_subset, 'masks')
                image_names = sorted(glob(os.path.join(image_dir, '*.tif')))
                instance_names = sorted(glob(os.path.join(instance_dir, '*.tif')))
                self.preprocess_progress_bar.reset
                for i in tqdm(np.arange(len(image_names))):

                    if self.one_hot_checkbox.isChecked():
                        process_one_hot(image_names[i], instance_names[i],
                                        os.path.join(self.crops_dir, os.path.split(self.data_dir)[1]), data_subset,
                                        int(self.crop_size_x_lineedit.text()), self.center_combobox.currentText(),
                                        one_hot=self.one_hot_checkbox.isChecked())
                    else:
                        process(image_names[i], instance_names[i],
                                os.path.join(self.crops_dir, os.path.split(self.data_dir)[1]), data_subset,
                                int(self.crop_size_x_lineedit.text()), self.center_combobox.currentText(),
                                one_hot=self.one_hot_checkbox.isChecked())
                    self.preprocess_progress_bar.setValue(100 * (i + 1) / len(image_names))
                print("=" * 15)
                print(
                    "Cropping of images, instances and centre_images for data_subset = `{}` done!".format(data_subset))
        elif self.data_type_3d_radio_button.isChecked():
            print("=" * 25)
            print("Processing train-test split")
            if os.path.isdir(os.path.join(self.data_dir, '/test')):
                print("=" * 15)
                print("The `test` directory already exists at this location : {}".format(
                    os.path.join(self.data_dir, '/test')))
                print("No splitting of training data was performed")
            else:
                split_train_test(os.path.split(self.data_dir)[0], os.path.split(self.data_dir)[1],
                                 self.train_test_lineedit.text(), subset=float(self.train_test_subset_lineedit.text()))

            print("=" * 25)
            print("Processing train-val split")
            if os.path.isdir(os.path.join(self.data_dir, '/val')):
                print("=" * 15)
                print("The `val` directory already exists at this location : {}".format(
                    os.path.join(self.data_dir, '/val')))
                print("No splitting of training data was performed")
            else:
                split_train_val(os.path.split(self.data_dir)[0], os.path.split(self.data_dir)[1],
                                self.train_val_lineedit.text(), subset=float(self.train_val_subset_lineedit.text()))
            print("=" * 25)
            print("Calculating dataset-specific properties")

            data_properties_dir = get_data_properties(os.path.split(self.data_dir)[0], os.path.split(self.data_dir)[1],
                                                      train_val_name=['train'], test_name=['test'], mode='3d',
                                                      one_hot=self.one_hot_checkbox.isChecked())
            data_properties_dir[
                'data_type'] = '16-bit' if self.image_datatype_combobox.currentText() == '16 bit (u)' else '8-bit'
            data_properties_dir['pixel_size_x_microns'] = float(self.voxel_size_x_lineedit.text())
            data_properties_dir['pixel_size_y_microns'] = float(self.voxel_size_y_lineedit.text())
            data_properties_dir['pixel_size_z_microns'] = float(self.voxel_size_z_lineedit.text())
            with open('data_properties.json', 'w') as outfile:
                json.dump(data_properties_dir, outfile)
                print("=" * 15)
                print("Dataset properies of the `{}` dataset is saved to `data_properties.json`".format(
                    os.path.split(self.data_dir)[1]))
            print("=" * 25)
            print("Generating crops")
            for data_subset in ['train', 'val']:
                image_dir = os.path.join(self.data_dir, data_subset, 'images')
                instance_dir = os.path.join(self.data_dir, data_subset, 'masks')
                image_names = sorted(glob(os.path.join(image_dir, '*.tif')))
                instance_names = sorted(glob(os.path.join(instance_dir, '*.tif')))
                self.preprocess_progress_bar.reset
                for i in tqdm(np.arange(len(image_names))):
                    process_3d(image_names[i], instance_names[i],
                               os.path.join(self.crops_dir, os.path.split(self.data_dir)[1]), data_subset,
                               crop_size_x=int(self.crop_size_x_lineedit.text()),
                               crop_size_y=int(self.crop_size_y_lineedit.text()),
                               crop_size_z=int(self.crop_size_z_lineedit.text()),
                               center=self.center_combobox.currentText(),
                               anisotropy_factor=data_properties_dir['pixel_size_z_microns'] / data_properties_dir[
                                   'pixel_size_x_microns'],
                               speed_up=3)
                    self.preprocess_progress_bar.setValue(100 * (i + 1) / len(image_names))
                print("=" * 15)
                print(
                    "Cropping of images, instances and centre_images for data_subset = `{}` done!".format(data_subset))


class Train(QScrollArea):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # define scroll
        # https: // www.mfitzp.com / qscrollarea /
        self.widget = QWidget()
        # define components
        logo_path = 'embedseg_napari/resources/embedseg_napari_logo.png'
        self.logo_label = QLabel(f'<h1><img src="{logo_path}">EmbedSeg</h1>')
        self.method_description_label = QLabel(
            '<small>Embedding-based Instance Segmentation <br> in Microscopy.<br>If you are using this in your research please <a href="https://github.com/juglab/EmbedSeg#citation" style="color:gray;">cite us</a>.</small><br><small><tt><a href="https://github.com/juglab/EmbedSeg" style="color:gray;">https://github.com/juglab/EmbedSeg</a></tt></small>')

        self.specify_path = QLabel('<h3>Specify path to crops and center embedding</h3>')
        self.crops_dir_label = QLabel('Data Directory')
        self.crops_dir_pushbutton = QPushButton('Browse')
        self.crops_dir_pushbutton.clicked.connect(self._prepare_data_dir)

        self.center_label = QLabel('Center')
        self.center_combobox = QComboBox(self)

        self.center_combobox.addItem('medoid')
        self.center_combobox.addItem('centroid')
        self.center_combobox.addItem('approximate-medoid')
        self.center_combobox.setMaximumWidth(280)
        self.center_combobox.setEditable(True)
        self.center_combobox.lineEdit().setAlignment(Qt.AlignCenter)
        self.center_combobox.lineEdit().setReadOnly(True)
        # self.center_combobox.setEditable(False)

        self.obtain_properties_label = QLabel('<h3>Obtain properties of the dataset</h3>')
        self.data_properties_label = QLabel('Data Properties')
        self.data_properties_pushbutton = QPushButton('Browse')
        self.data_properties_pushbutton.clicked.connect(self._prepare_data_dir)
        self.data_properties_pushbutton.setMaximumWidth(280)
        self.training_dataset_params_label = QLabel('<h3>Specify training dataset related parameters</h3>')
        self.train_size_label = QLabel('Train Size')
        self.train_size_lineedit = QLineEdit('1200')
        self.train_size_lineedit.setAlignment(Qt.AlignCenter)
        self.train_size_lineedit.setMaximumWidth(280)
        self.train_batch_size_label = QLabel('Train Batch Size')
        self.train_batch_size_lineedit = QLineEdit('1')
        self.train_batch_size_lineedit.setMaximumWidth(280)
        self.train_batch_size_lineedit.setAlignment(Qt.AlignCenter)
        self.train_virtual_batch_size_label = QLabel('Train Virtual Batch Size')
        self.train_virtual_batch_size_lineedit = QLineEdit('1')
        self.train_virtual_batch_size_lineedit.setMaximumWidth(280)
        self.train_virtual_batch_size_lineedit.setAlignment(Qt.AlignCenter)

        self.val_dataset_params_label = QLabel('<h3>Specify validation dataset related parameters</h3>')
        self.val_size_label = QLabel('Val Size')
        self.val_size_lineedit = QLineEdit('800')
        self.val_size_lineedit.setAlignment(Qt.AlignCenter)
        self.val_size_lineedit.setMaximumWidth(280)
        self.val_batch_size_label = QLabel('Val Batch Size')
        self.val_batch_size_lineedit = QLineEdit('1')
        self.val_batch_size_lineedit.setMaximumWidth(280)
        self.val_batch_size_lineedit.setAlignment(Qt.AlignCenter)
        self.val_virtual_batch_size_label = QLabel('Val Virtual Batch Size')
        self.val_virtual_batch_size_lineedit = QLineEdit('1')
        self.val_virtual_batch_size_lineedit.setMaximumWidth(280)
        self.val_virtual_batch_size_lineedit.setAlignment(Qt.AlignCenter)

        self.additional_params_label = QLabel('<h3>Specify additional parameters</h3>')
        self.n_epochs_label = QLabel('Number of Epochs')
        self.n_epochs_lineedit = QLineEdit('200')
        self.n_epochs_lineedit.setAlignment(Qt.AlignCenter)
        self.n_epochs_lineedit.setMaximumWidth(280)
        self.display_checkbox = QCheckBox('Display ?')
        self.display_embedding_checkbox = QCheckBox('Display Embedding ?')
        self.save_dir_label = QLabel('Save Directory')
        self.save_dir_pushbutton = QPushButton('Browse')
        self.save_dir_pushbutton.setMaximumWidth(280)
        self.save_dir_pushbutton.clicked.connect(self._prepare_data_dir)
        self.resume_path_label = QLabel('Resume Path')
        self.resume_path_pushbutton = QPushButton('Browse')
        self.resume_path_pushbutton.clicked.connect(self._prepare_data_dir)
        self.begin_training_label = QLabel('<h3>Begin training</h3>')
        self.begin_training_pushbutton = QPushButton('Begin training')
        self.begin_training_pushbutton.setMaximumWidth(280)
        self.begin_training_pushbutton.clicked.connect(self._start_training_notebook)
        self.stop_training_pushbutton = QPushButton('Stop training')
        self.stop_training_pushbutton.setMaximumWidth(280)

        self.epoch_label = QLabel('Epoch')
        self.epoch_lineedit = QLineEdit('')
        self.epoch_lineedit.setAlignment(Qt.AlignCenter)
        self.epoch_lineedit.setMaximumWidth(280)
        self.epoch_lineedit.setReadOnly(True)

        self.epoch_progress_bar = QProgressBar(self)

        self.train_loss_label = QLabel('Train Loss')
        self.train_loss_lineedit = QLineEdit('')
        self.train_loss_lineedit.setAlignment(Qt.AlignCenter)
        self.train_loss_lineedit.setMaximumWidth(280)
        self.train_loss_lineedit.setReadOnly(True)

        self.train_progress_bar = QProgressBar(self)

        self.val_loss_label = QLabel('Val Loss')
        self.val_loss_lineedit = QLineEdit('')
        self.val_loss_lineedit.setAlignment(Qt.AlignCenter)
        self.val_loss_lineedit.setReadOnly(True)
        self.val_loss_lineedit.setMaximumWidth(280)
        self.val_progress_bar = QProgressBar(self)

        # outer layout
        outer_layout = QVBoxLayout()

        # inner layout 0
        grid_0 = QGridLayout()
        grid_0.addWidget(self.logo_label, 0, 0, 1, 1)
        grid_0.addWidget(self.method_description_label, 0, 1, 1, 1)
        grid_0.setSpacing(10)

        # inner layout 1
        grid_1 = QGridLayout()
        grid_1.addWidget(self.specify_path, 0, 0, 1, 2)
        grid_1.addWidget(self.crops_dir_label, 1, 0)
        grid_1.addWidget(self.crops_dir_pushbutton, 1, 1)
        grid_1.addWidget(self.center_label, 2, 0)
        grid_1.addWidget(self.center_combobox, 2, 1)
        grid_1.setSpacing(10)

        # inner layout 2
        grid_2 = QGridLayout()
        grid_2.addWidget(self.obtain_properties_label, 0, 0, 1, 2)
        grid_2.addWidget(self.data_properties_label, 1, 0)
        grid_2.addWidget(self.data_properties_pushbutton, 1, 1)
        grid_2.setSpacing(10)

        # inner layout 3
        grid_3 = QGridLayout()
        grid_3.addWidget(self.training_dataset_params_label, 0, 0, 1, 2)
        grid_3.addWidget(self.train_size_label, 1, 0)
        grid_3.addWidget(self.train_size_lineedit, 1, 1)
        grid_3.addWidget(self.train_batch_size_label, 2, 0)
        grid_3.addWidget(self.train_batch_size_lineedit, 2, 1)
        grid_3.addWidget(self.train_virtual_batch_size_label, 3, 0)
        grid_3.addWidget(self.train_virtual_batch_size_lineedit, 3, 1)
        grid_3.setSpacing(10)

        # inner layout 4
        grid_4 = QGridLayout()
        grid_4.addWidget(self.val_dataset_params_label, 0, 0, 1, 2)
        grid_4.addWidget(self.val_size_label, 1, 0)
        grid_4.addWidget(self.val_size_lineedit, 1, 1)
        grid_4.addWidget(self.val_batch_size_label, 2, 0)
        grid_4.addWidget(self.val_batch_size_lineedit, 2, 1)
        grid_4.addWidget(self.val_virtual_batch_size_label, 3, 0)
        grid_4.addWidget(self.val_virtual_batch_size_lineedit, 3, 1)
        grid_4.setSpacing(10)

        # inner layout 5
        grid_5 = QGridLayout()
        grid_5.addWidget(self.additional_params_label, 0, 0, 1, 2)
        grid_5.addWidget(self.n_epochs_label, 1, 0)
        grid_5.addWidget(self.n_epochs_lineedit, 1, 1)
        grid_5.addWidget(self.display_checkbox, 2, 0)
        grid_5.addWidget(self.display_embedding_checkbox, 2, 1)
        grid_5.addWidget(self.save_dir_label, 3, 0)
        grid_5.addWidget(self.save_dir_pushbutton, 3, 1)
        grid_5.addWidget(self.resume_path_label, 4, 0)
        grid_5.addWidget(self.resume_path_pushbutton, 4, 1)
        grid_5.setSpacing(10)

        # inner layout 6
        grid_6 = QGridLayout()
        grid_6.addWidget(self.begin_training_label, 0, 0, 1, 2)
        grid_6.addWidget(self.begin_training_pushbutton, 1, 0)
        grid_6.addWidget(self.stop_training_pushbutton, 1, 1)
        grid_6.addWidget(self.epoch_label, 2, 0)
        grid_6.addWidget(self.epoch_lineedit, 2, 1)
        grid_6.addWidget(self.epoch_progress_bar, 3, 0, 1, 2)
        grid_6.addWidget(self.train_loss_label, 4, 0)
        grid_6.addWidget(self.train_loss_lineedit, 4, 1)
        grid_6.addWidget(self.train_progress_bar, 5, 0, 1, 2)
        grid_6.addWidget(self.val_loss_label, 6, 0)
        grid_6.addWidget(self.val_loss_lineedit, 6, 1)
        grid_6.addWidget(self.val_progress_bar, 7, 0, 1, 2)
        grid_6.setSpacing(10)

        outer_layout.addLayout(grid_0)
        outer_layout.addLayout(grid_1)
        outer_layout.addLayout(grid_2)
        outer_layout.addLayout(grid_3)
        outer_layout.addLayout(grid_4)
        outer_layout.addLayout(grid_5)
        outer_layout.addLayout(grid_6)

        outer_layout.setSpacing(20)
        self.widget.setLayout(outer_layout)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setWidget(self.widget)
        self.setFixedWidth(560)

    def _prepare_data_dir(self):
        if self.sender() == self.crops_dir_pushbutton:
            self.crops_dir = QFileDialog.getExistingDirectory(None, 'Open working directory', '/home/',
                                                              QFileDialog.ShowDirsOnly)
            print("=" * 25)
            print("Crops directory chosen is {}".format(self.crops_dir))
            if os.path.isdir(self.crops_dir):
                self.crops_dir_pushbutton.setStyleSheet("border :3px solid green")
        elif self.sender() == self.data_properties_pushbutton:
            self.data_properties_file_name = QFileDialog.getOpenFileName(None, 'Select data properties file', '/home/')[
                0]  # select first argument
            print("=" * 25)
            print("Data properties filename chosen is {}".format(self.data_properties_file_name))
            if os.path.isfile(self.data_properties_file_name):
                self.data_properties_pushbutton.setStyleSheet("border :3px solid green")
        elif self.sender() == self.save_dir_pushbutton:
            self.save_dir = QFileDialog.getExistingDirectory(None, 'Open directory to save', '/home/',
                                                             QFileDialog.ShowDirsOnly)
            print("=" * 25)
            print("Save directory chosen is {}".format(self.save_dir))
            if os.path.isdir(self.save_dir):
                self.save_dir_pushbutton.setStyleSheet("border :3px solid green")
        elif self.sender() == self.resume_path_pushbutton:
            self.resume_path = QFileDialog.getOpenFileName(None, 'Select path to model weights', '/home/')[
                0]  # select first argument
            print("=" * 25)
            print("Model weights path filename chosen is {}".format(self.resume_path))
            if os.path.isfile(self.resume_path):
                self.resume_path_pushbutton.setStyleSheet("border :3px solid green")
    def _start_training_notebook(self):
        # load data properties
        if os.path.isfile(self.data_properties_file_name):
            with open(self.data_properties_file_name) as json_file:
                data = json.load(json_file)
                if (data['n_z'] is None):
                    one_hot, data_type, foreground_weight, n_y, n_x = \
                        data['one_hot'], data['data_type'], float(data['foreground_weight']), \
                        int(data['n_y']), int(data['n_x'])
                    n_z = None
                    pixel_size_z_microns = 1.0
                    pixel_size_x_microns = 1.0
                else:
                    one_hot, data_type, foreground_weight, n_z, n_y, n_x, pixel_size_z_microns, pixel_size_x_microns = \
                        data['one_hot'], data['data_type'], float(data['foreground_weight']), int(data['n_z']), \
                        int(data['n_y']), int(data['n_x']), float(data['pixel_size_z_microns']), float(
                            data['pixel_size_x_microns'])

        normalization_factor = 65535 if data_type == '16-bit' else 255
        # create train dictionary
        if n_z is None:
            train_dataset_dict = create_dataset_dict(data_dir=os.path.split(self.crops_dir)[0],
                                                     project_name=os.path.split(self.crops_dir)[1],
                                                     center=self.center_combobox.currentText(),
                                                     size=int(self.train_size_lineedit.text()),
                                                     batch_size=int(self.train_batch_size_lineedit.text()),
                                                     virtual_batch_multiplier=int(
                                                         self.train_virtual_batch_size_lineedit.text()),
                                                     normalization_factor=normalization_factor,
                                                     one_hot=one_hot,
                                                     type='train')

            # create val dictionary
            val_dataset_dict = create_dataset_dict(data_dir=os.path.split(self.crops_dir)[0],
                                                   project_name=os.path.split(self.crops_dir)[1],
                                                   center=self.center_combobox.currentText(),
                                                   size=int(self.val_size_lineedit.text()),
                                                   batch_size=int(self.val_batch_size_lineedit.text()),
                                                   virtual_batch_multiplier=int(
                                                       self.val_virtual_batch_size_lineedit.text()),
                                                   normalization_factor=normalization_factor,
                                                   one_hot=one_hot,
                                                   type='val')
        else:
            train_dataset_dict = create_dataset_dict(data_dir=os.path.split(self.crops_dir)[0],
                                                     project_name=os.path.split(self.crops_dir)[1],
                                                     center=self.center_combobox.currentText(),
                                                     size=int(self.train_size_lineedit.text()),
                                                     batch_size=int(self.train_batch_size_lineedit.text()),
                                                     virtual_batch_multiplier=int(
                                                         self.train_virtual_batch_size_lineedit.text()),
                                                     normalization_factor=normalization_factor,
                                                     one_hot=one_hot,
                                                     type='train', name='3d')
            val_dataset_dict = create_dataset_dict(data_dir=os.path.split(self.crops_dir)[0],
                                                   project_name=os.path.split(self.crops_dir)[1],
                                                   center=self.center_combobox.currentText(),
                                                   size=int(self.val_size_lineedit.text()),
                                                   batch_size=int(self.val_batch_size_lineedit.text()),
                                                   virtual_batch_multiplier=int(
                                                       self.val_virtual_batch_size_lineedit.text()),
                                                   normalization_factor=normalization_factor,
                                                   one_hot=one_hot,
                                                   type='val', name='3d')
        # create model dictionary
        if n_z is None:
            model_dict = create_model_dict(input_channels=1, num_classes=[4, 1], name='2d')
        else:
            model_dict = create_model_dict(input_channels=1, num_classes=[6, 1], name='3d')

        # create loss dictionary
        if n_z is None:
            loss_dict = create_loss_dict(n_sigma=2, foreground_weight=foreground_weight)
        else:
            loss_dict = create_loss_dict(n_sigma=3, foreground_weight=foreground_weight)

        if not hasattr(self, 'resume_path'):
            self.resume_path = None

        # create configs dictionary
        configs = create_configs(n_epochs=int(self.n_epochs_lineedit.text()),
                                 one_hot=one_hot,
                                 display=self.display_checkbox.isChecked(),
                                 display_embedding=self.display_embedding_checkbox.isChecked(),
                                 resume_path=self.resume_path,
                                 save_dir=self.save_dir,
                                 n_z=n_z,
                                 n_y=n_y,
                                 n_x=n_x,
                                 anisotropy_factor=pixel_size_z_microns / pixel_size_x_microns,
                                 display_zslice=16)

        new_cmap = np.load('embedseg_napari/cmaps/cmap_60.npy')
        new_cmap = ListedColormap(new_cmap)
        self._begin_train_val(train_dataset_dict, val_dataset_dict, model_dict, loss_dict, configs, color_map=new_cmap)

    def _begin_train_val(self, train_dataset_dict, val_dataset_dict, model_dict, loss_dict, configs, color_map='magma'):
        if configs['save']:
            if not os.path.exists(configs['save_dir']):
                os.makedirs(configs['save_dir'])

        # +++++++++++++++++++
        ## Originally
        # if configs['display']:
        #     plt.ion()
        # else:
        #     plt.ioff()
        #     plt.switch_backend("agg")
        # +++++++++++++++++

        # set device
        device = torch.device("cuda:0" if configs['cuda'] else "cpu")

        # define global variables
        global train_dataset_it, val_dataset_it, model, criterion, optimizer, visualizer, cluster

        # train dataloader

        train_dataset = get_dataset(train_dataset_dict['name'], train_dataset_dict['kwargs'])
        train_dataset_it = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataset_dict['batch_size'],
                                                       shuffle=True, drop_last=True,
                                                       num_workers=train_dataset_dict['workers'],
                                                       pin_memory=True if configs['cuda'] else False)

        # val dataloader
        val_dataset = get_dataset(val_dataset_dict['name'], val_dataset_dict['kwargs'])
        val_dataset_it = torch.utils.data.DataLoader(val_dataset, batch_size=val_dataset_dict['batch_size'],
                                                     shuffle=True,
                                                     drop_last=True, num_workers=val_dataset_dict['workers'],
                                                     pin_memory=True if configs['cuda'] else False)

        # set model
        model = get_model(model_dict['name'], model_dict['kwargs'])
        model.init_output(loss_dict['lossOpts']['n_sigma'])
        model = torch.nn.DataParallel(model).to(device)

        if (configs['grid_z'] is None):
            criterion = get_loss(grid_z=None, grid_y=configs['grid_y'], grid_x=configs['grid_x'], pixel_z=None,
                                 pixel_y=configs['pixel_y'], pixel_x=configs['pixel_x'],
                                 one_hot=configs['one_hot'], loss_opts=loss_dict['lossOpts'])
        else:
            criterion = get_loss(configs['grid_z'], configs['grid_y'], configs['grid_x'],
                                 configs['pixel_z'], configs['pixel_y'], configs['pixel_x'],
                                 configs['one_hot'], loss_dict['lossOpts'])
        criterion = torch.nn.DataParallel(criterion).to(device)

        # set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=configs['train_lr'], weight_decay=1e-4)

        def lambda_(epoch):
            return pow((1 - ((epoch) / 200)), 0.9)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_, )

        if (configs['grid_z'] is None):
            # clustering
            cluster = Cluster(configs['grid_y'], configs['grid_x'], configs['pixel_y'], configs['pixel_x'],
                              configs['one_hot'])
        else:
            # clustering
            cluster = Cluster_3d(configs['grid_z'], configs['grid_y'], configs['grid_x'], configs['pixel_z'],
                                 configs['pixel_y'],
                                 configs['pixel_x'], configs['one_hot'])
        # +++++++++++++++++++
        # Visualizer
        # visualizer = Visualizer(('image', 'groundtruth', 'prediction', 'center'), color_map)  # 5 keys
        # +++++++++++++++++++

        # Logger
        self.logger = Logger(('train', 'val', 'iou'), 'loss')

        # resume
        start_epoch = 0
        self.best_iou = 0
        if configs['resume_path'] is not None and os.path.exists(configs['resume_path']):
            print('Resuming model from {}'.format(configs['resume_path']))
            state = torch.load(configs['resume_path'])
            start_epoch = state['epoch'] + 1
            self.best_iou = state['best_iou']
            model.load_state_dict(state['model_state_dict'], strict=True)
            optimizer.load_state_dict(state['optim_state_dict'])
            self.logger.data = state['logger_data']

        self.current_epoch = start_epoch
        self.max_epochs = int(configs['n_epochs'])
        self.display = configs['display']
        self.display_embedding = configs['display_embedding']
        self.display_it = configs['display_it']
        self.one_hot = configs['one_hot']
        self.n_sigma = loss_dict['lossOpts']['n_sigma']
        self.grid_x = configs['grid_x']
        self.grid_y = configs['grid_y']
        self.pixel_x = configs['pixel_x']
        self.pixel_y = configs['pixel_y']
        self.args = loss_dict['lossW']
        self.color_map = color_map
        self.save_checkpoint_frequency = configs['save_checkpoint_frequency']
        self.save = configs['save']

        if (configs['grid_z'] is None):  # 2d training
            self.worker = self._train_vanilla()
            self.worker.yielded.connect(self.show_intermediate_result)
            self.worker.returned.connect(self.update_train_loss)
            self.worker.finished.connect(self.restart)
            self.stop_training_pushbutton.clicked.connect(self.worker.quit)
            self.worker.finished.connect(self.stop_training_pushbutton.clicked.disconnect)
            self.train_mode = True
            self.type = '2d'
            self.worker.start()
        else:  # 3d training
            self.worker = self._train_vanilla_3d()
            self.worker.yielded.connect(self.show_intermediate_result)
            self.worker.returned.connect(self.update_train_loss)
            self.worker.finished.connect(self.restart)
            self.stop_training_pushbutton.clicked.connect(self.worker.quit)
            self.worker.finished.connect(self.stop_training_pushbutton.clicked.disconnect)
            self.train_mode = True
            self.type = '3d'
            self.worker.start()

    @thread_worker
    def _train_vanilla(self):
        # define meters
        loss_meter = AverageMeter()
        # put model into training mode
        model.train()

        for param_group in optimizer.param_groups:
            print('learning rate: {}'.format(param_group['lr']))

        for i, sample in enumerate(tqdm(train_dataset_it)):

            im = sample['image']
            instances = sample['instance'].squeeze(1)  # 1YX (not one-hot) or 1DYX (one-hot)
            class_labels = sample['label'].squeeze(1)  # 1YX
            center_images = sample['center_image'].squeeze(1)  # 1YX
            output = model(im)  # B 5 Y X
            loss = criterion(output, instances, class_labels, center_images, **self.args)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            if self.display and i % self.display_it == 0:
                with torch.no_grad():
                    predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=self.n_sigma)
                    if self.one_hot:
                        instance = self.invert_one_hot(instances[0].cpu().detach().numpy())
                        yield np.round(100 * (i + 1) / len(train_dataset_it)).astype(int), self.color_map, im[
                            0, 0].cpu().detach().numpy(), instance, predictions.cpu().detach().numpy()
                    else:
                        yield np.round(100 * (i + 1) / len(train_dataset_it)).astype(int), self.color_map, im[
                            0].cpu().detach().numpy(), instances[
                                  0].cpu().detach().numpy(), predictions.cpu().detach().numpy()  # TODO im[0] or im[0,0]
            else:
                yield np.round(100 * (i + 1) / len(train_dataset_it)).astype(int), self.color_map
        return loss_meter.avg

    @thread_worker
    def _val_vanilla(self):
        # define meters
        loss_meter, iou_meter = AverageMeter(), AverageMeter()
        # put model into eval mode
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(val_dataset_it)):
                im = sample['image']
                instances = sample['instance'].squeeze(1)
                class_labels = sample['label'].squeeze(1)
                center_images = sample['center_image'].squeeze(1)
                output = model(im)
                loss = criterion(output, instances, class_labels, center_images, **self.args, iou=True,
                                 iou_meter=iou_meter)
                loss = loss.mean()
                if self.display and i % self.display_it == 0:
                    with torch.no_grad():
                        predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=self.n_sigma)
                        if self.one_hot:
                            instance = self.invert_one_hot(instances[0].cpu().detach().numpy())
                            yield np.round(100 * (i + 1) / len(val_dataset_it)).astype(int), self.color_map, im[
                                0, 0].cpu().detach().numpy(), instance, predictions.cpu().detach().numpy()
                        else:
                            yield np.round(100 * (i + 1) / len(val_dataset_it)).astype(int), self.color_map, im[0], \
                                  instances[0].cpu().detach().numpy(), predictions.cpu().detach().numpy()

                loss_meter.update(loss.item())

        return loss_meter.avg, iou_meter.avg

    @thread_worker
    def _train_vanilla_3d(self):
        # define meters
        loss_meter = AverageMeter()
        # put model into training mode
        model.train()

        for param_group in optimizer.param_groups:
            print('learning rate: {}'.format(param_group['lr']))

        for i, sample in enumerate(tqdm(train_dataset_it)):

            im = sample['image']  # BCZYX
            instances = sample['instance'].squeeze(1)  # BZYX
            class_labels = sample['label'].squeeze(1)  # BZYX
            center_images = sample['center_image'].squeeze(1)  # BZYX
            output = model(im)  # B 7 Z Y X
            loss = criterion(output, instances, class_labels, center_images, **self.args)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

            if self.display and i % self.display_it == 0:
                with torch.no_grad():
                    zslice = im.shape[2] // 2
                    predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=self.n_sigma)
                    # TODO --> Add support for 3d onehot
                    yield np.round(100 * (i + 1) / len(train_dataset_it)).astype(int), self.color_map, im[
                        0, 0, zslice].cpu().detach().numpy(), instances[0, zslice].cpu().detach().numpy(), predictions[
                              zslice, ...].cpu().detach().numpy()
            else:
                yield np.round(100 * (i + 1) / len(train_dataset_it)).astype(int), self.color_map

        return loss_meter.avg

    @thread_worker
    def _val_vanilla_3d(self):
        # define meters
        loss_meter, iou_meter = AverageMeter(), AverageMeter()
        # put model into eval mode
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(val_dataset_it)):
                im = sample['image']  # BCZYX
                instances = sample['instance'].squeeze(1)  # BZYX
                class_labels = sample['label'].squeeze(1)  # BZYX
                center_images = sample['center_image'].squeeze(1)  # BZYX
                output = model(im)
                loss = criterion(output, instances, class_labels, center_images, **self.args, iou=True,
                                 iou_meter=iou_meter)
                loss = loss.mean()
                loss_meter.update(loss.item())
                if self.display and i % self.display_it == 0:
                    with torch.no_grad():
                        zslice = im.shape[2] // 2
                        predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=self.n_sigma)
                        yield np.round(100 * (i + 1) / len(val_dataset_it)).astype(int), self.color_map, im[
                            0, 0, zslice].cpu().detach().numpy(), instances[0, zslice].cpu().detach().numpy(), \
                              predictions[zslice, ...].cpu().detach().numpy()
                else:
                    yield np.round(100 * (i + 1) / len(val_dataset_it)).astype(int), self.color_map

        return loss_meter.avg, iou_meter.avg

    def invert_one_hot(self, image):
        instance = np.zeros((image.shape[1], image.shape[2]), dtype="uint16")
        for z in range(image.shape[0]):
            instance = np.where(image[z] > 0, instance + z + 1, instance)  # TODO - not completely accurate!
        return instance

    def save_checkpoint(self, state, is_best, epoch, save_dir, save_checkpoint_frequency, name='checkpoint.pth'):
        print('=> saving checkpoint')
        file_name = os.path.join(save_dir, name)
        torch.save(state, file_name)
        if (save_checkpoint_frequency is not None):
            if (epoch % int(save_checkpoint_frequency) == 0):
                file_name2 = os.path.join(save_dir, str(epoch) + "_" + name)
                torch.save(state, file_name2)
        if is_best:
            shutil.copyfile(file_name, os.path.join(
                save_dir, 'best_iou_model.pth'))

    def show_intermediate_result(self, yielded_data):
        if self.train_mode:
            self.train_progress_bar.setValue(yielded_data[0])
        else:
            self.val_progress_bar.setValue(yielded_data[0])
        if (len(yielded_data) > 2):
            fig = plt.figure(constrained_layout=True, figsize=(20, 20))
            spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
            f_ax1 = fig.add_subplot(spec[0, 0])
            f_ax2 = fig.add_subplot(spec[0, 1])
            f_ax4 = fig.add_subplot(spec[1, 1])
            if yielded_data[2].ndim > 2:
                f_ax1.imshow(np.moveaxis(yielded_data[2], 0, -1), cmap='magma')  # channel dimension should be last
            else:
                f_ax1.imshow(yielded_data[2], cmap='magma')
            f_ax2.imshow(yielded_data[3], cmap=yielded_data[1])
            f_ax4.imshow(yielded_data[4], cmap=yielded_data[1])
            f_ax1.tick_params(
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off
            f_ax1.tick_params(
                axis='x',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                top=False,  # ticks along the bottom edge are off
                bottom=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            f_ax2.tick_params(
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off
            f_ax2.tick_params(
                axis='x',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                top=False,  # ticks along the bottom edge are off
                bottom=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off

            f_ax4.tick_params(
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelleft=False)  # labels along the bottom edge are off
            f_ax4.tick_params(
                axis='x',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                top=False,  # ticks along the bottom edge are off
                bottom=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off

            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            try:
                self.viewer.layers['intermediate_results'].data = data
            except KeyError:
                self.viewer.add_image(data, name='intermediate_results')

    def update_train_loss(self, train_loss):
        print("=" * 25)
        print('===> train loss @ epoch {}: {:.2f}'.format(self.current_epoch, train_loss))
        self.logger.add('train', train_loss)

    def update_val_loss(self, val_loss_iou):

        print("=" * 25)
        print('===> val loss: {:.2f}, val iou: {:.2f}'.format(val_loss_iou[0], val_loss_iou[1]))
        self.logger.add('val', val_loss_iou[0])
        self.logger.add('iou', val_loss_iou[1])
        self.logger.plot(save=self.save, save_dir=self.save_dir)
        is_best = val_loss_iou[1] > self.best_iou
        self.best_iou = max(val_loss_iou[1], self.best_iou)

        if self.save:
            state = {
                'epoch': self.current_epoch,
                'best_iou': self.best_iou,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'logger_data': self.logger.data,
            }
        self.save_checkpoint(state, is_best, self.current_epoch, save_dir=self.save_dir,
                             save_checkpoint_frequency=self.save_checkpoint_frequency)
        self.current_epoch += 1

    def restart(self):
        if self.current_epoch < self.max_epochs:
            if self.type == '2d':
                if self.train_mode:  # previously train

                    self.train_mode = False  # switch to val
                    self.worker = self._val_vanilla()
                    self.worker.yielded.connect(self.show_intermediate_result)
                    self.worker.returned.connect(self.update_val_loss)
                    self.worker.finished.connect(self.restart)
                    self.stop_training_pushbutton.clicked.connect(self.worker.quit)
                    self.worker.finished.connect(self.stop_training_pushbutton.clicked.disconnect)
                    self.worker.start()
                elif not self.train_mode:  # previously val
                    self.train_mode = True  # switch to train
                    self.worker = self._train_vanilla()
                    self.worker.yielded.connect(self.show_intermediate_result)
                    self.worker.returned.connect(self.update_train_loss)
                    self.worker.finished.connect(self.restart)
                    self.stop_training_pushbutton.clicked.connect(self.worker.quit)
                    self.worker.finished.connect(self.stop_training_pushbutton.clicked.disconnect)
                    self.worker.start()
                    self.train_progress_bar.setValue(0)
                    self.val_progress_bar.setValue(0)
                    self.epoch_progress_bar.setValue(np.round(self.current_epoch / self.max_epochs * 100).astype(int))
            elif self.type == '3d':
                if self.train_mode:  # previously train
                    self.train_mode = False  # switch to val
                    self.worker = self._val_vanilla_3d()
                    self.worker.yielded.connect(self.show_intermediate_result)
                    self.worker.returned.connect(self.update_val_loss)
                    self.worker.finished.connect(self.restart)
                    self.stop_training_pushbutton.clicked.connect(self.worker.quit)
                    self.worker.finished.connect(self.stop_training_pushbutton.clicked.disconnect)
                    self.worker.start()
                elif not self.train_mode:  # previously val
                    self.train_mode = True  # switch to train
                    self.worker = self._train_vanilla_3d()
                    self.worker.yielded.connect(self.show_intermediate_result)
                    self.worker.returned.connect(self.update_train_loss)
                    self.worker.finished.connect(self.restart)
                    self.stop_training_pushbutton.clicked.connect(self.worker.quit)
                    self.worker.finished.connect(self.stop_training_pushbutton.clicked.disconnect)
                    self.worker.start()
                    self.train_progress_bar.setValue(0)
                    self.val_progress_bar.setValue(0)
                    self.epoch_progress_bar.setValue(np.round(self.current_epoch / self.max_epochs * 100).astype(int))

            self.scheduler.step()


class Predict(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # # define components
        logo_path = 'embedseg_napari/resources/embedseg_napari_logo.png'
        self.logo_label = QLabel(f'<h1><img src="{logo_path}">EmbedSeg</h1>')
        self.method_description_label = QLabel(
            '<small>Embedding-based Instance Segmentation <br> in Microscopy.<br>If you are using this in your research please <a href="https://github.com/juglab/EmbedSeg#citation" style="color:gray;">cite us</a>.</small><br><small><tt><a href="https://github.com/juglab/EmbedSeg" style="color:gray;">https://github.com/juglab/EmbedSeg</a></tt></small>')

        self.specify_path_images_label = QLabel('<h3>Specify path to the evaluation images </h3>')
        self.data_dir_label = QLabel('Data Directory')
        self.data_dir_pushbutton = QPushButton('Browse')
        self.data_dir_pushbutton.setMaximumWidth(280)
        self.data_dir_pushbutton.clicked.connect(self._prepare_data_dir)

        self.specify_path_model_label = QLabel('<h3>Specify path to the model weights and data properties </h3>')
        self.checkpoint_path_label = QLabel('Path to model weights')
        self.checkpoint_pushbutton = QPushButton('Browse')
        self.checkpoint_pushbutton.setMaximumWidth(280)
        self.checkpoint_pushbutton.clicked.connect(self._prepare_data_dir)

        self.data_properties_path_label = QLabel('Path to data properties')
        self.data_properties_pushbutton = QPushButton('Browse')
        self.data_properties_pushbutton.clicked.connect(self._prepare_data_dir)

        self.evaluation_params_path_label = QLabel('<h3>Specify evaluation parameters</h3>')
        self.tta_checkbox = QCheckBox('Test Time Augmentation ?')
        self.ap_val_label = QLabel('Average Precision Label')
        self.ap_val_lineedit = QLineEdit('0.5')
        self.ap_val_lineedit.setAlignment(Qt.AlignCenter)
        self.ap_val_lineedit.setMaximumWidth(280)
        self.seed_thresh_label = QLabel('Seediness Threshold')
        self.seed_thresh_lineedit = QLineEdit('0.90')
        self.seed_thresh_lineedit.setAlignment(Qt.AlignCenter)
        self.seed_thresh_lineedit.setMaximumWidth(280)
        self.save_images_checkbox = QCheckBox('Save Images ?')
        self.save_results_checkbox = QCheckBox('Save Results ?')
        self.save_images_label = QLabel('Path to Results')
        self.save_images_pushbutton = QPushButton('Browse')
        self.save_images_pushbutton.clicked.connect(self._prepare_data_dir)
        self.begin_evaluating_label = QLabel('<h3>Begin evaluating</h3>')
        self.begin_evaluating_pushbutton = QPushButton('Predict')
        self.begin_evaluating_pushbutton.setMaximumWidth(280)
        self.begin_evaluating_pushbutton.clicked.connect(self._start_testing_notebook)
        self.begin_evaluating_progress_bar = QProgressBar(self)
        self.begin_evaluating_progress_bar.setMaximumWidth(280)

        # outer layout
        outer_layout = QVBoxLayout()

        # inner layout 0
        grid_0 = QGridLayout()
        grid_0.addWidget(self.logo_label, 0, 0, 1, 1)
        grid_0.addWidget(self.method_description_label, 0, 1, 1, 1)
        grid_0.setSpacing(10)

        grid_1 = QGridLayout()
        grid_1.addWidget(self.specify_path_images_label, 0, 0, 1, 2)
        grid_1.addWidget(self.data_dir_label, 1, 0)
        grid_1.addWidget(self.data_dir_pushbutton, 1, 1)

        grid_2 = QGridLayout()
        grid_2.addWidget(self.specify_path_model_label, 0, 0, 1, 2)
        grid_2.addWidget(self.checkpoint_path_label, 1, 0)
        grid_2.addWidget(self.checkpoint_pushbutton, 1, 1)
        grid_2.addWidget(self.data_properties_path_label, 2, 0)
        grid_2.addWidget(self.data_properties_pushbutton, 2, 1)

        grid_3 = QGridLayout()
        grid_3.addWidget(self.evaluation_params_path_label, 0, 0, 1, 2)
        grid_3.addWidget(self.tta_checkbox, 1, 0)
        grid_3.addWidget(self.ap_val_label, 2, 0)
        grid_3.addWidget(self.ap_val_lineedit, 2, 1)
        grid_3.addWidget(self.seed_thresh_label, 3, 0)
        grid_3.addWidget(self.seed_thresh_lineedit, 3, 1)
        grid_3.addWidget(self.save_images_checkbox, 4, 0)
        grid_3.addWidget(self.save_results_checkbox, 4, 1)
        grid_3.addWidget(self.save_images_label, 5, 0)
        grid_3.addWidget(self.save_images_pushbutton, 5, 1)

        grid_4 = QGridLayout()
        grid_4.addWidget(self.begin_evaluating_label, 0, 0, 1, 2)
        grid_4.addWidget(self.begin_evaluating_pushbutton, 1, 0)

        grid_4.addWidget(self.begin_evaluating_progress_bar, 1, 1)

        outer_layout.addLayout(grid_0)
        outer_layout.addLayout(grid_1)
        outer_layout.addLayout(grid_2)
        outer_layout.addLayout(grid_3)
        outer_layout.addLayout(grid_4)
        outer_layout.setSpacing(20)
        self.setLayout(outer_layout)
        self.setFixedWidth(560)

    def _prepare_data_dir(self):
        if self.sender() == self.data_dir_pushbutton:
            self.test_dir = QFileDialog.getExistingDirectory(None, 'Open directory containing evaluation images',
                                                             '/home/',
                                                             QFileDialog.ShowDirsOnly)
            print("=" * 25)
            print("Test images directory chosen is {}".format(self.test_dir))
            if os.path.isdir(self.test_dir):
                self.data_dir_pushbutton.setStyleSheet("border :3px solid green")
        elif self.sender() == self.save_images_pushbutton:
            self.save_dir = QFileDialog.getExistingDirectory(None, 'Path to directory containing results',
                                                             '/home/',
                                                             QFileDialog.ShowDirsOnly)
            print("=" * 25)
            print("Instance segmentation results directory chosen is {}".format(self.save_dir))
            if os.path.isdir(self.save_dir):
                self.save_images_pushbutton.setStyleSheet("border :3px solid green")
        elif self.sender() == self.checkpoint_pushbutton:
            self.checkpoint_path = QFileDialog.getOpenFileName(None, 'Select model weights file', '/home/')[
                0]  # select first argument
            print("=" * 25)
            print("Model weights filename chosen is {}".format(self.checkpoint_path))
            if os.path.isfile(self.checkpoint_path):
                self.checkpoint_pushbutton.setStyleSheet("border :3px solid green")
        elif self.sender() == self.data_properties_pushbutton:
            self.data_properties_file_name = QFileDialog.getOpenFileName(None, 'Select data properties file', '/home/')[
                0]  # select first argument
            print("=" * 25)
            print("Data properties filename chosen is {}".format(self.data_properties_file_name))
            if os.path.isfile(self.data_properties_file_name):
                self.data_properties_pushbutton.setStyleSheet("border :3px solid green")


    def _start_testing_notebook(self):
        if os.path.isfile(self.data_properties_file_name):
            with open(self.data_properties_file_name) as json_file:
                data = json.load(json_file)
                if data['n_z'] is None:  # 2d
                    one_hot, data_type, min_object_size, n_y, n_x, avg_bg = data['one_hot'], data['data_type'], int(
                        data['min_object_size']), int(data['n_y']), int(data['n_x']), float(
                        data['avg_background_intensity'])

                    normalization_factor = 65535 if data_type == '16-bit' else 255

                    test_configs = create_test_configs_dict(data_dir=os.path.split(self.test_dir)[0],
                                                            checkpoint_path=self.checkpoint_path,
                                                            tta=self.tta_checkbox.isChecked(),
                                                            ap_val=float(self.ap_val_lineedit.text()),
                                                            seed_thresh=float(self.seed_thresh_lineedit.text()),
                                                            min_object_size=min_object_size,
                                                            save_images=self.save_images_checkbox.isChecked(),
                                                            save_results=self.save_results_checkbox.isChecked(),
                                                            save_dir=self.save_dir,  # TODO
                                                            normalization_factor=normalization_factor,
                                                            one_hot=one_hot,
                                                            n_y=n_y,
                                                            n_x=n_x, name='2d')

                    self._begin_evaluating(test_configs, verbose=False, avg_bg=avg_bg / normalization_factor)
                else:  # 3d
                    one_hot = data['one_hot']
                    data_type = data['data_type']
                    min_object_size = int(data['min_object_size'])
                    foreground_weight = float(data['foreground_weight'])
                    n_z = int(data['n_z'])
                    n_y = int(data['n_y'])
                    n_x = int(data['n_x'])
                    pixel_size_z_microns = float(data['pixel_size_z_microns'])
                    pixel_size_y_microns = float(data['pixel_size_y_microns'])
                    pixel_size_x_microns = float(data['pixel_size_x_microns'])
                    avg_background_intensity = float(data['avg_background_intensity'])
                    normalization_factor = 65535 if data_type == '16-bit' else 255
                    if (data['mask_start_x'] is None):
                        mask_start_x = None
                        mask_start_y = None
                        mask_start_z = None
                        mask_end_x = None
                        mask_end_y = None
                        mask_end_z = None
                        mask_intensity = None
                    else:
                        mask_start_x = int(data['mask_start_x'])
                        mask_start_y = int(data['mask_start_y'])
                        mask_start_z = int(data['mask_start_z'])
                        mask_end_x = int(data['mask_end_x'])
                        mask_end_y = int(data['mask_end_y'])
                        mask_end_z = int(data['mask_end_z'])
                        mask_intensity = avg_background_intensity / normalization_factor

                    test_configs = create_test_configs_dict(data_dir=os.path.split(self.test_dir)[0],
                                                            checkpoint_path=self.checkpoint_path,
                                                            tta=self.tta_checkbox.isChecked(),
                                                            ap_val=float(self.ap_val_lineedit.text()),
                                                            seed_thresh=float(self.seed_thresh_lineedit.text()),
                                                            min_object_size=min_object_size,
                                                            save_images=self.save_images_checkbox.isChecked(),
                                                            save_results=self.save_results_checkbox.isChecked(),
                                                            save_dir=self.save_dir,  # TODO
                                                            normalization_factor=normalization_factor,
                                                            one_hot=one_hot,
                                                            n_z=n_z,
                                                            n_y=n_y,
                                                            n_x=n_x,
                                                            anisotropy_factor=pixel_size_z_microns / pixel_size_x_microns,
                                                            name='3d')

                    self._begin_evaluating(test_configs, verbose=False,
                                           mask_region=[[mask_start_z, mask_start_y, mask_start_x],
                                                        [mask_end_z, mask_end_y, mask_end_x]],
                                           mask_intensity=mask_intensity,
                                           avg_bg=avg_background_intensity / normalization_factor)

    def _begin_evaluating(self, test_configs, verbose=True, mask_region=None, mask_intensity=None, avg_bg=None):
        global n_sigma, ap_val, min_mask_sum, min_unclustered_sum, min_object_size
        global tta, seed_thresh, model, dataset_it, save_images, save_results, save_dir

        n_sigma = test_configs['n_sigma']
        ap_val = test_configs['ap_val']
        min_mask_sum = test_configs['min_mask_sum']
        min_unclustered_sum = test_configs['min_unclustered_sum']
        min_object_size = test_configs['min_object_size']
        tta = test_configs['tta']
        seed_thresh = test_configs['seed_thresh']
        save_images = test_configs['save_images']
        save_results = test_configs['save_results']
        save_dir = test_configs['save_dir']

        # set device
        device = torch.device("cuda:0" if test_configs['cuda'] else "cpu")

        # dataloader
        dataset = get_dataset(test_configs['dataset']['name'], test_configs['dataset']['kwargs'])
        dataset_it = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4,
                                                 pin_memory=True if test_configs['cuda'] else False)

        # load model
        model = get_model(test_configs['model']['name'], test_configs['model']['kwargs'])
        model = torch.nn.DataParallel(model).to(device)

        # load snapshot
        if os.path.exists(test_configs['checkpoint_path']):
            state = torch.load(test_configs['checkpoint_path'])
            model.load_state_dict(state['model_state_dict'], strict=True)
        else:
            assert (False, 'checkpoint_path {} does not exist!'.format(test_configs['checkpoint_path']))

        self.verbose = verbose
        self.grid_x = test_configs['grid_x']
        self.grid_y = test_configs['grid_y']
        if test_configs['grid_z'] is None:
            self.grid_z = None
        else:
            self.grid_z = test_configs['grid_z']

        self.pixel_x = test_configs['pixel_x']
        self.pixel_y = test_configs['pixel_y']
        if test_configs['pixel_z'] is None:
            self.pixel_z = None
        else:
            self.pixel_z = test_configs['pixel_z']

        self.one_hot = test_configs['dataset']['kwargs']['one_hot']
        self.avg_bg = avg_bg
        self.n_sigma = n_sigma
        if mask_region is None:
            self.mask_region = None
        else:
            self.mask_region = mask_region

        if mask_intensity is None:
            self.mask_intensity = None
        else:
            self.mask_intensity = mask_intensity

        if (test_configs['name'] == '2d'):
            self.worker = self._test()
            self.worker.yielded.connect(self.show_intermediate_result)
            self.worker.start()
        elif (test_configs['name'] == '3d'):
            self.worker = self._test_3d()
            self.worker.yielded.connect(self.show_intermediate_result)
            self.worker.start()

    def show_intermediate_result(self, val):
        self.begin_evaluating_progress_bar.setValue(val[0])
        self.viewer.add_image(val[1], name=os.path.basename(val[2])[:-4] + '_pred.tif', colormap='magma')
        self.viewer.add_image(val[3], name=os.path.basename(val[2])[:-4] + '_im.tif', colormap='viridis')

    @thread_worker
    def _test(self):
        """
            :param verbose: if True, then average prevision is printed out for each image
            :param grid_y:
            :param grid_x:
            :param pixel_y:
            :param pixel_x:
            :param one_hot: True, if the instance masks are encoded in a one-hot fashion
            :param avg_bg: Average Background Image Intensity
            :return:
            """
        model.eval()

        # cluster module
        cluster = Cluster(self.grid_y, self.grid_x, self.pixel_y, self.pixel_x)

        with torch.no_grad():
            resultList = []
            imageFileNames = []
            for i, sample in enumerate(tqdm(dataset_it)):

                im = sample['image']  # B 1 Y X
                multiple_y = im.shape[2] // 8
                multiple_x = im.shape[3] // 8

                if im.shape[2] % 8 != 0:
                    diff_y = 8 * (multiple_y + 1) - im.shape[2]
                else:
                    diff_y = 0
                if im.shape[3] % 8 != 0:
                    diff_x = 8 * (multiple_x + 1) - im.shape[3]
                else:
                    diff_x = 0
                p2d = (
                    diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)  # last dim, second last dim

                im = F.pad(im, p2d, "constant", self.avg_bg)
                if ('instance' in sample):
                    instances = sample['instance'].squeeze()  # Y X  (squeeze takes away first two dimensions) or DYX
                    instances = F.pad(instances, p2d, "constant", 0)

                if (tta):
                    output = apply_tta_2d(im, model)
                else:
                    output = model(im)

                instance_map, predictions = cluster.cluster(output[0],
                                                            n_sigma=n_sigma,
                                                            seed_thresh=seed_thresh,
                                                            min_mask_sum=min_mask_sum,
                                                            min_unclustered_sum=min_unclustered_sum,
                                                            min_object_size=min_object_size)

                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                imageFileNames.append(base)

                if (self.one_hot):
                    if ('instance' in sample):
                        results = obtain_AP_one_hot(gt_image=instances.cpu().detach().numpy(),
                                                    prediction_image=instance_map.cpu().detach().numpy(), ap_val=ap_val)
                        if (self.verbose):
                            print("Accuracy: {:.03f}".format(results), flush=True)
                        resultList.append(results)
                else:
                    if ('instance' in sample):
                        results = matching_dataset(y_true=[instances.cpu().detach().numpy()],
                                                   y_pred=[instance_map.cpu().detach().numpy()], thresh=ap_val,
                                                   show_progress=False)
                        if (self.verbose):
                            print("Accuracy: {:.03f}".format(results.accuracy), flush=True)
                        resultList.append(results.accuracy)

                if save_images and ap_val == 0.5:
                    if not os.path.exists(os.path.join(save_dir, 'predictions/')):
                        os.makedirs(os.path.join(save_dir, 'predictions/'))
                        print("Created new directory {}".format(os.path.join(save_dir, 'predictions/')))
                    if not os.path.exists(os.path.join(save_dir, 'ground-truth/')):
                        os.makedirs(os.path.join(save_dir, 'ground-truth/'))
                        print("Created new directory {}".format(os.path.join(save_dir, 'ground-truth/')))
                    if not os.path.exists(os.path.join(save_dir, 'embedding/')):
                        os.makedirs(os.path.join(save_dir, 'embedding/'))
                        print("Created new directory {}".format(os.path.join(save_dir, 'embedding/')))

                    base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                    instances_file = os.path.join(save_dir, 'predictions/', base + '.tif')
                    imsave(instances_file, instance_map.cpu().detach().numpy().astype(np.uint16))
                    if ('instance' in sample):
                        gt_file = os.path.join(save_dir, 'ground-truth/', base + '.tif')
                        imsave(gt_file, instances.cpu().detach().numpy().astype(np.uint16))
                    embedding_file = os.path.join(save_dir, 'embedding/', base + '.tif')
                yield np.round(100 * (i + 1) / len(dataset_it)).astype(int), instance_map.cpu().detach().numpy().astype(
                    np.uint16), instances_file, im.cpu().detach().numpy()

            if save_results and 'instance' in sample:
                if not os.path.exists(os.path.join(save_dir, 'results/')):
                    os.makedirs(os.path.join(save_dir, 'results/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'results/')))
                txt_file = os.path.join(save_dir,
                                        'results/combined_AP-' + '{:.02f}'.format(ap_val) + '_tta-' + str(tta) + '.txt')
                with open(txt_file, 'w') as f:
                    f.writelines(
                        "image_file_name, min_mask_sum, min_unclustered_sum, min_object_size, seed_thresh, intersection_threshold, accuracy \n")
                    f.writelines("+++++++++++++++++++++++++++++++++\n")
                    for ind, im_name in enumerate(imageFileNames):
                        im_name_png = im_name + '.png'
                        score = resultList[ind]
                        f.writelines(
                            "{} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.05f} \n".format(im_name_png, min_mask_sum,
                                                                                           min_unclustered_sum,
                                                                                           min_object_size, seed_thresh,
                                                                                           ap_val, score))
                    f.writelines("+++++++++++++++++++++++++++++++++\n")
                    f.writelines("Average Precision (AP)  {:.02f} {:.05f}\n".format(ap_val, np.mean(resultList)))

                print("Mean Average Precision at IOU threshold = {}, is equal to {:.05f}".format(ap_val,
                                                                                                 np.mean(resultList)))

    @thread_worker
    def _test_3d(self):

        model.eval()
        # cluster module
        cluster = Cluster_3d(self.grid_z, self.grid_y, self.grid_x, self.pixel_z, self.pixel_y, self.pixel_x)

        with torch.no_grad():
            resultList = []
            imageFileNames = []
            for i, sample in enumerate(tqdm(dataset_it)):
                im = sample['image']
                if (self.mask_region is not None and self.mask_intensity is not None):
                    im[:, :, int(self.mask_region[0][0]):, : int(self.mask_region[1][1]),
                    int(self.mask_region[0][2]):] = self.mask_intensity  # B 1 Z Y X
                else:
                    pass

                multiple_z = im.shape[2] // 8
                multiple_y = im.shape[3] // 8
                multiple_x = im.shape[4] // 8

                if im.shape[2] % 8 != 0:
                    diff_z = 8 * (multiple_z + 1) - im.shape[2]
                else:
                    diff_z = 0
                if im.shape[3] % 8 != 0:
                    diff_y = 8 * (multiple_y + 1) - im.shape[3]
                else:
                    diff_y = 0
                if im.shape[4] % 8 != 0:
                    diff_x = 8 * (multiple_x + 1) - im.shape[4]
                else:
                    diff_x = 0
                p3d = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2, diff_z // 2,
                       diff_z - diff_z // 2)  # last dim, second last dim, third last dim!

                im = F.pad(im, p3d, "constant", self.avg_bg)
                if ('instance' in sample):
                    instances = sample['instance'].squeeze()
                    instances = F.pad(instances, p3d, "constant", 0)

                if (tta):
                    for iter in range(16):
                        if iter == 0:
                            output_average = apply_tta_3d(im, model, iter)  # iter
                        else:
                            output_average = 1 / (iter + 1) * (
                                    output_average * iter + apply_tta_3d(im, model, iter))  # iter
                    output = torch.from_numpy(output_average).float().cuda()
                else:
                    output = model(im)

                instance_map, predictions = cluster.cluster(output[0],
                                                            n_sigma=n_sigma,
                                                            seed_thresh=seed_thresh,
                                                            min_mask_sum=min_mask_sum,
                                                            min_unclustered_sum=min_unclustered_sum,
                                                            min_object_size=min_object_size,
                                                            )

                if (self.one_hot):
                    if ('instance' in sample):
                        sc = obtain_AP_one_hot(gt_image=instances.cpu().detach().numpy(),
                                               prediction_image=instance_map.cpu().detach().numpy(), ap_val=ap_val)
                        if (self.verbose):
                            print("Accuracy: {:.03f}".format(sc), flush=True)
                        resultList.append(sc)
                else:
                    if ('instance' in sample):
                        sc = matching_dataset(y_true=[instances.cpu().detach().numpy()],
                                              y_pred=[instance_map.cpu().detach().numpy()], thresh=ap_val,
                                              show_progress=False)
                        if (self.verbose):
                            print("Accuracy: {:.03f}".format(sc.accuracy), flush=True)
                        resultList.append(sc.accuracy)

                if save_images and ap_val == 0.5:
                    if not os.path.exists(os.path.join(save_dir, 'predictions/')):
                        os.makedirs(os.path.join(save_dir, 'predictions/'))
                        print("Created new directory {}".format(os.path.join(save_dir, 'predictions/')))
                    if not os.path.exists(os.path.join(save_dir, 'ground-truth/')):
                        os.makedirs(os.path.join(save_dir, 'ground-truth/'))
                        print("Created new directory {}".format(os.path.join(save_dir, 'ground-truth/')))
                    if not os.path.exists(os.path.join(save_dir, 'seeds/')):
                        os.makedirs(os.path.join(save_dir, 'seeds/'))
                        print("Created new directory {}".format(os.path.join(save_dir, 'seeds/')))
                    if not os.path.exists(os.path.join(save_dir, 'images/')):
                        os.makedirs(os.path.join(save_dir, 'images/'))
                        print("Created new directory {}".format(os.path.join(save_dir, 'images/')))

                    base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                    imageFileNames.append(base)

                    instances_file = os.path.join(save_dir, 'predictions/', base + '.tif')
                    imsave(instances_file, instance_map.cpu().detach().numpy().astype(np.uint16))
                    if ('instance' in sample):
                        gt_file = os.path.join(save_dir, 'ground-truth/', base + '.tif')
                        imsave(gt_file, instances.cpu().detach().numpy().astype(np.uint16))

                    seeds_file = os.path.join(save_dir, 'seeds/', base + '.tif')
                    imsave(seeds_file, torch.sigmoid(output[0, -1, ...]).cpu().detach().numpy())

                    im_file = os.path.join(save_dir, 'images/', base + '.tif')
                    imsave(im_file, im[0, 0].cpu().detach().numpy())
                yield np.round(100 * (i + 1) / len(dataset_it)).astype(int), instance_map.cpu().detach().numpy().astype(
                    np.uint16), instances_file, im.cpu().detach().numpy()
            if save_results and 'instance' in sample:
                if not os.path.exists(os.path.join(save_dir, 'results/')):
                    os.makedirs(os.path.join(save_dir, 'results/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'results/')))
                txt_file = os.path.join(save_dir,
                                        'results/combined_AP-' + '{:.02f}'.format(ap_val) + '_tta-' + str(tta) + '.txt')
                with open(txt_file, 'w') as f:
                    f.writelines(
                        "image_file_name, min_mask_sum, min_unclustered_sum, min_object_size, seed_thresh, intersection_threshold, accuracy \n")
                    f.writelines("+++++++++++++++++++++++++++++++++\n")
                    for ind, im_name in enumerate(imageFileNames):
                        im_name_png = im_name + '.png'
                        score = resultList[ind]
                        f.writelines(
                            "{} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.05f} \n".format(im_name_png, min_mask_sum,
                                                                                           min_unclustered_sum,
                                                                                           min_object_size, seed_thresh,
                                                                                           ap_val, score))
                    f.writelines("+++++++++++++++++++++++++++++++++\n")
                    f.writelines("Average Precision (AP)  {:.02f} {:.05f}\n".format(ap_val, np.mean(resultList)))

                print("Mean Average Precision at IOU threshold = {}, is equal to {:.05f}".format(ap_val,
                                                                                                 np.mean(resultList)))


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [Preprocess, Train, Predict]
