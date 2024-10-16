# -*- coding: utf-8 -*-
'''
Date : January 2017

Authors : Pierre-Marc Jodoin from the University of Sherbrooke

Description : code used to parse the MIO-TCD classification dataset,  classify
            each image and save results in the proper csv format.  Please see
            http://tcd.miovision.com/ for more details on the dataset

Execution : simply type the following command in a terminal:

   >> python parse_classification_dataset.py ./train/ your_results_train.csv
or
   >> python parse_classification_dataset.py ./test/ your_results_test.csv


NOTE: this code was developed and tested with Python 3.5.2 and Linux
      (Ubuntu 14.04)

Disclamer:

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
from os import listdir
from os.path import isfile, join, splitext
from tqdm import tqdm
import numpy as np
import csv
import sys

classes = ['articulated_truck', 'bicycle', 'bus', 'car',
           'motorcycle', 'non-motorized_vehicle', 'pedestrian',
           'pickup_truck', 'single_unit_truck', 'work_van', 'background']


def classify_image(path_to_image):
    '''
    Classify the image contained in 'path_to_image'.

    You may replace this line with a call to your classification method
    '''

    label = classes[np.random.randint(11)]
    return label


def parse_dataset(path_to_dataset):
    '''
    Parse every image contained in 'path_to_dataset' (a path to the training
    or testing set), classify each image and save in a csv file the resulting
    assignment

    dataset_result: dict structure returned by the function.  It contains the
            label of each image
    '''
    llist = listdir(path_to_dataset)
    dataset_result = {}

    for name in tqdm(llist):
        dn = join(path_to_dataset, name)
        if isfile(dn):
            label = classify_image(dn)
            file_nb, file_ext = splitext(name)
            dataset_result[file_nb] = label

        else:
            file_list = listdir(dn)
            for file_name in file_list:
                file_name_with_path = join(dn, file_name)
                label = classify_image(file_name_with_path)
                file_nb, file_ext = splitext(file_name)
                if file_nb in dataset_result.keys():
                    print('error! ', file_nb, dataset_result[file_nb], ' vs ', file_name_with_path)
                dataset_result[file_nb] = label

    return dataset_result


def save_classification_result(dataset_result, output_csv_file_name):
    '''
    save the dataset_result (a dict structure containing the class of every image)
    into a valid csv file.
    '''

    csvfile = open(output_csv_file_name, 'w')
    fieldnames = ['file_name', 'class_label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for file_number in dataset_result.keys():
        writer.writerow({'file_name': str(file_number), 'class_label': dataset_result[file_number]})

    csvfile.close()

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("\nUsage : \n\t python parse_classification_dataset.py PATH OUTPUT_CSV_FILE_NAME\n")
        print("\t PATH : path to the training or the testing dataset")
        print("\t OUTPUT_CSV_FILE_NAME : name of the resulting csv file\n")
    else:
        print('\nProcessing: ', sys.argv[1], '\n')
        dataset_result = parse_dataset(sys.argv[1])
        save_classification_result(dataset_result, sys.argv[2])
