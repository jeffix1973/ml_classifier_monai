import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
import seaborn as sns
from itertools import cycle
plt.style.use('ggplot')
import pdfkit
from yattag import Doc
from yattag import indent
import os
import sys
import json

print('#############################################"', sys.path)
from report_publisher.templates import focused_performance_report

doc, tag, text, line = Doc().ttl()

def buildHTMLfile(html, root_path, output_dir, model_name):
    html_file_name = os.path.join(root_path, output_dir, 'out', model_name, 'test', model_name + '_performance_report.html')
    f = open(html_file_name,'w')
    f.write(html)
    f.close()
    print('>>>> HTML performance report file', html_file_name, 'has been generated...')
    return html_file_name

def test_and_create_dir (outdir):
    # Check if outdir directory exists
    isExist = os.path.exists(outdir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(outdir)
        return 'New directory : ' + outdir + ' has been created'
    else:
        return outdir + ' allready exists...'

def generate(path):

    print('>>>> Loading', path)
    # Opening settings JSON file
    try:
        f = open(path, "r")
    except:
        print('File does not exist. The program has been stopped...')
        sys.exit(0)
    
    # Collect JSON part for this framework
    file = json.load(f)
    VAR = file['report_publisher']
    VAR1 = file['class_monai_dcm']
    print('>>>> Variables have been successfuly loaded...')

    server_name = VAR['server_name']
    sidviewer_url = VAR['sidviewer_url']
    root_path = VAR['root_path']
    output_dir = VAR['output_dir']
    working_folder_path = VAR['working_folder_path']
    csv_files = VAR1['test_csv_files']
    model_name = VAR['model_name']
    labels = VAR['labels']
    focus_labels = VAR['focus_labels']
    metric_charts = VAR['metric_charts']
    
    # Init variables
    total_count = 0
    FN_links = []
    FN_img_links = []

    # Read result file
    path2csv = os.path.join(root_path, output_dir, "out", model_name, "test", "results.csv")
    # Read JSON log file
    path2json = os.path.join(root_path, output_dir, "out", model_name, 'log.json')
    log = {}
    try:
        f_json = open(path2json, "r")
        log = json.load(f_json)
    except:
        print('JSON log file not found...')
    
    # Test if file exists
    isExist = os.path.isfile(path2csv)
    if not isExist:
        print(path2csv, 'does not exist...')
    else:
        # Init result line
        print('>>> Loading file : ', path2csv)
        
        test_results = pd.read_csv(path2csv)
        print(test_results.head())

        for i in range(len(test_results)):

            # Build paths
            full_path = str(test_results.loc[i, 'Path'])
            path_parts = full_path.split('/')
            StudyInstanceUID = path_parts[-3]
            SeriesInstanceUID = path_parts[-2]

            # Get the database name corresponding to this DICOM directory
            database = ''
            dicom_directory_path = '/'.join(path_parts[:-3])  # Get the directory path excluding instance UID
            for csv_file in csv_files:
                if os.path.normpath(csv_file[1]) == dicom_directory_path:
                    database = csv_file[0]
                    break  # Found the matching database, no need to continue the loop
     
            # Build preview path
            if os.path.basename(full_path).endswith('.dcm'): 
                preview_filename = os.path.basename(full_path).replace('.dcm', '.png')
            else:
                preview_filename = os.path.basename(full_path) + '.png'
            preview_path = os.path.join(root_path, output_dir, "out", model_name, "test", "previews", preview_filename)

            # Determine if prediction is valid
            valid = str(test_results.loc[i, 'GT']) == str(test_results.loc[i, 'Prediction'])
            total_count += 1

            if not valid:
                # Append failed detection case
                detection = f"{test_results.loc[i, 'GT']} >> {test_results.loc[i, 'Prediction']}@{test_results.loc[i, 'Max_Prob']}%"
                series_description = str(test_results.loc[i, 'SeriesDescription'])
                sidviewer_link = f"{sidviewer_url}/#/datatable/{database}/FullSIDStatistics-{StudyInstanceUID}"

                FN_links.append((
                    series_description,  # Series Description
                    sidviewer_link,  # Sidviewer URL
                    detection,  # Detection Details
                    preview_path,  # Preview Image Path
                    test_results.loc[i, 'Max_Prob']))  # Max Probability

                FN_img_links.append(preview_path)  # New Simplified Preview Path

        
        print('>>> ', len(FN_links), ' failed detections / ', total_count, 'tested images')
        expected = np.asarray(test_results['GT']).astype(str)
        detected = np.asarray(test_results['Prediction']).astype(str)
        
        unique_classes = np.unique(np.concatenate((expected, detected)))
        print("Unique classes found: ", unique_classes)
        print("Number of unique classes: ", len(unique_classes))
        
        expected_binarized = preprocessing.label_binarize(expected, classes=sorted(labels))
        detected_binarized = preprocessing.label_binarize(detected, classes=sorted(labels))
    
        roc_auc = metrics.roc_auc_score(expected_binarized, detected_binarized,average='macro',multi_class='ovo')
        print('##########           COUNTS         ##########')
        print('----------------------------------------------')
        print('Total Nb of series:', total_count, '/ FAILED detection:', len(FN_links))
        print('##########    MACRO ROC AUC SCORE   ##########')
        print('----------------------------------------------')
        print(round(roc_auc, 4))
    
        conf_matrix = metrics.confusion_matrix(expected, detected)
        print('############    CONFUSION MATRIX   ###########')
        print('----------------------------------------------')
        print(conf_matrix)
    
        #print accuracy of model
        print('############        ACCURACY       ###########')
        print('----------------------------------------------')
        accuracy = round(metrics.accuracy_score(expected, detected), 4)
        print(accuracy)
    
        #print precision value of model
        print('############       PRECISION       ###########')
        print('----------------------------------------------')
        precision = round(metrics.precision_score(expected, detected, average='weighted'), 4)
        print(precision)
    
        #print recall value of model
        print('############        F1 Score       ###########')
        print('----------------------------------------------')
        f1score = round(metrics.f1_score(expected, detected, average='weighted'), 4)
        print(f1score)
    
        #print recall value of model
        print('############ Classification Report ###########')
        print('----------------------------------------------')
        report = metrics.classification_report(expected, detected, 
                                        target_names=sorted(labels))
        print(report)
    
        #### CHART GENERATION ####
    
        ## CONFUSION MATRIX
        #sns.heatmap(conf_matrix, annot=True)
        plt.clf()
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="", xticklabels=sorted(labels), yticklabels=sorted(labels))
        plt.title('Test results / ' + str(len(test_results)) + ' series' + 
                '\nAccuracy={} / Precision={} / F1Score={}'.format(accuracy, precision, f1score))
        plt.ylabel('EXPECTED (gt)')
        plt.xlabel('PREDICTED\n')
    
        # Save confusion matrix as PNG images
        plt.savefig(os.path.join(root_path, output_dir, "out", model_name, "test", "confusion_matrix_for_report.png"), format='png', bbox_inches = 'tight')
        print('>>>> ', os.path.join(root_path, output_dir, "out", model_name, "test", "confusion_matrix_for_report.png"), 'has been saved...')
    
        ## ROC - AUC CURVES
        plt.clf()
        n_classes = expected_binarized.shape[1]
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(expected_binarized[:, i], detected_binarized[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        colors = cycle(['blue', 'red', 'green', 'grey'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                    label='ROC curve / {0} (area = {1:0.2f})'
                    ''.format(sorted(labels)[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC(Receiver Operating Characteristic) - AUC')
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(root_path, output_dir, "out", model_name, "test", "roc_auc.png"), format='png', bbox_inches = 'tight')
        print('>>>> ', os.path.join(root_path, output_dir, "out", model_name, "test", "roc_auc.png"), 'has been saved...')

        # Sort FN Links
        sorter = lambda x: (x[4])
        sorted_FN_links = sorted(FN_links, key=sorter, reverse=True)
        
        # Generate PDF file
        printPDF(focus_labels, server_name, root_path, output_dir, log, working_folder_path, model_name, total_count, sorted_FN_links, labels, metric_charts)


def printPDF(focus_labels, server_name, root_path, output_dir, log, working_folder_path, model_name, total_count, FN_links, labels, metric_charts):
    
    # Get labals variables
    failed_expected_labels = []
    failed_detected_labels = []
    # Get classes where either expected or detected failure occurred
    for l in range(len(FN_links)):
        expected = FN_links[l][2].split(' >> ')[0]
        detected = FN_links[l][2].split(' >> ')[1].split('@')[0]
        if expected not in failed_expected_labels:
            failed_expected_labels.append(expected)
        if detected not in failed_detected_labels:
            failed_detected_labels.append(detected)
    # Declare counter
    failed_expected_counters = []
    failed_detected_counters = []
    for i in range(len(failed_expected_labels)):
        failed_expected_counters.append(0)
    for i in range(len(failed_detected_labels)):
        failed_detected_counters.append(0)
    # Fill-in counters
    for l in range(len(FN_links)):
        expected = FN_links[l][2].split(' >> ')[0]
        detected = FN_links[l][2].split(' >> ')[1].split('@')[0]
        if expected in failed_expected_labels and expected != detected:
            ind = failed_expected_labels.index(expected)
            failed_expected_counters[ind] += 1
        if detected in failed_detected_labels and expected != detected:
            ind = failed_detected_labels.index(detected)
            failed_detected_counters[ind] += 1

    document = focused_performance_report(focus_labels, server_name, root_path, output_dir, working_folder_path, model_name, labels, metric_charts, log, total_count, FN_links, failed_expected_labels, failed_expected_counters, failed_detected_labels, failed_detected_counters)
    
    # Return clean indented variable
    html = indent(document.getvalue())
    
    # Build and save to HTML file
    html_file_name = buildHTMLfile(html, root_path, output_dir, model_name)
    
    # Generate PDF file
    options = { 
        '--margin-top' : '20mm',
        '--margin-bottom' : '20mm',
        '--margin-left' : '15mm',
        '--margin-right' : '15mm',
        '--page-size' : 'A4',
        '--enable-local-file-access' : None,
        '--header-html' : 'assets/header.html',
        '--footer-html' : 'assets/footer.html'
    }

    # Test and create DIR
    test_and_create_dir(os.path.join(root_path, output_dir,'out', 'reports'))
    pdf_file_name = os.path.join(root_path, output_dir, 'out', 'reports', model_name + '_performance_report_focused.pdf')
    
    pdfkit.from_file(html_file_name, pdf_file_name, options=options, verbose=True)

    print('>>>> Model performance report', pdf_file_name, 'has been generated...')

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Passing json file path')
    parser.add_argument('--path', type=str, required=True,
                        help='path to the setting JSON file')
    
    args = parser.parse_args()

    generate(path=args.path)