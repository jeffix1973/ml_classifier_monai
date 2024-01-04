import os
from yattag import Doc
from yattag import indent

doc, tag, text, line = Doc().ttl()


def performance_report(server_name, root_path, output_dir, working_folder_path, model_name, labels, metric_charts, log, total_count, FN_links, FN_img_links, failed_expected_labels, failed_expected_counters, failed_detected_labels, failed_detected_counters):
    
    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('meta'):
            doc.stag('link', href=os.path.join(working_folder_path, 'assets/style.css'), rel="stylesheet")
        with tag('body'):
            # header
            doc.stag('img', src=os.path.join(working_folder_path, 'img/logo.png'), klass="logo")
            with tag('div', klass="report_title"):
                line('span', 'Model Performance Report', klass="title")
                doc.stag('br')
                line('span', server_name + '>> ' + root_path + output_dir + '/out/' + model_name, klass="sub_title")
            # Statistics section
            with tag('h2'):
                text('1 - Statistics')
            with tag('p', klass="stats"):
                text('Path to the test results file:')
                doc.stag('br')
                with tag('b'):
                    text(os.path.join(root_path, output_dir, 'out', model_name, 'test', 'test_results.csv'))
                doc.stag('br')
            with tag('p', klass="stats"):
                with tag('b'):
                    text(str(total_count))
                text(' is the total number of series predicted by the model for this test. Among them:')
            with tag('p', klass="stats"):
                with tag('b'):
                    text(str(len(FN_links)))
                text(' have wrongly been detected.')
            if log != {}:
                with tag('p', klass="stats"):
                    text('Log details for this model:')
                    doc.stag('br')
                    for item in log:
                        text(str(item))
                        doc.stag('br')
                        with tag('b'):
                            if isinstance(log[item], list):
                                res = ''
                                for el in log[item]:
                                    res += (str(el) + ', ')
                                text(res)
                            else:
                                text(log[item])
                        doc.stag('br')
            # Links and preview of False Negative detection
            if len(FN_links) != 0:
                with tag('h4', klass="new-page"):
                    text('Details of the ' + str(len(FN_links)) + ' failed dectections:')
                for l in range(len(failed_expected_labels)):
                    with tag('h5'):
                        text('> ' + str(failed_expected_counters[l]) + ' expected ' + str(failed_expected_labels[l]) + ' not correctly detected:')
                    for i in range(len(FN_links)):
                        if FN_links[i][2].split(' >> ')[0] == failed_expected_labels[l]:
                            with tag('div', klass="preview_wrapper"):
                                doc.stag('img', src=FN_img_links[i])
                                with tag('span', klass="description"):
                                    text(str(FN_links[i][0]))
                                with tag('span', klass="detection"):
                                    text(str(FN_links[i][2]))
                                with tag('a', href=str(FN_links[i][1])):
                                    text('> Sidviewer Link')
                for l in range(len(failed_detected_labels)):
                    with tag('h5'):
                        text('> ' + str(failed_detected_counters[l]) + ' detected ' + str(failed_detected_labels[l]) + ' not corresponding to the expected class:')
                    for i in range(len(FN_links)):
                        if FN_links[i][2].split(' >> ')[1].split('@')[0] == failed_detected_labels[l]:
                            with tag('div', klass="preview_wrapper"):
                                doc.stag('img', src=FN_img_links[i])
                                with tag('span', klass="description"):
                                    text(str(FN_links[i][0]))                                
                                with tag('span', klass="detection"):
                                    text(str(FN_links[i][2]))
                                with tag('a', href=str(FN_links[i][1])):
                                    text('> Sidviewer Link')
            # Confusion matrix and ROC AUC sections
            with tag('h2', klass="new-page"):
                text('2 - Confusion Matrix')
            with tag('p'):
                text('The model has been trained with ' + str(len(labels)) + ' classes and the tests gives the following confusion matrix:')
            doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'test/confusion_matrix_for_report.png'), klass="chart_full_page")
            with tag('h2', klass="new-page"):
                text('3 - ROC (Area Under the Curve')
            with tag('p'):
                text('The ROC indicator has been plot for each classes in One against all mode')
            doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'test/roc_auc.png'), klass="chart_full_page")
            # Model training details section
            with tag('h2', klass="new-page"):
                text('4 - Model training details')
            with tag('p'):
                text('Series distribution in the training dataset:')
            for i in range(len(labels)):
                doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'split', 'label_' + str(i) + '.png'), klass="chart_half_page")
            with tag('p', klass="new-page"):
                text('Model metrics curves:')
            for i in range(len(metric_charts)):
                doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'train', str(metric_charts[i]) + '.png'), klass="chart_half_page")
    
    return doc


def focused_performance_report(focus_labels, server_name, root_path, output_dir, working_folder_path, model_name, labels, metric_charts, log, total_count, FN_links, failed_expected_labels, failed_expected_counters, failed_detected_labels, failed_detected_counters):
    
    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('meta'):
            doc.stag('link', href=os.path.join(working_folder_path, 'assets/style.css'), rel="stylesheet")
        with tag('body'):
            # header
            doc.stag('img', src=os.path.join(working_folder_path, 'img/logo.png'), klass="logo")
            with tag('div', klass="report_title"):
                line('span', 'Model Performance Report / Focused', klass="title")
                doc.stag('br')
                line('span', server_name + '>> ' + root_path + output_dir + '/out/' + model_name, klass="sub_title")
            # Statistics section
            with tag('h2'):
                text('1 - Statistics')
            with tag('p', klass="stats"):
                text('Path to the test results file:')
                doc.stag('br')
                with tag('b'):
                    text(os.path.join(root_path, output_dir, 'out', model_name, 'test', 'test_results.csv'))
                doc.stag('br')
            with tag('p', klass="stats"):
                with tag('b'):
                    text(str(total_count))
                text(' is the total number of series predicted by the model for this test. Among them:')
            with tag('p', klass="stats"):
                with tag('b'):
                    text(str(len(FN_links)))
                text(' have wrongly been detected.')
            if log != {}:
                with tag('p', klass="stats"):
                    text('Log details for this model:')
                    doc.stag('br')
                    for item in log:
                        text(str(item))
                        doc.stag('br')
                        with tag('b'):
                            if isinstance(log[item], list):
                                res = ''
                                for el in log[item]:
                                    res += (str(el) + ', ')
                                text(res)
                            else:
                                text(log[item])
                        doc.stag('br')
            # Links and preview of False Negative detection
            if len(FN_links) != 0:
                with tag('h4', klass="new-page"):
                    text('Details of the failed dectections focused on ' + str(focus_labels) + ' labels:')
                for l in range(len(failed_expected_labels)):
                    if failed_expected_labels[l] in focus_labels:
                        with tag('h5'):
                            if len(focus_labels) > 2:
                                text('> ' + str(failed_expected_labels[l]) + ' not correctly detected (False Negative):')
                            else:
                                text('> ' + str(failed_expected_labels[l]) + ' not correctly detected:')
                        for i in range(len(FN_links)):
                            if FN_links[i][2].split(' >> ')[0] == failed_expected_labels[l] and FN_links[i][2].split(' >> ')[1].split('@')[0] in focus_labels:
                                with tag('div', klass="preview_wrapper"):
                                    doc.stag('img', src=FN_links[i][3])
                                    with tag('span', klass="description"):
                                        text(str(FN_links[i][0]))
                                    if float(FN_links[i][4]) >= 90:
                                        with tag('span', klass="detection red"):
                                            text(str(FN_links[i][2]))
                                    elif float(FN_links[i][4]) >= 80 and float(FN_links[i][4]) < 90:
                                        with tag('span', klass="detection orange"):
                                            text(str(FN_links[i][2]))
                                    else:
                                        with tag('span', klass="detection green"):
                                            text(str(FN_links[i][2]))                                                                            
                                    with tag('a', href=str(FN_links[i][1])):
                                        text('> Sidviewer Link')
                if len(focus_labels) > 2:
                    for l in range(len(failed_detected_labels)):
                        if failed_detected_labels[l] in focus_labels:
                            with tag('h5'):
                                text('> ' + str(failed_detected_labels[l]) + ' detections not corresponding to the expected class (False Positive):')
                            for i in range(len(FN_links)):
                                if FN_links[i][2].split(' >> ')[1].split('@')[0] == failed_detected_labels[l] and FN_links[i][2].split(' >> ')[0] in focus_labels:
                                    with tag('div', klass="preview_wrapper"):
                                        doc.stag('img', src=FN_links[i][3])
                                        with tag('span', klass="description"):
                                            text(str(FN_links[i][0]))
                                        if float(FN_links[i][4]) >= 90:
                                            with tag('span', klass="detection red"):
                                                text(str(FN_links[i][2]))
                                        elif float(FN_links[i][4]) >= 80 and float(FN_links[i][4]) < 90:
                                            with tag('span', klass="detection orange"):
                                                text(str(FN_links[i][2]))
                                        else:
                                            with tag('span', klass="detection green"):
                                                text(str(FN_links[i][2])) 
                                        with tag('a', href=str(FN_links[i][1])):
                                            text('> Sidviewer Link')
            # Confusion matrix and ROC AUC sections
            with tag('h2', klass="new-page"):
                text('2 - Confusion Matrix')
            with tag('p'):
                text('The model has been trained with ' + str(len(labels)) + ' classes and the tests gives the following confusion matrix:')
            doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'test/confusion_matrix_for_report.png'), klass="chart_full_page")
            with tag('h2', klass="new-page"):
                text('3 - ROC (Area Under the Curve')
            with tag('p'):
                text('The ROC indicator has been plot for each classes in One against all mode')
            doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'test/roc_auc.png'), klass="chart_full_page")
            # Model training details section
            with tag('h2', klass="new-page"):
                text('4 - Model training details')
            with tag('p'):
                text('Series distribution in the training dataset:')
            for i in range(len(labels)):
                doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'split', 'label_' + str(i) + '.png'), klass="chart_half_page")
            with tag('p', klass="new-page"):
                text('Model metrics curves:')
            for i in range(len(metric_charts)):
                doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'train', str(metric_charts[i]) + '.png'), klass="chart_half_page")
    
    return doc 
