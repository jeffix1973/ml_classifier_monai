import os
from yattag import Doc

# define errors as a global variable
errors = 0

def performance_report(server_name, root_path, output_dir, working_folder_path, model_name, labels, metric_charts, log, total_count, FN_links, FN_img_links, failed_expected_labels, failed_expected_counters, failed_detected_labels, failed_detected_counters, report, DT, settings):
    
    doc, tag, text, line = Doc().ttl()
    
    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('meta'):
            doc.stag('link', href=os.path.join(working_folder_path, 'assets/style.css'), rel="stylesheet")
        with tag('body'):
            # header
            doc.stag('img', src=os.path.join(working_folder_path, 'img/logo.png'), klass="logo")
            with tag('div', klass="report_title"):
                line('span', 'Performance Report', klass="title")
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
            with tag('p', klass="stats"):
                text('>>> Adjacent labels list for counting errors (only non-adjacent expected-detected classes pairs will be considered as ERRORS in the report):')
                # print adjacent labels list as a list
                doc.stag('br')
                with tag('b'):
                    # display head_labels list as string with commas
                    text(', '.join(head_labels))
                
            # Links and preview of False Negative detection
            if len(FN_links) != 0:
                with tag('h2', klass="new-page"):
                    text('2 - Details of the ' + str(len(FN_links)) + ' failed dectections:')
                for l in range(len(failed_expected_labels)):
                    with tag('h3'):
                        text('> ' + str(failed_expected_counters[l]) + ' expected ' + str(failed_expected_labels[l]) + ' not correctly detected:')
                    for i in range(len(FN_links)):
                        if FN_links[i][2].split(' >> ')[0] == failed_expected_labels[l]:
                            with tag('div', klass="preview_wrapper"):
                                doc.stag('img', src=FN_img_links[i])
                                with tag('span', klass="description"):
                                    text(str(FN_links[i][0]))
                                if float(FN_links[i][4]) >= DT:
                                    with tag('span', klass="detection red"):
                                        text(str(FN_links[i][2]))
                                elif float(FN_links[i][4]) >= DT*0.8 and float(FN_links[i][4]) < DT:
                                    with tag('span', klass="detection orange"):
                                        text(str(FN_links[i][2]))
                                else:
                                    with tag('span', klass="detection green"):
                                        text(str(FN_links[i][2])) 
                                with tag('a', href=str(FN_links[i][1])):
                                    text('> Sidviewer Link')
                for l in range(len(failed_detected_labels)):
                    with tag('h3'):
                        text('> ' + str(failed_detected_counters[l]) + ' detected ' + str(failed_detected_labels[l]) + ' not corresponding to the expected class:')
                    for i in range(len(FN_links)):
                        if FN_links[i][2].split(' >> ')[1].split('@')[0] == failed_detected_labels[l]:
                            with tag('div', klass="preview_wrapper"):
                                doc.stag('img', src=FN_img_links[i])
                                with tag('span', klass="description"):
                                    text(str(FN_links[i][0]))                                
                                if float(FN_links[i][4]) >= DT:
                                    with tag('span', klass="detection red"):
                                        text(str(FN_links[i][2]))
                                elif float(FN_links[i][4]) >= DT*0.8 and float(FN_links[i][4]) < DT:
                                    with tag('span', klass="detection orange"):
                                        text(str(FN_links[i][2]))
                                else:
                                    with tag('span', klass="detection green"):
                                        text(str(FN_links[i][2])) 
                                with tag('a', href=str(FN_links[i][1])):
                                    text('> Sidviewer Link')
            # Confusion matrix and ROC AUC sections
            with tag('h2', klass="new-page"):
                text('3 - Confusion Matrix')
            with tag('p'):
                text('The model has been trained with ' + str(len(labels)) + ' classes and the tests gives the following confusion matrix:')
            doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'test/confusion_matrix_for_report.png'), klass="chart_full_page")
            
            # Including the classification report
            report_lines = report.split('\n')
            with tag('h2', klass="new-page"):
                text('4 - Classification Report')
            with tag('table', klass='classification-report'):
                for i, line in enumerate(report_lines):
                    with tag('tr'):
                        # For the first line, add an empty cell at the beginning
                        if i == 0:
                            doc.stag('td')
                        for item in line.split():
                            with tag('td'):
                                text(item)
                
            with tag('h2', klass="new-page"):
                text('5 - ROC (Area Under the Curve')
            with tag('p'):
                text('The ROC indicator has been plot for each classes in One against all mode')
            doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'test/roc_auc.png'), klass="chart_full_page")
            # Model training details section
            with tag('h2', klass="new-page"):
                text('6 - Model training details')
            with tag('h3'):
                text('Series distribution in the training dataset:')
            for i in range(len(labels)):
                doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'split', 'label_' + str(i) + '.png'), klass="chart_half_page")
            with tag('h3', klass="new-page"):
                text('Model metrics curves:')
            # for i in range(len(metric_charts)):
            #     doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'train', str(metric_charts[i]) + '.png'), klass="chart_half_page")
                doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'train', 'training_curves.png'), klass="chart_full_page")
            with tag('h3'):
                text('Model parameters:')
            with tag('p'):
                text('The model has been trained with the following parameters:')
            with tag('table', klass='model-params'):
                # use settings key value pairs
                for key, value in settings.items():
                    if key in settings_keys:
                        with tag('tr'):
                            with tag('td'):
                                text(key)
                            with tag('td'):
                                # if value is a list, convert it to a string
                                if isinstance(value, list):
                                    # if list of int, convert to string
                                    if isinstance(value[0], int):
                                        vals = [str(val) for val in value]
                                        value = ', '.join(vals)
                                    else:
                                        value = ', '.join(value)
                                else:
                                    value = str(value)
                                text(value)
   
    return doc

def focused_performance_report(focus_labels, server_name, root_path, output_dir, working_folder_path, model_name, labels, metric_charts, log, total_count, FN_links, failed_expected_labels, failed_expected_counters, failed_detected_labels, failed_detected_counters, report, DT, settings):
    
    doc, tag, text, line = Doc().ttl()
    
    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('meta'):
            doc.stag('link', href=os.path.join(working_folder_path, 'assets/style.css'), rel="stylesheet")
        with tag('body'):
            # header
            doc.stag('img', src=os.path.join(working_folder_path, 'img/logo.png'), klass="logo")
            with tag('div', klass="report_title"):
                line('span', 'Focused Performance Report', klass="title")
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
            with tag('p', klass="stats"):
                text('>>> Adjacent labels list for counting errors (only non-adjacent expected-detected classes pairs will be considered as ERRORS in the report):')
                # print adjacent labels list as a list
                doc.stag('br')
                with tag('b'):
                    # display head_labels list as string with commas
                    text(', '.join(head_labels))
                        
            # Links and preview of False Negative detection
            if len(FN_links) != 0:
                with tag('h2', klass="new-page"):
                    text('2 - Details of the failed dectections focused on ' + str(focus_labels) + ' labels:')
                for l in range(len(failed_expected_labels)):
                    if failed_expected_labels[l] in focus_labels:
                        errors = 0
                        with tag('h3'):
                            if len(focus_labels) > 2:
                                text('> ' + str(failed_expected_labels[l]) + ' not correctly detected (False Negative):')
                            else:
                                text('> ' + str(failed_expected_labels[l]) + ' not correctly detected:')                       
                        for i in range(len(FN_links)):
                            expected = FN_links[i][2].split(' >> ')[0]
                            detected = FN_links[i][2].split(' >> ')[1].split('@')[0]
                            # if FN_links[i][2].split(' >> ')[0] == failed_expected_labels[l] and FN_links[i][2].split(' >> ')[1].split('@')[0] in focus_labels:
                            
                            if expected == failed_expected_labels[l] and expected in focus_labels:
                                if expected in head_labels and detected in head_labels and is_label_adjacent(expected, detected, head_labels):
                                    continue # Skip prediction with adjacent labels for HEAD
                                errors += 1
                                with tag('div', klass="preview_wrapper"):
                                    doc.stag('img', src=FN_links[i][3])
                                    with tag('span', klass="description"):
                                        text(str(FN_links[i][0]))
                                    if float(FN_links[i][4]) >= DT:
                                        with tag('span', klass="detection red"):
                                            text(str(FN_links[i][2]))
                                    elif float(FN_links[i][4]) >= DT*0.8 and float(FN_links[i][4]) < DT:
                                        with tag('span', klass="detection orange"):
                                            text(str(FN_links[i][2]))
                                    else:
                                        with tag('span', klass="detection green"):
                                            text(str(FN_links[i][2]))                                                                            
                                    with tag('a', href=str(FN_links[i][1])):
                                        text('> Sidviewer Link')
                        if errors == 0:
                            with tag('span', klass="detection green"):
                                text('No non-adjancent errors found for expected {}'.format(failed_expected_labels[l]))
                if len(focus_labels) > 2:
                    for l in range(len(failed_detected_labels)):
                        if failed_detected_labels[l] in focus_labels:
                            errors = 0
                            with tag('h3'):
                                text('> ' + str(failed_detected_labels[l]) + ' detections not corresponding to the expected class (False Positive):')
                            for i in range(len(FN_links)):
                                expected = FN_links[i][2].split(' >> ')[0]
                                detected = FN_links[i][2].split(' >> ')[1].split('@')[0]
                                if detected == failed_detected_labels[l] and detected in focus_labels:
                                    if expected in head_labels and detected in head_labels and is_label_adjacent(expected, detected, head_labels):
                                        continue # Skip cases with prediction with adjacent labels for HEAD
                                    errors += 1
                                    with tag('div', klass="preview_wrapper"):
                                        doc.stag('img', src=FN_links[i][3])
                                        with tag('span', klass="description"):
                                            text(str(FN_links[i][0]))
                                        if float(FN_links[i][4]) >= DT:
                                            with tag('span', klass="detection red"):
                                                text(str(FN_links[i][2]))
                                        elif float(FN_links[i][4]) >= DT*0.8 and float(FN_links[i][4]) < DT:
                                            with tag('span', klass="detection orange"):
                                                text(str(FN_links[i][2]))
                                        else:
                                            with tag('span', klass="detection green"):
                                                text(str(FN_links[i][2])) 
                                        with tag('a', href=str(FN_links[i][1])):
                                            text('> Sidviewer Link')
                            if errors == 0:
                                with tag('span', klass="detection green"):
                                    text('No non-adjacent errors found for detected {}'.format(failed_detected_labels[l]))                            
            # Confusion matrix and ROC AUC sections
            with tag('h2', klass="new-page"):
                text('3 - Confusion Matrix')
            with tag('p'):
                text('The model has been trained with ' + str(len(labels)) + ' classes and the tests gives the following confusion matrix:')
            doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'test/confusion_matrix_for_report.png'), klass="chart_full_page")

            # Including the classification report
            report_lines = report.split('\n')
            with tag('h2', klass="new-page"):
                text('4 - Classification Report')
            with tag('table', klass='classification-report', style="width: 30%;"):
                for i, line in enumerate(report_lines):
                    with tag('tr'):
                        # For the first line, add an empty cell at the beginning
                        if i == 0:
                            doc.stag('td')
                        for item in line.split():
                            with tag('td'):
                                text(item)
                                
            with tag('h2', klass="new-page"):
                text('5 - ROC Area Under the Curve')
            with tag('p'):
                text('The ROC indicator has been plot for each classes in One against all mode')
            doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'test/roc_auc.png'), klass="chart_full_page")
            # Model training details section
            with tag('h2', klass="new-page"):
                text('6 - Model training details')
            with tag('h3'):
                text('6.1 - Series distribution in the training dataset:')
            for i in range(len(labels)):
                doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'split', 'label_' + str(i) + '.png'), klass="chart_half_page")
            with tag('h3', klass="new-page"):
                text('6.2 - Model metrics curves:')
            # for i in range(len(metric_charts)):
            #     doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'train', str(metric_charts[i]) + '.png'), klass="chart_half_page")
                doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'train', 'training_curves.png'), klass="chart_full_page")
            with tag('h3'):
                text('6.3 - Model metrics curves:')
            with tag('p'):
                text('The model has been trained with the following parameters:')
            with tag('table', klass='model-params'):
                # use settings key value pairs
                for key, value in settings.items():
                    if key in settings_keys:
                        with tag('tr'):
                            with tag('td'):
                                text(key)
                            with tag('td'):
                                # if value is a list, convert it to a string
                                if isinstance(value, list):
                                    # if list of int, convert to string
                                    if isinstance(value[0], int):
                                        vals = [str(val) for val in value]
                                        value = ', '.join(vals)
                                    else:
                                        value = ', '.join(value)
                                else:
                                    value = str(value)
                                text(value)
            
            
    return doc 

def summary_performance_report(focus_labels, server_name, root_path, output_dir, working_folder_path, model_name, labels, metric_charts, log, total_count, FN_links, failed_expected_labels, failed_expected_counters, failed_detected_labels, failed_detected_counters, report, DT, settings):
    
    doc, tag, text, line = Doc().ttl()
    
    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('meta'):
            doc.stag('link', href=os.path.join(working_folder_path, 'assets/style.css'), rel="stylesheet")
        with tag('body'):
            # header
            doc.stag('img', src=os.path.join(working_folder_path, 'img/logo.png'), klass="logo")
            with tag('div', klass="report_title"):
                line('span', 'Summary Performance Report', klass="title")
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

            with tag('p', klass="stats"):
                text('>>> Adjacent labels list for counting errors (only non-adjacent expected-detected classes pairs will be considered as ERRORS in the report):')
                # print adjacent labels list as a list
                doc.stag('br')
                with tag('b'):
                    # display head_labels list as string with commas
                    text(', '.join(head_labels))
                    
            # Links and preview of False Negative detection
            if len(FN_links) != 0:
                with tag('h2', klass="new-page"):
                    text('2 - Details of the failed dectections >= DT focused on ' + str(focus_labels) + ' labels:')
                for l in range(len(failed_expected_labels)):
                    if failed_expected_labels[l] in focus_labels:
                        errors = 0
                        with tag('h3'):
                            if len(focus_labels) > 2:
                                text('> ' + str(failed_expected_labels[l]) + ' not correctly detected (False Negative):')
                            else:
                                text('> ' + str(failed_expected_labels[l]) + ' not correctly detected:')                       
                        for i in range(len(FN_links)):
                            expected = FN_links[i][2].split(' >> ')[0]
                            detected = FN_links[i][2].split(' >> ')[1].split('@')[0]
                            # if FN_links[i][2].split(' >> ')[0] == failed_expected_labels[l] and FN_links[i][2].split(' >> ')[1].split('@')[0] in focus_labels:
                            
                            if expected == failed_expected_labels[l] and expected in focus_labels:
                                if expected in head_labels and detected in head_labels and is_label_adjacent(expected, detected, head_labels):
                                    continue # Skip prediction with adjacent labels for HEAD
                                if float(FN_links[i][4]) < DT:
                                    continue # Skip cases with prediction < DT%
                                errors += 1
                                with tag('div', klass="preview_wrapper"):
                                    doc.stag('img', src=FN_links[i][3])
                                    with tag('span', klass="description"):
                                        text(str(FN_links[i][0]))
                                    if float(FN_links[i][4]) >= DT:
                                        with tag('span', klass="detection red"):
                                            text(str(FN_links[i][2]))
                                    # elif float(FN_links[i][4]) >= DT*0.8 and float(FN_links[i][4]) < DT:
                                    #     with tag('span', klass="detection orange"):
                                    #         text(str(FN_links[i][2]))
                                    # else:
                                    #     with tag('span', klass="detection green"):
                                    #         text(str(FN_links[i][2]))                                                                            
                                    with tag('a', href=str(FN_links[i][1])):
                                        text('> Sidviewer Link')
                        if errors == 0:
                            with tag('span', klass="detection green"):
                                text('No non-adjacent errors found for expected {} with threshold >= {}%'.format(failed_expected_labels[l], DT))
                if len(focus_labels) > 2:
                    errors = 0
                    for l in range(len(failed_detected_labels)):
                        if failed_detected_labels[l] in focus_labels:
                            errors = 0
                            with tag('h3'):
                                text('> ' + str(failed_detected_labels[l]) + ' detections not corresponding to the expected class (False Positive):')
                            for i in range(len(FN_links)):
                                expected = FN_links[i][2].split(' >> ')[0]
                                detected = FN_links[i][2].split(' >> ')[1].split('@')[0]
                                if detected == failed_detected_labels[l] and detected in focus_labels:
                                    if expected in head_labels and detected in head_labels and is_label_adjacent(expected, detected, head_labels):
                                        continue # Skip cases with prediction with adjacent labels for HEAD
                                    if float(FN_links[i][4]) < DT:
                                        continue # Skip cases with prediction < DT%
                                    errors += 1
                                    with tag('div', klass="preview_wrapper"):
                                        doc.stag('img', src=FN_links[i][3])
                                        with tag('span', klass="description"):
                                            text(str(FN_links[i][0]))
                                        if float(FN_links[i][4]) >= DT:
                                            with tag('span', klass="detection red"):
                                                text(str(FN_links[i][2]))
                                        # elif float(FN_links[i][4]) >= DT*0.8 and float(FN_links[i][4]) < DT:
                                        #     with tag('span', klass="detection orange"):
                                        #         text(str(FN_links[i][2]))
                                        # else:
                                        #     with tag('span', klass="detection green"):
                                        #         text(str(FN_links[i][2])) 
                                        with tag('a', href=str(FN_links[i][1])):
                                            text('> Sidviewer Link')
                        if errors == 0:
                            with tag('span', klass="detection green"):
                                text('No non-adjacent errors found for detected {} with threshold >= {}%'.format(failed_detected_labels[l], DT))                               
            # Confusion matrix and ROC AUC sections
            with tag('h2', klass="new-page"):
                text('3 - Confusion Matrix')
            with tag('p'):
                text('The model has been trained with ' + str(len(labels)) + ' classes and the tests gives the following confusion matrix:')
            doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'test/confusion_matrix_for_report.png'), klass="chart_full_page")
            
            # Including the classification report
            report_lines = report.split('\n')
            with tag('h2', klass="new-page"):
                text('4 - Classification Report')
            with tag('table', klass='classification-report', style="width: 30%;"):
                for i, line in enumerate(report_lines):
                    with tag('tr'):
                        # For the first line, add an empty cell at the beginning
                        if i == 0:
                            doc.stag('td')
                        for item in line.split():
                            with tag('td'):
                                text(item)
                                
            with tag('h2', klass="new-page"):
                text('5 - ROC Area Under the Curve')
            with tag('p'):
                text('The ROC indicator has been plot for each classes in One against all mode')
            doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'test/roc_auc.png'), klass="chart_full_page")
            # Model training details section
            with tag('h2', klass="new-page"):
                text('6 - Model training details')
            with tag('h3'):
                text('6.1 - Series distribution in the training dataset:')
            for i in range(len(labels)):
                doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'split', 'label_' + str(i) + '.png'), klass="chart_half_page")
            with tag('h3', klass="new-page"):
                text('6.2 - Model metrics curves:')
            # for i in range(len(metric_charts)):
            #     doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'train', str(metric_charts[i]) + '.png'), klass="chart_half_page")
                doc.stag('img', src=os.path.join(root_path, output_dir, 'out', model_name, 'train', 'training_curves.png'), klass="chart_full_page")

            with tag('h3'):
                text('6.3 - Model metrics curves:')
            with tag('p'):
                text('The model has been trained with the following parameters:')
            with tag('table', klass='model-params'):
                # use settings key value pairs
                for key, value in settings.items():
                    if key in settings_keys:
                        with tag('tr'):
                            with tag('td'):
                                text(key)
                            with tag('td'):
                                # if value is a list, convert it to a string
                                if isinstance(value, list):
                                    # if list of int, convert to string
                                    if isinstance(value[0], int):
                                        vals = [str(val) for val in value]
                                        value = ', '.join(vals)
                                    else:
                                        value = ', '.join(value)
                                else:
                                    value = str(value)
                                text(value)

    return doc

def is_label_adjacent(expected_label, detected_label, label_list):
    expected_index = label_list.index(expected_label)
    detected_index = label_list.index(detected_label)

    # Check if the detected label is the same, just above, or just below the expected label
    if abs(expected_index - detected_index) <= 1:
        return True
    else:
        return False

# Labels list
head_labels = [
    "HEAD.ZONE_BRAIN",
    "HEAD.ZONE_EYES",
    "HEAD.ZONE_NOSE",
    "HEAD.ZONE_MOUTH",
    "CERVICAL",
    "HEAD.ZONE_CENTER",
    "HEAD.ZONE_FACE",
]

settings_keys = [
    "runtype",
    "reproductible_split",
    "network",
    "label_column",
    "model_focus",
    "balanced_datasets",
    "balanced_max_factor",
    "max_series_per_class",
    "max_test_series_per_class",
    "patience",
    "eval_ratio",
    "test_ratio",
    "size",
    "lr",
    "batch_size",
    "num_workers",
    "epochs"
]