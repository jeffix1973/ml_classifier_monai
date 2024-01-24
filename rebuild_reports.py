import argparse
import datetime

import report_publisher.model_performance_report as report
import report_publisher.model_performance_report_focused as focused_report

def main():
    parser = argparse.ArgumentParser(description="Re build reports without regenerating the previews of errors.")
    parser.add_argument("--json_path", required=True, help="Path to the model_ax, sag or cor configuration file.")
    
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    # print duration in hours, minutes and seconds
    print('Publishing reports...')
    report.generate(args.json_path)
    focused_report.generate(args.json_path)
    report_duration = datetime.datetime.now() - start_time
    # print duration in hours, minutes and seconds
    print('Report generation duration: {}'.format(report_duration))

if __name__ == "__main__":
    main()