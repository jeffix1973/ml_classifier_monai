import argparse
import train
import test
import datetime

def main():
    parser = argparse.ArgumentParser(description="Run training and/or testing.")
    parser.add_argument("--mode", choices=["train", "test", "both"], required=True, help="Select mode: train, test, or both.")
    parser.add_argument("--json_path", required=True, help="Path to the configuration file.")
    # add a boolean argument to specify whether to use local data
    parser.add_argument("--data_root_path", help="Optional : root path for local DICOM files location.")
    
    
    args = parser.parse_args()
    
    start_time = datetime.datetime.now()
    
    if args.mode in ["train", "both"]:
        train.run_training(args)
        train_duration = datetime.datetime.now() - start_time
        # print duration in hours, minutes and seconds
        print('Training duration: {}'.format(train_duration))
    
    if args.mode in ["test", "both"]:
        test.run_testing(args)
        test_duration = datetime.datetime.now() - start_time
        # print duration in hours, minutes and seconds
        print('Testing duration: {}'.format(test_duration))

if __name__ == "__main__":
    main()
