import pandas as pd
import argparse
from src.metrics import print_metrics

def main(datasets, output_corrections, ocr_predictions):
  for dataset in datasets:
    for ocr_prediction in ocr_predictions:
      df_corrections = pd.read_csv(f'{output_corrections}/{dataset}_{ocr_prediction}.csv')
      df_predictions = pd.read_csv(f'datasets/{dataset}/{ocr_prediction}.csv')
      
      inputs = df_predictions['prediction'].tolist()
      outputs = df_predictions['ground_truth'].tolist()
      corrections = df_corrections['correction'].tolist()
      
      print_metrics(inputs, outputs, corrections, dataset, ocr_prediction)
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--datasets', nargs='+', type=str, default=['rimes', 'iam', 'bressay'])
  parser.add_argument('--output_corrections', type=str, default='phi4')
  parser.add_argument('--ocr_predictions', nargs='+', type=str, default=['bluche', 'flor', 'puigcerver'])
  
  args = parser.parse_args()
  
  main(args.datasets, args.output_corrections, args.ocr_predictions)