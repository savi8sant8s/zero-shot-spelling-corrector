import pandas as pd
import os
import argparse
from dotenv import load_dotenv
from src.metrics import print_metrics
from src.format import get_sentences
from src.model import SpellingModel
import time

load_dotenv()

def process_dataset(dataset, ocr_prediction, provider, llm, output_folder, prompt_type):
    print('-' * 50)
    print(f'Dataset: {dataset} - OCR Prediction: {ocr_prediction}')
    
    input_path = f'datasets/{dataset}/{ocr_prediction}.csv'
    os.makedirs(output_folder, exist_ok=True)
    
    df = pd.read_csv(input_path)
    
    sentences = get_sentences(df, dataset)

    spelling_model = SpellingModel(provider, llm, prompt_type)
    
    start_time = time.time()
    result = spelling_model.predict(sentences, len(sentences))
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    corrections = []
    for data in result:
        corrections.extend(data)
        
    print_metrics(df['prediction'].tolist(), df['ground_truth'].tolist(), corrections, dataset, ocr_prediction)
    
    output_file = f'{output_folder}/{dataset}_{ocr_prediction}.csv'
    pd.DataFrame({'correction': corrections}).to_csv(output_file, index=False)
    print(f'Saved corrections to {output_file}')
    
    time_log_path = os.path.join(output_folder, 'timing_log.txt')
    with open(time_log_path, 'a') as f:
        f.write(f'{dataset},{ocr_prediction},{len(sentences)} sentences,{elapsed_time:.2f} seconds\n')
    
    print(f'Time taken: {elapsed_time:.2f} seconds (logged to {time_log_path})')

def main(args):
    datasets = args.datasets
    ocr_predictions = args.ocr_predictions
    
    for dataset in datasets:
        for ocr_prediction in ocr_predictions:
            process_dataset(dataset, ocr_prediction, args.provider, args.llm, args.output_folder, args.prompt_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', type=str, default='openai')
    parser.add_argument('--llm', type=str, default='gpt-4o-mini')
    parser.add_argument('--output_folder', type=str, default='gpt')
    parser.add_argument('--datasets', nargs='+', type=str, default=['rimes', 'iam', 'bressay'])
    parser.add_argument('--ocr_predictions', nargs='+', type=str, default=['bluche', 'flor', 'puigcerver'])
    parser.add_argument('--prompt_type', type=int, default=1)
    
    args = parser.parse_args()
    
    main(args)
