import editdistance
import numpy as np
from src.format import format_correction

def get_cer(word1, word2):
  if type(word1) is not str or type(word2) is not str:
    return 1

  gt_cer, pd_cer = list(word1), list(word2)
  cer_distance = editdistance.eval(gt_cer, pd_cer)
  return cer_distance / max(len(gt_cer), 1)

def get_wer(word1, word2):
  if type(word1) is not str or type(word2) is not str:
    return 1

  gt_wer, pd_wer = word1.split(), word2.split()
  wer_distance = editdistance.eval(gt_wer, pd_wer)
  return wer_distance / max(len(gt_wer), 1)

def print_metrics(inputs, outputs, corrections, dataset, ocr_prediction):
    cer_base = np.mean([get_cer(gt, pred) for gt, pred in zip(outputs, inputs)])
    wer_base = np.mean([get_wer(gt, pred) for gt, pred in zip(outputs, inputs)])
    cer_corr = np.mean([get_cer(gt, format_correction(corr, dataset)) for gt, corr in zip(outputs, corrections)])
    wer_corr = np.mean([get_wer(gt, format_correction(corr, dataset)) for gt, corr in zip(outputs, corrections)])
    
    print('-' * 50)
    print(f"Dataset: {dataset} - OCR Prediction: {ocr_prediction}")
    print(f"Total: {len(corrections)}")
    print(f'CER Baseline: {(cer_base * 100):.2f}%')
    print(f'WER Baseline: {(wer_base * 100):.2f}%')
    print(f'CER Correct.: {(cer_corr * 100):.2f}%')
    print(f'WER Correct.: {(wer_corr * 100):.2f}%')
    