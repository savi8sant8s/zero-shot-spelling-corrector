
import re

def format_correction(correction, dataset):
  if dataset == 'iam':
    punctuation_pattern = r'([,;."?!:])'
    formatted_correction = re.sub(punctuation_pattern, r' \1', correction)
    return formatted_correction
  return correction

def get_sentences(df, ocr_prediction):
    sentences = {}
    for _, row in df.iterrows():
        filename = row['filename']
        if ocr_prediction  == 'bressay':
            sentence_id = filename[:11]
        if ocr_prediction == 'rimes':
            sentence_id = filename[:11]
        elif ocr_prediction == 'iam':
            sentence_id = filename[:9]
        
        if sentence_id not in sentences:
            sentences[sentence_id] = []
        
        sentences[sentence_id].append(row['prediction'])
    
    return list(sentences.values())