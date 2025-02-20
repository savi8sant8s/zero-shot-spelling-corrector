import os
import re
import string
import openai
import numpy as np
import tensorflow as tf
from ollama import chat, ChatResponse
from google import genai
from google.genai import types

class SpellingModel():
    """
    A model for correcting spelling errors in text using LLMs.
    """

    def __init__(self, provider='openai', model='gpt-4o-mini', prompt_type=1, env_file='.env'):
        """
        Initializes the spelling model.

        Parameters
        ----------
        provider : str, optional
            The provider of the LLM. Options are 'openai', 'ollama', or 'gemini'.
        model : str, optional
            The model to use for correction.
        prompt_type : int, optional
            The type of prompt to use. Options are 1 or 2.
        env_file : str, optional
            Path to the environment file.
        """

        self.env_file = env_file
        self.provider = provider
        self.model = model
        self.prompt_type = prompt_type
        self.instruction = self._get_instruction()

        if provider == 'openai':
            openai.api_key = self._get_openai_api_key()

    def _get_instruction(self):
        """
        Generates the instruction for the model based on the prompt type.
        Returns
        -------
        str
            Instruction for the model.
        """
        
        if self.prompt_type == 1:
            return ('Correct only obvious spelling mistakes in words within tags. '
                    'Keep the number of tags the same. '
                    'Do not add extra text or change correct text. '
                    'Maintain the unique and historical style of the text.')
        elif self.prompt_type == 2:
            return ('Act as an document analyst specializing in OCR correction. '
                    'Your task is to correct OCR errors in the text. '
                    'Guidelines: '
                    '1. Ensure corrections accurately reflect the text language and conventions. '
                    '2. Keep punctuation marks from the original text, and do not add new punctuation. '
                    '3. Preserve original word splits. '
                    '4. Keep the hyphenation in the original text. '
                    '5. Do not delete words unless duplicated. '
                    '6. Do not modify the end of the text. '
                    '7. Do not correct numbers.')
        else:
            raise ValueError("Invalid prompt type. Choose either 1 or 2.")
        
    def _get_openai_api_key(self):
        """
        Retrieves the API key from the environment file or environment variables.

        Returns
        -------
        str
            Retrieved API key.
        """

        if os.path.isfile(self.env_file):
            with open(self.env_file, 'r') as file:
                for line in file:
                    if line.startswith(f"OPENAI_API_KEY="):
                        return line.split('=', 1)[1].strip()

        return os.getenv('OPENAI_API_KEY')

    def _encode_batch(self, batch):
        """
        Encodes a batch of text.

        Parameters
        ----------
        batch : list
            Batch of text data.

        Returns
        -------
        list
            Encoded text data.
        """

        tokens_length = 0
        encoded = [[]]

        for i, data in enumerate(batch):
            for j, top_path in enumerate(data):
                for u, text in enumerate(top_path.split('\n')):
                    pp_text = f'<{i}.{j}.{u}> {text} </{i}.{j}.{u}>'

                    pp_text_tokens = re.sub(f'([{re.escape(string.punctuation)}])', r' \1 ', pp_text).split()
                    pp_text_tokens_length = len(pp_text_tokens)

                    tokens_length += pp_text_tokens_length
                    encoded[-1].append(pp_text)

        return encoded

    def _decode_batch(self, batch, encoded_batch, corrected_encoded_batch):
        """
        Decodes a processed batch of text.

        Parameters
        ----------
        batch : list
            Batch of text data.
        encoded_batch : list
            Encoded batch of text data for fallback.
        corrected_encoded_batch : list
            Processed and corrected text data.

        Returns
        -------
        list
            Decoded and corrected text data.
        """
        if not isinstance(batch, list):
            batch = batch.tolist()
        
        if len(corrected_encoded_batch) != len(encoded_batch):
            return batch

        pattern = re.compile(r'<([0-9]+\.[0-9]+\.[0-9]+)>(.*?)<\/\1>', re.DOTALL)

        for i, (corrected, fallback) in enumerate(zip(corrected_encoded_batch, encoded_batch)):
            corrected_text = ''.join(corrected) if corrected else ''
            fallback_text = ''.join(fallback) if fallback else ''
            
            corrected_matches = pattern.findall(corrected_text)
            fallback_matches = pattern.findall(fallback_text)
            
            if len(corrected_matches) != len(fallback_matches):
                corrected_encoded_batch[i] = fallback_text
                corrected_matches = fallback_matches
            
            for match in corrected_matches:
                tags = tuple(map(int, match[0].split('.')))
                batch_item = batch[tags[0]][tags[1]] if tags[0] < len(batch) and tags[1] < len(batch[tags[0]]) else None
                
                if batch_item is None or not isinstance(batch_item, (str, list)):
                    continue
                
                text_content = match[1].strip()
                if not text_content:
                    corrected_encoded_batch[i] = fallback_text
                    continue
                
                if isinstance(batch_item, str):
                    batch[tags[0]][tags[1]] = []
                
                batch[tags[0]][tags[1]].append(text_content)

        for i, data in enumerate(batch):
            for j, item in enumerate(data):
                if isinstance(item, list):
                    batch[i][j] = '\n'.join(item)
        
        return batch


    def _request_ollama(self, batch):
        """
        Sends a request to the Ollama API.

        Parameters
        ----------
        batch : list
            Batch of text data to send.

        Returns
        -------
        list
            Processed text responses.
        """
        
        messages = [
            {'role': 'system', 'content': self.instruction},
            {'role': 'user', 'content': '\n\n'.join(batch)},
        ]
        try:
           response: ChatResponse = chat(
                model=self.model,
                messages=messages,
                keep_alive='1m',
                options={'temperature': 0}
            )
           output = response['message']['content']
           return output.strip()
        except Exception as err:
            print(f"Ollama response error: {err}")

        return batch
    
    def _request_openai(self, batch):
        """
        Sends a request to the OpenAI API and handles retries.

        Parameters
        ----------
        batch : list
            Batch of text data to send.

        Returns
        -------
        list
            Processed text responses.
        """

        messages = [
            {'role': 'system', 'content': self.instruction},
            {'role': 'user', 'content': '\n\n'.join(batch)},
        ]

        response = openai.chat.completions.create(model=self.model,
                                                    messages=messages,
                                                    temperature=0)

        return response.choices[0].message.content.strip().split('\n')

    def _request_gemini(self, batch):
        gemini_api_key = os.getenv('GEMINI_API_KEY')   
        client = genai.Client(api_key=gemini_api_key)

        response = client.models.generate_content(
            model=self.model, 
            config=types.GenerateContentConfig(
                system_instruction=self.instruction,
                temperature=0,
            ),
            contents=['\n\n'.join(batch)],
        )
        
        return response.text
    
    def predict(self, x, steps, verbose=1):
        """
        Predicts the corrections for the given data of texts.

        Parameters
        ----------
        x : list
            Data of texts.
        steps : int
            Number of steps for processing.
        verbose : int, optional
            Verbosity mode.

        Returns
        -------
        np.ndarray
            Array of corrected texts.
        """
        
        progbar = tf.keras.utils.Progbar(target=steps, unit_name='spelling', verbose=verbose)

        corrections = []
        batch_index = 0
        batch_size = int(np.ceil(len(x) / steps))

        for step in range(steps):
            progbar.update(step)

            batch = x[batch_index:batch_index + batch_size]
            batch_index += batch_size

            request_ = self._request_ollama
            if self.provider == 'ollama':
                request_ = self._request_ollama
            elif self.provider == 'gemini':
                request_ = self._request_gemini
            else:
                request_ = self._request_openai
                
            encoded_batch = self._encode_batch(batch)
            corrected_encoded_batch_batch = [request_(item) for item in encoded_batch]
            corrected_batch = self._decode_batch(batch, encoded_batch, corrected_encoded_batch_batch)

            corrections.extend(corrected_batch)
            progbar.update(step + 1)

        corrections = np.array(corrections, dtype=object)

        return corrections
    