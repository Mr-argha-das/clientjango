from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import Foundation
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
matplotlib.use('Agg')  # Set the backend to non-interactive

import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from django.http import HttpResponse


def home(request):
  template = loader.get_template('index.html')
  return HttpResponse(template.render())



def pie_chart(request):
    true_data = pd.read_csv('comp702/True.csv')
    fake_data = pd.read_csv('comp702/Fake.csv')
    true_data['Target'] = ['True'] * len(true_data)
    fake_data['Target'] = ['Fake'] * len(fake_data)
    
    # Concatenate the two DataFrames
    data = pd.concat([true_data, fake_data]).sample(frac=1).reset_index(drop=True)
    data['label'] = pd.get_dummies(data.Target)['Fake']
    label_sizes = [data['label'].sum(), len(data['label']) - data['label'].sum()]
    
    # Create the pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(label_sizes, explode=[0.1, 0.1], colors=['green', 'navy'], startangle=90, shadow=True, labels=['Fake', 'True'], autopct='%1.1f%%')
    plt.axis('equal')

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode('utf-8')

    plt.close()  # Close the plot to release resources
    #graph 
    train_text, temp_text, train_labels, temp_labels = train_test_split(data['title'], data['label'], random_state=2018, test_size=0.3, stratify=data['Target'])
    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels)
    bert = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    seq_lengths = [len(title.split()) for title in train_text]
    pd.Series(seq_lengths).hist(bins=40, color='royalblue')
    plt.xlabel('Number of Words')
    plt.ylabel('Number of Titles')
    
    # Define the maximum title length
    MAX_LENGTH = 15
    
    # Tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True
    )
    
    # Tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True
    )
    tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length=MAX_LENGTH,
    pad_to_max_length=True,
    truncation=True
    )
        # response = HttpResponse(content_type='image/png')
        # response['Content-Disposition'] = 'attachment; filename="pie_chart.png"'
        # response.write(base64.b64decode(img_data))
    context = {'image_url': f"data:image/png;base64,{img_data}"}
    return render(request, 'chart.html', context)
  
if __name__ == "__main__":
    Foundation.NSApplication.sharedApplication()
    Foundation.NSApplication.sharedApplication().run()