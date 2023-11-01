from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
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
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from django.http import HttpResponse


def home(request):
  template = loader.get_template('index.html')
  return HttpResponse(template.render())

def page1(request):
    template = loader.get_template('assign-page1.html')
    return HttpResponse(template.render())

def page2(request):
    template = loader.get_template('assign-page2.html')
    return HttpResponse(template.render())

def page3(request):
    template = loader.get_template('assign-page3.html')
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
    image_buf = BytesIO()
    plt.savefig(image_buf, format='png')
    image_buf.seek(0)
    image = Image.open(image_buf)
    image.save('image_grahp.png')
    with open('image_grahp.png', 'rb') as image_file:
        image_binary = image_file.read()
    image_base64 = base64.b64encode(image_binary).decode('utf-8')
    context = {
        'image_url': f"data:image/png;base64,{img_data}", 
        'image_grahp': f"data:image/png;base64,{image_base64}"
        }
    return render(request, 'chart.html', context)

if __name__ == "__main__":
    Foundation.NSApplication.sharedApplication()
    Foundation.NSApplication.sharedApplication().run()