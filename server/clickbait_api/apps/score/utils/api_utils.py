from django.conf import settings
import os
import csv

#APP_DIR = "apps/score"
DATASET_PATH = "apps/score/data/source"


def create_dataset(title, text):
    #print(APP_DIR)
    print(DATASET_PATH)
    with open(os.path.join(settings.BASE_DIR, DATASET_PATH, "processing_article.csv"), "w", encoding='utf-8-sig') as f:
        header = ['id', 'title', 'text', 'label']
        writer = csv.DictWriter(f, fieldnames=header, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerow({'id':'1_0001', 'title':title, 'text':text, 'label': 0})
