import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from transformers import TrainerCallback
from evaluate import load
from datasets import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve, auc, f1_score
from tqdm import tqdm
import ast

def read_fasta_to_dataframe(file_path):
    names = []
    sequences = []
    labels = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):
            name = lines[i].strip().lstrip('>')  # 去掉行首的 '>' 和多余的空格
            sequence = lines[i + 1].strip()     # 去掉多余的空格
            label = lines[i + 2].strip()        # 去掉多余的空格
            names.append(name)
            sequences.append(sequence)
            labels.append(label)
    # 创建 DataFrame
    df = pd.DataFrame({
        'name': names,
        'sequence': sequences,
        'label': labels
    })

    def convert_to_list(label_str):
        return ast.literal_eval(label_str)
        
    df["sequence"]=df["sequence"].str.replace('|'.join(["O","B","U","Z"]),"X",regex=True)
    df['sequence']=df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    df['label'] = df['label'].apply(convert_to_list)

    return df

def create_dataset(tokenizer, seqs, labels, max_length=1024):
    tokenized = tokenizer(seqs, max_length=max_length, padding=False, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    # we need to cut of labels after 1023 positions for the data collator to add the correct padding (1023 + 1 special tokens)
    labels = [l[:max_length-1] for l in labels] 
    dataset = dataset.add_column("labels", labels)
    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    metrics = {}

    labels = labels.reshape(-1)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    preds = np.argmax(logits, axis=2)
    preds = preds.reshape((-1,))

    # 过滤掉标签为-100的样本（通常是padding位置）
    valid_mask = labels != -100
    valid_preds = preds[valid_mask]
    valid_labels = labels[valid_mask]

    # 计算准确率和F1分数
    metrics["acc"] = accuracy_score(valid_labels, valid_preds)
    metrics['f1'] = f1_score(valid_labels, valid_preds)
    
    # 计算AUC
    # 提取正类的概率 [B*L, 2] -> [B*L]
    pos_probs = probs.reshape((-1, 2))[valid_mask, 1]
    
    # 计算二分类AUC
    metrics['auc'] = roc_auc_score(valid_labels, pos_probs)

    return metrics


''' Earyly stop '''
class EarlyStoppingCallBack(TrainerCallback):
    def __init__(self, patience=1):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_loss = metrics.get('eval_loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True

def get_evaluate_results(y_true, y_pred, y_pred_classes):
    results = {}
    str_unique_labels = [0,1]
    results['Accuracy'] = accuracy_score(y_true, y_pred_classes)
    results['AUC'] = roc_auc_score(y_true, y_pred)

    matrix = pd.DataFrame(confusion_matrix(y_true, y_pred_classes, labels = np.arange(2)), index = str_unique_labels, \
                    columns = str_unique_labels)
    
    tn, fp, fn, tp = np.array(matrix).ravel()
    tn, tp, fp, fn = int(tn), int(tp), int(fp), int(fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*precision*recall / (precision+recall)
    results['Sn'] = tp / (tp + fn)
    results['Sp'] = tn / (tn + fp)
    results['BACC'] = (tp / (tp + fn) + tn / (tn + fp)) / 2
    results['MCC'] = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    results['F1'] = f1
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    results['AUCPR'] = auc(recall, precision)

    return results, matrix

def evaluate(model, device, test_dataloader):
    model = model.to(device)
    model.eval()
    # Make predictions on the test dataset
    predictions = []
    # We need to collect the batch["labels"] as well, this allows us to filter out all positions with a -100 afterwards
    padded_labels = []
    # The probability value predicted for the positive class
    scores = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # Padded labels from the data collator
            padded_labels += batch['labels'].tolist()
            # Add batch results(logits) to predictions, we take the argmax here to get the predicted class
            predictions += model(input_ids, attention_mask=attention_mask, labels=None).logits.argmax(dim=-1).tolist()
            # Get the score
            score = torch.softmax(model(input_ids, attention_mask=attention_mask, labels=None).logits, dim=-1)
            scores += score[:,:,1].tolist()
    # to make it easier we flatten both the label and prediction lists
    def flatten(l):
        return [item for sublist in l for item in sublist]
    # flatten and convert to np array for easy slicing in the next step
    predictions = np.array(flatten(predictions))
    scores = np.array(flatten(scores))
    padded_labels = np.array(flatten(padded_labels))
    # Filter out labels 2
    mask = padded_labels != 2
    predictions = predictions[mask]
    scores = scores[mask]
    padded_labels = padded_labels[mask]
    # Filter out all invalid (label = -100) values
    predictions = predictions[padded_labels!=-100]
    scores = scores[padded_labels!=-100]
    padded_labels = padded_labels[padded_labels!=-100]
    # # Calculate metirics
    results, confusion_matrix = get_evaluate_results(y_true=padded_labels, y_pred=scores, y_pred_classes=predictions)
    return results, confusion_matrix