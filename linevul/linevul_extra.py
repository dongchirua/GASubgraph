import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from linevul_helpers import clean_special_token_values, get_word_att_scores

def extract_line_attention(attentions, all_tokens):
    # attentions: a tuple with of one Tensor with 4D shape (batch_size, num_heads, sequence_length, sequence_length)
    attentions = attentions[0][0]
    attention = None
    # go into the layer
    for i in range(len(attentions)):
        layer_attention = attentions[i]
        # summerize the values of each token dot other tokens
        layer_attention = sum(layer_attention)
        if attention is None:
            attention = layer_attention
        else:
            attention += layer_attention
    # clean att score for <s> and </s>
    attention = clean_special_token_values(attention, padding=True)
    # attention should be 1D tensor with seq length representing each token's attention value
    # word_att_scores -> [[token, att_value], [token, att_value], ...]
    word_att_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attention)


    # go through each line
    separator = ["Ċ", " Ċ", "ĊĊ", " ĊĊ"]
    ignore_tokens = ['<pad>', '<s>' , '</s>']
    score_sum = 0
    line = ""
    score_sum = 0
    lines_with_score = []  # line_idx, content, score
    line_idx = 0
    for i in range(len(word_att_scores)):
        if word_att_scores[i][0] not in ignore_tokens:
            score_sum += word_att_scores[i][1].detach().item()
        if ((word_att_scores[i][0] in separator) or (i == (len(word_att_scores) - 1))) and score_sum > 0:
            lines_with_score.append((line_idx, line, score_sum))
            line = ""
            score_sum = 0
            line_idx += 1
        elif (word_att_scores[i][0] not in separator) and word_att_scores[i][0] not in ignore_tokens:
            line += word_att_scores[i][0]
    return sorted(lines_with_score, key=lambda x: x[2], reverse=True), line_idx  # last line_idx mean total lines

def linevul_predict(model, dataloader, device, threshold=0.5):
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in dataloader:
        (inputs_ids, labels) = [x.to(device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > threshold
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)   
    f1 = f1_score(y_trues, y_preds)             
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold":threshold,
    }
    return result, y_trues, y_preds

