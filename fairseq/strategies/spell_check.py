import os
import torch
from tqdm import tqdm
from pytorch_pretrained_bert import BertModel, BertForMaskedLM, BertTokenizer, BasicTokenizer

from utils import read_file, write_file, read_vocab, is_similar_token


def check_spelling_error(sentence):

    base_tokens = basic_tokenizer.tokenize(sentence)
    base_tokens = ["[CLS]"] + base_tokens + ["[SEP]"]

    tokens = [x if x in bert_tokenizer.vocab else "[UNK]" for x in base_tokens]

    if len(tokens) > 180:
        return sentence, 0

    ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    mask_ids = bert_tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

    mask_position, input_tokens = [], []
    for i in range(1, len(tokens) - 1):
        token = tokens[i]
        mask_position.append(i)
        input_tokens.append(ids[:])
        input_tokens[-1][i] = mask_ids

    if len(input_tokens) == 0:
        return sentence, 0
    
    bert_predictions = None
    batch_size = 1
    for i in range(0, len(input_tokens), batch_size):
        batch_tokens = input_tokens[i: i + batch_size]
        if len(batch_tokens) == 0:
            continue
        #print("batch_toekns",torch.LongTensor(batch_tokens).size())
        predictions = bert_model(torch.LongTensor(batch_tokens).cuda()).cpu()
       # print("after bert", predictions.size())
        bert_predictions = predictions if bert_predictions is None else torch.cat([bert_predictions, predictions], dim=0)

    correct_tokens = base_tokens[:]
    correct_num = 0
    f=open("./record.txt","a",encoding="utf-8")
    result = base_tokens[1:-1]
    #print (result)
    result_sen = []
   # f3 = open("./aftercorrection.txt","w",encoding = "utf-8")
    for i, mask_p in enumerate(mask_position):
        prediction = bert_predictions[i, mask_p]
        score, index = torch.topk(prediction, k=1)
        pred_tokens = bert_tokenizer.convert_ids_to_tokens(index.tolist())
        result_sen.append(pred_tokens[0])
        #print(pred_tokens, score)
        if(result[i] not in pred_tokens):
            result[i]="<mask>"
    #    print(result)

        if base_tokens[mask_p] in pred_tokens:
            pred_tokens = pred_tokens[:1]

        for pt in pred_tokens:
            if pt == "[UNK]" or pt == base_tokens[mask_p]:
                continue
            if is_similar_token(pt, tokens[mask_p]):
                correct_tokens[mask_p] = pt
                correct_num += 1

                tokens[mask_p]=pt
                f.writelines(tokens)
                break
    print(result_sen) 
    sen = ' '.join(result_sen)
    f3.writelines(sen+'\n')
    f2.writelines(' '.join(result)+'\n')
        
        

        # mask_tokens = tokens[:]
        # mask_tokens[mask_p] = "[MASK]"
        # print("".join(mask_tokens))
        # print(pred_tokens)

    correct_sentence = "".join(correct_tokens[1: -1])

    return correct_sentence, correct_num


if __name__ == '__main__':

    data = read_file('./codataz_spanc_msk2/test.2')

    bert_model = BertForMaskedLM.from_pretrained("/home/sunrui/bert-base-chinese/").cuda()
    bert_model.eval()

    bert_tokenizer = BertTokenizer.from_pretrained("/home/sunrui/bert-base-chinese/vocab_bert.txt")
    basic_tokenizer = BasicTokenizer(do_lower_case=False)

    #vocab = read_vocab("/home/sunrui/Mask-Predict-main/vocab.txt")
    f2=open("./mask2.txt","a+",encoding="utf-8")
    # s, n = check_spelling_error("随着中国经济突飞猛近，建造工业与日俱增。", vocab)
    f3 = open("./aftercorrection.txt","a+",encoding = "utf-8")
    new_data = []
    change_num = 0
    for sentence in tqdm(data):
         correct_sentence, num = check_spelling_error(sentence)
         change_num += num

    write_file("./%s.src" % data_type, new_data, join_mark=" ")
    print(">>> Change number: %d" % change_num)
