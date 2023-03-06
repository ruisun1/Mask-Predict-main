import os
from fuzzychinese import Radical, Stroke
from pypinyin import pinyin, lazy_pinyin, Style

radical = Radical()
stroke = Stroke()


def read_vocab(file_path):

    vocab = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            word, num = line[0], int(line[1])
            vocab[word] = num

    return vocab


def read_file(file_path):
    if not os.path.exists(file_path):
        print("File not exists: %s" % file_path)
        raise

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(line.strip())

    return data


def write_file(file_path, data, join_mark=""):
    with open(file_path, "w", encoding="utf-8") as f:
        for line in data:
            text = join_mark.join(line)
            # text = text.replace("##", "")
            f.write("%s\n" % text)


def read_ids_vocab(file_path):

    vocab = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            if len(line) < 3:
                continue
            token, part = line[1], line[2]
            vocab[token] = part

    return vocab


def pinyin_similar(a, b):

    a_pinyins, b_pinyins = pinyin([a, b], style=Style.TONE3, heteronym=True)

    if len(a_pinyins) == 0 or len(b_pinyins) == 0:
        return False

    ap, bp = a_pinyins[0][:-1], b_pinyins[0][:-1]
    ap = ap.replace("ing", "in")
    bp = bp.replace("ing", "in")
    if ap == bp:
        return True

    return False


def get_glyph(token):

    subtoken = radical.get_radical(token)
    if subtoken == "":
        return ""
    if subtoken == token or token in subtoken:
        return token

    result = ""
    for part in subtoken:
        s = stroke.get_stroke(part)
        if len(s) == 1:
            return token
        if len(s) > 7:
            part = get_glyph(part)
        result += part

    return result


def glyph_similar(a, b):

    if len(a) > 1 or len(b) > 1:
        return False

    a_glyph = get_glyph(a)
    b_glyph = get_glyph(b)

    union_part = set(a_glyph).intersection(b_glyph)
    union_stroke = sum([len(stroke.get_stroke(part)) for part in union_part])

    a_stroke = sum([len(stroke.get_stroke(part)) for part in a_glyph])
    b_stroke = sum([len(stroke.get_stroke(part)) for part in b_glyph])

    if a_stroke == 0 or b_stroke == 0:
        return False

    rate = union_stroke / (a_stroke + b_stroke)
    return rate > 0.38


def is_similar_token(a, b):

    return pinyin_similar(a, b) or glyph_similar(a, b)


if __name__ == '__main__':

    print(pinyin_similar("金", "今"))
