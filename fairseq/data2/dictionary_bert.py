# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from collections import Counter
from multiprocessing import Pool
from pypinyin import lazy_pinyin, Style

import torch
from fairseq import utils
from fairseq.binarizer import safe_readline
from fairseq.data import data_utils
# from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line


class DictionaryBert:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        bert_dictionary_file,
        bos="[CLS]",
        pad="[PAD]",
        eos="[SEP]",
        unk="[UNK]",
        mask="[MASK]",
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word, self.mask_word = bos, unk, pad, eos, mask
        self.symbols = []
        self.count = []
        self.indices = {}
        self.pinyin_dict = {}

        self.add_from_file(bert_dictionary_file)
        assert len(self.symbols) == 21128, "Bert dictionary should contains 21128 words, but get %d" % len(self.symbols)

        # bert模型不需要将特殊符号设定在下标0-3
        # self.bos_index = self.add_symbol(bos)
        # self.pad_index = self.add_symbol(pad)
        # self.eos_index = self.add_symbol(eos)
        # self.unk_index = self.add_symbol(unk)

        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        # self.nspecial = len(self.symbols)

        try:
            self.bos_index = self.index(bos)
            self.pad_index = self.index(pad)
            self.eos_index = self.index(eos)
            self.unk_index = self.index(unk)
            self.mask_index = self.index(mask)
        except:
            raise Exception("Special token [PAD], [CLS], [SEP], [UNK], [MASK] not found ")

        assert self.bos_index == 101, "index of [CLS] should be 101 but get %d" % self.bos_index
        assert self.pad_index == 0, "index of [PAD] should be 0 but get %d" % self.pad_index
        assert self.eos_index == 102, "index of [SEP] should be 102 but get %d" % self.eos_index
        assert self.unk_index == 100, "index of [UNK] should be 100 but get %d" % self.unk_index
        assert self.mask_index == 103, "index of [MASK] should be 103 but get %d" % self.mask_index

        self.nspecial = 0

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
    ):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(t, bpe_symbol, escape_unk, extra_symbols_to_ignore, include_eos=include_eos)
                for t in tensor
            )

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        extra_symbols_to_ignore.add(self.eos())

        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, "bos_index"):
            extra_symbols_to_ignore.add(self.bos())

        sent = " ".join(
            token_string(i)
            for i in tensor
            if utils.item(i) not in extra_symbols_to_ignore
        )
        
        return data_utils.process_bpe_symbol(sent, bpe_symbol)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""

        self.add_pinyin(word, n)
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def add_pinyin(self, word, n):
        pinyin = lazy_pinyin(word, style=Style.NORMAL)
        if len(pinyin) == 0:
            return
        pinyin = pinyin[0]
        if pinyin in self.pinyin_dict:
            self.pinyin_dict[pinyin][word] = self.pinyin_dict[pinyin][word] + n if word in self.pinyin_dict[pinyin] else n
        else:
            self.pinyin_dict[pinyin] = {word: n}

    def get_random_word(self, use_freq_as_weight=False):

        def valid(index):
            if 1 <= index <= 99:
                return False
            if 173 <= index <= 670:
                return False
            if self.count[index] == 0:
                return False
            return True

        if not use_freq_as_weight:
            index = random.randint(self.nspecial, len(self.symbols) - 1)
            while not valid(index):
                index = random.randint(self.nspecial, len(self.symbols) - 1)
        else:
            index_list, weights = [n for n in range(len(self.symbols))], self.count
            index_list, weights = index_list[self.nspecial:], weights[self.nspecial:]
            index = random.choices(index_list, weights=weights)[0]
            while not valid(index):
                index = random.choices(index_list, weights=weights)[0]

        return index

    def get_homophone(self, index):
        word = self.symbols[index]
        pinyin = lazy_pinyin(word, style=Style.NORMAL)[0]
        if pinyin not in self.pinyin_dict or len(self.pinyin_dict[pinyin]) == 1:
            homophone = word
        else:
            words, weights = [], []
            for word, num in self.pinyin_dict[pinyin].items():
                words.append(word)
                weights.append(num)
            homophone = random.choices(words, weights=weights)[0]

        return self.index(homophone)

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = Counter(
            dict(
                sorted(zip(self.symbols[self.nspecial :], self.count[self.nspecial :]))
            )
        )
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

        self.pinyin_dict = {}
        for i in range(self.nspecial, len(self.symbols)):
            word, count = self.symbols[i], self.count[i]
            self.add_pinyin(word, count)

        self.pad_to_multiple_(padding_factor)

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        if padding_factor > 1:
            i = 0
            while len(self) % padding_factor != 0:
                symbol = "madeupword{:04d}".format(i)
                self.add_symbol(symbol, n=0)
                i += 1

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    def mask(self):
        return self.mask_index

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls(f)
        # d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)

        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file.".format(word)
                    )
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            if not os.path.exists(os.path.dirname(f)):
                os.makedirs(os.path.dirname(f))
            with open(f, "w", encoding="utf-8") as fd:
                return self.save(fd)
        for k, v in kv_iterator:
            print("{} {}".format(k, v), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(
            f,
            zip(
                ex_keys + self.symbols[self.nspecial :],
                ex_vals + self.count[self.nspecial :],
            ),
        )

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ) -> torch.IntTensor:
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    @staticmethod
    def _add_file_to_dictionary_single_worker(
        filename, tokenize, eos_word, worker_id=0, num_workers=1
    ):
        counter = Counter()
        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                # f.tell() returns only an opaque number which can
                # return to the position in the file via f.seek()
                # and does not necessarily represent a byte position
                # in the file. However, f.tell() is faithful to the
                # byte position _most of the time_. Thus we can just
                # check against the file size to prevent early exit.
                if f.tell() > end and f.tell() < size:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        Dictionary._add_file_to_dictionary_single_worker,
                        (filename, tokenize, dict.eos_word, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                Dictionary._add_file_to_dictionary_single_worker(
                    filename, tokenize, dict.eos_word
                )
            )
