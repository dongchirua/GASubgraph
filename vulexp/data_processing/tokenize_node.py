#!/usr/bin/env python
# coding: utf-8

import re
import nltk
from vulexp.data_processing.python_keywords import l_funcs, keywords, puncs, operators


def symbolic_tokenize(code, symbol_table):
    """
    Method that tokenizes code
    :param code:
    :param symbol_table: this is global mapping is used for storing
    :return: tokens and updated symbol_table
    """
    tmp = []
    for v in code:
        if v == '(' or v == '[' or v == ')' or v == ']':
            tmp.append(' ')
            tmp.append(v)
            tmp.append(' ')
        else:
            tmp.append(v)
    code = ''.join(tmp)

    tokens = nltk.wordpunct_tokenize(code)
    c_tokens = []
    for t in tokens:
        if t.strip() != '':
            c_tokens.append(t.strip())
    f_count = symbol_table.get('f_count', 1)
    var_count = symbol_table.get('var_count', 1)
    # symbol_table = {}
    final_tokens = []
    for idx in range(len(c_tokens)):
        t = c_tokens[idx]
        if t in keywords:
            final_tokens.append(t)
        elif t in puncs:
            final_tokens.append(t)
        elif t in l_funcs:
            final_tokens.append(t)
        elif idx < len(c_tokens) - 1 and c_tokens[idx + 1] == '(':
            if t in keywords:
                final_tokens.append(t)
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t])
            idx += 1

        elif t.endswith('('):
            t = t[:-1]
            if t in keywords:
                final_tokens.append(t + '(')
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t] + '(')
        elif t.endswith('()'):
            t = t[:-2]
            if t in keywords:
                final_tokens.append(t + '()')
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t] + '()')
        elif re.match("^\"*\"$", t) is not None:
            final_tokens.append("STRING")
        elif re.match("^[0-9]+(\.[0-9]+)?$", t) is not None:
            final_tokens.append("NUMBER")
        elif t in operators:
            final_tokens.append(t)
        else:
            if t not in symbol_table.keys():
                symbol_table[t] = "VAR" + str(var_count)
                var_count += 1
            final_tokens.append(symbol_table[t])
    symbol_table['var_count'] = var_count
    symbol_table['f_count'] = f_count
    return ' '.join(final_tokens).split(), symbol_table
