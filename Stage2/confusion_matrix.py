import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 给定的混淆矩阵
# city='wuhan'
city='guangzhou'
# algorithm_list=['sourceonly','aggregation','conmix','ours','MT-MTDA','graph','ALT','D3GU'] #aggregation,sourceonly,ours
algorithm_list=['D3GU'] #aggregation,sourceonly,ours


for algorithm in algorithm_list:
    if city=='guangzhou':
        if algorithm == 'sourceonly':
            #source-only guangzhou
            confusion_matrix = np.array([
        [ 193,  236,   11,  136,  128,    6,   27,    4,  92,    3,    0,   14,   1,  39,  5,   1],
        [   8,  280,   64,   26,   89,  154,   41,   41,  145,    6,    2,   20,   0,  34,  22,  8],
        [  10,  147,  375,   38,   82,  177,  101,  126,  198,    5,   19,   47,   12,  141,  32,  50],
        [  42,  109,    2,  218,  297,   25,   20,    8,   70,   10,   14,   16,   10,   39,  19,  19],
        [  15,  208,    8,   36,  157,  151,   58,   14,   68,   17,   10,   16,    6,   18,   4,   7],
        [   0,    6,   27,   13,   23,  236,    9,   64,   29,    2,   18,   41,    3,   44,   8,  31],
        [  15,   43,    3,   14,   38,   67,  1725,   7,  424,    7,   24,   33,   38,   88,  13,  31],
        [   2,   25,    5,   19,   59,   24,   25,   75,   36,   14,  134,  111,   25,   88,  18,  44],
        [  16,   28,    3,    0,    9,   25,  150,    2,  272,    1,   16,   11,   14,   44,  11,  26],
        [   1,    5,    2,    7,   10,    3,    5,    3,    8,  908,  307,  120,   31,   34,    5,  50],
        [  10,   12,    0,   10,   17,   10,   20,   11,   43,   44,  335,  198,   52,  101,   17,  71],
        [   1,    9,    1,    5,   10,    1,   33,    2,   61,   23,   63,   87,   21,  147,   33,  51],
        [   3,    2,    0,    5,    9,    4,   50,    4,   33,   24,   54,  259,  130,   79,   12,  220],
        [   5,   27,    5,    9,   11,    3,  203,    3,  166,   21,   15,   40,    8,   356,   26,  29],
        [   6,    3,    0,    2,   17,    6,   23,    5,  102,   22,   34,   149,   15,   179,  406, 131],
        [   5,    4,    3,   20,   15,    2,   12,    2,   17,   29,   97,   63,   35,   39,   18, 1681]
    ])

        elif algorithm=='D3GU':
            confusion_matrix = np.array([
    [507, 56, 13, 241, 21, 0, 26, 0, 12, 10, 0, 0, 2, 8, 0, 0],
    [26, 376, 132, 296, 62, 7, 9, 3, 1, 7, 0, 0, 3, 8, 0, 0],
    [49, 190, 720, 363, 31, 28, 7, 9, 3, 66, 24, 0, 6, 35, 11, 5],
    [114, 25, 7, 592, 85, 1, 3, 7, 3, 7, 12, 1, 11, 46, 2, 2],
    [21, 112, 2, 306, 219, 7, 27, 7, 13, 25, 29, 0, 8, 13, 3, 0],
    [7, 27, 86, 80, 44, 146, 0, 33, 0, 37, 25, 2, 8, 21, 13, 16],
    [19, 16, 8, 28, 4, 0, 2099, 19, 148, 85, 18, 3, 14, 69, 15, 10],
    [8, 4, 18, 35, 36, 7, 9, 133, 1, 91, 189, 5, 52, 50, 22, 28],
    [20, 5, 11, 14, 6, 1, 147, 6, 336, 31, 20, 1, 3, 12, 5, 9],
    [0, 0, 3, 6, 5, 4, 4, 6, 1, 1100, 152, 29, 16, 19, 17, 118],
    [3, 0, 6, 42, 13, 1, 5, 19, 2, 173, 389, 62, 79, 71, 25, 52],
    [1, 1, 0, 11, 1, 0, 14, 0, 26, 83, 117, 40, 27, 82, 105, 40],
    [0, 1, 0, 49, 2, 0, 10, 11, 12, 66, 105, 80, 400, 20, 15, 101],
    [2, 2, 6, 8, 8, 0, 174, 0, 82, 41, 16, 4, 31, 472, 31, 15],
    [1, 7, 0, 22, 1, 0, 22, 5, 38, 58, 77, 15, 12, 130, 562, 145],
    [6, 0, 0, 63, 3, 0, 7, 9, 21, 61, 94, 13, 30, 39, 6, 1687]
])
    
        elif algorithm=='ALT':
            confusion_matrix = np.array([
    [  72,   45,    0,    0,    1,    0,   11,    0,    0,  166,   35,  247,  222,    7,  6,    0],
    [   7,  263,    0,    6,    0,    2,   25,   11,    1,   75,    6,    5,   17,   28,  139,    0],
    [   2,    1,  735,  473,   39,    9,   32,    1,   45,    0,    0,    1,    8,    1,    0,    0],
    [   3,   18,   20,  504,   67,   42,  135,    5,   36,    2,    1,    2,   13,    2,    4,    4],
    [   3,   48,   16,  206,   30,   14,  118,    9,   23,    8,    3,    7,    4,    0,    8,    0],
    [   2,   28,    3,  249,  139,  130,   36,   13,  110,    0,    2,    3,    2,    0,   68,    1],
    [   3,  107,    0,   57,   14,    8,  350,   22,    7,   24,   17,   16,   20,   16,  144,    0],
    [  15,   45,   16,  339,   25,    7,   82,  364,   49,    6,    1,    2,   24,    7,   16,    2],
    [   3,   27,  152,  274,   23,   15,   57,   21, 1234,    0,    0,    3,   19,    3,   11,    0],
    [   1,   20,    0,    1,    0,    0,   24,    1,    0,  248,  331,   17,   90,   37,   71,    0],
    [   4,   28,    0,   34,    6,    1,   27,    2,    5,   80, 1089,    1,   43,   17,   47,   28],
    [   7,   21,    1,   10,    0,    0,   24,    0,    0,   85,    4,  378,  304,   12,    5,    2],
    [   6,    5,    8,    6,    0,    0,   23,    0,    2,  268,   72,   41,  188,   66,   32,    2],
    [   1,    0,    0,   25,    2,    0,   12,    0,   10,   15,  153,   12,   31,  187,    0,   33],
    [   4,  213,    0,   10,    9,   11,   49,    8,    6,   61,    7,    4,   15,    2,  1919,   11],
    [   6,    3,    3,  201,   24,   10,   71,   12,   12,   10,   59,   13,   45,   42,    7,  102]
        ])

        elif algorithm=='ours':
            # ours guangzhou
            confusion_matrix = np.array([
            [619,  15,   4,  219,  31,   3,   0,   0,   3,   0,   0,   1,   0,   1,   0,   0],
            [ 46, 419, 298,   18,  18, 106,   4,   0,   7,   0,   2,   0,   0,  22,   0,   0],
            [ 18,  79, 1227,   7,  32, 107,   6,   2,   5,  17,  10,  24,   2,  15,   5,   4],
            [136,   2,   1,  613, 150,   5,   0,   0,   0,   0,   2,   5,   1,   2,   1,   0],
            [ 37, 292,  44,  80, 191, 119,   8,   0,   5,   0,   5,   1,   0,   8,   3,   0],
            [  3,   1,  97,   9,   1, 306,   0,  19,  12,   3,  14,  57,  13,  10,   5,   4],
            [ 17,   9,  17,  25,   6,   0, 2291,   0,  80,   1,   9,  12,   8,  90,   3,   2],
            [  3,  13,  21,  34,  16,   8,  15,  45,   5,  31, 206, 114,  92,  77,  22,   2],
            [ 45,   1,   1,  42,   2,  19, 150,   0, 261,   0,   2,   4,  14,  66,  20,   1],
            [  2,   0,   0,   6,   0,   2,   2,   0,   0, 852, 378,  63,  27,   7,   6, 154],
            [  5,   3,   1,  26,   2,   3,   6,   0,  18,  60, 412, 181, 103,  77,  43,  11],
            [ 10,   0,   0,   4,   2,   0,   5,   0,  26,  35,  44, 148,  33, 147,  69,  25],
            [ 32,   1,   2,   0,   0,   0,   9,   0,   3,   5,  57, 147, 590,   9,   2,  31],
            [ 21,   1,   5,  26,   6,   6, 126,   0,  99,  15,   5,  46,  34, 457,  65,  15],
            [  4,   0,   0,  15,  23,   0,   2,   0,  21,  27,  13, 205,  32, 120, 579,  59],
            [  3,   0,   0,   9,   0,   5,   2,   0,   0,  43,  57,  50, 282,  32,  15, 1544]
        ])

        elif algorithm=='aggregation':
            confusion_matrix = np.array([
        [  12,  340,   27,  427,   37,    1,    0,    1,   44,    0,    0,    3,    4,    0,    0,    0],
        [   3,  468,  274,   13,   69,   59,    5,   33,    2,    0,    0,    1,    7,    5,    1,    0],
        [   2,   99,  678,    2,   74,  314,    4,  327,    9,    3,    1,    1,   34,   10,    1,    1],
        [  12,   55,    0,  571,  211,    4,   31,    2,    6,    0,    0,    4,   21,    1,    0,    0],
        [  46,  292,   35,   23,  258,   81,   12,    1,    4,    0,    0,    0,   36,    5,    0,    0],
        [  15,    1,   11,    0,   13,  350,    0,  148,    0,    0,    0,    2,    6,    7,    0,    1],
        [  95,   44,    2,    7,   10,    2, 1262,  139,  786,   13,    6,    9,   71,  115,    3,    6],
        [   6,    5,    1,    0,   46,   51,   11,  253,    1,    4,  160,   70,   80,   10,    5,    1],
        [ 415,    5,    3,    0,    6,    0,   23,   20,   87,    1,    0,    7,   35,   16,    9,    1],
        [   4,    0,    1,    0,   13,    3,    0,    1,    0,  1107,  278,   19,   23,   14,   11,   25],
        [  14,    0,    2,   13,   40,    2,    0,    0,    9,   196,  389,  100,   90,   69,   16,   11],
        [  14,    2,    0,    3,    6,    2,    2,    2,   31,   40,   58,  105,   49,   89,  123,   22],
        [   3,    3,    0,    1,    5,    1,    2,    7,   20,   99,   42,  520,  126,    9,    6,   44],
        [  83,    2,    3,    3,   13,    7,   20,    5,  193,   43,   12,   25,   94,   366,   43,   15],
        [  44,    0,    1,    2,    2,    1,    1,    9,   26,  111,   10,   42,   27,   85,  708,   31],
        [   6,    0,    2,    0,   14,    7,    0,    3,    0,   22,   75,   59,   33,   39,   60, 1722]
    ])
        elif algorithm=='conmix':
            confusion_matrix = np.array([
        [ 485,  135,    0,  146,   95,    0,    2,    0,   30,    0,    0,    1,    0,    0,    0,    2],
        [  26,  574,  108,   12,   21,  160,   15,    2,   11,    0,    1,    1,    1,    6,    2,    0],
        [  12,  222,  752,   11,   57,  321,   27,   78,   14,    1,    4,   36,    6,    5,    2,   12],
        [ 141,   63,    0,  453,  226,    8,    6,    0,    3,    2,    3,    0,    3,    3,    0,    7],
        [  27,  383,    5,   62,  134,  113,   33,    0,    4,    3,    8,    0,    7,    5,    0,    9],
        [   7,   18,   36,   15,   20,  320,    2,   66,    1,    0,   13,   19,   18,    6,    2,   11],
        [  32,   18,    4,    4,    5,    8, 2231,    1,  156,    4,   16,    3,    7,   22,   10,   49],
        [  25,    6,   14,   25,   49,   14,   17,   88,    2,   27,  200,   82,   82,   29,   15,   29],
        [  57,    5,    0,    2,    6,    9,  162,    0,  323,    0,    0,    7,   11,   12,   14,   20],
        [   4,    1,    3,    0,    7,    0,    2,    0,    1, 1099,  240,   29,   39,    9,    2,   63],
        [   8,    7,    1,   19,    8,    5,   12,    0,   15,  132,  381,  109,  109,   64,   19,   62],
        [   5,    4,    1,    8,    3,    0,   18,    0,   39,   45,   51,   68,   28,  126,   81,   71],
        [   2,    3,    0,    1,    6,    2,   15,    0,    6,   84,   41,  121,  471,   15,    1,  120],
        [  42,    3,    1,   25,   17,    0,  220,    0,  169,   34,   23,   14,   26,  236,   69,   48],
        [  28,    0,    0,   12,    3,    1,   37,    2,   39,   35,   15,  104,   32,   83,  626,   83],
        [   0,    0,    1,    5,    3,    2,   12,    0,    6,   45,   50,   18,   68,    4,    1, 1827]
    ])
        elif algorithm =='MT-MTDA':
            confusion_matrix = np.array([
        [ 185,   13,    0,    0,    0,    0,    4,    0,    0,  484,   22,  128,   34,    2,   24,    0],
        [  28,  163,    0,    5,   13,    3,   14,    0,    2,   69,   23,   10,    8,    6,  284,    0],
        [  20,    2,  883,  262,  210,   30,   10,    0,   43,    3,    0,    1,   24,    6,    5,    0],
        [  15,   19,   56,  371,  201,   77,   77,    1,   35,    8,    0,    7,   41,    4,   32,    7],
        [  39,   53,   11,   95,  109,   32,   80,   14,   52,   16,    0,    6,   11,    1,   28,    1],
        [   2,   35,   13,   45,  350,  213,   10,    1,  134,   12,    2,    1,    8,    0,   62,    0],
        [  23,  107,    8,    6,   36,    5,  260,   15,   15,   52,    8,   19,   19,    1,  353,    0],
        [  31,   87,   16,  100,  131,   15,  137,  421,   98,    8,    2,   12,   11,    0,   31,    0],
        [  41,   14,   37,  115,   78,    6,   23,    0,  1621,    1,    0,   20,   73,    2,   11,    0],
        [   3,   10,    0,    0,    3,    0,   15,    0,    0,  292,  463,   51,   39,   45,   19,    0],
        [  15,   15,    3,   11,   10,    0,   22,    2,   12,  171,  1169,   34,   14,   65,    4,   13],
        [  97,    5,    0,    3,    2,    0,    8,    0,    0,  200,    2,  379,  196,   12,   11,    3],
        [  11,   10,    0,    4,    0,   10,   10,    0,    0,  252,   67,   73,  129,  170,   41,   16],
        [  10,    0,    1,   21,   27,    0,   11,    0,   22,   27,  120,   22,   19,  239,    5,   30],
        [  20,  111,    1,    3,   12,    2,   25,    0,    3,  168,   13,    8,    4,    1,  2197,    2],
        [  22,    8,   10,  150,  116,   42,   28,    5,    4,   35,   63,   18,   64,   75,   23,   41]
    ])
        elif algorithm=='graph':
            confusion_matrix = np.array([
        [  62,  100,    3,  589,  140,    0,    2,    0,    0,    0,    0,    0,    0,    0,    0,    0],
        [   2,  467,  225,  115,   92,    0,   23,    0,    0,    0,    0,    6,    0,   10,    0,    0],
        [  10,  311, 1089,   24,   10,    1,    9,    0,    0,    0,   12,    0,    9,   79,    4,    2],
        [  45,   28,    0,  529,  316,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
        [   2,  271,   56,  126,  217,   10,   97,    0,    4,    0,    8,    0,    2,    0,    0,    0],
        [   0,   31,  133,   88,   77,  104,   13,    3,    0,    0,   85,    1,   10,    0,    0,    9],
        [   0,    8,    5,   15,    7,    0, 2473,    0,   38,    0,    5,    5,    3,    4,    0,    7],
        [  43,   17,    9,   12,   42,    0,   23,    9,    0,    0,  397,   35,   55,    0,    2,   60],
        [  12,    2,    0,   11,   16,   13,  346,    0,  205,    0,   12,    4,    0,    6,    0,    1],
        [  17,    0,    0,    4,   81,    0,    0,    0,    0,   56,  773,  124,    1,    1,    0,  442],
        [  11,    0,    0,   24,   19,    0,    3,    0,    4,    0,  658,  118,   13,    7,    5,   89],
        [  47,    0,    0,    5,    3,    0,   63,    0,    9,    0,  142,   73,   15,   20,   60,  111],
        [  65,    2,    0,    0,    4,    0,   11,    0,   29,    1,   16,  111,  235,    6,    1,  407],
        [  18,    0,    1,   18,   38,    0,  496,    0,    0,    0,   12,   15,    1,   238,   35,   55],
        [  98,    0,    0,   36,    6,    0,   74,    0,   13,    0,   70,   70,    0,    30,  514,  189],
        [   1,    0,    0,    0,    7,    0,    0,    0,    0,    0,   60,    4,    0,    7,    0, 1963]
    ])

    elif city=='wuhan':
        if algorithm=='sourceonly':
            confusion_matrix = np.array([
        [ 565,   25,    0,  250,   23,    1,    6,    2,    4,    0,    1,    0,    2,   20,    5,   19],
        [  48,  171,   49,   21,  355,   16,   30,    7,    0,    1,    2,    0,    3,   28,   10,    3],
        [  46,  520,  438,    3,   69,  118,   11,   10,    9,    2,    1,    2,    6,   26,    6,    3],
        [ 269,   24,    1,  410,   53,    3,    9,   25,    6,   11,    6,    8,    1,   35,   12,   18],
        [ 115,   79,    4,   44,  345,   34,   50,   45,    2,   21,    5,    0,    2,   27,    5,   10],
        [  19,   48,   80,    7,   76,  203,    7,   27,    3,    2,    8,    1,   21,   17,    2,    1],
        [  16,   27,    3,    4,    6,    0, 2113,   18,   20,    0,    4,    2,   24,   39,    7,    6],
        [  12,   56,   43,    5,   19,   55,   44,   89,    6,   23,   54,   11,   76,   41,   13,    4],
        [  60,    7,    4,    5,   13,    3,  330,   54,   71,    1,   10,    4,    9,   41,   54,    4],
        [   9,    7,    9,    1,    7,    0,   11,   19,    1, 1041,  131,    3,   66,   32,    9,   73],
        [   3,   10,    7,    5,   14,    4,   23,   27,    1,  112,  336,   17,  159,   35,   20,   28],
        [   2,   16,    1,    4,    7,    0,   67,    8,    1,   19,  107,   38,   54,   39,  138,   13],
        [   4,    4,    3,    0,    6,    3,   19,    8,    1,    9,   26,   12,  382,   10,   11,   19],
        [  12,   20,    3,   33,    2,    6,  323,   10,   16,   13,   23,   27,   31,  190,   84,   16],
        [   2,    9,    6,   11,    2,    0,   57,    7,    1,   15,   81,   37,   82,   40,  521,   30],
        [   0,    4,    6,    3,    2,    3,   86,   23,    1,   16,   45,   17,  120,   13,   31, 1336]
    ])

        elif algorithm=='D3GU':
            confusion_matrix = np.array([
        [617, 70, 22, 143, 0, 0, 8, 0, 11, 0, 5, 0, 14, 23, 4, 6],
        [31, 292, 123, 22, 177, 4, 61, 0, 5, 9, 5, 0, 3, 10, 1, 1],
        [7, 176, 964, 6, 23, 41, 9, 3, 0, 4, 1, 0, 5, 12, 6, 0],
        [306, 48, 23, 335, 52, 4, 1, 11, 7, 5, 23, 1, 16, 39, 6, 14],
        [66, 157, 63, 53, 219, 24, 53, 38, 3, 36, 30, 2, 10, 29, 3, 2],
        [9, 21, 221, 2, 24, 155, 2, 24, 1, 5, 19, 1, 13, 12, 4, 0],
        [4, 12, 15, 6, 1, 0, 2095, 19, 33, 3, 13, 14, 17, 35, 18, 3],
        [6, 26, 50, 7, 4, 67, 16, 70, 1, 22, 91, 16, 92, 58, 20, 5],
        [4, 1, 14, 6, 0, 1, 271, 14, 193, 2, 19, 50, 20, 42, 31, 2],
        [0, 2, 16, 0, 2, 3, 4, 10, 0, 1016, 202, 4, 60, 8, 7, 79],
        [0, 0, 21, 3, 3, 6, 16, 24, 0, 179, 341, 27, 142, 23, 4, 12],
        [0, 1, 6, 6, 1, 3, 37, 4, 5, 38, 110, 64, 80, 21, 121, 17],
        [0, 0, 7, 0, 0, 0, 10, 5, 1, 2, 35, 4, 425, 9, 0, 19],
        [6, 5, 13, 7, 0, 2, 239, 13, 9, 5, 28, 29, 24, 344, 60, 19],
        [0, 8, 3, 6, 0, 2, 41, 4, 0, 63, 85, 29, 55, 37, 536, 29],
        [0, 1, 10, 2, 1, 21, 17, 17, 0, 58, 58, 39, 123, 14, 21, 1324]
    ])

        elif algorithm=='ALT':
            confusion_matrix = np.array([
            [443, 7, 0, 6, 0, 3, 10, 0, 1, 30, 18, 273, 28, 1, 9, 1],
            [29, 44, 0, 54, 3, 27, 44, 42, 0, 44, 37, 16, 6, 6, 256, 14],
            [6, 0, 572, 115, 7, 141, 9, 4, 372, 1, 11, 0, 4, 0, 15, 31],
            [0, 0, 82, 312, 10, 226, 9, 22, 29, 0, 11, 7, 7, 0, 12, 16],
            [0, 2, 19, 121, 35, 93, 27, 105, 20, 6, 4, 4, 3, 0, 31, 2],
            [0, 0, 0, 26, 0, 437, 4, 1, 5, 0, 6, 0, 0, 1, 22, 0],
            [4, 8, 0, 52, 11, 70, 220, 58, 6, 29, 17, 17, 4, 1, 218, 16],
            [0, 1, 16, 155, 12, 166, 60, 346, 46, 1, 8, 3, 1, 2, 25, 4],
            [0, 0, 19, 86, 11, 264, 10, 4, 1081, 0, 5, 9, 0, 10, 27, 46],
            [25, 1, 0, 0, 0, 0, 0, 6, 0, 332, 103, 31, 168, 5, 20, 7],
            [1, 0, 0, 2, 0, 0, 6, 0, 0, 264, 850, 10, 7, 32, 6, 1],
            [119, 1, 4, 14, 0, 0, 9, 5, 1, 22, 4, 535, 98, 3, 3, 13],
            [31, 0, 15, 8, 0, 10, 7, 3, 0, 95, 52, 89, 336, 7, 37, 29],
            [2, 0, 1, 9, 0, 9, 3, 0, 0, 30, 159, 6, 57, 145, 0, 36],
            [10, 32, 0, 10, 1, 19, 37, 9, 1, 26, 13, 9, 4, 0, 1933, 9],
            [2, 2, 11, 64, 1, 127, 3, 8, 0, 12, 127, 3, 11, 29, 9, 104]
        ])

        elif algorithm=='aggregation':
            confusion_matrix = np.array([
        [ 769,    1,   12,   85,    7,    1,    0,    8,    9,    0,    0,   19,    0,   12,    0,    0],
        [  41,   14,  174,   15,  411,   15,    0,    5,   58,    2,    0,    0,    0,    9,    0,    0],
        [   2,  388,  562,    2,   28,  251,    0,    7,    0,    0,    3,    8,    0,   15,    3,    1],
        [ 275,    0,   18,  509,   21,    0,    0,   32,    2,    9,    5,   13,    0,    7,    0,    0],
        [ 112,   31,    6,  118,  372,   50,    0,   54,   21,    3,    5,    0,    0,   16,    0,    0],
        [   6,   57,   26,    4,   15,  315,    0,   74,    0,    0,    9,    3,    0,   13,    0,    0],
        [   4,   12,    5,    6,   28,    1, 1499,   27,  431,    0,    1,   43,   49,  178,    3,    2],
        [   1,   73,   10,    1,    2,   53,    0,  238,    8,    5,   43,   25,   81,    9,    2,    0],
        [   7,    1,    5,    2,    1,    1,   68,   37,  310,    0,    2,   51,   24,  148,   13,    0],
        [   0,    5,    0,    0,    0,    0,   55,   17,    0,  898,  377,   14,   27,    2,    11,   13],
        [   0,    5,    0,    3,    5,    2,    4,   52,    2,   43,  462,   63,  137,    3,    13,    7],
        [   0,    0,    0,    7,    1,    1,    3,    2,   11,    2,   77,  202,   61,    9,   119,   19],
        [   6,    0,    1,    0,    2,    3,    5,    0,    0,    0,   28,   18,  441,    1,    0,   12],
        [   6,    4,   10,   20,    2,    4,   30,   21,   34,    7,   15,  152,   14,   449,   33,    8],
        [   3,    0,    1,   13,    1,    1,    4,   16,    4,    2,   21,  114,   37,   25,  646,   13],
        [  30,    4,    5,    1,    1,    9,    3,   33,    2,   17,   44,   32,   67,    2,    7, 1449]
    ])
        elif algorithm=='conmix':
            confusion_matrix = np.array([
        [ 437,    4,    0,  317,   98,    0,   12,   15,   16,    0,    1,    0,    0,   14,    0,    9],
        [  18,   58,   71,   17,  486,   15,   31,   20,    4,    2,    1,    1,    0,   17,    0,    3],
        [   8,  295,  698,   13,  113,   64,   21,   10,    3,    2,    5,    3,    6,   14,    2,   13],
        [ 154,   10,    3,  492,  108,    0,   10,   62,   13,   10,    3,    0,    1,    4,    0,   21],
        [  44,   43,    4,   72,  433,   28,   44,   63,   12,   10,   11,    0,    0,    7,    0,   17],
        [   1,   24,  106,   10,   70,  218,    4,   43,    0,    0,   19,    4,   14,    3,    0,    6],
        [   2,    2,    2,    4,   14,    0, 2103,   17,   26,    1,    2,    1,    7,   42,    9,   57],
        [   0,   45,   15,    3,    8,   27,    9,  137,    7,   37,   79,   23,  120,    1,    9,   31],
        [  10,    0,    2,   11,    5,    0,  227,   43,  285,    0,    1,    1,    7,   34,   24,   20],
        [   1,    0,    0,    0,    0,    0,    1,   10,    0, 1083,   98,  139,    7,    0,    4,   76],
        [   2,    1,    0,    6,   15,    0,    7,   32,    4,   204,  337,   29,  102,    3,    5,   54],
        [   0,    0,    0,    3,    8,    1,   31,    5,    4,   28,  115,   59,   43,   26,  128,   63],
        [   0,    0,    1,    0,    2,    2,   11,    2,    0,   10,   27,    2,  378,    2,    4,   76],
        [  10,    2,    1,    7,    4,    1,  235,   16,   14,   11,   31,   17,   19,  288,  107,   46],
        [   1,    2,    0,   10,    1,    0,   36,   15,    7,    9,   78,   40,   25,   26,  572,   79],
        [   0,    0,    2,    0,    1,    0,   20,   17,   19,   13,   49,    0,   30,    0,   13, 1542]
    ])
        elif algorithm=='ours':
            confusion_matrix = np.array([
        [ 408,    5,    3,  338,   66,    0,    2,   38,   29,    1,    0,    0,    3,   22,    5,    3],
        [  16,  112,   94,   18,  431,   22,   13,    7,    6,    0,    0,    0,    2,   15,    8,    0],
        [   2,  304,  721,    4,   46,  132,    5,    2,    3,    0,    5,    6,    6,   13,   15,    6],
        [ 128,   14,    5,  541,   99,    9,    0,   41,    5,    1,    9,    2,    1,    7,   28,    1],
        [  40,   88,    1,   56,  387,   56,   14,  106,   10,    0,    8,    0,    0,   15,    5,    2],
        [   0,   35,   68,    0,   41,  294,    0,   29,    0,    2,    9,    3,   23,   13,    2,    3],
        [   1,    4,    0,    7,   14,    0, 2061,   15,   42,    0,    3,    7,   38,   71,   25,    1],
        [   0,   45,    2,    1,    0,   40,   12,  114,    0,   19,  110,   17,  179,    3,    8,    1],
        [   1,    0,    2,    0,    0,    1,  186,   90,  263,    0,    9,    3,   11,   60,   44,    0],
        [   1,    0,    0,    2,    0,    3,    0,    2,    0, 1024,  113,  139,   23,    0,   29,   83],
        [   0,    1,    0,    0,    3,    0,    2,   12,    3,   151,  351,   39,  178,   18,   38,    5],
        [   1,    0,    0,    2,    0,    0,    4,    4,    4,   12,   68,  122,   91,   17,  171,   18],
        [   0,    0,    0,    0,    0,    2,    3,    0,    0,    4,   33,    2,  460,    2,    4,    7],
        [   0,    0,    5,    4,    0,    3,  133,   22,   12,    6,   18,   42,   36,  347,  172,    9],
        [   0,    0,    0,    2,    1,    2,    9,   23,    4,    5,   37,   40,   59,   17,   679,   23],
        [   0,    0,    3,    0,    1,    0,    6,    5,    0,    6,   50,    6,  167,    7,   36, 1419]
    ])

        elif algorithm=='MT-MTDA':
            confusion_matrix = np.array([
        [ 433,    8,    0,    0,    0,    5,   11,    1,    6,   85,   34,  290,   20,    0,   29,    1],
        [  15,   24,    0,    7,   48,   43,   60,   13,    0,   21,   22,   17,    2,    0,  381,   17],
        [   0,    0,  881,  140,    1,   32,    6,    3,  335,    0,    5,    4,    0,    0,    9,    3],
        [   1,    0,  187,  378,   34,  103,   20,   12,   23,    1,    3,    1,    0,    3,   13,   22],
        [   0,    0,   12,  123,   58,   97,   30,  135,   32,    0,    0,    3,    1,    0,   22,    1],
        [   0,    0,    2,   40,    5,  420,    7,    6,   28,    0,    0,    0,    0,    1,    7,    1],
        [   0,   21,    2,   28,   40,   36,  283,   44,   16,   11,    7,   10,    0,    0,  304,    7],
        [   0,    8,   13,  123,   53,   59,   56,  500,   44,    0,    0,   19,    1,    0,   24,    1],
        [   0,    2,   24,   71,    9,  214,    7,   17, 1320,    0,    2,    7,    1,    5,   17,   10],
        [  20,    1,    0,    2,    0,    1,   11,    1,    0,  296,  163,   21,  171,    9,   45,    3],
        [   5,    0,    0,    6,    3,    1,   12,    5,    0,  177,  996,    2,   13,   42,    8,    0],
        [  71,    2,    2,    4,    5,    2,   39,   23,    6,   16,   24,  615,   29,   17,    4,   32],
        [  25,    4,    7,   18,    1,   11,   22,    6,    3,   82,   48,  119,  230,   56,   58,   98],
        [   7,    0,    1,   22,    1,   18,    5,    3,    1,    0,  212,    4,    3,  203,    0,   42],
        [   8,   34,    0,    3,    4,   24,   26,   18,    5,   20,    7,    4,    5,    1, 2121,    9],
        [   0,    4,    8,  153,    7,  106,   12,    6,    0,    2,   72,    3,    1,   54,    3,  120]
    ])
        elif algorithm=='graph':
            confusion_matrix = np.array([
        [ 447,   69,    6,  385,    0,    0,    4,    0,    0,    0,    0,    0,    0,    8,    0,    4],
        [  47,  396,   94,   29,   98,    0,   61,    3,    0,    0,    0,    0,    0,    4,   12,    0],
        [   4,  254,  958,    5,    4,    4,    6,    0,    0,    0,    0,    0,   26,    7,    0,    2],
        [  94,   32,   13,  715,   12,    6,    0,    8,    0,    0,    0,    0,    0,   10,    0,    1],
        [  50,  192,   56,  158,  140,   11,   45,   79,    0,    6,   40,    0,    1,    4,    6,    0],
        [   4,   30,  224,    4,   13,   88,    4,   63,    0,    0,   16,    0,   60,   16,    0,    0],
        [   0,    0,    4,    2,    0,    0, 2232,    1,    8,    0,    2,    0,   11,   13,   15,    1],
        [   0,    0,   34,    0,    0,    1,   13,   52,    0,   20,  145,    0,  267,    0,    9,   10],
        [  12,    1,    0,    5,    0,    0,  566,   26,    0,    0,   14,    0,   25,    2,   19,    0],
        [   1,    0,    3,    0,    0,    0,    0,    0,    0,  577,  382,    0,   55,    6,    0,  395],
        [   0,    0,    8,    0,    0,    0,    6,    0,    0,   29,  562,    0,  156,   25,    5,   10],
        [   0,    0,   20,    0,    0,    0,    4,    4,    0,    0,   13,    0,  464,    1,    0,   11],
        [   4,    1,    1,    3,    0,    0,  399,    0,    0,    0,   40,    0,   25,  238,   77,   21],
        [   0,    0,    0,    0,    0,    0,   42,    0,    0,    0,  256,    0,   45,   23,  483,   52],
        [   0,    0,    5,    0,    0,    0,    2,    0,    0,    0,   57,    0,   76,   39,   16, 1511]
    ])
        



    # 设置类别的标签（根据数据调整为实际的类别数量）
    num_classes = confusion_matrix.shape[0]
    class_labels = ['LCZ1', 'LCZ2', 'LCZ3', 'LCZ4', 'LCZ5', 'LCZ6', 'LCZ8', 'LCZ9', 'LCZ10', 'LCZA', 'LCZB', 'LCZC', 'LCZD', 'LCZE', 'LCZF', 'LCZG']

    # 设置绘图
    plt.figure(figsize=(10, 8))

    # 绘制热力图
    # max_value = np.max(confusion_matrix)
    max_value =1.00
    # log_confusion_matrix = np.log1p(confusion_matrix)
    # sns.heatmap(log_confusion_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=class_labels, yticklabels=class_labels, cbar=True, linewidths=0.5)
    confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    sns_heatmap =sns.heatmap(confusion_matrix_normalized, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=class_labels, yticklabels=class_labels, cbar=True, linewidths=0.5, vmax=max_value,annot_kws={"size": 13})
    colorbar = sns_heatmap.collections[0].colorbar
    # colorbar.tick_params(labelsize=15)
    colorbar.ax.tick_params(labelsize=13)  # 设置colorbar刻度标签的字体大小
    plt.xticks(fontsize=15) 
    plt.yticks(fontsize=15) 

    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    # 设置标题和轴标签
    # plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=16)
    plt.ylabel("True Labels", fontsize=16)

    # 显示图形
    plt.tight_layout()
    # plt.show()

    plt.savefig(f'confusion_matrix/{city}_0112/{algorithm}_normalize_YlGnBu.png', dpi=300, bbox_inches='tight')

# plt.savefig('sourceonly_int_YlGnBu.png', dpi=300, bbox_inches='tight')

