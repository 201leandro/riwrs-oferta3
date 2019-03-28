stopwords = ['de',
 'https',
 'demais',
 'rt',
 'a',
 'o',
 'que',
 'e',
 'do',
 'da',
 'em',
 'um',
 'para',
 'com',
 'uma',
 'os',
 'no',
 'se',
 'na',
 'por',
 'as',
 'dos',
 'como',
 'mas',
 'ao',
 'ele',
 'das',
 'à',
 'seu',
 'sua',
 'ou',
 'quando',
 'nos',
 'já',
 'eu',
 'também',
 'só',
 'pelo',
 'pela',
 'até',
 'isso',
 'ela',
 'entre',
 'depois',
 'sem',
 'mesmo',
 'aos',
 'seus',
 'quem',
 'nas',
 'me',
 'esse',
 'eles',
 'você',
 'essa',
 'num',
 'nem',
 'suas',
 'meu',
 'às',
 'minha',
 'numa',
 'pelos',
 'elas',
 'qual',
 'nós',
 'lhe',
 'deles',
 'essas',
 'esses',
 'pelas',
 'este',
 'dele',
 'tu',
 'te',
 'vocês',
 'vos',
 'lhes',
 'meus',
 'minhas',
 'teu',
 'tua',
 'teus',
 'tuas',
 'nosso',
 'nossa',
 'nossos',
 'nossas',
 'dela',
 'delas',
 'esta',
 'estes',
 'estas',
 'aquele',
 'aquela',
 'aqueles',
 'aquelas',
 'isto',
 'aquilo',
 'estou',
 'está',
 'estamos',
 'estão',
 'estive',
 'esteve',
 'estivemos',
 'estiveram',
 'estava',
 'estávamos',
 'estavam',
 'estivera',
 'estivéramos',
 'esteja',
 'estejamos',
 'estejam',
 'estivesse',
 'estivéssemos',
 'estivessem',
 'estiver',
 'estivermos',
 'estiverem',
 'hei',
 'há',
 'havemos',
 'hão',
 'houve',
 'houvemos',
 'houveram',
 'houvera',
 'houvéramos',
 'haja',
 'hajamos',
 'hajam',
 'houvesse',
 'houvéssemos',
 'houvessem',
 'houver',
 'houvermos',
 'houverem',
 'houverei',
 'houverá',
 'houveremos',
 'houverão',
 'houveria',
 'houveríamos',
 'houveriam',
 'sou',
 'somos',
 'são',
 'era',
 'éramos',
 'eram',
 'fui',
 'foi',
 'fomos',
 'foram',
 'fora',
 'fôramos',
 'seja',
 'sejamos',
 'sejam',
 'fosse',
 'fôssemos',
 'fossem',
 'for',
 'formos',
 'forem',
 'serei',
 'será',
 'seremos',
 'serão',
 'seria',
 'seríamos',
 'seriam',
 'tenho',
 'tem',
 'temos',
 'tém',
 'tinha',
 'tínhamos',
 'tinham',
 'tive',
 'teve',
 'tivemos',
 'tiveram',
 'tivera',
 'tivéramos',
 'tenha',
 'tenhamos',
 'tenham',
 'tivesse',
 'tivéssemos',
 'tivessem',
 'tiver',
 'tivermos',
 'tiverem',
 'terei',
 'terá',
 'teremos',
 'terão',
 'teria',
 'teríamos',
 'teriam',
 'i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]