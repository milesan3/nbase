'''
朴素贝叶斯的一般过程
 ⑴收集数据：可以使用任何方法。本章使用RSS源
(2) 准备数据：需要数值型或者布尔型数据。
(3) 分析数据：有大量特征时，绘制特征作用不大，此时使用直方图效果更好。
(4) 训练算法：计算不同的独立特征的条件概率。
(5) 测试算法：计算错误率。
(6) 使用算法：一个常见的朴素贝叶斯应用是文档分类。可以在任意的分类场景中使用朴素贝叶斯命类器，不一定非要是文本。
#准备数据从文本中构建词向量
#训练算法从词向量计算概率
#测试算法根据现实情况修改分类器
#准备数据文档词袋模型
***工作原理：
提取所有文档中的词条并进行去重
获取文档的所有类别
计算每个类别中的文档数目
对每篇训练文档:
    对每个类别:
        如果词条出现在文档中-->增加该词条的计数值（for循环或者矩阵相加）
        增加所有词条的计数值（此类别下词条总数）
对每个类别:
    对每个词条:
        将该词条的数目除以总词条数目得到的条件概率（P(词条|类别)）
返回该文档属于每个类别的条件概率（P(类别|文档的所有词条)）
'''
from numpy import *
import numpy
#import feedparser

#=========================从文本中构建词向量：词表到向暈的转换函数==========================
#切分数据集和类别标签
def loadDataSet():
    '''
    创建实验样本，真实样本可能差很多，需要对真实样本做一些处理，如
    去停用词(stopwords)，词干化(stemming)等等，处理完后得到更"clear"的数据集，
    方便后续处理
    '''
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]  #进行词条切分后的文档集合，去掉了标点符号
    classVec = [0,1,0,1,0,1]  #类别标签集合，两类：1表示侮辱类，0表示不属于
    return postingList,classVec  #返回词条切分后的分档和类别标签

#创建词汇表
def createVocabList(dataSet):  #创建一个包含在所有文档中出现的不重复词的列表，将词条列表输给set构造函数
    vocabSet = set([])   #创建一个空集合，set是返回不带重复词的list，set()去重
    for document in dataSet:
        vocabSet = vocabSet | set(document)    #创建两个集合的并集，将每篇文档返回的新词集合添加到该集合中，操作符丨用于求两个集合的并集，这也是一个按位或操 作 符
    return list(vocabSet)   #返回词汇表

#判断词是否出现
def setOfWords2Vec(vocabList,inputSet):  #判断某个词条在文档中是否出现，输人参数为词汇表及某个文档
    '''
    词1,词2,XXX，词n    #词表vocabList
    doc1:  1, 0,...,1   #inputSet的输出结果
    doc2:  0, 1,...,0
    '''
    returnVec = [0]*len(vocabList)   #创建同vocabList同样长度的全0列表(文档向量)，也可[0 for i in range(len(vocabList))]
    for word in inputSet:
        if word in vocabList:   #针对某篇inpustSet处理
            returnVec[vocabList.index(word)] = 1   #找到某篇文档的词，其在词表中出现的位置，将其改为1
        else:
            print('the word:%s is not in my Vocabulary!' % word)
    return returnVec  #返回文档向量 表示某个词是否在输入文档中出现过 1/0


#=======================训练分类器,原始的朴素贝叶斯，没有优化===================
'''
伪代码：
计算每个类别中的文档数目
对每篇训练文档：
    对每个类别：
        如果词条出现文档中―增加该词条的计数值
        增加所有词条的计数值
    对每个类别：
        对每个词条：
        将该词条的数目除以总词条数目得到条件概率
    返回每个类别的条件概率
'''
#输入trainMatrix：词向量数据集，文件单词矩阵 [[1,0,1,1,1....],[],[]...]
#输入trainCategory：数据集对应的类别标签，文件对应的类别[0,1,1,0....]，列表长度等于单词矩阵数，其中的1代表对应的文件是侮辱性文件，0代表不是侮辱性文件
#输出p0Vect：词汇表中各个单词在正常言论中的类条件概率密度
#输出p1Vect：词汇表中各个单词在侮辱性言论中的类条件概率密度
#输出pAbusive：侮辱性言论在整个数据集中的比例
#朴贝叶斯分类区训练函数
def trainNB00(trainMatrix,trainCategory):   #trainMatrix：文档矩阵；trainCategory：由每篇文档的类别标签所构成的向量[0,1,0,1,0,1]，就是listClasses
    numTrainDocs = len(trainMatrix)    #文档的个数  训练集总条数  总文件数
    numWords = len(trainMatrix[0])     #词表的个数,一个文档中的单词数  训练集中所有不重复单词总数  总单词数
    pAbusive = sum(trainCategory)/float(numTrainDocs)  #侮辱类的概率(侮辱类占总训练数据的比例),sum([0,1,0,1,0,1])=3,也即是trainCategory里面1的个数
    p0Num = zeros(numWords)    #type(p0Num)为numpy.array类型，p0Num是1行numWords列的数组，正常言论的类条件概率密度 p(某单词|正常言论)=p0Num/p0Denom
    p1Num = zeros(numWords)  # type(p1Num)为numpy.array类型，p1Num是1行numWords列的数组，侮辱性言论的类条件概率密度 p(某单词|侮辱性言论)=p1Num/p1Denom
    p0Denom = 0.0   #初始化分子为0
    p1Denom = 0.0   #初始化分母置为0
    for i in range(numTrainDocs):   #遍历训练集数据
        if trainCategory[i] ==1:   #统计所有侮辱类文档中的各个单词总数
            p1Num += trainMatrix[i]   #如果是侮辱性文件，对侮辱性文件的向量进行加和，[0,1,1,....] + [0,1,1,....]->[0,2,2,...]
            p1Denom += sum(trainMatrix[i])  #p1Denom侮辱类总单词数，对向量中的所有元素进行求和，也就是计算所有侮辱性文件中出现的单词总数
        else:
            p0Num += trainMatrix[i]  #统计正常类所有文档中的各个单词总数
            p0Denom += sum(trainMatrix[i])   #p0Denom正常类总单词数
        numpy.seterr(divide = 'ignore',invalid = 'ignore')
        p1Vec = p1Num/p1Denom   #词汇表中的单词在侮辱性言论文档中的类条件概率，计算p(w0|ci)；侮辱性文档的[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]列表
        p0Vec = p0Num/p0Denom   #词汇表中的单词在正常性言论文档中的类条件概率，计算p(wn|ci)；即正常文档的[P(F1|C0),P(F2|C0),P(F3|C0),P(F4|C0),P(F5|C0)....]列表
    return p0Vec,p1Vec,pAbusive   #返回p0Vec，p1Vec都是矩阵，对应每个词在文档总体中出现概率，pAb对应文档属于1的概率


#============================训练分类器，优化处理===================贝叶斯估计防止概率为0，取对数防止下溢
#输入trainMatrix：词向量数据集
#输入trainCategory：数据集对应的类别标签
#输出p0Vect：词汇表中各个单词在正常言论中的类条件概率密度
#输出p1Vect：词汇表中各个单词在侮辱性言论中的类条件概率密度
#输出pAbusive：侮辱性言论在整个数据集中的比例
def trainNB0(trainMatrix,trainCategory):   #trainMatrix：文档矩阵；trainCategory：由每篇文档的类别标签所构成的向量[0,1,0,1,0,1]，就是listClasses
    numTrainDocs = len(trainMatrix)    #文档的个数  训练集总条数   # 总文件数
    numWords = len(trainMatrix[0])     #词表的个数,一个文档中的单词数  训练集中所有不重复单词总数  总单词数
    pAbusive = sum(trainCategory)/float(numTrainDocs)  #侮辱类的概率(侮辱类占总训练数据的比例),sum([0,1,0,1,0,1])=3,也即是trainCategory里面1的个数
    # *拉普拉斯平滑防止类条件概率为0，初始化分子为1，分母为2
    p0Num = ones(numWords)    #type(p0Num)为numpy.array类型，p0Num是1行numWords列的数组，正常言论的类条件概率密度 p(某单词|正常言论)=p0Num/p0Denom
    p1Num = ones(numWords)  # type(p1Num)为numpy.array类型，p1Num是1行numWords列的数组，侮辱性言论的类条件概率密度 p(某单词|侮辱性言论)=p1Num/p1Denom
    p0Denom = 2.0   #初始化分子为2
    p1Denom = 2.0   #初始化分母置为2
    for i in range(numTrainDocs):   #遍历训练集数据
        if trainCategory[i] ==1:   #统计侮辱类所有文档中的各个单词总数
            p1Num += trainMatrix[i]   #[0,1,1,....] + [0,1,1,....]->[0,2,2,...]，累加辱骂词的频次
            p1Denom += sum(trainMatrix[i])  #p1Denom侮辱类总单词数
        else:
            p0Num += trainMatrix[i]  #统计正常类所有文档中的各个单词总数
            p0Denom += sum(trainMatrix[i])   #p0Denom正常类总单词数
        numpy.seterr(divide = 'ignore',invalid = 'ignore')
        # 数据取log，即单个单词的p(x1|c1)取log，防止下溢出
        p1Vect = log(p1Num/p1Denom)   #词汇表中的单词在侮辱性言论文档中的类条件概率，计算p(w0|ci)；侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
        p0Vect = log(p0Num/p0Denom)   #词汇表中的单词在正常性言论文档中的类条件概率，计算p(wn|ci)；正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    return p0Vect,p1Vect,pAbusive    #返回p0Vect，p1Vect都是矩阵，对应每个词在文档总体中出现概率，pAbusive对应文档属于1的概率


#====================示例：对社区留言板言论进行分类***区分正常文档和侮辱性文档===================
"""
    使用算法：
        # 将乘法转换为加法
        乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    :param vec2Classify: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0Vec: 类别0，词汇表中每个单词在训练样本的正常言论中的类条件概率密度，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1Vec: 类别1，词汇表中每个单词在训练样本的侮辱性言论中的类条件概率密度，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param pClass1: 类别1，侮辱性文件的出现概率
    :return: 类别1 or 0
"""
#给定词向量 判断类别
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):  #输入为：vec2Classify是待分类文档，要分类的向量vec2Classify(0,1组合二分类矩阵，将每个词与其对应的概率相关联起来)，对应词汇表各个词是否出现以及使用函trainNB0()计算得到的三个概率
    p1 = sum(vec2Classify * p1Vec)+log(pClass1)    #vec2Classify * p1Vec，两个向量相乘指的是对应元素相乘；# P(w|c1) * P(c1) ，即贝叶斯准则的分子
    p0 = sum(vec2Classify * p0Vec)+log(1.0-pClass1)   # P(w|c0) * P(c0) ，即贝叶斯准则的分子·
    if p1>p0:
        return 1
    else:
        return 0
# 封装的bayes测试函数
def testingNB():
    # 1. 加载数据集
    listOPosts,listClasses = loadDataSet()  #导入数据，第一个存储文档，第二个存储文档标记类别
    # 2. 创建单词集合
    myVocabList = createVocabList(listOPosts)  #所有词汇总list，不含重复的
    # 3. 计算单词是否出现并创建数据矩阵
    trainMat=[]   #构建矩阵，存放训练数据
    for postinDoc in listOPosts:  #生成文档对应词的矩阵 每个文档一行，每行内容为词向量；遍历原始数据，转换为词向量，构成数据训练矩阵
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))  #每个词在文档中是否出现，生成1、0组合的词向量；数据转换后存入数据训练矩阵trainMat中
    # 4. 训练数据
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))  #根据现有数据输出词对应的类别判定和概率；训练分类器
    # 5. 测试数据
    testEntry = ['love','my','dalmation']   #===测试数据（1）
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))   #判断测试词条在词汇list中是否出现，生成词向量；测试数据转为词向量
    print(testEntry,'classfied as:',classifyNB(thisDoc,p0V,p1V,pAb))  #根据贝叶斯返回的概率，将测试向量与之乘，输出分类结果
    testEntry=['stupid','garbage']    #===测试数据（2）
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))   #测试数据转为词向量
    print(testEntry, 'classfied as:', classifyNB(thisDoc, p0V, p1V, pAb))  #输出分类结果


#==============================词袋模型：考虑单词出现的次数=========================
#vocabList：词汇表
#inputSet ：某个文档向量
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)   #创建所含元素全为0的向量
    for word in inputSet:    #依次取出文档中的单词与词汇表进行对照，统计单词在文档中出现的次数
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1    #单词在文档中出现的次数
        else:
            print('the word:%s is not in my Vocabulary!' % word)# 若测试文档的单词，不在词汇表中，显示提示信息，该单词出现次数用0表示
    return returnVec


#==============================示例：过滤垃圾邮件======================================
'''
示例：使用朴素贝叶斯对电子邮件进行分类 ' 
(1) 收集数据：提供文本文件。
(2) 准备数据：将文本文件解析成词条向量。
(3) 分析数据：检查词条确保解析的正确性。
(4) 训练算法：使用我们之前建立的trainNB0(trainMatrix, classLabel)函数。
(5) 测试算法：使用 clasSifyNB(> ，并且构建一个新的测试函数来计算文档集的错误率。
(6) 使用算法：构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上。
'''
#准备数据，切分文本，按空格切分出词，单词长度小于或等于2的全部丢弃
def textParse(bigString):
    import re
    #regEx = re.compile('\W*')     \\W*表示除了单词数字外的任意字符串
    #listOfTokens = regEx.split(bigString)
    listOfTokens = re.split('\W', bigString)  #！！！！一定要去掉原来的*号！！！此式同时实现切分和去掉出单词、数字之外的任意字符串，https://blog.csdn.net/hawkerou/article/details/53518154
    return [tok.lower() for tok in listOfTokens if len(tok)>2]   #tok.lower() 将整个词转换为小写

#过滤邮件 训练+测试
#使用朴素贝叶斯进行交叉验证
def spamTest():
    docList = []  #文章按篇存放
    classList = []  #存放文章类别
    fullText = []    #存放所有文章内容
    for i in range(1,26):
        # 1.切分，解析数据，并归类为 1 类别
        #wordList = textParse(open('email/spam/%d.txt'%i).read())  #读取垃圾邮件
        wordList = textParse(open('email/spam/%d.txt' % i, "rb").read().decode('GBK', 'ignore'))   # wordList=textParse(open('email/spam/%d.txt' %i).read()) 书上这行代码有些问题 unicode error
        docList.append(wordList)    #docList按篇存放文章；不融合格式
        fullText.extend(wordList)   #fullText邮件内容存放到一起；添加元素 去掉数组格式
        classList.append(1)    #垃圾邮件类别标记为1
        # 2.切分，解析数据，并归类为 0 类别
        #wordList = textParse(open('email/ham%d.txt'%i).reda())   #读取正常邮件
        wordList = textParse(open('email/ham/%d.txt' % i, "rb").read().decode('GBK', 'ignore'))  #同样处理
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)   #正常邮件类别标记为0
    # 3.创建词汇表
    vocabList = createVocabList(docList)   #创建词列表
    #trainingSet = range(50)  #训练集共50篇文章
    trainingSet = list(range(50))   #python3 del不支持返回数组对象 而是range对象
    testSet = []    #创建测试集
    # 4.随机取 10 个邮件用来测试
    for i in range(10):   #随机选取10篇文章为测试集，测试集中文章从训练集中删除
        randIndex = int(random.uniform(0,len(trainingSet)))  #0-50间产生一个随机数
        testSet.append(trainingSet[randIndex])   #从训练集中找到对应文章，加入测试集中
        del(trainingSet[randIndex])   #删除对应文章
    # 5.准备数据，用于训练分类器
    trainMat = []  #训练数据
    trainClasses = []  #类别标签
    for docIndex in trainingSet:    # 遍历训练集中文章数据
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))  #每篇文章转为词袋向量模型，存入trainMat数据矩阵中
        trainClasses.append(classList[docIndex])     #trainClasses存放每篇文章的类别
    # 6.训练分类器
    p0V,p1V,pSam = trainNB0(array(trainMat),array(trainClasses))   #得到概率
    errorCount = 0  #errorCount记录测试数据出错次数
    for docIndex in testSet:  #遍历测试数据集，每条数据相当于一条文本
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])  #文本转换为词向量模型
        if classifyNB(array(wordVector),p0V,p1V,pSam)!=classList[docIndex]:   #模型给出的分类结果与本身类别不一致时，说明模型出错，errorCount数加1
            errorCount += 1
            print('classification error',docList[docIndex])   # 输出出错的文章
    print('the error rate is:',float(errorCount)/len(testSet))   #输出错误率，即出错次数/总测试次数

'''
#===========================示例：使用朴素贝叶斯分类器从个人广告中获取区域倾向===========================

⑴收集数据：从RSS源收集内容，这里需要对RSS源构建一个接口.Universal Feed Parser是Python中最常用的RSS程序库
(2) 准备数据：将文本文件解析成词条向量。
(3) 分析数据：检查词条确保解析的正确性。
(4) 训练算法：使用我们之前建立的 trainNB0 函数。
(5) 测试算法：观察错误率，确保分类器可用。可以修改切分程序，以降低错误率，提高分类结果。
(6) 使用算法：构建一个完整的程序，封装所有内容。给定两个RSS源，该程序会显示最常用的公共词。

#RSS源分类器及高频词去除函数：通过函数calcMostFreq()改变要移除的单词数目
def calcMostFreq(vocabList,fullText):  #对所有词出现频率进行排序，返回排序后出现频率最高的前30个
    import operator
    freqDict={}
    for token in vocabList:  #遍历词汇表中的每个词
        freqDict[token]=fullText.count(token)   #统计每个词在文本中出现的次数
    sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)  #根据每个词出现的次数从高到底对字典进行排序True=降序排列
    return sortedFreq[:30]    #返回出现次数最高的30个单词

def localWords(feed1,feed0): #两个RSS源作为参数,与spamTest差别不大
    import feedparser
    docList=[];classList=[];fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):   #访问RSS源
        wordList = textParse(feed1['entries'][i]['summary'])   #每次访问一条RSS源
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words=calcMostFreq(vocabList,fullText)
    for pairW in top30Words:   #去掉出现频数最高的钱30个词
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen)) #python3修改替换trainSet=range(2*minLen)
    testSet=[]
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V

#最具代表性的词汇显示函数：显示地域相关的用词
#可以先对向量 pSF 与 pNY 进行排序，然后按照顺序将词打印出来
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)    #使用两个RSS源作为输人，然后训练并测试朴素贝叶斯分类器，返回使用的概率值
    topNY=[];topSF=[]    #创建两个列表用于元组的存储
    for i in range(len(p0V)):
        if p0V[i]>-1.0: topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-1.0: topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print (item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print (item[0])
'''


if __name__ == '__main__':
    '''
    #从个人广告中获取区域倾向
    ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sy=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocabList,pSF,pNY=localWords(ny,sf)  #多次进行上述实验，然后取平均值
    vocabList,pSF,pNY=localWords(ny,sf)
    vocabList,pSF,pNY=localWords(ny,sf)
    #测试显示地域相关的词
    getTopWords(ny,sf)
    '''
    '''
    #使用loadDataSet中的文档进行测试
    listOPosts,listClasses = loadDataSet()  #从预先加载值调入数据
    myVocabList = createVocabList(listOPosts)   #构建包含所有词表的myVocabList
    list = setOfWords2Vec(myVocabList, listOPosts[0])  #判断某个词条在文档中是否出现
    print(listOPosts,'\n',listClasses,'\n',myVocabList,'\n',list)
    # 自定义词条文档进行测试
    postingList = [['mother','want','has','a','dog'],['has', 'big', 'or']]
    listf = setOfWords2Vec(myVocabList,postingList[1])
    print(listf)
    #测试trainNB00(或者是trainNB0)
    listOPosts, listClasses = loadDataSet()  #从预先加载值调入数据
    myVocabList = createVocabList(listOPosts)  #构建包含所有词表的myVocabList
    print(myVocabList)  #输出词汇表
    trainMat=[]   #构建空列表
    for postinDoc in listOPosts:   #使用词向量填充trainMat列表
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))   #构建训练矩阵
    print(trainMat)
    p0V,p1V,pAb = trainNB00(trainMat,listClasses)
    print('p0V=',p0V,'\n')
    print('p1V=',p1V,'\n')
    print('pAb=',pAb)   #pAb是任意文档属于侮辱性文档的概率
    #测试分类文档函数,测试留言板文档，即在线社区的留言板的正常文档和侮辱性文档分类
    print('===================================')
    testingNB()
    #测试去掉英文中的标点符号并进行切片
    emailText = open('email/ham/6.txt').read()
    listOfTokens = textParse(emailText)
    print(listOfTokens)
    #测试分类垃圾邮件
    print('===================================')
    spamTest()
   
    '''

