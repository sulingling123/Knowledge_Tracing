# Knowledge_Tracing
=====
- Paper ： 知识追踪相关论文
    - Deep Knowledge Tracing:
        - 首次提出将RNN用于知识追踪，并能够基于复杂的知识联系进行建模（如构建知识图谱）
    -  How Deep is Knowledge Tracing
        - 探究DKT利用到的统计规律并拓展BKT，从而使BKT拥有能够与DKT相匹配的能力
    - Going Deeper with Deep Knowledge Tracing
        - 对DKT和PFA，BKT进行了模型比较，对DKT模型能碾压其他两种模型的结果进行了怀疑并加以论证，进一步讨论了原论文能够得出上述结果的原因，对进一步使用DKT模型提供了参考。
    - Incorporating Rich Features Into Deep Knowledge Tracing
        - 对DKT使用上进行数据层扩展，扩展学生和问题层的数据输入，包括结合自动编码器对输入进行转换
    - Addressing Two Problems in Deep Knowledge Tracing viaPrediction-Consistent Regularization
        - 指出DKT模型现存缺点：对输入序列存在重构问题和预测结果的波动性，进而对上述问题提出了改善方法
    - Exercise-Enhanced Sequential Modeling for Student Performance Prediction
        - 将题面信息引入，不仅作为输入送入模型，而且将题目编码后的向量计算cosine相似度作为atention的socre
        
