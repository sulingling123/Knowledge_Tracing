# Knowledge_Tracing
- Paper ： 知识追踪相关论文
    - [Deep Knowledge Tracing](https://github.com/ZoeYuhan/Knowledge_Tracing/blob/master/Paper/deep%20Knowledge%20Tracing.pdf):
        - 首次提出将RNN用于知识追踪，并能够基于复杂的知识联系进行建模（如构建知识图谱）
    -  [How Deep is Knowledge Tracing](https://github.com/ZoeYuhan/Knowledge_Tracing/blob/master/Paper/How%20Deep%20is%20Knowledge%20Tracing%3F.pdf)
        - 探究DKT利用到的统计规律并拓展BKT，从而使BKT拥有能够与DKT相匹配的能力
    - [Going Deeper with Deep Knowledge Tracing](https://github.com/ZoeYuhan/Knowledge_Tracing/blob/master/Paper/Going%20Deeper%20with%20Deep%20Knowledge%20Tracing%20.pdf)
        - 对DKT和PFA，BKT进行了模型比较，对DKT模型能碾压其他两种模型的结果进行了怀疑并加以论证，进一步讨论了原论文能够得出上述结果的原因，对进一步使用DKT模型提供了参考。
    - [Incorporating Rich Features Into Deep Knowledge Tracing](https://github.com/ZoeYuhan/Knowledge_Tracing/blob/master/Paper/Incorporating%20rich%20features%20into%20Deep%20knowledge%20tracing.pdf)
        - 对DKT使用上进行数据层扩展，扩展学生和问题层的数据输入，包括结合自动编码器对输入进行转换
    - [Addressing Two Problems in Deep Knowledge Tracing viaPrediction-Consistent Regularization](https://github.com/ZoeYuhan/Knowledge_Tracing/blob/master/Paper/Addressing%20Two%20Problems%20in%20Deep%20Knowledge%20Tracing%20via%20Prediction-Consistent%20Regularization.pdf)
        - 指出DKT模型现存缺点：对输入序列存在重构问题和预测结果的波动性，进而对上述问题提出了改善方法
    - [Exercise-Enhanced Sequential Modeling for Student Performance Prediction](https://github.com/ZoeYuhan/Knowledge_Tracing/blob/master/Paper/Exercise-Enhanced%20Sequential%20Modeling%20for%20Student%20Performance%20Prediction.pdf)
        - 将题面信息引入，不仅作为输入送入模型，而且将题目编码后的向量计算cosine相似度作为atention的socre
        
----
### Acknowledgement
- Blog:
    - [深度知识追踪](https://blog.csdn.net/Zoe_Su/article/details/84481651)
    - [论文导读：Exercise-Enhanced Sequential Modeling for Student Performance Prediction](https://blog.csdn.net/Zoe_Su/article/details/84566409)