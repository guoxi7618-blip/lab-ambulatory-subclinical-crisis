可用数据全局实验复算｜2026-10-28

先读：大学生亚临床心理危机研究_全局复算与文献核查_20261028.html
文章框架：文章框架与修改清单_20261028.txt
数值核对：audit_20261028_claim_audit.csv

工作目录：仓库根目录
本次解释器：D:\python3.11\python.exe。软件具体版本见audit_20261028_run1_manifest.json。
所需包：numpy、pandas、scipy、scikit-learn、statsmodels。H1无需MNE；执行的是5000次max-T替代敏感性，不是原cluster方法。
数据全部读取data目录；不得移动、更改、删除原始数据。运行会创建新产物，已存在文件会触发拒绝覆盖。

统一入口：
python -B audit_20261028_rerun_all_v2.py --list
python -B audit_20261028_rerun_all_v2.py --tag rerun_20261029
tag请每次换新值。默认tag含时间。程序不会删除旧结果、终止其他任务、安装依赖或修改代理。
入口为保留历史代码，只在内存中替换各脚本的字符串产物前缀，然后逐阶段执行；方程、随机种子及输入路径不变。源码可逐份审查。
入口已实际测试完整性阶段；各计算阶段分别实际执行成功。文献阅读/综合判断不是自动训练过程，不因重跑代码自动重新审稿。

单独运行（使用新前缀）：
python -B audit_20261028_integrity_v2.py --prefix integrity_new_run
python -B audit_20261028_h1.py --prefix h1_new_run --permutations 5000
python -B audit_20261028_models.py --prefix model_new_run
python -B audit_20261028_calibration_sensitivity.py --prefix calibration_new_run --bootstrap 2000
跨脚本依赖的后续细化建议用统一入口，以确保输入结果前缀一致。

权威结果优先级：
1. integrity_v2 + integrity_addendum：EO/EC和D0/全部会话口径已澄清；早期integrity文件只留作历史。
2. h1_student_markers为对齐原表主口径；h1_run1_markers是Welch敏感性。h1_definition_audit解释调整效应与ICC定义。
3. run1_provided_prediction_metrics：原19预测计分；calibration_intercept字段是联合斜率估计截距。原稿对应calibration-in-the-large应看metric_definitions.csv。
4. run1_h2中的随机截距拟合有奇异解，不引用该拟合；使用refined_h2的有效随机截距+斜率或GEE。run1_h2人级相关仍有效。
5. daily_refined_*采用有效生理prompt聚合并复现3787/3691样本；run1_within_person和run1_exam_associations用全已完成EMA/不同模型，是历史重建，不等于原文。
6. refined_h4披露GEE相关结构敏感性。Table5修正值看table5_corrections.csv；时段4行没有数据不可复算。
7. supplementary_audit_v2.py为成功代码；原supplementary_audit.py的索引错误版本保留，不作为交付运行入口。
8. run1_retrained_*为新的训练，不是原MSAF。refined_model_convergence确认8个最终模型均收敛、预测一致。

不确定性与统计口径：
固定预测bootstrap仅反映测试参与者抽样不确定性，不含重训随机性。H3 2000次普通参与者配对bootstrap；校准敏感性为2000次按标签分层配对bootstrap；日级重采样整个参与者簇。
新简单模型在core训练内5折选C，阈值来自参与者级嵌套OOF；extension不选超参数。原宽网格中部分未选中高C候选有收敛警告，最终选中模型均经100000上限复核收敛。
基率18%是应用假定；原表实际隐含18.111587%。模型增量p值是名义配对检验，不可自动理解为跨所有分析校正后成立。
没有原MSAF权重、窗级输入与原脚本，不把网络结构的文字重写或现成概率计分称为MSAF重训。

原文件保持不变。输入39CSV的SHA256在manifest和verification中复核一致。
