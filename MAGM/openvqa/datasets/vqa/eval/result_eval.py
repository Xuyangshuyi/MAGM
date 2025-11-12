from openvqa.datasets.vqa.eval.vqa import VQA
from openvqa.datasets.vqa.eval.vqaEval import VQAEval
import json, pickle
import numpy as np


def eval(__C, dataset, ans_ix_list, pred_list, result_eval_file, ensemble_file, log_file, valid=False):
    result_eval_file = result_eval_file + '.json'

    qid_list = [ques['question_id'] for ques in dataset.ques_list]
    ans_size = dataset.ans_size

    result = []
    for qix in range(len(qid_list)):
        pred_ix = int(ans_ix_list[qix])
        try:
            ans_str = dataset.ix_to_ans[str(pred_ix)]
        except Exception:
            ans_str = dataset.ix_to_ans[pred_ix]
        result.append({'answer': ans_str, 'question_id': int(qid_list[qix])})

    print('Save the result to file: {}'.format(result_eval_file))
    json.dump(result, open(result_eval_file, 'w'))

    if valid:
        ans_file_path  = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val'] + '-anno']
        ques_file_path = __C.RAW_PATH[__C.DATASET][__C.SPLIT['val']]
        with open(ans_file_path, 'r') as f:
            ann_json = json.load(f)
        base_qids = [int(a['question_id']) for a in ann_json.get('annotations', [])]

        raw = json.load(open(result_eval_file, 'r'))
        q2a = {}
        for r in raw:
            q2a[int(r['question_id'])] = r['answer']

        pred_qids = [int(x['question_id']) for x in raw]
        uniq = len(q2a)
        dup_cnt = len(pred_qids) - uniq
        extra = sorted(set(q2a.keys()) - set(base_qids))
        miss  = sorted(set(base_qids) - set(q2a.keys()))
        print(f'[VAL ALIGN] BASE={len(base_qids)}  PRED(raw)={len(pred_qids)}  '
              f'UNIQUE={uniq}  DUP={dup_cnt}  MISSING={len(miss)}  EXTRA={len(extra)}')

        aligned = [{'answer': q2a.get(q, ''), 'question_id': q} for q in base_qids]
        json.dump(aligned, open(result_eval_file, 'w'))
        print('[VAL ALIGN] result file has been realigned to ANNOTATION question ids.')

    if valid:
        vqa = VQA(ans_file_path, ques_file_path)
        vqaRes = vqa.loadRes(result_eval_file, ques_file_path)
        vqaEval = VQAEval(vqa, vqaRes, n=2)
        vqaEval.evaluate()

        print("\nOverall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        print("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        print("\n")
        print('Write to log file: {}'.format(log_file))
        with open(log_file, 'a+') as logfile:
            logfile.write("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
            for ansType in vqaEval.accuracy['perAnswerType']:
                logfile.write("%s : %.02f " % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            logfile.write("\n\n")

    if __C.TEST_SAVE_PRED:
        print('Save the prediction vector to file: {}'.format(ensemble_file))
        pred_list = np.array(pred_list).reshape(-1, ans_size)
        result_pred = [{'pred': pred_list[qix], 'qid': int(qid_list[qix])}
                       for qix in range(len(qid_list))]
        pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)



