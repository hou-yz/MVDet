import numpy as np


# import matlab.engine
# from multiview_detector.evaluation.pyeval.evaluateDetection import evaluateDetection_py


# def matlab_eval(res_fpath, gt_fpath, dataset='wildtrack'):
#     eng = matlab.engine.start_matlab()
#     eng.cd('multiview_detector/evaluation/motchallenge-devkit')
#     res = eng.evaluateDetection(res_fpath, gt_fpath, dataset)
#     recall, precision, moda, modp = np.array(res['detMets']).squeeze()[[0, 1, -2, -1]]
#     return recall, precision, moda, modp
#
#
# def python_eval(res_fpath, gt_fpath, dataset='wildtrack'):
#     modp, moda, recall, precision = evaluateDetection_py(res_fpath, gt_fpath, dataset)
#     return recall, precision, moda, modp


def evaluate(res_fpath, gt_fpath, dataset='wildtrack'):
    try:
        import matlab.engine

        eng = matlab.engine.start_matlab()
        eng.cd('multiview_detector/evaluation/motchallenge-devkit')
        res = eng.evaluateDetection(res_fpath, gt_fpath, dataset)
        recall, precision, moda, modp = np.array(res['detMets']).squeeze()[[0, 1, -2, -1]]
    except:
        from multiview_detector.evaluation.pyeval.evaluateDetection import evaluateDetection_py

        recall, precision, moda, modp = evaluateDetection_py(res_fpath, gt_fpath, dataset)
    return recall, precision, moda, modp


if __name__ == "__main__":
    import os

    res_fpath = os.path.abspath('test-demo.txt')
    gt_fpath = os.path.abspath('gt-demo.txt')
    os.chdir('../..')
    print(os.path.abspath('.'))

    # recall, precision, moda, modp = matlab_eval(res_fpath, gt_fpath, 'Wildtrack')
    # print(f'matlab eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')
    # recall, precision, moda, modp = python_eval(res_fpath, gt_fpath, 'Wildtrack')
    # print(f'python eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')

    recall, precision, moda, modp = evaluate(res_fpath, gt_fpath, 'Wildtrack')
    print(f'eval: MODA {moda:.1f}, MODP {modp:.1f}, prec {precision:.1f}, rcll {recall:.1f}')
