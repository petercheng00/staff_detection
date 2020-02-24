import cv2

pred_prefix = 'Test_Pred_GRAY/GR_'
gt_prefix = 'Test_GT/GT_'

TP = 0
FP = 0
FN = 0
# for index in range(1, 2001):
for index in range(1000, 2001):
    if index % 100 == 0:
        print(f'{index} / 2000')
    pred_image = cv2.imread(pred_prefix + ('%04d.png' % index), 0)
    gt_image = cv2.imread(gt_prefix + ('%04d.png' % index), 0)

    and_image = pred_image & gt_image
    FP_image = pred_image & cv2.bitwise_not(gt_image)
    FN_image = cv2.bitwise_not(pred_image) & gt_image

    TP += cv2.countNonZero(and_image)
    FP += cv2.countNonZero(FP_image)
    FN += cv2.countNonZero(FN_image)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
print(f'precision: {precision} recall: {recall}')
print(f'f1 score: {2 * precision * recall / (precision + recall)}')
