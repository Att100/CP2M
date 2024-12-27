import paddle
import paddle.nn.functional as F


class Metrics:
    def __init__(self, num_classes, imsize) -> None:
        self.num_classes = num_classes
        self.imsize = imsize
        self.n_samples = 0
        self.total_miou = dict()
        self.total_acc = dict()
        self.current_miou = 0
        self.current_acc = 0
        self.reset()
    
    def update(self, preds, labels):
        current_intersection = paddle.zeros([self.num_classes], dtype='int64')
        current_union = paddle.zeros([self.num_classes], dtype='int64')
        
        for i in range(self.num_classes):
            preds_cls = (preds == i)
            labels_cls = (labels == i)

            intersection_cls = paddle.logical_and(preds_cls, labels_cls)
            union_cls = paddle.logical_or(preds_cls, labels_cls)
            
            current_intersection[i] = int(paddle.sum(intersection_cls.astype('int32')))
            current_union[i] = int(paddle.sum(union_cls.astype('int32')))
            self.total_miou['total_intersection'][i] += current_intersection[i]
            self.total_miou['total_union'][i] += current_union[i]
        
        self.n_samples += preds.shape[0]
        self.current_miou = float(self._miou(current_intersection, current_union))
        self.current_acc = paddle.sum((preds == labels).astype('float64'))
        self.total_acc['n_correct'] += self.current_acc
        self.current_acc = float(self.current_acc / (preds.shape[0] * self.imsize[0] * self.imsize[1]))
    
    def reset(self):
        self.n_samples = 0
        self.total_miou = {
            'value': 0,
            'total_intersection': paddle.zeros([self.num_classes], dtype='int64'),
            'total_union': paddle.zeros([self.num_classes], dtype='int64')
        }
        self.total_acc = {
            'value': 0,
            'n_correct': 0,
        }
        self.current_miou = 0
        self.current_acc = 0
        
    def total_values(self):
        self.total_miou['value'] = self._miou(
            self.total_miou['total_intersection'], 
            self.total_miou['total_union'])
        self.total_acc['value'] = self.total_acc['n_correct'] / (self.n_samples*self.imsize[0]*self.imsize[1])
        return dict(miou=float(self.total_miou['value']), acc=float(self.total_acc['value']))
    
    def _miou(self, intersection, union):
        union = union.astype('float64')
        union = paddle.clip(union, min=1.0)

        iou = intersection / union
        iou = paddle.where(union == 0, paddle.zeros_like(iou), iou)

        return paddle.mean(iou)
    
    
def per_class_precision_and_iou(model, test_loader, num_classes):
    total_TP = paddle.zeros([num_classes], dtype='int64')
    total_FP = paddle.zeros([num_classes], dtype='int64')
    total_FN = paddle.zeros([num_classes], dtype='int64')

    for img, label in test_loader():
        pred = model(img)
        pred = paddle.argmax(F.softmax(pred, axis=1), axis=1)
        for cls in range(num_classes):
            cls_pred = pred == cls
            cls_label = label == cls

            total_TP[cls] += paddle.sum(cls_pred & cls_label)
            total_FP[cls] += paddle.sum(cls_pred & ~cls_label)
            total_FN[cls] += paddle.sum(~cls_pred & cls_label)

    per_class_precision = total_TP / paddle.clip(total_TP + total_FP, min=1)
    per_class_iou = total_TP / paddle.clip(total_TP + total_FP + total_FN, min=1)
    
    return per_class_precision, per_class_iou