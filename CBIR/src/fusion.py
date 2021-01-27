# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import my_evaluate_class
from DB import Database

from color import Color
from edge import Edge
from gabor import Gabor

import numpy as np
import itertools
import os

d_type = 'd1'
depth = 30
#Ci-dessous, les différents modules recodés permettant de tester la fusion
feat_pools = ['color', 'edge'] #'gabor', 'daisy', 'hog', 'vgg', 'res']

# result dir
result_dir = 'result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

#Classe permettant de fusionner les classes choisies pour le traitement
class FeatureFusion(object):

    def __init__(self, features):
        assert len(features) > 1, "need to fuse more than one feature!"
        self.features = features
        self.samples = None
        self.testSamples = None

    def make_samples(self, db, db_name, verbose=True):
        if verbose:
            print("Use features {}".format(" & ".join(self.features)))

        if db_name == "train":
            if self.samples is None:
                feats = []
                for f_class in self.features:
                    feats.append(self._get_feat(db, db_name, f_class))
                samples = self._concat_feat(db, feats)
                self.samples = samples  # cache the result
            return self.samples
        else:
            if self.testSamples is None:
                feats = []
                for f_class in self.features:
                    feats.append(self._get_feat(db, db_name, f_class))
                testSamples = self._concat_feat(db, feats)
                self.testSamples = testSamples  # cache the result
            return self.testSamples

    #fonction permettant de récuperer la classe en fonction du nom
    def _get_feat(self, db,db_name, f_class):
        if f_class == 'color':
            f_c = Color()
        # elif f_class == 'daisy':
        # f_c = Daisy()
        elif f_class == 'edge':
            f_c = Edge()
            """  elif f_class == 'gabor':
            f_c = Gabor()
              elif f_class == 'hog':
      f_c = HOG()
    elif f_class == 'vgg':
      f_c = VGGNetFeat()
    elif f_class == 'res':
      f_c = ResNetFeat()"""
        return f_c.make_samples(db, db_name, verbose=False)

    def _concat_feat(self, db, feats):
        samples = feats[0]
        delete_idx = []
        for idx in range(len(samples)):
            for feat in feats[1:]:
                feat = self._to_dict(feat)
                key = samples[idx]['img']
                if key not in feat:
                    delete_idx.append(idx)
                    continue
                assert feat[key]['cls'] == samples[idx]['cls']
                samples[idx]['hist'] = np.append(samples[idx]['hist'], feat[key]['hist'])
        for d_idx in sorted(set(delete_idx), reverse=True):
            del samples[d_idx]
        if delete_idx != []:
            print("Ignore %d samples" % len(set(delete_idx)))

        return samples

    def _to_dict(self, feat):
        ret = {}
        for f in feat:
            ret[f['img']] = {
                'cls': f['cls'],
                'hist': f['hist']
            }
        return ret


def evaluate_feats(db1, db2, N, feat_pools=feat_pools, d_type='d1', depths=[200, 100, 50, 30, 10, 5, 3, 1]):
    result = open(os.path.join(result_dir, 'feature_fusion-{}-{}feats.csv'.format(d_type, N)), 'w')
    for i in range(N):
        result.write("feat{},".format(i))
    result.write("depth,distance,MMAP")
    combinations = itertools.combinations(feat_pools, N)
    for combination in combinations:
        sommeBonnesReponsesCombinaison = 0
        fusion = FeatureFusion(features=list(combination))
        for d in depths:
            APs, prevision = my_evaluate_class(db1, db2, f_instance=fusion, d_type=d_type, depth=d)

            sommeBonnesReponses = 0

            for i in range(0, len(db_test)):
                #print("Prevision {}, {}".format(db_test.data.img[i], prevision[i]))
                if prevision[i] in db_test.data.img[i]:  # Ayant trié les données de tests, je suis en mesure de savoir si mon modèle récupère la bonne réponses. Avec les données rentrées, la moyenne est de 78%
                    sommeBonnesReponses += 1
            print("Moyennes bonnes réponses = {}".format(sommeBonnesReponses / len(db_test) * 100))
            sommeBonnesReponsesCombinaison += sommeBonnesReponses / len(db_test)
            cls_MAPs = []
            for cls, cls_APs in APs.items():
                MAP = np.mean(cls_APs)
                cls_MAPs.append(MAP)
            r = "{},{},{},{}".format(",".join(combination), d, d_type, np.mean(cls_MAPs))
            print(r)

            result.write('\n' + r)
        print("Moyennes {} bonnes réponses tout depth= {}".format(",".join(combination), sommeBonnesReponsesCombinaison/len(depths) * 100))
    result.close()


if __name__ == "__main__":
    # On crée les deux bases, celle de test et celle de train
    DB_train_dir_param = "../../ReseauDeNeurones/data/train"
    DB_train_csv_param = "database/data_train.csv"

    db_train = Database(DB_train_dir_param, DB_train_csv_param)

    DB_test_dir_param = "../../ReseauDeNeurones/data/test_classés"
    DB_test_csv_param = "database/data_test.csv"

    db_test = Database(DB_test_dir_param, DB_test_csv_param)

    # evaluate features double-wise
    evaluate_feats(db_train,db_test, N=2, d_type='d1')

    # evaluate features triple-wise
    evaluate_feats(db_train,db_test, N=3, d_type='d1')

    # evaluate features quadra-wise
    evaluate_feats(db_train,db_test, N=4, d_type='d1')

    # evaluate features penta-wise
    evaluate_feats(db_train,db_test, N=5, d_type='d1')

    # evaluate features hexa-wise
    evaluate_feats(db_train,db_test, N=6, d_type='d1')

    # evaluate features hepta-wise
    evaluate_feats(db_train,db_test, N=7, d_type='d1')

    # evaluate database
    fusion = FeatureFusion(features=['color', 'edge'])
    APs, prevision = my_evaluate_class(db_train, db_test, f_instance=fusion, d_type=d_type, depth=depth)
    cls_MAPs = []

    sommeBonnesReponses = 0

    for i in range(0, len(db_test)):
        print("Prevision {}, {}".format(db_test.data.img[i], prevision[i]))
        if prevision[i] in db_test.data.img[i]:  # Ayant trié les données de tests, je suis en mesure de savoir si mon modèle récupère la bonne réponses. Avec les données rentrées, la moyenne est de 78%
            sommeBonnesReponses += 1

    print("Moyennes bonnes réponses = {}".format(sommeBonnesReponses / len(db_test) * 100))
