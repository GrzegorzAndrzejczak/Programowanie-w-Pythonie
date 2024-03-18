import numpy as np
from numbers import Number
from sklearn.preprocessing import OrdinalEncoder, LabelBinarizer
from sklearn.naive_bayes import CategoricalNB, GaussianNB


class BoundedCategoricalNB(CategoricalNB):

    def _check_X(self, X):
        max_cats = self.n_categories_ - 1
        # print(max_cats)
        X = super()._check_X(X)
        for i in range(X.shape[1]):
            X[X[:, i]>max_cats[i], i] = max_cats[i]
        return X


class OrdinalNanEncoder(OrdinalEncoder):        # Dopuszcza nan's dla uczących i dla transformacji
                                                # W trakcie uczenia ewent. None's traktuje jak etykiety,
                                                # - transformacja zamienia None's i nieznane etykiety na nan's
    def fit(self, X, y=None):
        self.handle_unknown = "error"
        return super().fit(X,y)

    def transform(self, X):
        self.handle_unknown = 'use_encoded_value'
        return super().transform(X)


class CategoricalNanNB(CategoricalNB):
    enc: OrdinalNanEncoder
    n_features_in_: int                     # pole jest w BaseEstimator

    def _fit(self, y):
        classes, class_counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.class_log_prior_ = np.log(class_counts) - np.log(class_counts.sum())

    def _check_X(self, X):
        return self.enc.transform(X)

    def fit(self, X, y, encoder: OrdinalNanEncoder = None, **kwargs):   # zamiast `super().fit(X,y)`
        self._fit(y)
        self.enc = OrdinalNanEncoder() if encoder is None else encoder
        enc = self.enc
        if not hasattr(enc, "categories_"):
            enc.fit(X)
        self.n_features_in_ = X.shape[1]
        for j in range(self.n_features_in_):
            cat = enc.categories_[j][-1]
            if isinstance(cat, Number) and np.isnan(cat):       # pomijamy "nan", zostają  None - jeśli są
                enc.categories_[j] = enc.categories_[j][:-1]    # można usunąć końce z nan ...
        self._count(X, y)

    def _count(self, X, y):
        X_ = self._check_X(X)
        n_classes = self.classes_.size
        n_features = self.n_features_in_
        n_categories = [self.enc.categories_[j].shape[0] for j in range(n_features)]
        self.category_count_ = [np.zeros((n_classes, n_categories[j])) for j in range(n_features)]
        for i in range(n_classes):
            mask_i = y == self.classes_[i]
            for j in range(n_features):
                cat_count = self.category_count_[j]
                indices, counts = np.unique(X_[mask_i, j], return_counts=True)  # zlicza też nan's
                if np.isnan(indices[-1]):
                    indices, counts = indices[:-1], counts[:-1]                 # ... ale pomija wartości
                cat_count[i, indices.astype(int)] += counts
        self._update_feature_log_prob(self.alpha)

    def _joint_log_likelihood(self, Z):
        clp = self.cond_log_proba(Z)
        return clp + self.class_log_prior_

    def cond_log_proba(self, Z):    # log( P(x|Y=i) ), potrzebne dla MixedNB
        Z_ = self._check_X(Z)       # categorial --> numbered / nan's
        n_classes = self.classes_.size
        clp = np.zeros((Z_.shape[0], n_classes))  # joint_log_likelihood
        for j in range(self.n_features_in_):
            z_j = Z_[:, j]
            j_nans = np.isnan(z_j)
            z_j_ints = z_j[~j_nans].astype(int)    # TE są całkowite
            clp[~j_nans, :] += self.feature_log_prob_[j][:, z_j_ints].T
        return clp

    def predict_log_proba(self, Z):
        from scipy.special import logsumexp
        jll = self._joint_log_likelihood(Z)
        log_prob = logsumexp(jll, axis=1)
        return jll - log_prob[:, None]

    def predict(self, Z):
        prob = self.predict_proba(Z)
        return self.classes_.take(prob.argmax(1))


from sklearn.base import ClassifierMixin, BaseEstimator


class MixedNB(ClassifierMixin, BaseEstimator):
    cmask: np.ndarray
    claNB: CategoricalNanNB = CategoricalNanNB()

    def __init__(self, classifier, cat_nb=None):
        self.classifier = classifier        # obiekt, nie klasa
        if cat_nb: self.claNB = cat_nb
        self.n_classes, self.n_features = 2, 0

    def fit(self, X, y, nums: list[int] = None, **kwargs):      # raz wpisana maska staje się domyślna
        num_a = X.shape[1]
        self.n_features = num_a
        if nums is not None:
            nmask = np.asarray([(i in nums) or (i-num_a in nums) for i in range(num_a)])
            self.cmask = ~nmask
        else:
            nmask = ~self.cmask
        Xc, Xn = X[:, ~nmask], X[:, nmask]     # nominal, numeric(gaussian?)
        claNB = self.claNB
        claNB.fit(Xc, y)
        self.n_classes = self.claNB.classes_.size
        cla = self.classifier
        cla.fit(Xn, y)

    def predict_log_proba(self, Z):
        Zc = Z[:, self.cmask]
        Zn = Z[:, ~self.cmask]
        clp_c = self.claNB.cond_log_proba(Zc)  # sum(log P_NB(x|c))
        log_pr_n = self.classifier.predict_log_proba(Zn)  # log P_LR(c|x)
        cll = clp_c + log_pr_n
        # normalize by sumexp
        from scipy.special import logsumexp
        log_prob_rel = logsumexp(cll, axis=1)
        return cll - log_prob_rel[:, None]

    def predict_proba(self, Z):
        return np.exp(self.predict_log_proba(Z))

    def predict(self, Z, threshold=.5):
        res = self.predict_proba(Z)
        y_pred = self.claNB.classes_.take(res.argmax(1))
        return y_pred


def xStratification(y, n_splits=10, rand=np.random.default_rng(), alloc=False):
    # n_inclass = np.bincount(y_)  #
    classes, n_inclass = np.unique(y, return_counts=True)  # [0, 1], [497, 266]
    n_classes = len(classes)  # 2
    n_samples = y.size  # 763

    y_order = np.arange(n_classes).repeat(n_inclass)  # [0..(497)..0, 1..(266)..1]
    allocation = np.asarray([
        np.bincount(y_order[i:: n_splits], minlength=n_classes)
        for i in range(n_splits)
    ])
    if alloc:
        print(allocation.T)  #
    test_folds = np.empty(n_samples, dtype="i")
    for j in range(n_classes):
        folds_for_class = np.arange(n_splits).repeat(allocation[:, j])
        if rand:
            rand.shuffle(folds_for_class)
        test_folds[y == classes[j]] = folds_for_class
    return test_folds


def xValidation(clf, X, y, test_folds):     # , coeff=False):
    n_splits = max(test_folds) + 1
    y_prob = np.empty(X.shape[0], dtype="f")  # the array of probabilities
    for i in range(n_splits):  # i-th portion
        mask_i = test_folds == i
        test_index = np.where(mask_i)[0]
        train_index = np.where(~mask_i)[0]

        nn = clf.fit(X[train_index], y[train_index])  # as classifies do
        y_prob_i = clf.predict_proba(X[test_index])[:, 1]
        y_prob[test_folds == i] = y_prob_i
    return y_prob


def ROC(y_true, y_score, details=False):
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_sc = y_score[desc_score_indices]
    y_tr = y_true[desc_score_indices]
    tps = np.cumsum(y_tr)
    # fps = 1 + np.arange(y_true.size) - tps
    distinct_value_idxs = np.where(np.diff(y_sc))  # points where the score changes
    # append the last index
    threshold_idxs = np.append(distinct_value_idxs, y_true.size - 1)
    tp = tps[threshold_idxs]
    # fp = fps[threshold_idxs]        # or
    fp = 1 + threshold_idxs - tp  # clever simplification
    if details:
        return fp, tp, y_sc[threshold_idxs]
    return fp, tp


def conf_mtx(y_true, y_pred, tbl=None):
    if tbl is None:
        nn = int(np.max(y_true) + 1)
        tbl = np.zeros((nn, nn), dtype=np.int32)
    for y0, y1 in zip(y_true, y_pred):
        tbl[y0, y1] += 1
    return tbl


def AUC(FP, TP):
    fp, tp = FP/FP[-1],  TP/TP[-1]
    return np.dot(tp[:-1]+tp[1:], fp[1:]-fp[:-1]) / 2
