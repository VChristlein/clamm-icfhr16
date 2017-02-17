"""
Copyright(c) 2016 Vincent Christlein <vincent.christlein@fau.de>
No warranty for anything, usage with own risk.

License: GPLv.3


Estimates for each image of the input folder the containing script type and
writes all estimations in the files 'belonging_class.csv' and
'belonging_matrix.csv' (written to the current folder)

Necessary files in the `trained` folder:
    - pretrained principal component analysis (pca.pkl.gz)
    - pretrained Gaussian mixture model (gmm.pkl.gz)
    - pretrained total-variability space (tv.pkl.gz)
    - pretrained within-class-covariance-normalization (wccn.pkl.gz)
    - pretrained support vector machines (svm.pkl.gz)
    - optional: pretrained linear discriminant analysis (lda.pkl.gz)
    --Note: in the competition lda was used for task2, however note that svm achieves
    better results
usage:
    python2 <inputdir> 

where <inputdir> is the folder of the input images. 

You can also evaluate on your own dataset using a labelfile, see '-h' for more
possible options.
"""
import os
import glob
import argparse
import cv2
import numpy as np
import multiprocessing
import gzip
import cPickle
import scipy.spatial.distance as spdistance
from sklearn import preprocessing
import zca # Note: contains actually only a class for regularized PCA 

def parserArguments(parser):
    parser.add_argument('inputdir', help='input dir containing the images')
    parser.add_argument('--parallel', action='store_true',
                        help='parallel forks')
    parser.add_argument('--nprocs', type=int, 
                         default=multiprocessing.cpu_count(),
                         help='number of parallel instances')
    parser.add_argument('-l', '--labelfile', 
                        help='labelfile each row: <FILENAME> <CLASS_LABEL>,'
                        ' where class_label is a numeric number')
    parser.add_argument('-s', '--suffix', default='' ,
                        help='image suffix, if not given: all files are assumed'
                        ' to be images in the inputdir')
    parser.add_argument('--progress', action='store_true', 
                        help='show a progress bar')
    return parser

def spawn(f):
    def fun(q_in,q_out):
        while True:
            i,x = q_in.get()
            if i == None:
                break
            q_out.put((i,f(x)))
    return fun

def parmap(f, X, nprocs = multiprocessing.cpu_count()):
    q_in   = multiprocessing.Queue(1)
    q_out  = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=spawn(f),args=(q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]

def getPosteriors(gmm, data, 
		  nbest=0):
    """
    compute the posterior probability (assignment) for each sample

    parameters:
        gmm: scikit-learn computed gmm
        data: feature-vectors row-wise
        nbest: keep only the nbest, i.e. n highest posteriors
    """

    posteriors = gmm.predict_proba(data)

    if nbest > 0:
        # create grid
        gr = np.ogrid[:posteriors.shape[0], :posteriors.shape[1]]        
        # partial  sorting
        indices = np.argpartition(-posteriors, nbest, axis=1)
        # replace axis 1 with all but the nbest 
        gr[1] = indices[:,nbest:]        
        # ... and set them to 0
        posteriors[gr] = 0.0

        # and re-normalize them such that all posteriors sum
        # up to 1 again
        # Note: seems not to be needed
        #posteriors = preprocessing.normalize(posteriors, norm='l1', copy=False)

    return posteriors   

def compute_bw_stats(data, ubm, posteriors):
    nmix, ndim = ubm.means_.shape
    # code assumes everything column-based not row-based
    m = ubm.means_.T.reshape(-1,1)
    idx_sv = np.arange(nmix).repeat(ndim).reshape(-1)

    N, F = expectation(data, posteriors)
    #  centralized first order statistics
    F = F - N[idx_sv] * m
    return N, F

def expectation(data, posteriors):
    N = np.sum(posteriors,0).reshape(-1,1)
    F = data.T.dot(posteriors).T.reshape(-1,1)
    return N, F

def extract_ivector(N, F, ubm, T):
    """
    extracts i-vector for N, F - stats with ubm and T matrix 
    """
    nmix, ndim = ubm.means_.shape
    S = ubm.covars_.T.reshape(-1,1)
    # same as:
#    S = np.reshape(ubm.covars_, (-1,1), order='F')
    idx_sv = np.arange(nmix).repeat(ndim).reshape(-1)

    tv_dim = T.shape[0]
    I = np.eye(tv_dim)
    T_invS = T / S.T 

    tmp = T_invS * N[idx_sv].T
    L = I + tmp.dot(T.T)
    B = T_invS.dot(F)
    # TODO maybe use solve here to gain some speed?
    #x = np.linalg.pinv(L).dot(B)
    x = np.linalg.solve(L, B)

    return x

def load(filename):
    if not filename.endswith('.pkl.gz'):
        filename += '.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        ret = cPickle.load(f)
    return ret

def getFiles(folder, pattern, labelfile=None):
    assert(folder is not None)
    if not os.path.exists(folder):
        raise ValueError('folder {} doesnt exist'.format(folder))    
    if not os.path.isdir(folder):
        raise ValueError('folder {} is not a directory'.format(folder))    

    if labelfile:
        labels = []
        with open(labelfile, 'r') as f:
            all_lines = f.readlines()
        all_files = []
        check = True
        for line in all_lines:
            try:
                img_name, class_id = line.split()
            except ValueError:
                if check:
                    print ('WARNING: labelfile apparently doesnt contain a label, '
                          ' you can ignore this warning if you dont use the'
                          ' label, e.g. if you just want the'
                          ' files to be read in a certain order')
                    check = False
                img_name = line.strip(' \t\n\r')
                class_id = None
            except:
                raise
            if pattern != '':
                file_name = os.path.join(folder, os.path.splitext(img_name)[0]\
                                         + pattern )
            else:
                file_name = os.path.join(folder, img_name )

            all_files.append(file_name)
            labels.append(class_id)                

        return all_files, labels

    return glob.glob(os.path.join(folder, '*' + pattern )), None

def computeDistances(descriptors, distance=True, parallel=True,
                     nprocs=None, normalize=False):
    num_desc = len(descriptors)

    if np.isnan(descriptors).any():
        raise ValueError('nan in descr!')
    if np.isinf(descriptors).any():
        raise ValueError('inf in descr!')

    for i in range(len(descriptors)):
#        if np.count_nonzero(descriptors[i]) == 0:
        if not descriptors[i].any(): # faster
            print 'WARNING: complete row {} is 0'.format(i)

    indices = [(y,x) for y in range(num_desc-1) for x in range(y+1, num_desc)]
    splits = np.array_split(np.array(indices), 8)
    def loop(inds): 
        dists = []
        for ind in inds:
            dist = spdistance.cosine( descriptors[ ind[0]], descriptors[ ind[1]])
#            dist = 1.0 - np.dot(descriptors[ind[0]], descriptors[ind[1]]) / \
#                ( np.sqrt(descriptors[ind[0]]**2) *\
#                 np.sqrt(descriptors[ind[1]]**2))
            dists.append(dist)
        return dists

    if parallel:
        dists = parmap(loop, splits, nprocs)
    else:
        dists = map(loop, splits) 
  
    # convert densed vector-form to matrix
    dense_vector = np.concatenate( dists )
    if spdistance.is_valid_y(dense_vector, warning=True):
        dist_matrix = spdistance.squareform( dense_vector )
    else:
        print 'ERROR: not a valid condensed distance matrix!'
        n = dense_vector.shape[0]
        d = int(np.ceil(np.sqrt(n * 2)))
        should = d * (d - 1) / 2
        raise ValueError('{} != {}, num: {}'.format(should, n, num_desc))
                    
    # do some checks
    if np.isnan(dist_matrix).any():
        print 'WARNING have a nan in the dist-matrix'
    if np.isinf(dist_matrix).any():
        print 'WARNING have a inf in the dist-matrix'
    

    if normalize:
        dist_matrix /= np.sum(dist_matrix)
#    if distance:
#        if np.count_nonzero(dist_matrix == np.finfo(float).max) > 0:
#            raise ValueError('there is already a float-maximum')
#        if normalize:
#            dist_matrix /= np.sum(dist_matrix)
#        np.fill_diagonal(dist_matrix, np.finfo(float).max)        
#    else:
#        if np.count_nonzero(dist_matrix == np.finfo(float).min) > 0:
#            raise ValueError('there is already a float-min')
#        if normalize:
#            dist_matrix /= np.sum(dist_matrix)
#        np.fill_diagonal(dist_matrix, np.finfo(float).min)

    return dist_matrix #, dist_m

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encoding')
    parser = parserArguments(parser)
    args = parser.parse_args()

    print 'Copyright(c) 2016 Vincent Christlein <vincent.christlein@fau.de>'

    files, labels = getFiles(args.inputdir, args.suffix, args.labelfile)
    if len(files) == 0: raise ValueError('no image files found')
  
    pca = load(os.path.join('trained','pca'))
    gmm = load(os.path.join('trained','gmm'))
    tv_space = load(os.path.join('trained','tv'))
    wccn = load(os.path.join('trained','wccn'))

    svm = load(os.path.join('trained','svm'))
    if os.path.exists('trained', 'lda.pkl.gz'):
        lda = load(os.path.join('trained','lda'))
    else: 
        lda = None

    detector = cv2.FeatureDetector_create('SIFT')
    descriptor_ex = cv2.DescriptorExtractor_create('SIFT')

    if args.progress:
        import progressbar
        widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
        progress = progressbar.ProgressBar(widgets=widgets,
                                       maxval=len(files))
    def encode(i):
        img = cv2.imread(files[i], cv2.CV_LOAD_IMAGE_UNCHANGED)
        if img is None:
            raise ValueError('ERROR: cannot read img: {}'.format(files[i]))

        kpts = detector.detect(img)
        if kpts is None or len(kpts) == 0:
            raise ValueError('ERROR: no keypoints detected in img: '
                             ' {}'.format(files[i]))

        # drop orientation
        for kp in kpts:
            kp.angle = 0.0

        _, features = descriptor_ex.compute(img, kpts)
        if features is None or len(features) == 0:
            raise ValueError('ERROR: no features extracted, img:'
                             ' {}'.format(files[i]))

        # hellinger: first L1 normalization then sqrt
        features = preprocessing.normalize(features, norm='l1')
        #features += np.finfo(np.float32).eps
        #features /= np.sum(features, axis=1)[:,np.newaxis]
        # square root - guess np.sign not needed here
        features = np.sign(features) * np.sqrt( np.abs(features) )

        # pca
        data = pca.transform(features)
        data = preprocessing.normalize(data, norm='l2')

        # i-vector-encoding
        posteriors = getPosteriors(gmm, data, nbest=10)
        N, F = compute_bw_stats(data, gmm, posteriors)
        enc = extract_ivector(N, F, gmm, tv_space).reshape(1,-1)

        # post-process
        enc = preprocessing.normalize(enc, norm='l2')
        enc = wccn.dot(enc.T).reshape(1,-1)
        enc = preprocessing.normalize(enc, norm='l2')

        if args.progress: progress.update(i+1)
        return enc
   
    print '=> start encoding files from inputfolder, this might take a while...'
    if args.progress: progress.start()
    if args.parallel:
        all_encs = parmap(encode, range(len(files)), args.nprocs)
    else:
        all_encs = map(encode, range(len(files)))
    if args.progress: progress.finish()

    all_encs = np.concatenate(all_encs, axis=0)

    # symmetrical distance matrix
    print '=> Compute distance matrix'
    dist_matrix = computeDistances(all_encs, distance=True, parallel=args.parallel,
                                  nprocs=args.nprocs, normalize=True)  
    np.savetxt('normalized_symmetrical_distance_matrix.csv', dist_matrix, delimiter=',')
    
    print '=> predict labels and probabilities'
    pred_labels = svm.predict(all_encs)
    if lda:
        #pred_labels = lda.predict(all_encs)
        pred_probas = lda.predict_proba(all_encs)
    else:
        pred_probas = svm.decision_function(all_encs)

    print '=> save files'
    with open('belonging_class.csv', 'w') as f:
        f.write('FILENAME,SCRIPT_TYPE\n')
        for e, fn in enumerate(files):
            if all_encs[e] is None:
                continue
            f.write('{},{}\n'.format(os.path.basename(fn), pred_labels[e] + 1))

    with open('belonging_matrix.csv', 'w') as f:
        first_line = 'FILENAME'
        for i in range(12):
            first_line += ',SCRIPT_TYPE{}'.format(i+1)
        f.write(first_line + '\n')
        for e, fn in enumerate(files):
            if all_encs[e] is None:
                continue
            f.write('{}'.format(os.path.basename(fn)))
            for i in range(12):
                f.write(',{}'.format(pred_probas[e,i]))
            f.write('\n')


    if labels is not None:
        from sklearn.metrics import accuracy_score
        new_labels = []
        for l in labels:
            new_labels.append( l if len(l) == 2 else '0'+l)
        t_labels = new_labels
        le = preprocessing.LabelEncoder()        
        t_labels = le.fit_transform(np.array(t_labels))

        print 'score:', accuracy_score(t_labels, pred_labels)


