# In this version, the concentration parameters alpha and gamma are given a gamma prior. Thus no need to specify alpha
# and gamma when initiate an instance of Gibbs sampler.
import numpy as np
import math
import pickle
from model_loglikelihood import dict2mix, mixture_logpdf, all_loglike
np.seterr(divide='ignore')


class Gaussian:
    def __init__(self, X=np.zeros((0, 2)), kappa_0=0., nu_0=1.0001, mu_0=None, Psi_0=None):
        self.n = X.shape[0]
        self.dim = X.shape[1]

        if mu_0 is None:  # initial mean for the cluster
            self._mu_0 = np.zeros((1, self.dim))
        else:
            self._mu_0 = mu_0
        assert(self._mu_0.shape == (1, self.dim))

        self._kappa_0 = kappa_0  # mean fraction

        self._nu_0 = nu_0  # degrees of freedom
        if self._nu_0 < self.dim:
            self._nu_0 = self.dim

        if Psi_0 is None:
            self._Psi_0 = 10*np.eye(self.dim)
        else:
            self._Psi_0 = Psi_0
        assert(self._Psi_0.shape == (self.dim, self.dim))

        self.mean = None
        self.covar = None
        if X.shape[0] > 0:
            self.fit(X)
        else:
            self.default()

    def default(self):
        self.mean = np.matrix(np.zeros((1, self.dim)))
        self.covar = 100.0 * np.matrix(np.eye(self.dim))

    def recompute_ss(self):
        self.n = self._X.shape[0]
        self.dim = self._X.shape[1]
        if self.n <= 0:
            self.default()
            return
        # update conjugate posterior parameters
        kappa_n = self._kappa_0 + self.n
        nu = self._nu_0 + self.n
        mu_hat = np.matrix(self._sum) / self.n
        mu_hat_mu_0 = mu_hat - self._mu_0
        C = self._square_sum - self.n*(mu_hat.transpose()*mu_hat)
        Psi = self._Psi_0 + C + \
            self._kappa_0 * self.n * mu_hat_mu_0.transpose() * mu_hat_mu_0 / (self._kappa_0 + self.n)

        self.mean = (self._kappa_0 * self._mu_0 + self.n * mu_hat) / (self._kappa_0 + self.n)
        self.covar = (kappa_n + 1) / (kappa_n * (nu - self.dim + 1)) * Psi

    def fit(self, X):  # fit data
        self._X = X
        self._sum = X.sum(axis=0)
        self._square_sum = np.matrix(X).transpose() * np.matrix(X)
        self.recompute_ss()

    def rm_point(self, x):
        """
        add a point to current Gaussian mixture
        @:param x: data point to be added
        """
        assert(self._X.shape[0] > 0)
        # Find the index of the point x in self._X
        indices = (abs(self._X - x)).argmin(axis=0)
        indices = np.matrix(indices)
        ind = indices[0, 0]
        for ii in indices:
            if (ii-ii[0] == np.zeros(len(ii))).all():
                ind = ii[0,0]
                break
        tmp = np.matrix(self._X[ind])
        self._sum -= self._X[ind]
        self._X = np.delete(self._X, ind, axis=0)
        self._square_sum -= tmp.transpose() * tmp
        self.recompute_ss()

    def add_point(self, x):
        """
        remove a point from current Gaussian mixture
        @:params x: data point to be removed
        """
        if self.n <= 0:
            self._X = np.array([x])
            self._sum = self._X.sum(0)
            self._square_sum = np.matrix(self._X).transpose() * np.matrix(self._X)
        else:
            self._X = np.append(self._X, [x], axis=0)
            self._sum += x
            self._square_sum += np.matrix(x).transpose() * np.matrix(x)
        self.recompute_ss()

    def pdf(self, x):  # compute the prob density of data point x
        size = len(x)
        assert size == self.mean.shape[1]
        assert (size, size) == self.covar.shape
        det = np.linalg.det(self.covar)
        assert det != 0
        norm_const = 1. / (np.power((2*np.pi), float(size)/2) * np.power(det, .5))
        x_mu = x - self.mean
        res = math.pow(math.e, -.5 * (x_mu * self.covar.I * x_mu.transpose()))
        return norm_const * res

    def logpdf(self, x):    # compute the log prob density of data point x
        size = len(x)
        assert size == self.mean.shape[1]
        assert (size, size) == self.covar.shape
        det = np.linalg.det(self.covar)
        assert det != 0
        norm_const = -np.log(np.power((2*np.pi), float(size)/2) * np.power(det, .5))
        x_mu = x - self.mean
        res = -.5 * (x_mu * self.covar.I * x_mu.transpose())
        return norm_const + res


class GibbsSampler(object):
    """
    @:param snapshot_interval: the interval for exporting a snapshot of the model
    """
    def __init__(self, snapshot_interval=1, compute_loglik=False):
        self._snapshot_interval = snapshot_interval
        self._flag_compute_loglik = compute_loglik

        self._table_info_title = "Table-information-"
        self._dish_info_title = "Dish-information-"
        self._hyper_parameter_title = "Hyper-parameter-"

        self.params = {}

    """
    @param data: a N-by-D np array object, defines N points of D dimension
    @param K: number of topics, number of broke sticks
    @param gamma: the smoothing value for a table to be assigned to a new topic
    @param alpha: the smoothing value for a word to be assigned to a new table

    """
    def _initialize(self, data):
        # initialize the documents
        self._corpus = data
        # initialize the size of the collection, i.e., total number of documents.
        self._D = len(self._corpus)

        # initialize the total number of topics.
        self._K = self._D

        # initialize the parameters of gamma prior of alpha
        self._a = 1.
        self._b = 1.
        self._w_alpha = np.random.random(self._D)
        self._s_alpha = np.random.binomial(1, .5, self._D)
        self._w_gamma = np.random.random()
        self._s_gamma = np.random.binomial(1, .5)

        self._alpha = np.random.gamma(self._a, self._b)
        self._gamma = np.random.gamma(self._a, self._b)

        # initialize the word count matrix indexed by topic id and word id, i.e., n_{\cdot \cdot k}^v
        # self._n_kv = np.zeros((self._K, self._V))
        # initialize the word count matrix indexed by topic id and document id, i.e., n_{j \cdot k}
        self._n_kd = np.zeros((self._K, self._D))
        # initialize the table count matrix indexed by topic id, i.e., m_{\cdot k}
        self._m_k = np.zeros(self._K)

        # initialize the table information vectors indexed by document id and word id, i.e., t{j i}
        self._t_dv = {}
        # initialize the topic information vectors indexed by document id and table id, i.e., k_{j t}
        self._k_dt = {}
        # initialize the word count vectors indexed by document id and table id, i.e., n_{j t \cdot}
        self._n_dt = {}

        # we assume all words in a document belong to one table which is assigned to topic d, where d is the index of
        # document
        for d in xrange(self._D):
            # initialize the table information vector indexed by document and records down which table a word belongs to
            self._t_dv[d] = np.zeros(len(self._corpus[d]), dtype=np.int)

            # self._k_dt records down which topic a table was assigned to
            self._k_dt[d] = np.array([d])
            assert(len(self._k_dt[d]) == len(np.unique(self._t_dv[d])))

            # word_count_table records down the number of words sit on every table
            self._n_dt[d] = np.zeros(1, dtype=np.int) + len(self._corpus[d])
            assert(len(self._n_dt[d]) == len(np.unique(self._t_dv[d])))
            assert(np.sum(self._n_dt[d]) == len(self._corpus[d]))

            self._n_kd[d, d] = len(self._corpus[d])

            # here, document index equals topic index
            self._m_k[d] = 1
            # initialize Gaussian mixtures, each document belongs to one cluster
            self.params[d] = Gaussian(X=self._corpus[d])

    """
    sample the data to train the parameters
    @param iteration: the maximum number of gibbs sampling iteration
    """
    def sample(self, iteration):
        # sample the total data
        iter = 0
        # store log-likelihood for each iteration
        self.log_likelihoods = np.zeros(iteration)
        while iter < iteration:
            iter += 1
            for document_index in np.random.permutation(xrange(self._D)):
                # sample customer assignment, see which table it should belong to
                for word_index in np.random.permutation(xrange(len(self._corpus[document_index]))):
                    self.update_params(document_index, word_index, -1)

                    # get the data at the index position
                    x = self._corpus[document_index][word_index]

                    # compute the log-likelihood
                    f = np.zeros(self._K, dtype=np.float128)
                    flog = np.zeros(self._K, dtype=np.float128)
                    f_new_table = 0.
                    for k in xrange(self._K):
                        flog[k] = self.params[k].logpdf(x)
                        f[k] = np.exp(flog[k])
                        f_new_table += self._m_k[k]*f[k]
                    base_distribution = Gaussian(X=np.zeros((0, len(x))))
                    f_new_topic = np.exp(base_distribution.logpdf(x))
                    f_new_table += self._gamma*f_new_topic
                    f_new_table /= (np.sum(self._m_k) + self._gamma)    # normalization
                    # assert f_new_table > 0.

                    # compute the prior probability of this word sitting at every table
                    # table_probability = np.zeros(len(self._k_dt[document_index]) + 1)
                    table_probability_log = np.zeros(len(self._k_dt[document_index]) + 1, dtype=np.float128)
                    for t in xrange(len(self._k_dt[document_index])):
                        if self._n_dt[document_index][t] > 0:
                            # if there are some words sitting on this table,
                            # the probability will be proportional to the population
                            assigned_topic = self._k_dt[document_index][t]
                            assert(assigned_topic >= 0 or assigned_topic < self._K)
                            # table_probability[t] = f[assigned_topic] * self._n_dt[document_index][t]
                            table_probability_log[t] = flog[assigned_topic] + np.log(self._n_dt[document_index][t])
                        else:
                            # if there are no words sitting on this table
                            # note that it is an old table, hence the prior probability is 0, not self._alpha
                            # table_probability[t] = 0.
                            table_probability_log[t] = np.log(0.)
                    # compute the prob of current word sitting on a new table, the prior probability is self._alpha
                    # sample new w_alpha and s_alpha
                    self._w_alpha = np.array([np.random.beta(self._alpha, np.sum(self._n_dt[j])) for j in xrange(self._D)])
                    self._s_alpha = np.array([np.random.binomial(1, self._alpha/(np.sum(self._n_dt[j])+self._alpha)) for j in xrange(self._D)])
                    # sample alpha from the gamma distribution
                    self._alpha = np.random.gamma(self._a+np.sum(self._m_k)-np.sum(self._s_alpha),
                                                  self._b-np.sum(np.log(self._w_alpha)))
                    # table_probability[len(self._k_dt[document_index])] = self._alpha * f_new_table
                    table_probability_log[len(self._k_dt[document_index])] = np.log(self._alpha) + np.log(f_new_table)

                    # sample a new table this word should sit in
                    table_probability = np.exp(table_probability_log)
                    assert np.sum(table_probability) > 0.
                    table_probability /= np.sum(table_probability)
                    cdf = np.cumsum(table_probability)
                    new_table = np.uint8(np.nonzero(cdf >= np.random.random())[0][0])

                    # assign current word to new table
                    self._t_dv[document_index][word_index] = new_table

                    # if current word sits on a new table, we need to get the topic of that table
                    if new_table == len(self._k_dt[document_index]):
                        # expand the vectors to fit in new table
                        self._n_dt[document_index] = np.hstack((self._n_dt[document_index], np.zeros(1)))
                        self._k_dt[document_index] = np.hstack((self._k_dt[document_index], np.zeros(1)))

                        assert(len(self._n_dt) == self._D and np.all(self._n_dt[document_index] >= 0))
                        assert(len(self._k_dt) == self._D and np.all(self._k_dt[document_index] >= 0))
                        assert(len(self._n_dt[document_index]) == len(self._k_dt[document_index]))

                        # compute the probability of this table having every topic
                        # topic_probability = np.zeros(self._K + 1)
                        topic_probability_log = np.zeros(self._K + 1, dtype=np.float128)
                        for k in xrange(self._K):
                            # topic_probability[k] = self._m_k[k] * f[k]
                            topic_probability_log[k] = np.log(self._m_k[k]) + flog[k]
                        # topic_probability[self._K] = self._gamma * f_new_topic
                        # first sample gamma from a gamma distribution
                        self._w_gamma = np.random.beta(self._gamma, np.sum(self._m_k))
                        self._s_gamma = np.random.binomial(1, self._gamma/(np.sum(self._m_k)+self._gamma))
                        self._gamma = np.random.gamma(self._a+self._K-self._s_gamma, self._b-np.log(self._w_gamma))
                        topic_probability_log[self._K] = np.log(self._gamma) + np.log(f_new_topic)

                        # sample a new topic this table should be assigned
                        topic_probability = np.exp(topic_probability_log)
                        assert np.sum(topic_probability) > 0.
                        topic_probability /= np.sum(topic_probability)
                        cdf = np.cumsum(topic_probability)
                        new_topic = np.uint8(np.nonzero(cdf >= np.random.random())[0][0])

                        self._k_dt[document_index][new_table] = new_topic

                        # if current table requires a new topic
                        if new_topic == self._K:
                            # expand the matrices to fit in new topic
                            self._K += 1
                            self._n_kd = np.vstack((self._n_kd, np.zeros((1, self._D))))
                            assert(self._n_kd.shape == (self._K, self._D))
                            self._k_dt[document_index][-1] = new_topic
                            self._m_k = np.hstack((self._m_k, np.zeros(1)))
                            assert(len(self._m_k) == self._K)
                            self.params[new_topic] = Gaussian(X=np.zeros((0, len(x))))

                    self.update_params(document_index, word_index, +1)

                # sample table assignment, see which topic it should belong to
                for table_index in np.random.permutation(xrange(len(self._k_dt[document_index]))):
                    # if this table is not empty, sample the topic assignment of this table
                    if self._n_dt[document_index][table_index] > 0:
                        old_topic = self._k_dt[document_index][table_index]

                        # find the index of the words sitting on the current table
                        selected_word_index = np.nonzero(self._t_dv[document_index] == table_index)[0]
                        # find all the data associated with current table
                        selected_word = np.array([self._corpus[document_index][term]
                                                  for term in list(selected_word_index)])
                        # remove all the data in this table from their cluster
                        for x in selected_word:
                            self.params[old_topic].rm_point(x)

                        # compute the probability of assigning current table every topic
                        topic_probability = np.zeros(self._K + 1, dtype=np.float128)
                        # first compute the log-likelihood of a new topic
                        topic_probability[self._K] = 0.
                        for x in selected_word:
                            base_distribution = Gaussian(X=np.zeros((0, len(x))))
                            topic_probability[self._K] += base_distribution.logpdf(x)
                        # sample gamma from a gamma distribution
                        self._w_gamma = np.random.beta(self._gamma, np.sum(self._m_k))
                        self._s_gamma = np.random.binomial(1, self._gamma/(np.sum(self._m_k)+self._gamma))
                        self._gamma = np.random.gamma(self._a+self._K-self._s_gamma, self._b-np.log(self._w_gamma))
                        topic_probability[self._K] += np.log(self._gamma)

                        # compute the likelihood of each existing topic
                        for topic_index in xrange(self._K):
                            if topic_index == old_topic:
                                if self._m_k[topic_index] <= 1:
                                    # if current table is the only table assigned to current topic,
                                    # it means this topic is probably less useful or less generalizable to other documents,
                                    # it makes more sense to collapse this topic and hence assign this table to other topic.
                                    topic_probability[topic_index] = -1e300
                                else:
                                    # if there are other tables assigned to current topic
                                    # topic_probability[topic_index] = 0.
                                    for x in selected_word:
                                        assert self.params[topic_index].logpdf(x) > -np.inf
                                        topic_probability[topic_index] += self.params[topic_index].logpdf(x)
                                    # compute the prior if we move this table from this topic
                                    assert self._m_k[topic_index] - 1 > 0
                                    topic_probability[topic_index] += np.log(self._m_k[topic_index] - 1)
                            else:
                                # topic_probability[topic_index] = 0.
                                for x in selected_word:
                                    assert self.params[topic_index].logpdf(x) > -np.inf
                                    topic_probability[topic_index] += self.params[topic_index].logpdf(x)
                                # assert self._m_k[topic_index] > 0
                                topic_probability[topic_index] += np.log(self._m_k[topic_index])

                        # normalize the distribution and sample new topic assignment for this topic
                        # if len(np.where(topic_probability <= -np.inf)[0]) != 0:
                        #    print np.where(topic_probability <= -np.inf)[0]
                        #    print 'number of topics ', self._K
                        # assert np.all(topic_probability > -np.inf)
                        topic_probability = np.exp(topic_probability)
                        assert np.sum(topic_probability) > 0.
                        topic_probability = topic_probability/np.sum(topic_probability)

                        cdf = np.cumsum(topic_probability)
                        rdm = np.random.random()
                        if len(np.nonzero(cdf >= rdm)[0]) == 0:
                            print topic_probability
                        new_topic = np.uint8(np.nonzero(cdf >= rdm)[0][0])

                        # if the table is assigned to a new topic
                        if new_topic != old_topic:
                            # assign this table to new topic
                            self._k_dt[document_index][table_index] = new_topic

                            # if this table starts a new topic, expand all matrix
                            if new_topic == self._K:
                                self._K += 1
                                self._n_kd = np.vstack((self._n_kd, np.zeros((1, self._D))))
                                assert(self._n_kd.shape == (self._K, self._D))
                                self._m_k = np.hstack((self._m_k, np.zeros(1)))
                                assert(len(self._m_k) == self._K)
                                self.params[new_topic] = Gaussian(X=np.zeros((0, len(selected_word[0]))))

                            # adjust the statistics of all model parameter
                            self._m_k[old_topic] -= 1
                            self._m_k[new_topic] += 1
                            self._n_kd[old_topic, document_index] -= self._n_dt[document_index][table_index]
                            self._n_kd[new_topic, document_index] += self._n_dt[document_index][table_index]
                        # add data point to the cluster
                        for x in selected_word:
                            self.params[new_topic].add_point(x)

            # compact all the parameters, including removing unused topics and unused tables
            self.compact_params()
            if self._flag_compute_loglik:
                self.log_likelihoods[iter-1] = self.get_logpdf()
            if iter > 0 and iter % self._snapshot_interval == 0:
                print "sampling in progress %2d%%" % (100 * iter / iteration)
                print "total number of topics %i " % self._K
                # print 'model log-likelihood is ', self.log_likelihoods[iter-1]

    """
    @param document_index: the document index to update
    @param word_index: the word index to update
    @param update: the update amount for this document and this word
    @attention: the update table index and topic index is retrieved from self._t_dv and self._k_dt, so make sure these values were set properly before invoking this function
    """
    def update_params(self, document_index, word_index, update):
        # retrieve the table_id of the current word of current document
        table_id = self._t_dv[document_index][word_index]
        # retrieve the topic_id of the table that current word of current document sit on
        topic_id = self._k_dt[document_index][table_id]
        # get the data at the word_index of the document_index
        x = self._corpus[document_index][word_index]

        self._n_dt[document_index][table_id] += update
        assert(np.all(self._n_dt[document_index] >= 0))
        if update == -1:
            self.params[topic_id].rm_point(x)
        elif update == 1:
            self.params[topic_id].add_point(x)
        self._n_kd[topic_id, document_index] += update
        assert(np.all(self._n_kd >= 0))

        # if current table in current document becomes empty
        if update == -1 and self._n_dt[document_index][table_id] == 0:
            # adjust the table counts
            self._m_k[topic_id] -= 1

        # if a new table is created in current document
        if update == 1 and self._n_dt[document_index][table_id] == 1:
            # adjust the table counts
            self._m_k[topic_id] += 1

        assert(np.all(self._m_k >= 0))
        assert(np.all(self._k_dt[document_index] >= 0))

    """
    """
    def compact_params(self):
        # find unused and used topics
        unused_topics = np.nonzero(self._m_k == 0)[0]
        used_topics = np.nonzero(self._m_k != 0)[0]

        self._K -= len(unused_topics)
        assert(self._K >= 1 and self._K == len(used_topics))

        self._n_kd = np.delete(self._n_kd, unused_topics, axis=0)
        assert(self._n_kd.shape == (self._K, self._D))

        self._m_k = np.delete(self._m_k, unused_topics)
        assert(len(self._m_k) == self._K)

        for k in xrange(len(used_topics)):
            self.params[k] = self.params.pop(used_topics[k])
        for key in xrange(len(self.params)):
            if key >= self._K and key in unused_topics:
                del self.params[key]
        for d in xrange(self._D):
            # find the unused and used tables
            unused_tables = np.nonzero(self._n_dt[d] == 0)[0]
            used_tables = np.nonzero(self._n_dt[d] != 0)[0]

            self._n_dt[d] = np.delete(self._n_dt[d], unused_tables)
            self._k_dt[d] = np.delete(self._k_dt[d], unused_tables)

            # shift down all the table indices of all words in current document
            # @attention: shift the used tables in ascending order only.
            for t in xrange(len(self._n_dt[d])):
                self._t_dv[d][np.nonzero(self._t_dv[d] == used_tables[t])[0]] = t

            # shrink down all the topics indices of all tables in current document
            # @attention: shrink the used topics in ascending order only.
            for k in xrange(self._K):
                self._k_dt[d][np.nonzero(self._k_dt[d] == used_topics[k])[0]] = k

    def get_logpdf(self, data=None):
        if data is None:
            data = self._corpus
        weights, dists = dict2mix(self.params)
        tmp = 0.
        for X in data:
            tmp += all_loglike(X, weights, dists)
        return tmp

    def pickle(self, path, filename):
        with file(path + filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def unpickle(self, path):
        with file(path, 'rb') as f:
            return pickle.load(f)
