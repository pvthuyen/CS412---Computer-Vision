#ifndef QUANTIZE_H_INCLUDED
#define QUANTIZE_H_INCLUDED

#include "../configurations.h"


const double deltaSqr = 6250;
const int nKdTree = 8;
const int nChecks = 800;
const int dataKnn = 1;
const int queryKnn = 3;

cvflann::Matrix<double> dataset;
cvflann::Index<cvflann::L2<double>> *treeIndex;

void buildIndex(bool force = false) {
    cvflann::load_from_file(dataset, codebookFile, "clusters");

    cvflann::IndexParams *indexParams;

    if (!force && file_exists)
        indexParams = new cvflann::SavedIndexParams(indexFile);
    else
        indexParams = new cvflann::KDTreeIndexParams(nKdTree);

    treeIndex = new cvflann::Index<cvflann::L2<double>> (dataset, *indexParams);
    treeIndex->buildIndex();
    treeIndex->save(indexFile);
}

void buildBoW(const arma::mat &imageDesc, arma::vec &_weights, arma::uvec &_termID, const string &weightPath, const string &termIDPath, bool force = false) {
    if (!force && file_exists(weightPath)) {
        _weights.load(weightPath);
        _termID.load(termIDPath);
        return;
    }

    double *tmpData = new double[imageDesc.n_elem];
    memcpy(tmpData, imageDesc.memptr(), sizeof(double) * imageDesc.n_elem);
    cvflann::Matrix<double> query(tmpData, imageDesc.n_cols, imageDesc.n_rows);

    cvflann::Matrix<int> indices(new int[query.rows*queryKnn], query.rows, queryKnn);
    cvflann::Matrix<double> dists(new double[query.rows*queryKnn], query.rows, queryKnn);

    treeIndex->knnSearch(query, indices, dists, queryKnn, cvflann::SearchParams(nChecks));

    arma::umat bins(queryKnn, query.rows);
    std::copy(indices.data, indices.data + query.rows * queryKnn, bins.memptr());
    arma::mat sqrDists(queryKnn, query.rows);
    memcpy(sqrDists.memptr(), dists.data, query.rows * queryKnn * sizeof(double));

    _termID = arma::vectorise(bins, 0);

    arma::mat weights = exp(-sqrDists / (2 * deltaSqr));
    weights = weights / arma::repmat(sum(weights, 0), weights.n_rows, 1);
    _weights = arma::vectorise(weights, 0);

    _weights.save(weightPath);
    _termID.save(termIDPath);
    delete[] indices.data;
    delete[] query.data;
    delete[] dists.data;
}

#endif
