#ifndef EXTRACT_H_INCLUDED
#define EXTRACT_H_INCLUDED

#include "../configurations.h"
#include "../utils/hesaff/hesaff.h"

arma::mat inv2x2(arma::mat C) {
    arma::mat den = C.row(0) % C.row(2) - C.row(1) % C.row(1);
    arma::mat S = join_vert(join_vert(C.row(2), - C.row(1)), C.row(0)) / repmat(den.row(0), 3, 1);
    return S;
}

bool vl_ubcread(string file, arma::mat &f, arma::umat &d) {
    int numKeypoints, descLen;
    FILE *fid = fopen(file.c_str(), "r");

    if (fid == NULL)
        return false;

    fscanf(fid, "%d %d", &descLen, &numKeypoints);

    f = arma::mat(5, numKeypoints);

    d = arma::umat(descLen, numKeypoints);

    for (int k = 0; k < numKeypoints; ++k) {
        fscanf(fid, "%lf %lf %lf %lf %lf", &f(0, k), &f(1, k), &f(2, k), &f(3, k), &f(4, k));
        for (int i = 0; i < descLen; ++i)
            fscanf(fid, "%d", &d(i, k));
    }
    fclose(fid);

    f.rows(0, 1) = f.rows(0, 1) + 1;
    f.rows(2, 4) = inv2x2(f.rows(2, 4));

    return 1;
}

void extractFeatures(string imagePath, arma::mat &kpMat, arma::mat &siftMat, const string &kpPath, const string &siftPath, const string &tempPath, bool force = false) {
    if (!force && file_exists(siftPath)) {
        kpMat.load(kpPath);
        siftMat.load(siftPath);
        cout << kpMat.n_rows <<" "<<kpMat.n_cols << endl;
        cout << siftMat.n_rows <<" "<<siftMat.n_cols << endl;
        return;
    }

    string tempFile = "./temp.mat";

    hesaff(imagePath, tempFile);

    arma::mat clip_kp;
    arma::umat clip_desc;

    if (!vl_ubcread(tempFile, clip_kp, clip_desc)) {
        clip_kp = arma::mat(5, 0);
        clip_desc = arma::umat(128, 0);
    }

    arma::mat sift = arma::conv_to<arma::mat>::from(clip_desc);

    arma::mat sqrt_desc = sqrt(sift / repmat(sum(sift), 128, 1));

    kpMat = clip_kp;
    siftMat = sqrt_desc;
    kpMat.save(kpPath);
    siftMat.save(siftPath);
}

#endif
