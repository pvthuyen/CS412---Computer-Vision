#ifndef INITIALIZE_H_INCLUDED
#define INITIALIZE_H_INCLUDED


#include "configurations.h"
#include "app/appdata.h"
#include "extract/extract.h"
#include "quantize/quantize.h"


void extractAndQuantizeAll() {

    AppData *app = AppData::getInstance();

//    Get list of image files
    DIR *dir = opendir(dataFolder.c_str());
    while (dirent *pdir = readdir(dir)) {
        string fName = pdir->d_name;
        if (fName[0] == '.') continue;

        app->path.push_back(fName);
    }

    debugInfo("Extracting features");

//    Extract features
    app->path.shrink_to_fit();
    // app->kp.reserve(app->path.size());
    // app->sift.reserve(app->path.size());

    buildIndex(true);

    for (int i = 0; i < app->path.size(); ++i) {
        string imgPath = app->path[i];
        debugVar(imgPath);

        string tmp = imgPath;
        tmp.replace(tmp.size() - 3, 3, "mat");

        string kpPath = kpFolder + "/" + tmp;
        string siftPath = siftFolder + "/" + tmp;
        string tempPath = tempFolder + "/" + tmp;
        string weightPath = weightFolder + "/" + tmp;
        string termIDPath = termIDFolder + "/" + tmp;

        imgPath = dataFolder + "/" + imgPath;

        debugVar(imgPath);
        debugVar(kpPath);
        debugVar(siftPath);
        debugVar(weightPath);
        debugVar(termIDPath);

        arma::mat _kp, _sift;
        extractFeatures(imgPath, _kp, _sift, kpPath, siftPath, tempPath, true);

        arma::vec _weights;
        arma::uvec _termID;

        buildBoW(_sift, _weights, _termID, weightPath, termIDPath, true);
        
//        Insert to inverted index
        app->ivt.add(_weights, _termID, i);
    }

    //    Build TFIDF
    app->ivt.buildTfidf();
}

#endif
