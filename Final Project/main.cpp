#include "configurations.h"
#include "initialize.h"
#include "test.h"

#include "app/appdata.h"
#include "extract/extract.h"
#include "quantize/quantize.h"
#include "query/query.h"
#include "utils/score.h"
#include "utils/utils.h"


int main(int argc, char **argv) {

    // Initialize
    // extractAll();
    // quantizeAllData();

    extractAndQuantizeAll();

    runTest();

    return 0;
}
