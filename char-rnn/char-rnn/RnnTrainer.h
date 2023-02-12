
#include <string>

using namespace std;

class RnnTrainer{
public:
    RnnTrainer();
    ~RnnTrainer();
    string loadData(string filepath);
    void train();

    /*
    void setRnn(Rnn* rnn);
    void setTrainData(std::vector<std::vector<double>>* trainData);
    void setTrainLabel(std::vector<std::vector<double>>* trainLabel);
    void setTestData(std::vector<std::vector<double>>* testData);
    void setTestLabel(std::vector<std::vector<double>>* testLabel);
    void setBatchSize(int batchSize);
    void setEpoch(int epoch);
    void setLearningRate(double learningRate);
    void setLossFunction(LossFunction* lossFunction);
    void setOptimizer(Optimizer* optimizer);
    void setSavePath(std::string savePath);
    void setSaveInterval(int saveInterval);
    void setTestInterval(int testInterval);
    void setPrintInterval(int printInterval);
    void setTestFlag(bool testFlag);
    void setSaveFlag(bool saveFlag);
    void setPrintFlag(bool printFlag);
    void setTestResultPath(std::string testResultPath);
    void setTestResultFlag(bool testResultFlag);
    void setTestResultInterval(int testResultInterval);
    void setTestResultHeader(std::string testResultHeader);
    void setTestResultFooter(std::string testResultFooter);
    void setTestResultFormat(std::string testResultFormat);
    void setTestResultPrecision(int testResultPrecision);
    void setTestResultDelimiter(std::string testResultDelimiter);
    void setTestResultAppendFlag(bool testResultAppendFlag);
    void setTestResultSaveFlag(bool testResultSaveFlag);
    void setTestResultSavePath(std::string testResultSavePath);
    void setTestResultSaveInterval(int testResultSaveInterval);
    void setTestResultSaveHeader(std::string testResultSaveHeader);
    void setTestResultSaveFooter(std::string testResultSaveFooter);
    void setTestResultSaveFormat(std::string testResultSaveFormat);
    void setTestResultSavePrecision(int testResultSavePrecision);
    void setTestResultSaveDelimiter(std::string testResultSaveDelimiter);
    void setTestResultSaveAppendFlag(bool testResultSaveAppendFlag);
    void setTestResultSaveSaveFlag(bool testResultSaveSaveFlag);
    void setTestResultSaveSavePath(std::string testResultSaveSavePath);
    void setTestResultSaveSaveInterval(int testResultSaveSaveInterval);
    void setTestResultSave
    */
};
